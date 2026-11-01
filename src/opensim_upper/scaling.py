"""Segment lengths, and a sized copy of the camera-frame upper-body model.

``subject_lengths`` only records distances. It does not write a model.

``write_scaled_capture_model`` writes a new ``.osim`` for one recording.
The generic camera model is not edited. Measurements:

- shoulder-marker width is recorded and not applied. Those camera points sit
  inward of the outer shoulder, so they are not the tape width across the shoulders
- upper-arm length, from the elbow marker to the shoulder joint estimated
  along the shoulder line; a difference under 5 percent is left unchanged,
  because the shoulder marker is not the joint centre
- forearm length, from the elbow marker to the wrist study markers, compared
  with that same distance on the generic model at its default elbow pose

Torso height, torso depth, the head, and the hands are not measured here.
Mass and inertia are not changed. Observed wrist joints stay unmapped.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.opensim_upper.camera_model import CAMERA_MODEL, SOURCE_MODEL, rotation_from_body_fixed_xyz
from src.opensim_upper.model_xml import UPPER_MODEL_MARKERS, _local
from src.opensim_upper.trc import read_trc

PAIRS = (
    ("humerus_r", "right_shoulder", "right_elbow", "humerus_r", "observed joint-to-joint distance"),
    ("humerus_l", "left_shoulder", "left_elbow", "humerus_l", "observed joint-to-joint distance"),
    ("forearm_r", "right_elbow", "right_wrist", "radius_r, ulna_r", "observed joint-to-joint distance"),
    ("forearm_l", "left_elbow", "left_wrist", "radius_l, ulna_l", "observed joint-to-joint distance"),
    ("shoulder_width", "left_shoulder", "right_shoulder", "torso Z only", "mediolateral torso width, not applied"),
)

UNSCALED_BODIES = (
    "pelvis",
    "sacrum",
    "femur_r",
    "femur_l",
    "tibia_r",
    "tibia_l",
    "talus_r",
    "talus_l",
    "calcn_r",
    "calcn_l",
    "toes_r",
    "toes_l",
    "torso",
    "head",
    "hand_r",
    "hand_l",
)


def _point(summary: dict, name: str) -> np.ndarray:
    row = summary["landmarks"][name]
    return np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=np.float64)


def subject_lengths(summary: dict) -> dict:
    """Segment lengths in metres from the static camera-frame medians."""
    if summary.get("units") != "metres":
        raise ValueError("Static pose units must be metres")
    if summary.get("coord_frame") != "camera_3d_metre":
        raise ValueError("Static pose must stay in camera metres")
    lengths = {}
    for key, a, b, bodies, note in PAIRS:
        distance = float(np.linalg.norm(_point(summary, a) - _point(summary, b)))
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError(f"{key} distance is not positive")
        lengths[key] = {
            "distance_m": distance,
            "from": a,
            "to": b,
            "bodies": bodies,
            "note": note,
            "applied_to_model": False,
            "scale_factor": None,
            "reason_not_applied": (
                "Generic model marker distance depends on the default pose. "
                "No scaled model is written in this milestone."
            ),
        }
    study = {}
    for side, elbow_name, lat, med in (
        ("right", "right_elbow", "r_lelbow_study", "r_melbow_study"),
        ("left", "left_elbow", "L_lelbow_study", "L_melbow_study"),
    ):
        study[f"{side}_epicondyle_pair_m"] = float(
            np.linalg.norm(_point(summary, lat) - _point(summary, med))
        )
        study[f"{side}_elbow_to_lateral_study_m"] = float(
            np.linalg.norm(_point(summary, elbow_name) - _point(summary, lat))
        )
    return {
        "units": "metres",
        "lengths": lengths,
        "study_marker_distances_not_used_as_bone_length": study,
        "bodies_left_at_model_default": list(UNSCALED_BODIES),
        "participant_height_m": summary.get("participant_height_m"),
        "participant_mass_kg": summary.get("participant_mass_kg"),
        "height_used_for_scale": False,
        "mass_used_for_scale": False,
        "mass_note": (
            "Participant mass can later set the ScaleTool subject mass. "
            "It does not by itself scale segment lengths."
        ),
    }


_MIN_SAMPLES = 30
_FACTOR_MIN = 0.70
_FACTOR_MAX = 1.40
# The shoulder marker sits about 1.6 cm off the glenohumeral joint in the
# directions this estimate does not correct. A smaller upper-arm change is noise.
_HUMERUS_KEEP_GENERIC = 0.05
_SIDE_DISAGREEMENT = 0.10
_GENERIC_NAME = "UpperBody_CameraFrame"


def _find(root: ET.Element, tag: str, name: str) -> ET.Element:
    for element in root.iter():
        if _local(element.tag) == tag and element.get("name") == name:
            return element
    raise ValueError(f"The model has no {tag} named {name}.")


def _child_text(element: ET.Element, tag: str) -> str:
    for child in element:
        if _local(child.tag) == tag and child.text:
            return child.text.strip()
    raise ValueError(f"Missing {tag}.")


def _vec(text: str) -> np.ndarray:
    values = np.array([float(part) for part in text.split()], dtype=float)
    if values.shape != (3,):
        raise ValueError(f"Expected 3 values, got {text!r}.")
    return values


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.16g}" for value in values)


def _frame(joint: ET.Element, name: str) -> tuple[np.ndarray, np.ndarray]:
    for element in joint.iter():
        if _local(element.tag) == "PhysicalOffsetFrame" and element.get("name") == name:
            return _vec(_child_text(element, "orientation")), _vec(_child_text(element, "translation"))
    raise ValueError(f"The joint has no offset frame {name}.")


def _default_coordinate(root: ET.Element, name: str) -> float:
    coordinate = _find(root, "Coordinate", name)
    return float(_child_text(coordinate, "default_value"))


def _rz(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _pin_to_parent(
    parent_orientation: np.ndarray,
    parent_translation: np.ndarray,
    child_orientation: np.ndarray,
    angle: float,
    point: np.ndarray,
) -> np.ndarray:
    """Point in the parent body. OpenSim pin joints rotate about the frame Z axis."""
    rotation = (
        rotation_from_body_fixed_xyz(parent_orientation)
        @ _rz(angle)
        @ rotation_from_body_fixed_xyz(child_orientation).T
    )
    return rotation @ np.asarray(point, dtype=float) + parent_translation


def _arm_geometry(root: ET.Element) -> dict:
    elbow = _find(root, "PinJoint", "elbow_r")
    forearm = _find(root, "PinJoint", "radioulnar_r")
    shoulder = _find(root, "CustomJoint", "acromial_r")
    elbow_parent_orientation, elbow_parent_translation = _frame(elbow, "humerus_r_offset")
    elbow_child_orientation, _elbow_child_translation = _frame(elbow, "ulna_r_offset")
    radio_parent_orientation, radio_parent_translation = _frame(forearm, "ulna_r_offset")
    radio_child_orientation, _radio_child_translation = _frame(forearm, "radius_r_offset")
    _shoulder_orientation, shoulder_translation = _frame(shoulder, "torso_offset")
    humerus_child = _frame(shoulder, "humerus_r_offset")[1]
    if float(np.linalg.norm(humerus_child)) > 1e-8:
        raise ValueError("The humerus joint frame is not at the humerus origin.")
    elbow_marker = _vec(_child_text(_find(root, "Marker", "RElbow"), "location"))
    lateral = _vec(_child_text(_find(root, "Marker", "r_lwrist_study"), "location"))
    medial = _vec(_child_text(_find(root, "Marker", "r_mwrist_study"), "location"))
    return {
        "elbow_marker": elbow_marker,
        "elbow_parent_orientation": elbow_parent_orientation,
        "elbow_parent_translation": elbow_parent_translation,
        "elbow_child_orientation": elbow_child_orientation,
        "radio_parent_orientation": radio_parent_orientation,
        "radio_parent_translation": radio_parent_translation,
        "radio_child_orientation": radio_child_orientation,
        "wrist_midpoint": 0.5 * (lateral + medial),
        "elbow_angle": _default_coordinate(root, "elbow_flex_r"),
        "forearm_angle": _default_coordinate(root, "pro_sup_r"),
        "shoulder_right": _vec(_child_text(_find(root, "Marker", "RShoulder"), "location")),
        "shoulder_left": _vec(_child_text(_find(root, "Marker", "LShoulder"), "location")),
        "glenohumeral_right": shoulder_translation,
        "humerus_length_m": float(np.linalg.norm(elbow_marker)),
    }


def forearm_reference_distance(
    model_path: Path,
    forearm_scale: float = 1.0,
    humerus_scale: float = 1.0,
) -> float:
    """Elbow-marker to wrist-study midpoint at the model's default elbow pose."""
    geometry = _arm_geometry(ET.parse(Path(model_path)).getroot())
    return _forearm_distance(geometry, forearm_scale, humerus_scale)


def _forearm_distance(geometry: dict, forearm_scale: float, humerus_scale: float) -> float:
    humerus_factor = np.array([1.0, humerus_scale, 1.0])
    forearm_factor = np.array([1.0, forearm_scale, 1.0])
    elbow_marker = geometry["elbow_marker"] * humerus_factor
    wrist = geometry["wrist_midpoint"] * forearm_factor
    in_ulna = _pin_to_parent(
        geometry["radio_parent_orientation"],
        geometry["radio_parent_translation"] * forearm_factor,
        geometry["radio_child_orientation"],
        geometry["forearm_angle"],
        wrist,
    )
    in_humerus = _pin_to_parent(
        geometry["elbow_parent_orientation"],
        geometry["elbow_parent_translation"] * humerus_factor,
        geometry["elbow_child_orientation"],
        geometry["elbow_angle"],
        in_ulna,
    )
    return float(np.linalg.norm(in_humerus - elbow_marker))


def _median(values: list[float]) -> float | None:
    if len(values) < _MIN_SAMPLES:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _mean_sides(right: float | None, left: float | None) -> tuple[float | None, str | None]:
    if right is None and left is None:
        return None, "Fewer than 30 frames had both points."
    if right is None or left is None:
        return right if right is not None else left, None
    average = 0.5 * (right + left)
    if abs(right - left) / average > _SIDE_DISAGREEMENT:
        return None, "The right and left measurements disagree by more than 10 percent."
    return average, None


def _factor_in_range(factor: float) -> bool:
    return _FACTOR_MIN <= factor <= _FACTOR_MAX


def _solve_forearm_scale(geometry: dict, target_m: float, humerus_scale: float) -> tuple[float | None, str]:
    low = _forearm_distance(geometry, _FACTOR_MIN, humerus_scale)
    high = _forearm_distance(geometry, _FACTOR_MAX, humerus_scale)
    if high < low:
        raise ValueError("Forearm length did not increase when the bone was lengthened.")
    if not low <= target_m <= high:
        return None, (
            f"The measured forearm length {target_m:.3f} m is outside the range a "
            f"{_FACTOR_MIN:.2f} to {_FACTOR_MAX:.2f} scale can reach."
        )
    lo, hi = _FACTOR_MIN, _FACTOR_MAX
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _forearm_distance(geometry, mid, humerus_scale) < target_m:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), "applied"


def measure_body_scales(trc_path: Path, model_path: Path) -> dict:
    """Scale factors for one camera-metre TRC. Nothing is written."""
    root = ET.parse(Path(model_path)).getroot()
    model = root.find("Model")
    if model is None or model.get("name") != _GENERIC_NAME:
        raise ValueError("Scaling starts from the generic camera-frame model.")
    geometry = _arm_geometry(root)
    shoulder_model = float(np.linalg.norm(geometry["shoulder_right"] - geometry["shoulder_left"]))
    toward_left = geometry["shoulder_left"] - geometry["shoulder_right"]
    toward_left = toward_left / np.linalg.norm(toward_left)
    medial_generic = float(np.dot(geometry["glenohumeral_right"] - geometry["shoulder_right"], toward_left))
    if medial_generic <= 0:
        raise ValueError("The shoulder joint is not medial to the shoulder marker.")

    widths: list[float] = []
    humerus_right: list[float] = []
    humerus_left: list[float] = []
    forearm_right: list[float] = []
    forearm_left: list[float] = []
    for frame in read_trc(trc_path)["frames"]:
        points = frame["points"]
        right = points.get("RShoulder")
        left = points.get("LShoulder")
        if right is not None and left is not None:
            span = float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
            if span > 0.05:
                widths.append(span)
        elbow_r = points.get("RElbow")
        elbow_l = points.get("LElbow")
        wrists_r = (points.get("r_lwrist_study"), points.get("r_mwrist_study"))
        wrists_l = (points.get("L_lwrist_study"), points.get("L_mwrist_study"))
        if elbow_r is not None and all(item is not None for item in wrists_r):
            midpoint = 0.5 * (np.asarray(wrists_r[0]) + np.asarray(wrists_r[1]))
            forearm_right.append(float(np.linalg.norm(np.asarray(elbow_r) - midpoint)))
        if elbow_l is not None and all(item is not None for item in wrists_l):
            midpoint = 0.5 * (np.asarray(wrists_l[0]) + np.asarray(wrists_l[1]))
            forearm_left.append(float(np.linalg.norm(np.asarray(elbow_l) - midpoint)))

    width = _median(widths)
    torso_factor = None if width is None else width / shoulder_model
    # The camera shoulder points are not the outer tips of the shoulders.
    # A tape across the outer shoulders matches the generic marker width,
    # so that width is kept.
    torso_applied = False
    torso_reason = (
        "The camera shoulder points sit inward of the outer shoulders. "
        "Their distance is recorded and is not used to narrow the chest."
    )
    medial = medial_generic

    for frame in read_trc(trc_path)["frames"]:
        points = frame["points"]
        right = points.get("RShoulder")
        left = points.get("LShoulder")
        if right is None or left is None:
            continue
        right_v = np.asarray(right, dtype=float)
        left_v = np.asarray(left, dtype=float)
        span = float(np.linalg.norm(left_v - right_v))
        if span <= 0.05:
            continue
        direction = (left_v - right_v) / span
        if points.get("RElbow") is not None:
            joint = right_v + direction * medial
            humerus_right.append(float(np.linalg.norm(np.asarray(points["RElbow"]) - joint)))
        if points.get("LElbow") is not None:
            joint = left_v - direction * medial
            humerus_left.append(float(np.linalg.norm(np.asarray(points["LElbow"]) - joint)))

    humerus_subject, humerus_note = _mean_sides(_median(humerus_right), _median(humerus_left))
    humerus_factor = None if humerus_subject is None else humerus_subject / geometry["humerus_length_m"]
    if humerus_note:
        humerus_applied = False
        humerus_reason = humerus_note
    elif humerus_factor is None:
        humerus_applied = False
        humerus_reason = "The upper-arm length could not be measured."
    elif abs(humerus_factor - 1.0) <= _HUMERUS_KEEP_GENERIC:
        humerus_applied = False
        humerus_reason = (
            "The measured upper arm is within 5 percent of the generic bone. "
            "The shoulder marker is not the joint centre, so a smaller difference is not applied."
        )
    elif not _factor_in_range(humerus_factor):
        humerus_applied = False
        humerus_reason = (
            f"The upper-arm scale {humerus_factor:.3f} is outside {_FACTOR_MIN:.2f} to {_FACTOR_MAX:.2f}."
        )
    else:
        humerus_applied = True
        humerus_reason = "applied"

    forearm_subject, forearm_note = _mean_sides(_median(forearm_right), _median(forearm_left))
    forearm_model = _forearm_distance(geometry, 1.0, humerus_factor if humerus_applied else 1.0)
    if forearm_note or forearm_subject is None:
        forearm_factor = None
        forearm_applied = False
        forearm_reason = forearm_note or "The forearm length could not be measured."
    else:
        forearm_factor, forearm_reason = _solve_forearm_scale(
            geometry,
            forearm_subject,
            humerus_factor if humerus_applied else 1.0,
        )
        forearm_applied = forearm_factor is not None

    return {
        "clinical_accuracy": False,
        "biomechanical_reliability": False,
        "mass_changed": False,
        "inertia_changed": False,
        "generic_model_unchanged": True,
        "torso": {
            "axis": "mediolateral",
            "applied": torso_applied,
            "factor": torso_factor if torso_applied else 1.0,
            "measured_factor": torso_factor,
            "subject_m": width,
            "model_m": shoulder_model,
            "reason": torso_reason,
        },
        "humerus": {
            "axis": "long",
            "applied": humerus_applied,
            "factor": humerus_factor if humerus_applied else 1.0,
            "measured_factor": humerus_factor,
            "subject_m": humerus_subject,
            "model_m": geometry["humerus_length_m"],
            "right_m": _median(humerus_right),
            "left_m": _median(humerus_left),
            "reason": humerus_reason,
        },
        "forearm": {
            "axis": "long",
            "applied": forearm_applied,
            "factor": forearm_factor if forearm_applied else 1.0,
            "measured_factor": forearm_factor,
            "subject_m": forearm_subject,
            "model_m": forearm_model,
            "right_m": _median(forearm_right),
            "left_m": _median(forearm_left),
            "reason": forearm_reason,
            "markers": "RElbow/LElbow to the wrist study markers",
        },
    }


def _statement(measured: dict) -> str:
    torso = measured["torso"]
    humerus = measured["humerus"]
    forearm = measured["forearm"]
    sentences = []
    if torso["subject_m"] is not None:
        sentences.append(
            f"The chest was left at the generic width of {torso['model_m']:.3f} m. "
            f"The camera shoulder points are {torso['subject_m']:.3f} m apart, "
            "which is inward of the outer shoulders, so that distance was not used."
        )
    else:
        sentences.append("The chest width was left at the generic size.")
    if forearm["applied"] and forearm["subject_m"] is not None:
        sentences.append(
            f"The forearms were changed from {forearm['model_m']:.3f} m to {forearm['subject_m']:.3f} m, "
            "using the elbow point and the wrist estimate points."
        )
    else:
        sentences.append("The forearm length was left at the generic size.")
    if humerus["applied"] and humerus["subject_m"] is not None:
        sentences.append(
            f"The upper arms were changed from {humerus['model_m']:.3f} m to {humerus['subject_m']:.3f} m."
        )
    elif humerus["subject_m"] is not None:
        sentences.append(
            f"The upper arms measured {humerus['subject_m']:.3f} m, close to the generic "
            f"{humerus['model_m']:.3f} m, so they were not changed."
        )
    else:
        sentences.append("The upper-arm length was not changed.")
    sentences.append(
        "Inverse kinematics loads this sized copy. The generic model file is not loaded. "
        "This sizing is a model fit, not a clinical measurement."
    )
    return " ".join(sentences)


def _scale_body(root: ET.Element, body_name: str, factors: np.ndarray) -> None:
    body = _find(root, "Body", body_name)
    for child in body:
        if _local(child.tag) == "mass_center" and child.text:
            child.text = _fmt(_vec(child.text) * factors)
    for mesh in body.iter():
        if _local(mesh.tag) != "Mesh":
            continue
        for child in mesh:
            if _local(child.tag) == "scale_factors" and child.text:
                child.text = _fmt(_vec(child.text) * factors)
    for marker in root.iter():
        if _local(marker.tag) != "Marker":
            continue
        parent = _child_text(marker, "socket_parent_frame")
        if parent.rstrip("/").rsplit("/", 1)[-1] != body_name:
            continue
        for child in marker:
            if _local(child.tag) == "location" and child.text:
                child.text = _fmt(_vec(child.text) * factors)
    for frame in root.iter():
        if _local(frame.tag) != "PhysicalOffsetFrame":
            continue
        parent = _child_text(frame, "socket_parent")
        if parent.rstrip("/").rsplit("/", 1)[-1] != body_name:
            continue
        for child in frame:
            if _local(child.tag) == "translation" and child.text:
                child.text = _fmt(_vec(child.text) * factors)


def spacing_mapping(model_path: Path) -> dict:
    """Marker locations from the model file that inverse kinematics actually loads."""
    root = ET.parse(Path(model_path)).getroot()
    rows = []
    for marker in root.iter():
        if _local(marker.tag) != "Marker" or not marker.get("name"):
            continue
        name = str(marker.get("name"))
        if name not in UPPER_MODEL_MARKERS:
            continue
        parent = _child_text(marker, "socket_parent_frame")
        rows.append(
            {
                "target": name,
                "body": parent.rstrip("/").rsplit("/", 1)[-1],
                "location_m": [float(value) for value in _vec(_child_text(marker, "location"))],
            }
        )
    found = {row["target"] for row in rows}
    missing = [name for name in UPPER_MODEL_MARKERS if name not in found]
    if missing:
        raise ValueError(f"The model is missing markers: {missing}")
    return {"mapped": rows}


def write_scaled_capture_model(source: Path, dest: Path, trc_path: Path) -> dict:
    """Write a sized copy. Refuses to overwrite the generic models."""
    source_path = Path(source).resolve()
    dest_path = Path(dest).resolve()
    protected = {source_path, CAMERA_MODEL.resolve(), SOURCE_MODEL.resolve()}
    if dest_path in protected:
        raise ValueError("Refusing to overwrite a generic model with a sized copy.")
    measured = measure_body_scales(trc_path, source_path)
    factors = {
        "torso": np.array([1.0, 1.0, measured["torso"]["factor"] if measured["torso"]["applied"] else 1.0]),
        "humerus_r": np.array([1.0, measured["humerus"]["factor"] if measured["humerus"]["applied"] else 1.0, 1.0]),
        "humerus_l": np.array([1.0, measured["humerus"]["factor"] if measured["humerus"]["applied"] else 1.0, 1.0]),
        "ulna_r": np.array([1.0, measured["forearm"]["factor"] if measured["forearm"]["applied"] else 1.0, 1.0]),
        "ulna_l": np.array([1.0, measured["forearm"]["factor"] if measured["forearm"]["applied"] else 1.0, 1.0]),
        "radius_r": np.array([1.0, measured["forearm"]["factor"] if measured["forearm"]["applied"] else 1.0, 1.0]),
        "radius_l": np.array([1.0, measured["forearm"]["factor"] if measured["forearm"]["applied"] else 1.0, 1.0]),
    }
    tree = ET.parse(source_path)
    root = tree.getroot()
    model = root.find("Model")
    if model is None:
        raise ValueError("The model file has no Model element.")
    for body_name, scale in factors.items():
        if float(np.max(np.abs(scale - 1.0))) < 1e-8:
            continue
        _scale_body(root, body_name, scale)
    model.set("name", "UpperBody_CameraFrame_Scaled")
    credits = model.find("credits")
    if credits is not None and credits.text:
        credits.text = (
            credits.text.strip()
            + " Sized copy for one recording. The generic camera model was not edited."
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest_path, encoding="utf-8", xml_declaration=True)
    model_path = dest_path
    statement = _statement(measured)
    payload = {
        **measured,
        "statement": statement,
        "source_model": str(source_path),
        "model_path": str(model_path),
        "sized_copy_written": True,
        "ik_uses_generic_model": False,
    }
    report_path = dest_path.parent / "scale_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["scale_report"] = str(report_path)
    return payload
