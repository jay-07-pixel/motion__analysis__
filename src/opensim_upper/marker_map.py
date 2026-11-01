"""Explicit upper-body marker map.

Targets and local coordinates are read from ``Markers_LSTM.xml``. Names that
are not in that file are rejected. Observed wrists stay unmapped: the file
has no ``RWrist`` or ``LWrist`` marker, and the wrist study markers are
estimated radius stations, not wrist joint centres.

The eight LSTM outputs stay labeled as estimates. C7, pelvis, ASIS, PSIS,
shoulder-study, and hip markers are not added.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.opensim_upper.environment import POSE2SIM_MODEL
from src.opensim_upper.model_xml import UPPER_MODEL_MARKERS, inspect_model

# MediaPipe side names for markers that already exist in Markers_LSTM.xml.
# RWrist and LWrist are OpenPose names in the LSTM feature map, but they are
# not model markers, so they are absent here.
_OBSERVED_TARGETS = {
    "right_shoulder": "RShoulder",
    "left_shoulder": "LShoulder",
    "right_elbow": "RElbow",
    "left_elbow": "LElbow",
}

_PREDICTED_TARGETS = {
    "r_lelbow_study": "r_lelbow_study",
    "r_melbow_study": "r_melbow_study",
    "r_lwrist_study": "r_lwrist_study",
    "r_mwrist_study": "r_mwrist_study",
    "L_lelbow_study": "L_lelbow_study",
    "L_melbow_study": "L_melbow_study",
    "L_lwrist_study": "L_lwrist_study",
    "L_mwrist_study": "L_mwrist_study",
}

UNMAPPED_OBSERVED = {
    "right_wrist": (
        "Markers_LSTM.xml has no RWrist marker. The observed wrist joint is "
        "not mapped to r_lwrist_study or r_mwrist_study."
    ),
    "left_wrist": (
        "Markers_LSTM.xml has no LWrist marker. The observed wrist joint is "
        "not mapped to L_lwrist_study or L_mwrist_study."
    ),
}

_PLACEMENT = "copied_unchanged_from_Markers_LSTM_xml"
_STUDY_WEIGHT_NOTE = (
    "This number is the IKMarkerTask weight of this marker name in "
    "IK_Setup_Pose2Sim_withoutHands_LSTM.xml. The rest of that full-body "
    "task set is not used, and inverse kinematics is not run."
)
_NO_WEIGHT_NOTE = (
    "This marker is defined in Markers_LSTM.xml and is not an IKMarkerTask "
    "in the official full-body setup. No weight was invented."
)
_MARKER_NOTES = {
    "RShoulder": (
        "Existing torso marker. Not a glenohumeral joint centre and not r_shoulder_study."
    ),
    "LShoulder": (
        "Existing torso marker. Not a glenohumeral joint centre and not L_shoulder_study."
    ),
    "RElbow": (
        "Existing humerus marker. Not an epicondyle; those estimates are "
        "r_lelbow_study and r_melbow_study."
    ),
    "LElbow": (
        "Existing humerus marker. Not an epicondyle; those estimates are "
        "L_lelbow_study and L_melbow_study."
    ),
}


class UnknownMarkerError(KeyError):
    """A requested marker name is not one of the supported model markers."""


def study_marker_weights(markers_path: Path | None = None) -> dict[str, float | None]:
    """Read weights for the eight study markers only. Other IK tasks are ignored."""
    import xml.etree.ElementTree as ET

    path = Path(markers_path or POSE2SIM_MODEL).parent / "IK_Setup_Pose2Sim_withoutHands_LSTM.xml"
    weights: dict[str, float | None] = {name: None for name in _PREDICTED_TARGETS}
    if not path.is_file():
        return weights
    root = ET.parse(path).getroot()
    for task in root.iter():
        name = task.get("name")
        if name not in weights or not task.tag.endswith("IKMarkerTask"):
            continue
        text = task.findtext("weight")
        if text is None:
            continue
        weights[name] = float(text)
    return weights


def marker_mapping(model_path: Path | None = None, markers_path: Path | None = None) -> dict:
    """Mapping table. Local coordinates are the XML values, not fitted values."""
    info = inspect_model(model_path, markers_path)
    if info["wrist_joint_marker_defined"]:
        raise ValueError("Unexpected RWrist or LWrist marker. The mapping was not built.")
    defined = info["upper_markers"]
    weights = study_marker_weights(markers_path)
    rows = []
    for source, target in {**_OBSERVED_TARGETS, **_PREDICTED_TARGETS}.items():
        if target not in defined:
            raise UnknownMarkerError(target)
        spec = defined[target]
        if spec["location_m"] is None:
            raise ValueError(f"{target} has no location in Markers_LSTM.xml.")
        parent = str(spec["parent"])
        body = parent.rstrip("/").rsplit("/", 1)[-1]
        predicted = source in _PREDICTED_TARGETS
        weight = weights.get(target) if predicted else None
        rows.append(
            {
                "source": source,
                "target": target,
                "body": body,
                "parent_frame": parent,
                "location_m": list(spec["location_m"]),
                "fixed": bool(spec["fixed"]),
                "placement_method": _PLACEMENT,
                "source_type": "predicted" if predicted else "observed",
                "estimated_not_palpated": predicted,
                "marker_weight": weight,
                "weight_note": _STUDY_WEIGHT_NOTE if predicted else _NO_WEIGHT_NOTE,
                "official_ik_task_present": predicted and weight is not None,
                "ik_applied": False,
                "note": (
                    "LSTM study marker. Estimated, not palpated."
                    if predicted
                    else _MARKER_NOTES[target]
                ),
                "validity": (
                    "Source point must be finite metres from a validated registration. "
                    "Target name and location_m are taken from Markers_LSTM.xml and are not adjusted."
                ),
            }
        )
    return {
        "mapped": rows,
        "unmapped_observed": [
            {"source": name, "target": None, "reason": reason, "source_type": "observed"}
            for name, reason in UNMAPPED_OBSERVED.items()
        ],
        "not_added": [
            "C7_study",
            "r_shoulder_study",
            "L_shoulder_study",
            "r.ASIS_study",
            "L.ASIS_study",
            "r.PSIS_study",
            "L.PSIS_study",
            "RHJC_study",
            "LHJC_study",
            "RWrist",
            "LWrist",
        ],
        "placement_adjusted": False,
        "full_body_ik_configuration_used": False,
    }


def target_names(mapping: dict | None = None) -> tuple[str, ...]:
    table = mapping if mapping is not None else marker_mapping()
    return tuple(row["target"] for row in table["mapped"])


def require_known_target(name: str, mapping: dict | None = None) -> dict:
    """Return the row for a model marker name, or raise."""
    table = mapping if mapping is not None else marker_mapping()
    for row in table["mapped"]:
        if row["target"] == name:
            return row
    raise UnknownMarkerError(name)


def map_static_landmarks(transformed: dict, mapping: dict | None = None) -> dict:
    """Attach transformed landmarks to the explicit map. Unknown names raise."""
    if transformed.get("coord_frame") != "opensim_ground_metre":
        raise ValueError("Landmarks must already be in the registered model frame.")
    if transformed.get("model_registration") != "applied":
        raise ValueError("Registration was not applied. An unregistered pose is not mapped.")
    table = mapping if mapping is not None else marker_mapping()
    by_source = {row["source"]: row for row in table["mapped"]}
    allowed = set(by_source) | set(UNMAPPED_OBSERVED)
    assigned = []
    for name, point in transformed["landmarks"].items():
        if name not in allowed:
            raise UnknownMarkerError(name)
        model_m = [float(point["x_m"]), float(point["y_m"]), float(point["z_m"])]
        if not np.isfinite(model_m).all():
            raise ValueError(f"{name} is not finite after registration.")
        if name in UNMAPPED_OBSERVED:
            assigned.append(
                {
                    "source": name,
                    "target": None,
                    "mapped": False,
                    "reason": UNMAPPED_OBSERVED[name],
                    "source_type": "observed",
                    "estimated_not_palpated": False,
                    "model_m": model_m,
                    "placement_method": None,
                    "location_m": None,
                    "marker_weight": None,
                }
            )
            continue
        row = by_source[name]
        expected = "estimated_lstm_study" if row["source_type"] == "predicted" else "observed"
        if point.get("source") != expected:
            raise ValueError(
                f"{name} has source {point.get('source')!r}; the map requires {expected}."
            )
        assigned.append(
            {
                **row,
                "mapped": True,
                "model_m": model_m,
                "reason": None,
            }
        )
    mapped_targets = [row["target"] for row in assigned if row["mapped"]]
    if set(mapped_targets) != set(UPPER_MODEL_MARKERS):
        raise ValueError("Mapped targets do not match the supported upper-body marker set.")
    return {
        "assignments": assigned,
        "n_mapped": len(mapped_targets),
        "n_unmapped": sum(1 for row in assigned if not row["mapped"]),
        "placement_adjusted": False,
        "inverse_kinematics": False,
        "trc_written": False,
    }
