"""Read the installed Pose2Sim model XML. Does not load OpenSim and does not copy the file."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.opensim_upper.environment import POSE2SIM_MARKERS, POSE2SIM_MODEL

# Bodies this milestone is allowed to talk about. Pelvis and legs stay at the file defaults.
UPPER_BODIES = (
    "torso",
    "humerus_r",
    "ulna_r",
    "radius_r",
    "hand_r",
    "humerus_l",
    "ulna_l",
    "radius_l",
    "hand_l",
)

# Model markers that have a live upper-body source. Study shoulder and C7 do not.
UPPER_MODEL_MARKERS = (
    "RShoulder",
    "LShoulder",
    "RElbow",
    "LElbow",
    "r_lelbow_study",
    "r_melbow_study",
    "r_lwrist_study",
    "r_mwrist_study",
    "L_lelbow_study",
    "L_melbow_study",
    "L_lwrist_study",
    "L_mwrist_study",
)

# Present in Markers_LSTM.xml and required by the official full-body scale file.
# They are not produced by the upper LSTM and are not used here.
EXCLUDED_FULL_BODY_MARKERS = (
    "C7_study",
    "r_shoulder_study",
    "L_shoulder_study",
    "r.ASIS_study",
    "L.ASIS_study",
    "r.PSIS_study",
    "L.PSIS_study",
    "RHJC_study",
    "LHJC_study",
)

SHOULDER_COORDINATES = ("arm_flex_r", "arm_add_r", "arm_rot_r", "arm_flex_l", "arm_add_l", "arm_rot_l")
ELBOW_COORDINATES = ("elbow_flex_r", "elbow_flex_l", "pro_sup_r", "pro_sup_l")


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def inspect_model(model_path: Path | None = None, markers_path: Path | None = None) -> dict:
    """Bodies, upper-limb coordinates, and marker parents from the XML text."""
    model_path = Path(model_path or POSE2SIM_MODEL)
    markers_path = Path(markers_path or POSE2SIM_MARKERS)
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose2Sim model not found: {model_path}")
    if not markers_path.is_file():
        raise FileNotFoundError(f"Pose2Sim marker file not found: {markers_path}")

    model = ET.parse(model_path).getroot()
    model_node = model.find("Model")
    if model_node is None:
        raise ValueError(f"{model_path} has no Model element")

    gravity = (model_node.findtext("gravity") or "").split()
    length_units = (model_node.findtext("length_units") or "").strip()
    bodies = []
    for body in model_node.iter():
        if _local(body.tag) == "Body" and body.get("name"):
            bodies.append(body.get("name"))

    meshes = []
    for mesh in model_node.iter():
        if _local(mesh.tag) == "mesh_file" and mesh.text:
            meshes.append(mesh.text.strip())

    coordinates = {}
    for coord in model_node.iter():
        if _local(coord.tag) != "Coordinate" or not coord.get("name"):
            continue
        name = coord.get("name")
        if name not in SHOULDER_COORDINATES and name not in ELBOW_COORDINATES:
            continue
        coordinates[name] = {
            "default_value_rad": float(coord.findtext("default_value") or "nan"),
            "range_rad": (coord.findtext("range") or "").split(),
            "locked": (coord.findtext("locked") or "").strip().lower() == "true",
            "clamped": (coord.findtext("clamped") or "").strip().lower() == "true",
        }

    markers_root = ET.parse(markers_path).getroot()
    markers = {}
    for marker in markers_root.iter():
        if _local(marker.tag) != "Marker" or not marker.get("name"):
            continue
        name = marker.get("name")
        parent = marker.findtext("socket_parent_frame") or ""
        location = (marker.findtext("location") or "").split()
        markers[name] = {
            "parent": parent,
            "location_m": [float(v) for v in location] if len(location) == 3 else None,
            "fixed": (marker.findtext("fixed") or "").strip().lower() == "true",
        }

    return {
        "model_path": str(model_path),
        "markers_path": str(markers_path),
        "model_name": model_node.get("name"),
        "length_units": length_units,
        "gravity_m_s2": [float(v) for v in gravity] if len(gravity) == 3 else None,
        "n_bodies": len(bodies),
        "upper_bodies_present": [name for name in UPPER_BODIES if name in bodies],
        "coordinates": coordinates,
        "upper_markers": {name: markers[name] for name in UPPER_MODEL_MARKERS if name in markers},
        "excluded_present_in_xml": [name for name in EXCLUDED_FULL_BODY_MARKERS if name in markers],
        "n_meshes": len(meshes),
        "geometry_dir": str(model_path.parent / "Geometry"),
        "wrist_joint_marker_defined": "RWrist" in markers or "LWrist" in markers,
    }
