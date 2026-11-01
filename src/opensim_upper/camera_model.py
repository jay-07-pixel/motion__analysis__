"""Upper-body model whose ground axes are the RealSense colour axes.

The Pose2Sim upper file is only read. This writes a new model:

- ground origin is the colour camera
- ground +X is camera right, +Y is camera down, +Z is camera forward
- the torso is the root body, on a 6-degree free joint
- pelvis, sacrum, and both legs are omitted
- arm, forearm, and head joints stay as they are on the torso

A person facing the camera has anterior along camera -Z, up along camera -Y,
and their right along camera -X. That map is a proper rotation (determinant +1).
Joint axes below the torso stay in each bone's own frame.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.opensim_upper.model_xml import _local

REPO = Path(__file__).resolve().parents[2]
SOURCE_MODEL = REPO / "models" / "opensim_upper" / "Model_Pose2Sim_upper.osim"
CAMERA_MODEL = REPO / "models" / "opensim_upper" / "Model_upper_camera.osim"

REMOVED_BODIES = (
    "pelvis",
    "sacrum",
    "femur_r",
    "tibia_r",
    "talus_r",
    "calcn_r",
    "toes_r",
    "femur_l",
    "tibia_l",
    "talus_l",
    "calcn_l",
    "toes_l",
)
REMOVED_JOINTS = (
    "ground_pelvis",
    "sacrumjnt",
    "hip_r",
    "walker_knee_r",
    "ankle_r",
    "subtalar_r",
    "mtp_r",
    "hip_l",
    "walker_knee_l",
    "ankle_l",
    "subtalar_l",
    "mtp_l",
    "back",
)
KEPT_BODIES = (
    "torso",
    "head",
    "humerus_r",
    "ulna_r",
    "radius_r",
    "hand_r",
    "humerus_l",
    "ulna_l",
    "radius_l",
    "hand_l",
)
ROOT_JOINT = "ground_thorax"
ROOT_COORDINATES = (
    "thorax_rx",
    "thorax_ry",
    "thorax_rz",
    "thorax_tx",
    "thorax_ty",
    "thorax_tz",
)

# Columns are the camera coordinates of the anatomical axes at the default pose.
# anatomical +X anterior, +Y up, +Z model-right.
FACING_CAMERA_ROTATION = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for element in root.iter():
        if _local(element.tag) == tag and element.get("name") == name:
            return element
    raise ValueError(f"Missing {tag} {name}.")


def _child(element: ET.Element, tag: str) -> ET.Element:
    for child in element:
        if _local(child.tag) == tag:
            return child
    raise ValueError(f"Missing <{tag}> under {element.tag}.")


def _vec(text: str | None) -> np.ndarray:
    if text is None or not text.strip():
        raise ValueError("Expected a 3-vector.")
    values = np.array([float(part) for part in text.split()], dtype=float)
    if values.shape != (3,):
        raise ValueError(f"Expected 3 values, got {text!r}.")
    return values


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.16g}" for value in values)


def rotation_from_body_fixed_xyz(angles: np.ndarray) -> np.ndarray:
    """R = Rx(ax) Ry(ay) Rz(az), the OpenSim body-fixed XYZ convention."""
    ax, ay, az = (float(angle) for angle in angles)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz


def body_fixed_xyz(rotation: np.ndarray) -> np.ndarray:
    """Inverse of ``rotation_from_body_fixed_xyz``."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation must be 3x3.")
    sy = float(np.clip(matrix[0, 2], -1.0, 1.0))
    ay = math.asin(sy)
    if abs(math.cos(ay)) < 1e-8:
        ax = 0.0
        az = math.atan2(-matrix[1, 0], matrix[1, 1]) if sy < 0.0 else math.atan2(matrix[1, 0], matrix[1, 1])
    else:
        ax = math.atan2(-matrix[1, 2], matrix[2, 2])
        az = math.atan2(-matrix[0, 1], matrix[0, 0])
    return np.array([ax, ay, az], dtype=float)


def default_torso_origin_yup(root: ET.Element) -> np.ndarray:
    """Torso origin in the Rajagopal ground frame at the file's default pose.

    Pelvis orientation and the lumbar coordinates are zero in this file, and
    the back joint's child offset is the torso origin. Anything else is refused
    rather than approximated.
    """
    pelvis = _named(root, "CustomJoint", "ground_pelvis")
    for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation"):
        coordinate = _named(pelvis, "Coordinate", name)
        if abs(float(_child(coordinate, "default_value").text or "nan")) > 1e-8:
            raise ValueError(f"{name} default is not zero.")
    ty = float(_child(_named(pelvis, "Coordinate", "pelvis_ty"), "default_value").text or "nan")
    pelvis_frame = _named(pelvis, "PhysicalOffsetFrame", "pelvis_offset")
    if np.max(np.abs(_vec(_child(pelvis_frame, "translation").text))) > 1e-12:
        raise ValueError("ground_pelvis child offset is not at the pelvis origin.")
    if np.max(np.abs(_vec(_child(pelvis_frame, "orientation").text))) > 1e-12:
        raise ValueError("ground_pelvis child offset is rotated.")

    back = _named(root, "CustomJoint", "back")
    for name in ("L5_S1_Flex_Ext", "L5_S1_Lat_Bending", "L5_S1_axial_rotation"):
        coordinate = _named(back, "Coordinate", name)
        if abs(float(_child(coordinate, "default_value").text or "nan")) > 1e-8:
            raise ValueError(f"{name} default is not zero.")
    parent = _named(back, "PhysicalOffsetFrame", "pelvis_offset")
    child = _named(back, "PhysicalOffsetFrame", "torso_offset")
    if np.max(np.abs(_vec(_child(parent, "orientation").text))) > 1e-12:
        raise ValueError("The lumbar parent offset is rotated.")
    if np.max(np.abs(_vec(_child(child, "translation").text))) > 1e-12:
        raise ValueError("The torso offset is not the torso origin.")
    if np.max(np.abs(_vec(_child(child, "orientation").text))) > 1e-12:
        raise ValueError("The torso offset is rotated.")
    offset = _vec(_child(parent, "translation").text)
    return np.array([offset[0], ty + offset[1], offset[2]], dtype=float)


def camera_pose_of_default_torso(origin_yup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the torso rotation and origin in colour-camera metres."""
    rotation = np.array(FACING_CAMERA_ROTATION, dtype=float)
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-9:
        raise ValueError("The camera axis map is not a proper rotation.")
    origin = rotation @ np.asarray(origin_yup, dtype=float)
    return rotation, origin


def _retarget_root(joint: ET.Element, origin_camera: np.ndarray, orientation: np.ndarray) -> None:
    joint.set("name", ROOT_JOINT)
    _child(joint, "socket_parent_frame").text = "ground_camera_offset"
    _child(joint, "socket_child_frame").text = "torso_root_offset"
    coordinates = list(_child(joint, "coordinates"))
    if len(coordinates) != 6:
        raise ValueError("The pelvis joint does not have 6 coordinates.")
    ranges = ("-3.141592653589793 3.141592653589793",) * 3 + ("-5 5",) * 3
    for element, name, span in zip(coordinates, ROOT_COORDINATES, ranges):
        element.set("name", name)
        _child(element, "default_value").text = "0"
        _child(element, "range").text = span
        _child(element, "clamped").text = "false"
        _child(element, "locked").text = "false"
    frames = list(_child(joint, "frames"))
    if len(frames) != 2:
        raise ValueError("The pelvis joint does not have two offset frames.")
    frames[0].set("name", "ground_camera_offset")
    _child(frames[0], "translation").text = _fmt(origin_camera)
    _child(frames[0], "orientation").text = "0 0 0"
    _child(frames[0], "socket_parent").text = "/ground"
    frames[1].set("name", "torso_root_offset")
    _child(frames[1], "translation").text = "0 0 0"
    _child(frames[1], "orientation").text = _fmt(orientation)
    _child(frames[1], "socket_parent").text = "/bodyset/torso"
    axes = ("1 0 0", "0 1 0", "0 0 1", "1 0 0", "0 1 0", "0 0 1")
    transforms = [element for element in joint.iter() if _local(element.tag) == "TransformAxis"]
    if len(transforms) != 6:
        raise ValueError("The pelvis spatial transform does not have 6 axes.")
    for element, name, axis in zip(transforms, ROOT_COORDINATES, axes):
        _child(element, "coordinates").text = name
        _child(element, "axis").text = axis


def build_camera_model_tree(source: Path | None = None) -> ET.ElementTree:
    """Return the camera-frame model. The source file is not modified."""
    path = Path(source or SOURCE_MODEL)
    tree = ET.parse(path)
    root = tree.getroot()
    model = _child(root, "Model")
    origin_yup = default_torso_origin_yup(model)
    rotation, origin_camera = camera_pose_of_default_torso(origin_yup)
    joint = copy.deepcopy(_named(model, "CustomJoint", "ground_pelvis"))
    _retarget_root(joint, origin_camera, body_fixed_xyz(rotation))

    for parent in model.iter():
        for child in list(parent):
            tag = _local(child.tag)
            name = child.get("name")
            if tag == "Body" and name in REMOVED_BODIES:
                parent.remove(child)
            elif tag in {"CustomJoint", "PinJoint", "WeldJoint", "UniversalJoint"} and name in REMOVED_JOINTS:
                parent.remove(child)
            elif tag in {"ContactHalfSpace", "ContactSphere"}:
                parent.remove(child)
    objects = _child(_named(model, "JointSet", "jointset"), "objects")
    objects.insert(0, joint)
    model.set("name", "UpperBody_CameraFrame")
    _child(model, "gravity").text = "0 9.8066499999999994 0"
    _child(model, "credits").text = (
        "Upper body only. Thorax is the root. Ground axes match the RealSense "
        "colour camera: +X right, +Y down, +Z forward, origin at the camera. "
        "Derived from the Pose2Sim Rajagopal adaptation. Arm joints were not "
        "rewritten. Pelvis and legs are omitted. Gravity is +Y because +Y is down."
    )
    return tree


def write_camera_model(dest: Path | None = None, source: Path | None = None) -> Path:
    """Write the camera-frame model. Refuses to overwrite the source model."""
    path = Path(dest or CAMERA_MODEL)
    source_path = Path(source or SOURCE_MODEL).resolve()
    if path.resolve() == source_path:
        raise ValueError("Refusing to overwrite the source upper-body model.")
    tree = build_camera_model_tree(source_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
