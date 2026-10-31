"""2D image-plane pose measurements shared by validate_2d_pose.py and the game bridge.

All values come from colour-pixel keypoints. Nothing here uses depth, world
landmarks, or 3D DOFs. Left/right are MediaPipe names: the person's own sides.
A joint below min_confidence is treated as missing, and any measurement that
needs it is None. None is never replaced by 0.
"""

from __future__ import annotations

import math

from src.analysis.angles_2d import angle_at_vertex_deg
from src.pose.keypoints import Keypoint2D

# Joints the presenter needs to see by name. Index is the MediaPipe id.
IMPORTANT = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

# Image-plane angles. Vertex is the middle name. Not 3D biomechanical DOFs.
ANGLE_SPECS = (
    ("left elbow", "left_shoulder", "left_elbow", "left_wrist"),
    ("right elbow", "right_shoulder", "right_elbow", "right_wrist"),
    ("left knee", "left_hip", "left_knee", "left_ankle"),
    ("right knee", "right_hip", "right_knee", "right_ankle"),
    ("left shoulder", "left_hip", "left_shoulder", "left_elbow"),
    ("right shoulder", "right_hip", "right_shoulder", "right_elbow"),
)

# Game metric name -> ANGLE_SPECS label. Arm angle is the shoulder angle
# hip-shoulder-elbow: about 0 with the arm down, 90 out to the side, 180 overhead.
GAME_ANGLE_METRICS = (
    ("right_arm_angle_deg", "right shoulder"),
    ("left_arm_angle_deg", "left shoulder"),
    ("right_elbow_angle_deg", "right elbow"),
    ("left_elbow_angle_deg", "left elbow"),
    ("right_knee_angle_deg", "right knee"),
    ("left_knee_angle_deg", "left knee"),
)

_TORSO_NAMES = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def valid(by_name: dict[str, Keypoint2D], name: str, min_confidence: float) -> bool:
    kp = by_name.get(name)
    return kp is not None and kp.confidence >= min_confidence


def tracking_status(by_name: dict[str, Keypoint2D], min_confidence: float) -> str:
    if not by_name:
        return "POOR"
    if all(valid(by_name, name, min_confidence) for name in IMPORTANT):
        return "GOOD"
    return "POOR"


def image_plane_angles(
    by_name: dict[str, Keypoint2D], min_confidence: float
) -> dict[str, float | None]:
    angles: dict[str, float | None] = {}
    for label, a_name, b_name, c_name in ANGLE_SPECS:
        if not all(valid(by_name, name, min_confidence) for name in (a_name, b_name, c_name)):
            angles[label] = None
            continue
        a, b, c = by_name[a_name], by_name[b_name], by_name[c_name]
        angles[label] = angle_at_vertex_deg((a.u_px, a.v_px), (b.u_px, b.v_px), (c.u_px, c.v_px))
    return angles


def torso_midpoints(
    by_name: dict[str, Keypoint2D], min_confidence: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Shoulder centre and hip centre (u, v) pixels, or None if a torso joint is weak."""
    if not all(valid(by_name, name, min_confidence) for name in _TORSO_NAMES):
        return None
    shoulder_u = 0.5 * (by_name["left_shoulder"].u_px + by_name["right_shoulder"].u_px)
    shoulder_v = 0.5 * (by_name["left_shoulder"].v_px + by_name["right_shoulder"].v_px)
    hip_u = 0.5 * (by_name["left_hip"].u_px + by_name["right_hip"].u_px)
    hip_v = 0.5 * (by_name["left_hip"].v_px + by_name["right_hip"].v_px)
    return (shoulder_u, shoulder_v), (hip_u, hip_v)


def torso_lean_deg(by_name: dict[str, Keypoint2D], min_confidence: float) -> float | None:
    """Sideways torso lean in the image plane, degrees from image vertical.

    Hip centre -> shoulder centre, measured against image up. Positive = toward
    the person's LEFT, negative = toward the person's RIGHT. The side comes from
    the person's own right-hip -> left-hip axis, so a mirrored preview does not
    flip the sign. None if the torso is not visible or the shoulders are not
    above the hips in the image.
    """
    mids = torso_midpoints(by_name, min_confidence)
    if mids is None:
        return None
    (shoulder_u, shoulder_v), (hip_u, hip_v) = mids
    axis_u = by_name["left_hip"].u_px - by_name["right_hip"].u_px
    axis_v = by_name["left_hip"].v_px - by_name["right_hip"].v_px
    axis_len = math.hypot(axis_u, axis_v)
    if axis_len < 1e-6:
        return None
    du = shoulder_u - hip_u
    dv = shoulder_v - hip_v
    toward_left = (du * axis_u + dv * axis_v) / axis_len
    up = -dv
    if up <= 1e-6:
        return None
    return math.degrees(math.atan2(toward_left, up))


def game_metrics_2d(by_name: dict[str, Keypoint2D], min_confidence: float) -> dict[str, float | None]:
    """Metrics for the PatientBalanceAI imitation game. None = cannot be measured."""
    angles = image_plane_angles(by_name, min_confidence)
    metrics: dict[str, float | None] = {
        name: angles[label] for name, label in GAME_ANGLE_METRICS
    }
    metrics["torso_lean_deg"] = torso_lean_deg(by_name, min_confidence)
    return metrics
