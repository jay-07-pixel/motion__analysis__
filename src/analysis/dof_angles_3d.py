"""Initial upper-limb degrees of freedom beside the existing one-angle path.

Computed only from RealSense camera metres already stored on Keypoint2D.
The three angles are ISB-style readings of directions we can actually see:

    shoulder plane of elevation
    shoulder elevation
    signed elbow flexion

Internal rotation and pronation are not produced. Wrist flexion and
deviation are a separate practical reading of the hand frame against the
forearm long axis. They are not ISB wrist angles: the radius styloids are
not in the landmark set. Missing geometry stays missing, not zero.

Thresholds are arguments. config.yaml analysis.dof owns the numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.geometry.anatomical_frames import (
    Vec3,
    build_hand_frame,
    build_practical_forearm_frame,
    build_trunk_frame,
    require_landmark,
)
from src.pose.keypoints import Keypoint2D

DOF_NAMES: tuple[str, ...] = (
    "left_shoulder_plane",
    "left_shoulder_elevation",
    "left_elbow_flexion",
    "right_shoulder_plane",
    "right_shoulder_elevation",
    "right_elbow_flexion",
)

WRIST_DOF_NAMES: tuple[str, ...] = (
    "left_wrist_flexion",
    "left_wrist_deviation",
    "right_wrist_flexion",
    "right_wrist_deviation",
)

_MIN_LENGTH_M = 1e-6
_SIGN_EPS = 1e-8


@dataclass(frozen=True)
class DofAngle:
    """One named degree of freedom for one frame."""

    name: str
    degrees: float | None
    min_confidence: float | None
    available: bool
    reason: str | None


def unavailable_initial_dofs(reason: str) -> dict[str, DofAngle]:
    """Six keys, all unavailable. Used for 2D mode and a disabled block."""
    return {
        name: DofAngle(
            name=name,
            degrees=None,
            min_confidence=None,
            available=False,
            reason=reason,
        )
        for name in DOF_NAMES
    }


def _unavailable(name: str, reason: str, min_confidence: float | None = None) -> DofAngle:
    return DofAngle(
        name=name,
        degrees=None,
        min_confidence=min_confidence,
        available=False,
        reason=reason,
    )


def _ready(
    name: str,
    degrees: float,
    min_confidence: float,
    reason: str | None = None,
) -> DofAngle:
    return DofAngle(
        name=name,
        degrees=float(degrees),
        min_confidence=min_confidence,
        available=True,
        reason=reason,
    )


def _unit(v: Vec3) -> Vec3 | None:
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag < _MIN_LENGTH_M or not math.isfinite(mag):
        return None
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _lowest(confidences: list[float]) -> float:
    return min(confidences)


def compute_initial_dofs(
    keypoints: list[Keypoint2D],
    min_confidence: float,
    plane_singular_deg: float,
    elbow_sign_deadband_deg: float,
) -> dict[str, DofAngle]:
    """Plane, elevation, and signed elbow flexion for both arms.

    Args:
        keypoints: Joints with camera x_m, y_m, z_m where depth was valid.
        min_confidence: Skip a landmark below this.
        plane_singular_deg: Hide the plane when the arm is this close to
            vertical (hanging or overhead). From analysis.dof.
        elbow_sign_deadband_deg: Report 0° near full extension, where the
            flexion sign is unstable. From analysis.dof.

    Returns:
        Exactly the six DOF_NAMES. Unavailable angles have degrees None.
    """
    trunk = build_trunk_frame(keypoints, min_confidence)
    if not trunk.available or trunk.frame is None:
        reason = trunk.reason or "invalid_geometry"
        return unavailable_initial_dofs(reason)

    by_name = {kp.name: kp for kp in keypoints}
    rotation = np.array(trunk.frame.rotation_matrix, dtype=float)
    y_axis = trunk.frame.y_axis
    z_axis = trunk.frame.z_axis
    # Horizontal radius of a unit arm direction. Below this, plane is unstable.
    plane_limit = math.sin(math.radians(abs(float(plane_singular_deg))))

    results: dict[str, DofAngle] = {}
    for side in ("left", "right"):
        results.update(
            _side_dofs(
                side,
                by_name,
                min_confidence,
                rotation,
                y_axis,
                z_axis,
                plane_limit,
                float(elbow_sign_deadband_deg),
            )
        )
    return {name: results[name] for name in DOF_NAMES}


def _side_dofs(
    side: str,
    by_name: dict[str, Keypoint2D],
    min_confidence: float,
    rotation: np.ndarray,
    y_axis: Vec3,
    z_axis: Vec3,
    plane_limit: float,
    elbow_sign_deadband_deg: float,
) -> dict[str, DofAngle]:
    shoulder_name = f"{side}_shoulder"
    elbow_name = f"{side}_elbow"
    wrist_name = f"{side}_wrist"
    plane_name = f"{side}_shoulder_plane"
    elevation_name = f"{side}_shoulder_elevation"
    flexion_name = f"{side}_elbow_flexion"

    shoulder, shoulder_conf, shoulder_reason = require_landmark(
        by_name, shoulder_name, min_confidence
    )
    elbow, elbow_conf, elbow_reason = require_landmark(by_name, elbow_name, min_confidence)
    arm_reason = shoulder_reason or elbow_reason
    if shoulder is None or elbow is None or shoulder_conf is None or elbow_conf is None:
        failed = _unavailable(plane_name, arm_reason or "invalid_geometry")
        return {
            plane_name: failed,
            elevation_name: _unavailable(elevation_name, arm_reason or "invalid_geometry"),
            flexion_name: _unavailable(flexion_name, arm_reason or "invalid_geometry"),
        }

    arm_confidence = _lowest([shoulder_conf, elbow_conf])
    distal = _unit(_sub(elbow, shoulder))
    if distal is None:
        reason = "degenerate_segment"
        return {
            plane_name: _unavailable(plane_name, reason, arm_confidence),
            elevation_name: _unavailable(elevation_name, reason, arm_confidence),
            flexion_name: _unavailable(flexion_name, reason, arm_confidence),
        }

    # Columns of R are trunk X, Y, Z, so R.T @ u is the arm in the trunk frame.
    in_trunk = rotation.T @ np.asarray(distal, dtype=float)
    u_x = float(in_trunk[0])
    u_y = float(in_trunk[1])
    u_z = float(in_trunk[2])
    vertical = max(-1.0, min(1.0, -u_y))
    elevation = -math.degrees(math.acos(vertical))

    if math.hypot(u_x, u_z) < plane_limit:
        plane = _unavailable(plane_name, "singular_plane", arm_confidence)
    else:
        # Left uses -u_z so abduction stays 0° and forward stays +90°.
        lateral = u_z if side == "right" else -u_z
        plane = _ready(plane_name, math.degrees(math.atan2(u_x, lateral)), arm_confidence)

    elevation_angle = _ready(elevation_name, elevation, arm_confidence)
    flexion = _elbow_flexion(
        side,
        by_name,
        min_confidence,
        shoulder,
        elbow,
        shoulder_conf,
        elbow_conf,
        y_axis,
        z_axis,
        elbow_sign_deadband_deg,
        flexion_name,
        wrist_name,
    )
    return {
        plane_name: plane,
        elevation_name: elevation_angle,
        flexion_name: flexion,
    }


def _elbow_flexion(
    side: str,
    by_name: dict[str, Keypoint2D],
    min_confidence: float,
    shoulder: Vec3,
    elbow: Vec3,
    shoulder_conf: float,
    elbow_conf: float,
    y_axis: Vec3,
    z_axis: Vec3,
    elbow_sign_deadband_deg: float,
    flexion_name: str,
    wrist_name: str,
) -> DofAngle:
    wrist, wrist_conf, wrist_reason = require_landmark(by_name, wrist_name, min_confidence)
    if wrist is None or wrist_conf is None:
        return _unavailable(flexion_name, wrist_reason or "invalid_geometry")

    confidence = _lowest([shoulder_conf, elbow_conf, wrist_conf])
    y_humerus = _unit(_sub(shoulder, elbow))
    y_forearm = _unit(_sub(elbow, wrist))
    if y_humerus is None or y_forearm is None:
        return _unavailable(flexion_name, "degenerate_segment", confidence)

    bend_axis = _cross(y_humerus, y_forearm)
    bend_length = math.sqrt(_dot(bend_axis, bend_axis))
    alpha_mag = math.degrees(math.atan2(bend_length, _dot(y_humerus, y_forearm)))
    if alpha_mag < elbow_sign_deadband_deg:
        return _ready(flexion_name, 0.0, confidence, reason="near_extension")

    along = _dot(z_axis, y_humerus)
    z_perp = (
        z_axis[0] - y_humerus[0] * along,
        z_axis[1] - y_humerus[1] * along,
        z_axis[2] - y_humerus[2] * along,
    )
    flexion_axis = _unit(z_perp)
    if flexion_axis is None:
        # Upper arm lies along the shoulder line. Approximate axis only.
        if side == "right":
            flexion_axis = _unit(_cross(y_axis, y_humerus))
        else:
            flexion_axis = _unit(_cross(y_humerus, y_axis))
    if flexion_axis is None:
        return _unavailable(flexion_name, "degenerate_segment", confidence)

    sign_value = _dot(bend_axis, flexion_axis)
    if abs(sign_value) < _SIGN_EPS:
        return _unavailable(flexion_name, "undefined_sign", confidence)
    signed = math.copysign(alpha_mag, sign_value)
    return _ready(flexion_name, signed, confidence)


def compute_wrist_dofs(
    keypoints: list[Keypoint2D],
    min_confidence: float,
) -> dict[str, DofAngle]:
    """Signed wrist flexion and ulnar deviation for both hands.

    Both numbers are components of one vector: the hand's long axis
    expressed in the practical forearm frame. Flexion is the palmar
    component. Deviation is the ulnar component.

    Positive flexion moves the hand toward the palm. Positive deviation
    moves it toward the ulna (index → pinky). Extension and radial
    deviation are negative. The same meanings are used on both sides.
    The right-hand frame has X palmar; the left-hand frame has X dorsal,
    so only the flexion component changes sign with side.

    These are practical observable angles. They are not ISB wrist angles.
    """
    results: dict[str, DofAngle] = {}
    for side in ("left", "right"):
        results.update(_wrist_side(side, keypoints, min_confidence))
    return {name: results[name] for name in WRIST_DOF_NAMES}


def _wrist_side(
    side: str,
    keypoints: list[Keypoint2D],
    min_confidence: float,
) -> dict[str, DofAngle]:
    flexion_name = f"{side}_wrist_flexion"
    deviation_name = f"{side}_wrist_deviation"
    hand = build_hand_frame(keypoints, side, min_confidence)
    if not hand.available or hand.frame is None:
        reason = hand.reason or "invalid_geometry"
        return {
            flexion_name: _unavailable(flexion_name, reason, hand.min_confidence),
            deviation_name: _unavailable(deviation_name, reason, hand.min_confidence),
        }

    forearm = build_practical_forearm_frame(keypoints, side, hand.frame, min_confidence)
    if not forearm.available or forearm.frame is None or forearm.min_confidence is None:
        reason = forearm.reason or "degenerate_forearm"
        confidence = hand.min_confidence
        return {
            flexion_name: _unavailable(flexion_name, reason, confidence),
            deviation_name: _unavailable(deviation_name, reason, confidence),
        }

    # Hand long axis in the forearm frame. Columns of R are the forearm axes.
    hand_y = hand.frame.y_axis
    along_x = _dot(hand_y, forearm.frame.x_axis)
    along_y = _dot(hand_y, forearm.frame.y_axis)
    along_z = _dot(hand_y, forearm.frame.z_axis)
    if not all(math.isfinite(value) for value in (along_x, along_y, along_z)):
        confidence = min(hand.min_confidence or 0.0, forearm.min_confidence)
        return {
            flexion_name: _unavailable(flexion_name, "degenerate_forearm", confidence),
            deviation_name: _unavailable(deviation_name, "degenerate_forearm", confidence),
        }

    # Right: +X is palmar. Left: +X is dorsal, so palmar flexion is -X.
    flexion_sign = 1.0 if side == "right" else -1.0
    flexion_deg = math.degrees(math.atan2(flexion_sign * along_x, along_y))
    deviation_deg = math.degrees(math.atan2(along_z, along_y))
    confidence = min(hand.min_confidence or 0.0, forearm.min_confidence)
    if hand.min_confidence is None:
        confidence = forearm.min_confidence
    return {
        flexion_name: _ready(flexion_name, flexion_deg, confidence),
        deviation_name: _ready(deviation_name, deviation_deg, confidence),
    }
