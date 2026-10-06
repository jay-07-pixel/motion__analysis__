"""Project trunk frame in RealSense camera metres.

This is not the ISB thorax. C7, T8, the suprasternal notch, and the xiphoid
are not in the landmark set. The frame is built from the two shoulders and
the two hips only.

Axes are right-handed anatomical directions expressed in the camera frame
(X right, Y down, Z forward):

    Y up the trunk, Z toward the subject's right, X anterior.

The old joint_frame u/v/w axes are a different construction and are not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.pose.keypoints import Keypoint2D

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

_MIN_LENGTH_M = 1e-6
# Usable index-to-pinky width perpendicular to the finger axis.
# On 20261005-163819, 5 mm still left adjacent X/Z flips of 98° and 102°.
# 10 mm removed every adjacent jump over 90°, including the edge-on pairs
# at frames 394, 499, 510, and 1093. 15 mm and 20 mm removed no further
# jumps over 45° and rejected more palms whose width was already usable.
_MIN_PALM_WIDTH_M = 0.010


@dataclass(frozen=True)
class AnatomicalFrame:
    """One right-handed segment frame. Rotation columns are X, Y, Z."""

    name: str
    origin: Vec3
    x_axis: Vec3
    y_axis: Vec3
    z_axis: Vec3
    rotation_matrix: Mat3


@dataclass(frozen=True)
class TrunkFrameResult:
    """Either a trunk frame, or an explicit reason it could not be built."""

    frame: AnatomicalFrame | None
    available: bool
    reason: str | None


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(v: Vec3) -> Vec3 | None:
    mag = math.sqrt(_dot(v, v))
    if mag < _MIN_LENGTH_M or not math.isfinite(mag):
        return None
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _finite_xyz(kp: Keypoint2D) -> Vec3 | None:
    if kp.x_m is None or kp.y_m is None or kp.z_m is None:
        return None
    point = (float(kp.x_m), float(kp.y_m), float(kp.z_m))
    if not all(math.isfinite(value) for value in point):
        return None
    return point


def require_landmark(
    by_name: dict[str, Keypoint2D],
    name: str,
    min_confidence: float,
) -> tuple[Vec3 | None, float | None, str | None]:
    """Return camera XYZ, or a reason the landmark cannot be used.

    Reasons:
        low_confidence:<name>
        invalid_geometry:<name>
    """
    kp = by_name.get(name)
    if kp is None or kp.confidence < min_confidence:
        if kp is None:
            return None, None, f"invalid_geometry:{name}"
        return None, float(kp.confidence), f"low_confidence:{name}"
    point = _finite_xyz(kp)
    if point is None:
        return None, float(kp.confidence), f"invalid_geometry:{name}"
    return point, float(kp.confidence), None


def _column_matrix(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Mat3:
    """Rows of R = [X Y Z]."""
    return (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )


def build_trunk_frame(
    keypoints: list[Keypoint2D],
    min_confidence: float,
) -> TrunkFrameResult:
    """Build the project trunk from both shoulders and both hips.

    Y points from mid-hip to mid-shoulder. Z is the shoulder line with the
    part along Y removed, toward the right shoulder. X = Y × Z, so
    X × Y = Z.
    """
    by_name = {kp.name: kp for kp in keypoints}
    points: dict[str, Vec3] = {}
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        point, _confidence, reason = require_landmark(by_name, name, min_confidence)
        if point is None:
            return TrunkFrameResult(frame=None, available=False, reason=reason)
        points[name] = point

    origin = _scale(_add(points["left_shoulder"], points["right_shoulder"]), 0.5)
    hips = _scale(_add(points["left_hip"], points["right_hip"]), 0.5)
    y_axis = _unit(_sub(origin, hips))
    if y_axis is None:
        return TrunkFrameResult(frame=None, available=False, reason="degenerate_trunk")

    shoulder_width = _sub(points["right_shoulder"], points["left_shoulder"])
    z_raw = _sub(shoulder_width, _scale(y_axis, _dot(shoulder_width, y_axis)))
    z_axis = _unit(z_raw)
    if z_axis is None:
        return TrunkFrameResult(frame=None, available=False, reason="degenerate_trunk")

    x_axis = _unit(_cross(y_axis, z_axis))
    if x_axis is None:
        return TrunkFrameResult(frame=None, available=False, reason="degenerate_trunk")

    # Re-orthogonalise Z so a tiny numeric drift cannot break X × Y = Z.
    z_axis = _unit(_cross(x_axis, y_axis))
    if z_axis is None:
        return TrunkFrameResult(frame=None, available=False, reason="degenerate_trunk")

    handed = _cross(x_axis, y_axis)
    if _dot(handed, z_axis) < 1.0 - 1e-6:
        return TrunkFrameResult(frame=None, available=False, reason="degenerate_trunk")

    frame = AnatomicalFrame(
        name="trunk",
        origin=origin,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        rotation_matrix=_column_matrix(x_axis, y_axis, z_axis),
    )
    return TrunkFrameResult(frame=frame, available=True, reason=None)


@dataclass(frozen=True)
class HandFrameResult:
    """A right-handed hand frame, or an explicit reason it could not be built."""

    frame: AnatomicalFrame | None
    available: bool
    reason: str | None
    min_confidence: float | None


def _unavailable_hand(
    reason: str,
    min_confidence: float | None = None,
) -> HandFrameResult:
    return HandFrameResult(
        frame=None,
        available=False,
        reason=reason,
        min_confidence=min_confidence,
    )


def _lookup_hand_point(
    by_name: dict[str, Keypoint2D],
    name: str,
    min_confidence: float,
) -> tuple[Vec3 | None, float | None, str | None]:
    """Camera XYZ for one hand landmark, or a refusal reason.

    Reasons:
        missing_landmark:<name>
        low_confidence:<name>
        missing_xyz:<name>
    """
    kp = by_name.get(name)
    if kp is None:
        return None, None, f"missing_landmark:{name}"
    confidence = float(kp.confidence)
    if not math.isfinite(confidence) or confidence < min_confidence:
        stored = confidence if math.isfinite(confidence) else None
        return None, stored, f"low_confidence:{name}"
    point = _finite_xyz(kp)
    if point is None:
        return None, confidence, f"missing_xyz:{name}"
    return point, confidence, None


def _frame_is_stable(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> bool:
    """True when the three axes are unit, orthogonal, and right-handed."""
    axes = (x_axis, y_axis, z_axis)
    if not all(all(math.isfinite(value) for value in axis) for axis in axes):
        return False
    for axis in axes:
        if abs(_dot(axis, axis) - 1.0) > 1e-6:
            return False
    if abs(_dot(x_axis, y_axis)) > 1e-6:
        return False
    if abs(_dot(x_axis, z_axis)) > 1e-6:
        return False
    if abs(_dot(y_axis, z_axis)) > 1e-6:
        return False
    # det(R) = X · (Y × Z). +1 is right-handed.
    return abs(_dot(x_axis, _cross(y_axis, z_axis)) - 1.0) < 1e-6


def build_hand_frame(
    keypoints: list[Keypoint2D],
    side: str,
    min_confidence: float,
) -> HandFrameResult:
    """Build one hand frame from wrist and three knuckles.

    The same construction is used for both sides. Y runs from the wrist
    through middle_mcp. Z is the index-to-pinky palm width with the part
    along Y removed. X = Y × Z, so the columns of R are a right-handed
    camera-from-hand rotation.

    Args:
        keypoints: Joints with camera x_m, y_m, z_m where depth was valid.
        side: ``left`` or ``right``. Selects landmark names only.
        min_confidence: Skip a landmark below this.

    Returns:
        A frame named ``<side>_hand``, or unavailable with a reason.
        Reasons: missing_landmark, low_confidence, missing_xyz,
        degenerate_long_axis, degenerate_palm_width, insufficient_palm_width,
        unstable_frame.
    """
    by_name = {kp.name: kp for kp in keypoints}
    names = {
        "wrist": f"{side}_wrist",
        "middle": f"{side}_middle_mcp",
        "index": f"{side}_index_mcp",
        "pinky": f"{side}_pinky_mcp",
    }
    points: dict[str, Vec3] = {}
    confidences: list[float] = []
    for key, name in names.items():
        point, confidence, reason = _lookup_hand_point(by_name, name, min_confidence)
        if point is None or confidence is None:
            return _unavailable_hand(reason or f"missing_landmark:{name}", confidence)
        points[key] = point
        confidences.append(confidence)

    confidence = min(confidences)
    y_axis = _unit(_sub(points["middle"], points["wrist"]))
    if y_axis is None:
        return _unavailable_hand("degenerate_long_axis", confidence)

    width = _sub(points["pinky"], points["index"])
    if _unit(width) is None:
        return _unavailable_hand("degenerate_palm_width", confidence)
    width_perp = _sub(width, _scale(y_axis, _dot(width, y_axis)))
    perp_length = math.sqrt(_dot(width_perp, width_perp))
    if not math.isfinite(perp_length) or perp_length < _MIN_LENGTH_M:
        return _unavailable_hand("degenerate_palm_width", confidence)
    if perp_length < _MIN_PALM_WIDTH_M:
        return _unavailable_hand("insufficient_palm_width", confidence)
    z_axis = _unit(width_perp)
    if z_axis is None:
        return _unavailable_hand("degenerate_palm_width", confidence)

    # Y and Z are unit and perpendicular, so X = Y × Z has length 1 and
    # X · (Y × Z) = +1.
    x_axis = _unit(_cross(y_axis, z_axis))
    if x_axis is None or not _frame_is_stable(x_axis, y_axis, z_axis):
        return _unavailable_hand("unstable_frame", confidence)

    frame = AnatomicalFrame(
        name=f"{side}_hand",
        origin=points["wrist"],
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        rotation_matrix=_column_matrix(x_axis, y_axis, z_axis),
    )
    return HandFrameResult(
        frame=frame,
        available=True,
        reason=None,
        min_confidence=confidence,
    )


def build_practical_forearm_frame(
    keypoints: list[Keypoint2D],
    side: str,
    hand: AnatomicalFrame,
    min_confidence: float,
) -> HandFrameResult:
    """Practical forearm frame at the wrist. This is not an ISB radius frame.

    Measured:
        Y is the distal forearm, unit(wrist - elbow).
        Z is the hand's ulnar axis (index_mcp → pinky_mcp) with the part
        along Y removed. X = Y × Z, so R is right-handed and uses the same
        column layout as the hand frame.

    Unavailable:
        Radial styloid, ulnar styloid, and pronation/supination. The
        transverse axis is borrowed from the observed palm, not from the
        forearm bones. A wrist angle from this frame is a practical
        observable angle, not a full ISB wrist angle.

    The hand frame must already exist. Its origin is the wrist.
    """
    by_name = {kp.name: kp for kp in keypoints}
    elbow_name = f"{side}_elbow"
    elbow, elbow_confidence, elbow_reason = _lookup_hand_point(
        by_name, elbow_name, min_confidence
    )
    if elbow is None or elbow_confidence is None:
        return _unavailable_hand(elbow_reason or f"missing_landmark:{elbow_name}", elbow_confidence)

    y_axis = _unit(_sub(hand.origin, elbow))
    if y_axis is None:
        return _unavailable_hand("degenerate_forearm", elbow_confidence)

    along = _dot(hand.z_axis, y_axis)
    z_raw = _sub(hand.z_axis, _scale(y_axis, along))
    z_axis = _unit(z_raw)
    if z_axis is None:
        return _unavailable_hand("degenerate_forearm", elbow_confidence)

    x_axis = _unit(_cross(y_axis, z_axis))
    if x_axis is None or not _frame_is_stable(x_axis, y_axis, z_axis):
        return _unavailable_hand("degenerate_forearm", elbow_confidence)

    frame = AnatomicalFrame(
        name=f"{side}_forearm",
        origin=hand.origin,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        rotation_matrix=_column_matrix(x_axis, y_axis, z_axis),
    )
    return HandFrameResult(
        frame=frame,
        available=True,
        reason=None,
        min_confidence=elbow_confidence,
    )
