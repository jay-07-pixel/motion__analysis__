"""Reject physically impossible 3D hand landmarks before the hand frame is built.

CameraXyzSmoother already blends small depth jitter. It still accepts a
landmark that jumps by a metre, and then pulls later frames toward that
point. This gate does not smooth and does not reuse a stale point. A failed
observation leaves the hand frame unavailable.

Thresholds are from the right-hand landmarks in data/output/20261005-163819
(the controlled hand-frame recording). On frames whose four points stayed
hand-sized:

    wrist to middle_mcp     max 0.133 m
    index_mcp to pinky_mcp  max 0.112 m
    wrist to any knuckle    max 0.140 m
    depth span of the four  max 0.103 m
    frame-to-frame step     max 0.068 m
    speed                    max 0.98 m/s

The same frames had confidence above 0.5 during later jumps of 0.2–1.9 m,
so confidence is not reused as a depth check. Global depth max (6 m) also
lets those jumps through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.geometry.anatomical_frames import HandFrameResult, Vec3, build_hand_frame
from src.pose.keypoints import Keypoint2D

# Above the largest clean step (0.068 m) and below the outlier cluster.
_XYZ_JUMP_M = 0.12
# Above the largest clean depth step (0.048 m).
_DEPTH_JUMP_M = 0.08
# Above the fastest clean hand point (0.98 m/s). Used only to widen the
# step limit when the timestamp gap is longer than one frame.
_MAX_SPEED_M_S = 1.5
# A long pause must not turn a metre-scale teleport into an allowed step.
_MAX_STEP_M = 0.40
# Above the largest clean segment (0.140 m) and palm width (0.112 m).
_MAX_WRIST_SEGMENT_M = 0.20
_MAX_PALM_WIDTH_M = 0.15
# Above the largest clean depth span (0.103 m).
_MAX_DEPTH_SPAN_M = 0.12

_STEMS = ("wrist", "middle_mcp", "index_mcp", "pinky_mcp")


@dataclass
class HandQualityGate:
    """Remembers the last accepted hand. Rejected points are not stored."""

    _accepted: dict[str, tuple[Vec3, float | None]] = field(default_factory=dict)

    def accept(self, points: dict[str, Vec3], time_sec: float | None) -> None:
        """Store a hand that already produced a valid frame."""
        for name, point in points.items():
            self._accepted[name] = (point, time_sec)


def hand_frame_after_quality(
    keypoints: list[Keypoint2D],
    side: str,
    min_confidence: float,
    gate: HandQualityGate,
    time_sec: float | None = None,
) -> HandFrameResult:
    """Quality-check the four hand landmarks, then build the existing frame.

    On failure the gate is left unchanged and the frame is unavailable.
    On success the mathematical frame is exactly ``build_hand_frame``.
    """
    reason, confidence, points = inspect_hand_landmarks(
        keypoints, side, min_confidence, gate, time_sec
    )
    if reason is not None or points is None:
        return HandFrameResult(
            frame=None,
            available=False,
            reason=reason or "invalid_3d",
            min_confidence=confidence,
        )
    result = build_hand_frame(keypoints, side, min_confidence)
    if result.available:
        gate.accept(points, time_sec)
    return result


def inspect_hand_landmarks(
    keypoints: list[Keypoint2D],
    side: str,
    min_confidence: float,
    gate: HandQualityGate,
    time_sec: float | None = None,
) -> tuple[str | None, float | None, dict[str, Vec3] | None]:
    """Return a refusal reason, or the four camera points.

    The gate is not modified. Reasons:
        missing_landmark:<name>
        low_confidence:<name>
        missing_xyz:<name>
        invalid_3d:<name>
        depth_jump:<name>
        xyz_jump:<name>
        implausible_hand_geometry
        inconsistent_hand_depth
    """
    by_name = {kp.name: kp for kp in keypoints}
    points: dict[str, Vec3] = {}
    confidences: list[float] = []
    for stem in _STEMS:
        name = f"{side}_{stem}"
        point, confidence, reason = _usable_point(by_name.get(name), name, min_confidence)
        if point is None:
            return reason or f"invalid_3d:{name}", confidence, None
        jump = _jump_reason(name, point, gate, time_sec)
        if jump is not None:
            return jump, confidence, None
        points[name] = point
        if confidence is not None:
            confidences.append(confidence)

    confidence = min(confidences) if confidences else None
    geometry = _geometry_reason(points, side)
    if geometry is not None:
        return geometry, confidence, None
    return None, confidence, points


def _usable_point(
    kp: Keypoint2D | None,
    name: str,
    min_confidence: float,
) -> tuple[Vec3 | None, float | None, str | None]:
    if kp is None:
        return None, None, f"missing_landmark:{name}"
    confidence = float(kp.confidence)
    if not math.isfinite(confidence) or confidence < min_confidence:
        stored = confidence if math.isfinite(confidence) else None
        return None, stored, f"low_confidence:{name}"
    if kp.x_m is None or kp.y_m is None or kp.z_m is None:
        return None, confidence, f"missing_xyz:{name}"
    point = (float(kp.x_m), float(kp.y_m), float(kp.z_m))
    if not all(math.isfinite(value) for value in point):
        return None, confidence, f"invalid_3d:{name}"
    if kp.coord_frame != "camera_3d_metre":
        return None, confidence, f"invalid_3d:{name}"
    return point, confidence, None


def _step_allowance(previous_time: float | None, time_sec: float | None) -> float:
    allowance = _XYZ_JUMP_M
    if (
        previous_time is not None
        and time_sec is not None
        and math.isfinite(previous_time)
        and math.isfinite(time_sec)
    ):
        dt = time_sec - previous_time
        if dt > 0.0:
            allowance = max(allowance, min(_MAX_STEP_M, _MAX_SPEED_M_S * dt))
    return allowance


def _jump_reason(
    name: str,
    point: Vec3,
    gate: HandQualityGate,
    time_sec: float | None,
) -> str | None:
    previous = gate._accepted.get(name)
    if previous is None:
        return None
    last_point, last_time = previous
    step = math.dist(last_point, point)
    depth_step = abs(point[2] - last_point[2])
    allowance = _step_allowance(last_time, time_sec)
    depth_allowance = max(_DEPTH_JUMP_M, min(_MAX_STEP_M, allowance))
    if depth_step > depth_allowance and depth_step >= 0.5 * step:
        return f"depth_jump:{name}"
    if step > allowance:
        return f"xyz_jump:{name}"
    return None


def _geometry_reason(points: dict[str, Vec3], side: str) -> str | None:
    wrist = points[f"{side}_wrist"]
    middle = points[f"{side}_middle_mcp"]
    index = points[f"{side}_index_mcp"]
    pinky = points[f"{side}_pinky_mcp"]
    if math.dist(wrist, middle) > _MAX_WRIST_SEGMENT_M:
        return "implausible_hand_geometry"
    if math.dist(index, pinky) > _MAX_PALM_WIDTH_M:
        return "implausible_hand_geometry"
    for point in (middle, index, pinky):
        if math.dist(wrist, point) > _MAX_WRIST_SEGMENT_M:
            return "implausible_hand_geometry"
    depths = (wrist[2], middle[2], index[2], pinky[2])
    if max(depths) - min(depths) > _MAX_DEPTH_SPAN_M:
        return "inconsistent_hand_depth"
    return None
