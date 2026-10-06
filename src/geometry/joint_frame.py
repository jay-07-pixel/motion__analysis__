"""Local 3-axis frames on shoulder, elbow, and wrist.

Each frame lives in camera metres (origin = that joint's X, Y, Z).
The three axes are perpendicular:

    u = reference (shoulder: trunk; elbow: upper arm; wrist: forearm)
    v = second axis, locked to the body so it does not spin with the bend
    w = cross product, the third axis

The dial reads the moving bone inside this frame. Its three components
(on u, v, and w) give one angle. Turning the whole arm in space rotates
the frame with it, so the angle stays the bend and does not follow the camera.

At the shoulder, the left–right shoulder line is removed from the hip
vector before the angle is taken. A hip sits medial to the shoulder, so
the raw hip line is not straight down. That bias makes a sideways raise
and a forward raise read differently even when the arm is at the same height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.pose.keypoints import Keypoint2D

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class JointFrame:
    """One joint origin plus three unit axes in camera metres."""

    name: str
    origin: Vec3
    axis_u: Vec3
    axis_v: Vec3
    axis_w: Vec3
    degrees: float
    vertex_u_px: float
    vertex_v_px: float
    min_confidence: float


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _unit(v: Vec3) -> Vec3 | None:
    mag = math.sqrt(_dot(v, v))
    if mag < 1e-6:
        return None
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _any_perp(u: Vec3) -> Vec3 | None:
    """A unit vector perpendicular to u, used when every body hint is parallel."""
    helper = (1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else (0.0, 1.0, 0.0)
    return _unit(_cross(u, helper))


def _axes_locked_to_body(axis_u: Vec3, hints: list[Vec3]) -> tuple[Vec3, Vec3] | None:
    """Second and third axes perpendicular to u, from the first usable hint.

    A hint that lies along u (arm parallel to that body line) is skipped.
    The moving bone is not a hint, so the frame does not spin when the
    joint bends or the limb turns to face the camera.
    """
    for hint in hints:
        sideways = _sub(hint, _scale(axis_u, _dot(hint, axis_u)))
        axis_v = _unit(sideways)
        if axis_v is None:
            continue
        axis_w = _unit(_cross(axis_v, axis_u))
        if axis_w is None:
            continue
        return axis_v, axis_w
    axis_w = _any_perp(axis_u)
    if axis_w is None:
        return None
    axis_v = _unit(_cross(axis_w, axis_u))
    if axis_v is None:
        return None
    return axis_v, axis_w


def angle_from_frame_deg(axis_u: Vec3, axis_v: Vec3, axis_w: Vec3, measure: Vec3) -> float | None:
    """Angle of `measure` away from axis u, using all three axes.

    The limb is split into components on u, v, and w. The angle is
    atan2(length of the v/w part, component on u). A rigid turn of the
    whole body rotates the axes and the limb together, so this number
    stays put. Range is 0° (along u) to 180° (opposite u).
    """
    direction = _unit(measure)
    if direction is None:
        return None
    on_u = _dot(direction, axis_u)
    on_v = _dot(direction, axis_v)
    on_w = _dot(direction, axis_w)
    perpendicular = math.sqrt(on_v * on_v + on_w * on_w)
    return math.degrees(math.atan2(perpendicular, on_u))


def _xyz(kp: Keypoint2D) -> Vec3 | None:
    if kp.x_m is None or kp.y_m is None or kp.z_m is None:
        return None
    return (float(kp.x_m), float(kp.y_m), float(kp.z_m))


def _joint(
    by_name: dict[str, Keypoint2D],
    name: str,
    min_confidence: float,
) -> tuple[Keypoint2D, Vec3] | None:
    kp = by_name.get(name)
    if kp is None or kp.confidence < min_confidence:
        return None
    point = _xyz(kp)
    if point is None:
        return None
    return kp, point


def _pair_vector(
    by_name: dict[str, Keypoint2D],
    pair: list,
    min_confidence: float,
) -> Vec3 | None:
    """Vector from pair[0] to pair[1], or None if either joint is unusable."""
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    start = _joint(by_name, str(pair[0]), min_confidence)
    end = _joint(by_name, str(pair[1]), min_confidence)
    if start is None or end is None:
        return None
    return _sub(end[1], start[1])


def _build_one(
    by_name: dict[str, Keypoint2D],
    spec: dict,
    min_confidence: float,
) -> JointFrame | None:
    """Build one frame. None if a required joint has no depth."""
    origin_hit = _joint(by_name, str(spec.get("origin", "")), min_confidence)
    if origin_hit is None:
        return None
    origin_kp, origin = origin_hit
    reference = _pair_vector(by_name, spec.get("reference") or [], min_confidence)
    measure = _pair_vector(by_name, spec.get("measure") or [], min_confidence)
    if reference is None or measure is None:
        return None

    across_spec = spec.get("across")
    hint_specs = list(spec.get("plane_hints") or [])
    if across_spec:
        across = _unit(_pair_vector(by_name, across_spec, min_confidence) or (0.0, 0.0, 0.0))
        if across is None:
            return None
        # Drop the part of the hip line that runs along the shoulders.
        reference = _sub(reference, _scale(across, _dot(reference, across)))
        axis_u = _unit(reference)
        if axis_u is None:
            return None
        axis_v = across
        # across × trunk points out of the chest when the person faces the camera.
        axis_w = _unit(_cross(axis_v, axis_u))
        if axis_w is None:
            return None
    else:
        axis_u = _unit(reference)
        if axis_u is None:
            return None
        hints: list[Vec3] = []
        for pair in hint_specs:
            hint = _pair_vector(by_name, pair, min_confidence)
            if hint is not None:
                hints.append(hint)
        if hint_specs and not hints:
            return None
        locked = _axes_locked_to_body(axis_u, hints)
        if locked is None:
            return None
        axis_v, axis_w = locked

    degrees = angle_from_frame_deg(axis_u, axis_v, axis_w, measure)
    if degrees is None:
        return None
    used = [origin_kp]
    pairs = [spec.get("reference"), spec.get("measure"), spec.get("across"), *hint_specs]
    for pair in pairs:
        if not isinstance(pair, (list, tuple)):
            continue
        for name in pair:
            kp = by_name.get(str(name))
            if kp is not None:
                used.append(kp)
    return JointFrame(
        name=str(spec["name"]),
        origin=origin,
        axis_u=axis_u,
        axis_v=axis_v,
        axis_w=axis_w,
        degrees=degrees,
        vertex_u_px=origin_kp.u_px,
        vertex_v_px=origin_kp.v_px,
        min_confidence=min(kp.confidence for kp in used),
    )


def build_joint_frames(
    keypoints: list[Keypoint2D],
    frame_specs: list[dict],
    min_confidence: float,
) -> list[JointFrame]:
    """Build every frame listed in config. Specs that lack depth are skipped.

    Args:
        keypoints: Joints with camera x_m, y_m, z_m filled in.
        frame_specs: YAML joint_frames list.
        min_confidence: Skip a frame if any joint it uses is below this.
    """
    by_name = {kp.name: kp for kp in keypoints}
    frames: list[JointFrame] = []
    for spec in frame_specs:
        if "name" not in spec or "origin" not in spec:
            continue
        built = _build_one(by_name, spec, min_confidence)
        if built is not None:
            frames.append(built)
    return frames
