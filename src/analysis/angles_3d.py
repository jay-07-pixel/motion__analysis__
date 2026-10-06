"""3D joint angles in camera metres.

Preferred path: a local three-axis frame at the shoulder, elbow, and wrist
(see joint_frames in config). The dial is the moving bone read inside
that frame, so turning the limb in space does not change the bend.

If that frame cannot be built, fall back to the raw angle at the middle
of the three named points. Either way this is the bone angle in space,
not the angle in the photo. Missing depth skips the angle.
"""

from __future__ import annotations

import math

from src.analysis.angles_2d import Angle2D
from src.geometry.joint_frame import JointFrame, build_joint_frames
from src.pose.keypoints import Keypoint2D


def angle_at_vertex_3d_deg(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    point_c: tuple[float, float, float],
) -> float | None:
    """Interior angle ABC in 3D degrees, or None if the points collapse."""
    bax = point_a[0] - point_b[0]
    bay = point_a[1] - point_b[1]
    baz = point_a[2] - point_b[2]
    bcx = point_c[0] - point_b[0]
    bcy = point_c[1] - point_b[1]
    bcz = point_c[2] - point_b[2]
    mag_ba = math.sqrt(bax * bax + bay * bay + baz * baz)
    mag_bc = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    cosine = (bax * bcx + bay * bcy + baz * bcz) / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _xyz(kp: Keypoint2D) -> tuple[float, float, float] | None:
    """Return camera metres, or None if this joint has no depth."""
    if kp.x_m is None or kp.y_m is None or kp.z_m is None:
        return None
    return (float(kp.x_m), float(kp.y_m), float(kp.z_m))


def compute_configured_angles_3d(
    keypoints: list[Keypoint2D],
    angle_specs: list[dict],
    min_confidence: float,
    frame_specs: list[dict] | None = None,
) -> tuple[list[Angle2D], list[JointFrame]]:
    """Compute each YAML angle from the joint frame when it exists.

    Args:
        keypoints: Joints with camera metres where depth was valid.
        angle_specs: YAML angles list ({name, points}).
        min_confidence: Skip a joint below this.
        frame_specs: YAML joint_frames list. Empty keeps the raw 3-point angle.

    Returns:
        Angles for the dials, and the frames that were built (for drawing).
    """
    frames = build_joint_frames(keypoints, frame_specs or [], min_confidence)
    by_frame = {frame.name: frame for frame in frames}
    by_name = {kp.name: kp for kp in keypoints}
    results: list[Angle2D] = []
    for spec in angle_specs:
        framed = by_frame.get(str(spec.get("name", "")))
        if framed is not None:
            results.append(
                Angle2D(
                    name=framed.name,
                    degrees=framed.degrees,
                    vertex_u_px=framed.vertex_u_px,
                    vertex_v_px=framed.vertex_v_px,
                    min_confidence=framed.min_confidence,
                )
            )
            continue
        names = spec.get("points") or []
        if len(names) != 3:
            continue
        joints = [by_name.get(name) for name in names]
        if any(j is None for j in joints):
            continue
        conf = min(j.confidence for j in joints)
        if conf < min_confidence:
            continue
        a, b, c = joints
        pa, pb, pc = _xyz(a), _xyz(b), _xyz(c)
        if pa is None or pb is None or pc is None:
            continue
        degrees = angle_at_vertex_3d_deg(pa, pb, pc)
        if degrees is None:
            continue
        results.append(
            Angle2D(
                name=str(spec["name"]),
                degrees=degrees,
                vertex_u_px=b.u_px,
                vertex_v_px=b.v_px,
                min_confidence=conf,
            )
        )
    return results, frames
