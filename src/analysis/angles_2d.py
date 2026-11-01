"""2D joint angles from three camera-pixel keypoints.

What:
    Angle at joint B, using points A-B-C in the RGB image.

Why:
    Skeleton dots are not analysis. An elbow angle (degrees) is a motion
    number we can show and save. This is still 2D: it is the angle in the
    camera image, not a true 3D bone angle.

How:
    Vectors BA and BC. Angle = arccos(dot / (|BA| |BC|)).
    180 deg = nearly straight. Smaller = more bent (for an elbow).

Limitation:
    If the arm points toward the camera, this 2D angle is wrong. Face the
    camera for Step 5. Real 3D angles come later from depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.pose.keypoints import Keypoint2D


@dataclass(frozen=True)
class Angle2D:
    """One named 2D angle for one frame."""

    name: str
    degrees: float
    vertex_u_px: float
    vertex_v_px: float
    # Lowest of the three joint confidences (so we can skip weak angles).
    min_confidence: float


def angle_at_vertex_deg(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    point_c: tuple[float, float],
) -> float | None:
    """Return the interior angle ABC in degrees, or None if points are degenerate.

    Args:
        point_a: First arm of the angle (e.g. shoulder), (u, v) pixels.
        point_b: Vertex (e.g. elbow).
        point_c: Second arm (e.g. wrist).
    """
    bax = point_a[0] - point_b[0]
    bay = point_a[1] - point_b[1]
    bcx = point_c[0] - point_b[0]
    bcy = point_c[1] - point_b[1]
    mag_ba = math.hypot(bax, bay)
    mag_bc = math.hypot(bcx, bcy)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    cosine = (bax * bcx + bay * bcy) / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def compute_configured_angles(
    keypoints: list[Keypoint2D],
    angle_specs: list[dict],
    min_confidence: float,
) -> list[Angle2D]:
    """Compute every angle listed in config (names, not hardcoded joints).

    Args:
        keypoints: This frame's 2D joints.
        angle_specs: YAML list of {name, points: [A, B, C]} where B is vertex.
        min_confidence: Skip if any of the three joints is below this.

    Returns:
        List of Angle2D (skipped specs are omitted).
    """
    by_name = {kp.name: kp for kp in keypoints}
    results: list[Angle2D] = []
    for spec in angle_specs:
        names = spec["points"]
        if len(names) != 3:
            continue
        joints = [by_name.get(name) for name in names]
        if any(j is None for j in joints):
            continue
        conf = min(j.confidence for j in joints)
        if conf < min_confidence:
            continue
        a, b, c = joints
        degrees = angle_at_vertex_deg(
            (a.u_px, a.v_px),
            (b.u_px, b.v_px),
            (c.u_px, c.v_px),
        )
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
    return results


def fill_missing_from_picture(
    keypoints: list[Keypoint2D],
    angle_specs: list[dict],
    measured: dict[str, float | None],
    min_confidence: float,
) -> dict[str, float | None]:
    """Dial values only. A measured angle is kept.

    When that angle is missing, the picture angle is used if the three dots
    are visible. This dict must not replace the saved angle series.
    """
    specs = {str(spec.get("name", "")): spec for spec in angle_specs}
    by_name = {kp.name: kp for kp in keypoints}
    shown: dict[str, float | None] = {}
    for name, value in measured.items():
        if value is not None:
            shown[name] = float(value)
            continue
        spec = specs.get(name) or {}
        points = spec.get("points") or []
        if len(points) != 3:
            shown[name] = None
            continue
        joints = [by_name.get(str(item)) for item in points]
        if any(joint is None or joint.confidence < min_confidence for joint in joints):
            shown[name] = None
            continue
        first, vertex, last = joints
        shown[name] = angle_at_vertex_deg(
            (first.u_px, first.v_px),
            (vertex.u_px, vertex.v_px),
            (last.u_px, last.v_px),
        )
    return shown
