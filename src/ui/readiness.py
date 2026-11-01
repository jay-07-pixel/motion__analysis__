"""Whether the person in view is inside the configured depth range.

Distance limits come from ``camera.depth.min_m`` and ``max_m``. Visibility
uses ``pose.min_visibility``. No extra cutoff is added here.
"""

from __future__ import annotations

SHOULDER_NAMES = ("left_shoulder", "right_shoulder")


def capture_readiness(
    keypoints,
    min_visibility: float,
    min_m: float,
    max_m: float,
) -> dict:
    """Shoulder visibility and RealSense depth against the configured range."""
    by_name = {kp.name: kp for kp in keypoints}
    visible = []
    distances = []
    for name in SHOULDER_NAMES:
        point = by_name.get(name)
        if point is None or float(point.confidence) < float(min_visibility):
            continue
        visible.append(name)
        z_m = getattr(point, "z_m", None)
        if z_m is not None and z_m == z_m:
            distances.append(float(z_m))
    distance = sum(distances) / len(distances) if distances else None
    both = len(visible) == len(SHOULDER_NAMES)
    if not both:
        instruction = "Step into view so both shoulders are visible."
        in_range = False
    elif distance is None:
        instruction = "Shoulders are visible, but depth is missing. Check the depth stream."
        in_range = False
    elif distance < float(min_m):
        instruction = (
            f"Too close ({distance:.2f} m). Configured capture range is "
            f"{float(min_m):.1f}–{float(max_m):.1f} m."
        )
        in_range = False
    elif distance > float(max_m):
        instruction = (
            f"Too far ({distance:.2f} m). Configured capture range is "
            f"{float(min_m):.1f}–{float(max_m):.1f} m."
        )
        in_range = False
    else:
        instruction = f"Within capture range ({distance:.2f} m)."
        in_range = True
    return {
        "shoulders_visible": both,
        "distance_m": distance,
        "in_range": in_range,
        "ready": both and in_range,
        "instruction": instruction,
        "min_m": float(min_m),
        "max_m": float(max_m),
    }
