"""Shared RealSense depth helpers (live USB and .bag)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pyrealsense2 as rs


def color_intrinsics_from_profile(profile):
    """Read fx, fy, cx, cy from the colour stream on this pipeline."""
    for stream in profile.get_streams():
        if stream.stream_type() == rs.stream.color:
            return stream.as_video_stream_profile().get_intrinsics()
    return None


@dataclass(frozen=True)
class DepthSampleInspect:
    """Diagnostic view of one depth sample. None means unavailable, not zero."""

    in_image: bool | None
    n_valid: int | None
    depth_sample_m: float | None


def collect_depth_samples(depth_frame, u_px: float, v_px: float, window: int) -> tuple[bool | None, list[float]]:
    """Valid get_distance values in the window. An empty list is missing, not zero depth."""
    if depth_frame is None:
        return None, []
    if not math.isfinite(u_px) or not math.isfinite(v_px):
        return None, []
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    ui = int(round(u_px))
    vi = int(round(v_px))
    in_image = 0 <= ui < width and 0 <= vi < height
    half = max(0, int(window) // 2)
    samples: list[float] = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            x = ui + dx
            y = vi + dy
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            z_m = float(depth_frame.get_distance(x, y))
            if z_m > 0.0 and math.isfinite(z_m):
                samples.append(z_m)
    return in_image, samples


def robust_depth_from_samples(
    samples: list[float],
    *,
    min_valid: int,
    max_spread_m: float | None,
) -> float | None:
    """Median of one consistent depth cluster, or None when the window is not usable.

    A nearer cluster that is too small does not license the farther cluster.
    The farther readings are background on a silhouette edge. None is invalid,
    not a substitute depth.
    """
    valid = sorted(float(z) for z in samples if z > 0.0 and math.isfinite(z))
    if len(valid) < int(min_valid):
        return None
    if max_spread_m is None or valid[-1] - valid[0] <= float(max_spread_m):
        return valid[len(valid) // 2]
    clusters: list[list[float]] = []
    start = 0
    for index in range(1, len(valid)):
        if valid[index] - valid[index - 1] > float(max_spread_m):
            clusters.append(valid[start:index])
            start = index
    clusters.append(valid[start:])
    nearest = min(clusters, key=lambda group: group[0])
    if len(nearest) < int(min_valid):
        return None
    return nearest[len(nearest) // 2]


def accept_depth_change(
    previous_z_m: float | None,
    new_z_m: float,
    pixel_step_px: float,
    *,
    max_step_m: float,
    fast_px: float,
) -> bool:
    """Reject a large depth jump that is not matched by image motion.

    The previous depth is not returned in its place.
    """
    if previous_z_m is None or not math.isfinite(previous_z_m):
        return True
    if abs(float(new_z_m) - float(previous_z_m)) <= float(max_step_m):
        return True
    return float(pixel_step_px) > float(fast_px)


def torso_gap_ok(z_m: float, reference_z_m: float | None, max_gap_m: float | None) -> bool:
    """An elbow or wrist cannot sit much farther from the shoulder than an arm allows."""
    if reference_z_m is None or max_gap_m is None:
        return True
    if not math.isfinite(reference_z_m) or not math.isfinite(z_m):
        return False
    return abs(float(z_m) - float(reference_z_m)) <= float(max_gap_m)


def inspect_depth_sample(depth_frame, u_px: float, v_px: float, window: int) -> DepthSampleInspect:
    """Same window as median_depth_m, plus counts and bounds. Does not invent depth."""
    if depth_frame is None or not math.isfinite(u_px) or not math.isfinite(v_px):
        return DepthSampleInspect(in_image=None, n_valid=None, depth_sample_m=None)
    in_image, samples = collect_depth_samples(depth_frame, u_px, v_px, window)
    if not samples:
        return DepthSampleInspect(in_image=in_image, n_valid=0, depth_sample_m=0.0)
    samples.sort()
    return DepthSampleInspect(
        in_image=in_image,
        n_valid=len(samples),
        depth_sample_m=samples[len(samples) // 2],
    )


def median_depth_m(depth_frame, u_px: float, v_px: float, window: int) -> float:
    """Median of valid get_distance samples in a small window. 0 = missing."""
    if depth_frame is None:
        return 0.0
    _in_image, samples = collect_depth_samples(depth_frame, u_px, v_px, window)
    if not samples:
        return 0.0
    samples.sort()
    return samples[len(samples) // 2]
