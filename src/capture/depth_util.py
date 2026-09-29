"""Shared RealSense depth helpers (live USB and .bag)."""

from __future__ import annotations

import pyrealsense2 as rs


def color_intrinsics_from_profile(profile):
    """Read fx, fy, cx, cy from the colour stream on this pipeline."""
    for stream in profile.get_streams():
        if stream.stream_type() == rs.stream.color:
            return stream.as_video_stream_profile().get_intrinsics()
    return None


def median_depth_m(depth_frame, u_px: float, v_px: float, window: int) -> float:
    """Median of valid get_distance samples in a small window. 0 = missing."""
    if depth_frame is None:
        return 0.0
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    ui = int(round(u_px))
    vi = int(round(v_px))
    half = max(0, int(window) // 2)
    samples: list[float] = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            x = ui + dx
            y = vi + dy
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            z_m = float(depth_frame.get_distance(x, y))
            if z_m > 0.0:
                samples.append(z_m)
    if not samples:
        return 0.0
    samples.sort()
    return samples[len(samples) // 2]
