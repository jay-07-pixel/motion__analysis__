"""Deproject 2D pixels + RealSense depth into camera 3D metres.

Origin is the colour-camera optical centre (not the top-left of the image).
    X = right
    Y = down
    Z = forward (depth)

This uses aligned depth at (u, v). We do NOT use MediaPipe world landmarks.
Those are a model guess, not RealSense metres.
"""

from __future__ import annotations

from dataclasses import replace

import pyrealsense2 as rs

from src.pose.keypoints import Keypoint2D


def attach_camera_xyz(
    keypoints: list[Keypoint2D],
    source,
    min_depth_m: float,
    max_depth_m: float,
    sample_window: int,
) -> list[Keypoint2D]:
    """Copy each 2D joint and fill x_m, y_m, z_m when depth is valid.

    Args:
        keypoints: Pose/Hands joints in colour pixels.
        source: RGB source that can report aligned depth + colour intrinsics.
        min_depth_m: Ignore closer readings (noise / invalid).
        max_depth_m: Ignore farther readings (D455f useful range ~6 m).
        sample_window: Odd pixel window; median depth at the joint.

    Returns:
        New keypoint list. Joints with no depth keep x_m/y_m/z_m as None.
    """
    intrinsics = source.color_intrinsics()
    if intrinsics is None:
        return keypoints

    filled: list[Keypoint2D] = []
    for kp in keypoints:
        z_m = source.depth_distance_m(kp.u_px, kp.v_px, sample_window)
        if z_m < min_depth_m or z_m > max_depth_m:
            filled.append(replace(kp, x_m=None, y_m=None, z_m=None))
            continue
        point = rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [float(kp.u_px), float(kp.v_px)],
            float(z_m),
        )
        filled.append(
            replace(
                kp,
                x_m=float(point[0]),
                y_m=float(point[1]),
                z_m=float(point[2]),
                coord_frame="camera_3d_metre",
            )
        )
    return filled
