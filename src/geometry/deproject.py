"""Deproject 2D pixels + RealSense depth into camera 3D metres.

Origin is the colour-camera optical centre (not the top-left of the image).
    X = right
    Y = down
    Z = forward (depth)

This uses aligned depth at (u, v). We do NOT use MediaPipe world landmarks.
Those are a model guess, not RealSense metres.
"""

from __future__ import annotations

import math
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


class CameraXyzSmoother:
    """Calm camera X, Y, Z so a still joint does not flicker every frame.

    RealSense depth and MediaPipe both jitter by a few millimetres. A still
    hand then looks like it is moving. Small steps are held; larger steps
    are blended with the previous camera point.
    """

    def __init__(self, smooth: float, deadband_m: float, hold_frames: int) -> None:
        """Args come from camera.depth in config.yaml.

        Args:
            smooth: 0..1. Share of the new point (1 = raw, 0.3 = calmer).
            deadband_m: Moves smaller than this stay on the last point.
            hold_frames: Keep the last good point if depth drops briefly.
        """
        self.smooth = min(1.0, max(0.0, float(smooth)))
        self.deadband_m = max(0.0, float(deadband_m))
        self.hold_frames = max(0, int(hold_frames))
        self._points: dict[str, tuple[float, float, float]] = {}
        self._misses: dict[str, int] = {}

    def apply(self, keypoints: list[Keypoint2D]) -> list[Keypoint2D]:
        """Return keypoints with steadier camera metres. 2D pixels are unchanged."""
        smoothed: list[Keypoint2D] = []
        for kp in keypoints:
            if kp.x_m is None or kp.y_m is None or kp.z_m is None:
                smoothed.append(self._hold_last(kp))
                continue
            previous = self._points.get(kp.name)
            if previous is None:
                point = (float(kp.x_m), float(kp.y_m), float(kp.z_m))
            else:
                point = self._blend(previous, (float(kp.x_m), float(kp.y_m), float(kp.z_m)))
            self._points[kp.name] = point
            self._misses[kp.name] = 0
            smoothed.append(
                replace(kp, x_m=point[0], y_m=point[1], z_m=point[2], coord_frame="camera_3d_metre")
            )
        return smoothed

    def _blend(
        self,
        previous: tuple[float, float, float],
        current: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Hold tiny noise; ease real motion toward the new camera point."""
        dist = math.dist(previous, current)
        if dist < self.deadband_m:
            return previous
        alpha = self.smooth
        return tuple(alpha * now + (1.0 - alpha) * old for now, old in zip(current, previous))

    def _hold_last(self, kp: Keypoint2D) -> Keypoint2D:
        """Reuse the last camera point for a few frames when depth is missing."""
        previous = self._points.get(kp.name)
        if previous is None:
            return kp
        misses = self._misses.get(kp.name, 0) + 1
        if misses > self.hold_frames:
            self._points.pop(kp.name, None)
            self._misses.pop(kp.name, None)
            return kp
        self._misses[kp.name] = misses
        return replace(
            kp,
            x_m=previous[0],
            y_m=previous[1],
            z_m=previous[2],
            coord_frame="camera_3d_metre",
        )
