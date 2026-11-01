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

from src.capture.depth_util import accept_depth_change, robust_depth_from_samples, torso_gap_ok
from src.pose.keypoints import Keypoint2D

TORSO_REFERENCE = {
    "left_elbow": "left_shoulder",
    "left_wrist": "left_shoulder",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_shoulder",
}


class DepthGate:
    """Remember the last accepted depth. A rejection does not reuse it."""

    def __init__(self, max_step_m: float, fast_px: float, max_torso_gap_m: float | None) -> None:
        self.max_step_m = float(max_step_m)
        self.fast_px = float(fast_px)
        self.max_torso_gap_m = None if max_torso_gap_m is None else float(max_torso_gap_m)
        self._last: dict[str, tuple[float, float, float]] = {}

    def consider(
        self,
        name: str,
        z_m: float | None,
        u_px: float,
        v_px: float,
        reference_z_m: float | None = None,
    ) -> float | None:
        if z_m is None or not math.isfinite(z_m):
            return None
        if not torso_gap_ok(z_m, reference_z_m, self.max_torso_gap_m):
            return None
        previous = self._last.get(name)
        previous_z = None if previous is None else previous[0]
        pixel_step = 0.0 if previous is None else math.hypot(u_px - previous[1], v_px - previous[2])
        if not accept_depth_change(
            previous_z,
            z_m,
            pixel_step,
            max_step_m=self.max_step_m,
            fast_px=self.fast_px,
        ):
            return None
        self._last[name] = (float(z_m), float(u_px), float(v_px))
        return float(z_m)


def _window_depth(source, u_px: float, v_px: float, sample_window: int, min_valid_pixels: int, max_spread_m):
    samples = source.depth_samples(u_px, v_px, sample_window)
    if int(min_valid_pixels) <= 1 and max_spread_m is None:
        if not samples:
            return source.depth_distance_m(u_px, v_px, sample_window)
        samples = sorted(samples)
        return samples[len(samples) // 2]
    return robust_depth_from_samples(samples, min_valid=int(min_valid_pixels), max_spread_m=max_spread_m)


def attach_camera_xyz(
    keypoints: list[Keypoint2D],
    source,
    min_depth_m: float,
    max_depth_m: float,
    sample_window: int,
    *,
    depth_gate: DepthGate | None = None,
    min_valid_pixels: int = 1,
    max_spread_m: float | None = None,
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

    spatial: dict[str, float | None] = {}
    for kp in keypoints:
        z_m = _window_depth(source, kp.u_px, kp.v_px, sample_window, min_valid_pixels, max_spread_m)
        if z_m is None or z_m < min_depth_m or z_m > max_depth_m:
            spatial[kp.name] = None
        else:
            spatial[kp.name] = float(z_m)

    filled: list[Keypoint2D] = []
    for kp in keypoints:
        z_m = spatial.get(kp.name)
        reference = None
        ref_name = TORSO_REFERENCE.get(kp.name)
        if ref_name is not None:
            reference = spatial.get(ref_name)
        if depth_gate is not None:
            z_m = depth_gate.consider(kp.name, z_m, kp.u_px, kp.v_px, reference)
        elif reference is not None and z_m is not None and not torso_gap_ok(z_m, reference, None):
            z_m = None
        if z_m is None:
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
    """Optional filter for camera X, Y, Z. Raw measurements are not overwritten in place.

    Small steps are blended. A step of fast_m or more is copied so a fast arm
    is not trailed. A missing depth stays missing.
    """

    def __init__(self, smooth: float, deadband_m: float, hold_frames: int = 0, fast_m: float = 0.04) -> None:
        """Args come from camera.depth.filter in config.yaml.

        Args:
            smooth: 0..1. Share of the new point while the joint is moving slowly.
            deadband_m: Moves smaller than this stay on the last filtered point.
            hold_frames: Unused. A missing depth is left missing.
            fast_m: A step at least this large is copied as-is so a fast arm is not lagged.
        """
        self.smooth = min(1.0, max(0.0, float(smooth)))
        self.deadband_m = max(0.0, float(deadband_m))
        self.hold_frames = 0
        self.fast_m = max(0.0, float(fast_m))
        self._points: dict[str, tuple[float, float, float]] = {}
        self._misses: dict[str, int] = {}

    def apply(self, keypoints: list[Keypoint2D]) -> list[Keypoint2D]:
        """Return keypoints with steadier camera metres. 2D pixels are unchanged."""
        smoothed: list[Keypoint2D] = []
        for kp in keypoints:
            if kp.x_m is None or kp.y_m is None or kp.z_m is None:
                self._points.pop(kp.name, None)
                smoothed.append(replace(kp, x_m=None, y_m=None, z_m=None, filt_x_m=None, filt_y_m=None, filt_z_m=None))
                continue
            raw = (float(kp.x_m), float(kp.y_m), float(kp.z_m))
            previous = self._points.get(kp.name)
            point = raw if previous is None else self._blend(previous, raw)
            self._points[kp.name] = point
            smoothed.append(
                replace(
                    kp,
                    x_m=point[0],
                    y_m=point[1],
                    z_m=point[2],
                    filt_x_m=point[0],
                    filt_y_m=point[1],
                    filt_z_m=point[2],
                    coord_frame="camera_3d_metre",
                )
            )
        return smoothed

    def _blend(
        self,
        previous: tuple[float, float, float],
        current: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Hold tiny noise. A fast step is the raw point, so the filter does not trail the arm."""
        dist = math.dist(previous, current)
        if dist >= self.fast_m:
            return current
        if dist < self.deadband_m:
            return previous
        alpha = self.smooth
        return tuple(alpha * now + (1.0 - alpha) * old for now, old in zip(current, previous))
