"""Live RGB (and optional aligned depth) from the Intel RealSense D455f.

2D mode: colour stream only. Pixels (u, v) are 2D camera coordinates
(origin = top-left of RGB).

3D mode: colour + depth. Depth is aligned to colour so we can deproject
each (u, v) to camera metres (origin = colour optical centre).
"""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs

from src.capture.base_source import RGBSource
from src.capture.depth_util import color_intrinsics_from_profile, median_depth_m


class LiveRealSenseRGB(RGBSource):
    """Owns the RealSense pipeline. Colour always; depth only when 3D is on."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        enable_depth: bool = False,
        depth_width: int | None = None,
        depth_height: int | None = None,
        depth_fps: int | None = None,
    ) -> None:
        """Store stream settings from config (do not start the camera yet).

        Args:
            width: Colour frame width in pixels (e.g. 1280).
            height: Colour frame height in pixels (e.g. 720).
            fps: Requested colour frames per second (e.g. 30).
            enable_depth: True in 3D mode so we can read Z at each joint.
            depth_width: Depth stream width (defaults to colour width).
            depth_height: Depth stream height (defaults to colour height).
            depth_fps: Depth stream FPS (defaults to colour FPS).
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = bool(enable_depth)
        self.depth_width = int(depth_width or width)
        self.depth_height = int(depth_height or height)
        self.depth_fps = int(depth_fps or fps)

        self._pipeline = rs.pipeline()
        self._rs_config = rs.config()
        self._running = False
        self._align: rs.align | None = None
        self._depth_frame = None
        self._intrinsics = None

    @property
    def ended(self) -> bool:
        """Live USB never ends by itself (user presses q)."""
        return False

    @property
    def label(self) -> str:
        """HUD tag for the live D455f."""
        return "live"

    def start(self) -> None:
        """Enable RGB (and depth in 3D) and start the camera.

        Raises:
            RuntimeError: no D455f plugged in, Viewer already using it,
                USB 2 instead of USB 3, or width/height not supported.
        """
        try:
            connected = len(rs.context().query_devices())
        except Exception:
            connected = 0
        if connected == 0:
            raise RuntimeError(_CAMERA_MISSING)

        try:
            self._rs_config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )
            if self.enable_depth:
                self._rs_config.enable_stream(
                    rs.stream.depth,
                    self.depth_width,
                    self.depth_height,
                    rs.format.z16,
                    self.depth_fps,
                )
                self._align = rs.align(rs.stream.color)
            profile = self._pipeline.start(self._rs_config)
            self._intrinsics = color_intrinsics_from_profile(profile)
        except Exception as error:
            raise RuntimeError(_friendly_camera_error(error)) from error
        self._running = True

    def get_frame(self) -> np.ndarray | None:
        """Wait for the next colour frame (and aligned depth in 3D).

        Returns:
            NumPy array shaped (height, width, 3), dtype uint8, or None if
            this wait did not contain a colour frame (rare; skip and retry).
        """
        if not self._running:
            raise RuntimeError("Camera is not started. Call start() first.")

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        except Exception as error:
            raise RuntimeError(_friendly_camera_error(error)) from error

        if self._align is not None:
            frames = self._align.process(frames)
            self._depth_frame = frames.get_depth_frame()
        else:
            self._depth_frame = None

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        if self._intrinsics is None:
            self._intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        return np.asanyarray(color_frame.get_data())

    def color_intrinsics(self):
        """Colour intrinsics (fx, fy, cx, cy) after start(), else None."""
        return self._intrinsics

    def depth_distance_m(self, u_px: float, v_px: float, window: int = 1) -> float:
        """Median aligned depth in metres around a colour pixel."""
        return median_depth_m(self._depth_frame, u_px, v_px, window)

    def stop(self) -> None:
        """Release the camera so Viewer or another script can use it."""
        if self._running:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._running = False
            self._depth_frame = None
            self._align = None


_CAMERA_MISSING = (
    "Camera not connected / not found. "
    "Plug in the RealSense D455f on USB 3 and close RealSense Viewer."
)


def _friendly_camera_error(error: BaseException) -> str:
    """Turn a RealSense SDK exception into a short message for the GUI."""
    text = str(error).lower()
    if any(
        word in text
        for word in ("no device", "not found", "0 devices")
    ):
        return _CAMERA_MISSING
    if any(word in text for word in ("busy", "in use", "occupied", "failed to set power")):
        return (
            "Camera is in use. Close RealSense Viewer (or another app) and try Start again."
        )
    if "timeout" in text:
        return "Camera not connected / not found (no frames). Check the USB 3 cable."
    if "couldn't resolve" in text or "cannot resolve" in text:
        return (
            "Could not start colour+depth at this resolution. "
            "Use USB 3, close Viewer, or in config.yaml set camera.depth to 848x480."
        )
    return f"Could not start the camera: {error}"
