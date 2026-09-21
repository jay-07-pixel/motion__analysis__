"""Live RGB frames from the Intel RealSense D455f.

What this module does (Step 1):
    Opens the colour sensor, waits for frames, and returns each frame as a
    NumPy image that OpenCV can show. Depth is NOT started yet — sir said
    2D first, and 2D keypoints will run on this RGB image.

Why camera coordinates start here:
    Each frame has a width and height from config. A pixel (u, v) on this
    image is already in 2D camera coordinates (origin = top-left of RGB).
"""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs

from src.capture.base_source import RGBSource


class LiveRealSenseRGB(RGBSource):
    """Owns the RealSense pipeline for the colour (RGB) stream only.

    One object = one open camera. Call start(), then get_frame() in a loop,
    then stop() when the window closes.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        """Store stream settings from config (do not start the camera yet).

        Args:
            width: Colour frame width in pixels (e.g. 1280).
            height: Colour frame height in pixels (e.g. 720).
            fps: Requested colour frames per second (e.g. 30).
        """
        self.width = width
        self.height = height
        self.fps = fps

        # pipeline = the RealSense SDK object that talks to the USB camera.
        self._pipeline = rs.pipeline()
        # config = which streams to enable (here: colour only).
        self._rs_config = rs.config()
        self._running = False

    @property
    def ended(self) -> bool:
        """Live USB never ends by itself (user presses q)."""
        return False

    @property
    def label(self) -> str:
        """HUD tag for the live D455f."""
        return "live"

    def start(self) -> None:
        """Enable the RGB stream and start the camera.

        Raises:
            RuntimeError: no D455f, USB 2 instead of USB 3, or width/height
                not supported. Try 640x480 in config.yaml if this fails.
        """
        # rs.stream.color = the RGB sensor (same image you saw in Viewer 2D).
        # rs.format.bgr8  = Blue-Green-Red, 8 bits per channel — OpenCV native.
        self._rs_config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        self._pipeline.start(self._rs_config)
        self._running = True

    def get_frame(self) -> np.ndarray | None:
        """Wait for the next colour frame and return it as a BGR image.

        Returns:
            NumPy array shaped (height, width, 3), dtype uint8, or None if
            this wait did not contain a colour frame (rare; skip and retry).
        """
        if not self._running:
            raise RuntimeError("Camera is not started. Call start() first.")

        # Blocks until the device has a new set of frames.
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        # asanyarray copies the RealSense buffer into a NumPy image we can draw on.
        return np.asanyarray(color_frame.get_data())

    def stop(self) -> None:
        """Release the camera so Viewer or another script can use it.

        RealSense allows only one program at a time. Close Viewer before
        running this script, and always call stop() (the Step 1 script does
        this in a finally block).
        """
        if self._running:
            self._pipeline.stop()
            self._running = False
