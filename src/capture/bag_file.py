"""Uploaded RealSense recording (.bag).

What:
    Replays a .bag saved by RealSense Viewer (Record) and returns the
    colour stream as BGR images — same type as live USB.

Why:
    A .bag is the camera's own recording: RGB + depth + timestamps.
    Step 2 uses RGB only (2D). Depth stays in the file for later 3D.
    A plain .mp4 cannot do that.

How:
    pyrealsense2 plays the file with enable_device_from_file(). We read
    the colour frame only. Pixels are 2D camera coordinates on that RGB
    image (origin top-left).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from src.capture.base_source import RGBSource


class BagFileRGB(RGBSource):
    """Plays a RealSense .bag and yields the colour (RGB) stream."""

    def __init__(self, file_path: Path, loop: bool = False, realtime: bool = True) -> None:
        """Store path and playback options; the bag is opened in start().

        Args:
            file_path: Path to the .bag from Viewer Record.
            loop: If True, replay from the start when the bag ends.
            realtime: If True, play at recorded speed. If False, go as
                fast as the CPU (useful later for batch processing).
        """
        self.file_path = Path(file_path)
        self.loop = loop
        self.realtime = realtime
        self._pipeline: rs.pipeline | None = None
        self._ended = False

    @property
    def ended(self) -> bool:
        """True after the last colour frame, unless loop is enabled."""
        return self._ended

    @property
    def label(self) -> str:
        """HUD tag."""
        return "bag"

    def start(self) -> None:
        """Open the .bag as if it were the live D455f (colour stream).

        Raises:
            FileNotFoundError: path does not exist.
            RuntimeError: SDK could not play the bag.
        """
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Bag file not found: {self.file_path}")

        pipeline = rs.pipeline()
        rs_config = rs.config()
        # repeat_playback = loop: SDK restarts the bag when it ends.
        rs_config.enable_device_from_file(str(self.file_path), repeat_playback=self.loop)

        try:
            profile = pipeline.start(rs_config)
        except RuntimeError as error:
            raise RuntimeError(
                f"Could not play bag (not a RealSense recording?): {self.file_path}\n{error}"
            ) from error

        # Playback device: control speed. True = watch like a video.
        playback = profile.get_device().as_playback()
        playback.set_real_time(self.realtime)

        self._pipeline = pipeline
        self._ended = False

    def get_frame(self) -> np.ndarray | None:
        """Wait for the next colour frame from the bag.

        Returns None and sets ended when the recording is finished
        (and loop is False). Viewer bags are often RGB8, so we convert
        to BGR for OpenCV.
        """
        if self._pipeline is None:
            raise RuntimeError("Bag is not started. Call start() first.")

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            # Typical when the bag ends and loop is False.
            if not self.loop:
                self._ended = True
            return None

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        image = np.asanyarray(color_frame.get_data())
        # Match live_realsense.py: OpenCV windows expect BGR.
        if color_frame.profile.format() == rs.format.rgb8:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def stop(self) -> None:
        """Stop playback and release the bag file."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
