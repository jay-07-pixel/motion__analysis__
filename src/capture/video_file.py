"""Uploaded colour movie (.mp4, .avi, .mov, ...).

What:
    Reads a normal video file with OpenCV and returns RGB frames as BGR
    images — the same type live_realsense.py returns.

Why:
    Sir asked for uploaded video as well as live. An mp4 is colour-only:
    good for 2D keypoints, NOT for later 3D (there is no depth in an mp4).

How:
    cv2.VideoCapture reads one frame per get_frame() call. Pixel (u, v)
    is still 2D camera/image coordinates (origin = top-left of the frame).
"""

from __future__ import annotations

from pathlib import Path

import cv2

from src.capture.base_source import RGBSource


# Extensions OpenCV can usually open as a colour movie.
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


class VideoFileRGB(RGBSource):
    """Plays a colour video file as a sequence of BGR frames."""

    def __init__(self, file_path: Path, loop: bool = False) -> None:
        """Store the path; the file is opened in start().

        Args:
            file_path: Path to the uploaded movie.
            loop: If True, restart from frame 0 when the file ends.
        """
        self.file_path = Path(file_path)
        self.loop = loop
        self._capture: cv2.VideoCapture | None = None
        self._ended = False

    @property
    def ended(self) -> bool:
        """True after the last frame, unless loop is enabled."""
        return self._ended

    @property
    def label(self) -> str:
        """HUD tag, e.g. 'mp4' or 'avi'."""
        return self.file_path.suffix.lower().lstrip(".") or "video"

    def start(self) -> None:
        """Open the video file.

        Raises:
            FileNotFoundError: path does not exist.
            RuntimeError: OpenCV could not decode the file.
        """
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.file_path}")

        self._capture = cv2.VideoCapture(str(self.file_path))
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(
                f"Could not open video (codec / path problem): {self.file_path}"
            )
        self._ended = False

    def get_frame(self) -> np.ndarray | None:
        """Read the next frame. Sets ended when the movie is finished."""
        if self._capture is None:
            raise RuntimeError("Video is not started. Call start() first.")

        ok, frame = self._capture.read()
        if ok:
            return frame

        # End of file.
        if self.loop:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok:
                return frame

        self._ended = True
        return None

    def playback_fps(self) -> float:
        """FPS stored in the file, or 0 if unknown (caller then uses config)."""
        if self._capture is None:
            return 0.0
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 1.0 else 0.0

    def stop(self) -> None:
        """Close the file so another program can open it."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
