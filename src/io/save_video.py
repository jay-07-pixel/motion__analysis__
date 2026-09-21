"""Save the skeleton overlay as an .mp4 (optional Step 4 output).

Why:
    CSV is for numbers. An overlay video is for sir / handover: proof that
    the dots sat on the body, without needing the camera again.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class OverlayVideoWriter:
    """Writes BGR frames to an mp4. Opened on the first real frame."""

    def __init__(self, video_path: Path, fps: float, codec: str) -> None:
        """Store path and fps. The file is created on the first write().

        Args:
            video_path: e.g. data/output/.../overlay.mp4
            fps: Playback rate stored in the file.
            codec: FourCC string from config, usually 'mp4v'.
        """
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = video_path
        self.fps = max(1.0, float(fps))
        self.codec = codec
        self._writer: cv2.VideoWriter | None = None
        self.frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        """Append one overlay image. Size is taken from the first frame."""
        if self._writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                str(self.path),
                fourcc,
                self.fps,
                (width, height),
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not create overlay video: {self.path}")
        self._writer.write(frame)
        self.frame_count += 1

    def close(self) -> None:
        """Finish the file. Safe to call even if no frame was written."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
