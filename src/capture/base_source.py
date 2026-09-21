"""Shared interface for every RGB source (live camera, mp4, bag).

Why this exists:
    Sir asked for live AND uploaded. Pose (Step 3) must not care which one
    produced the image. Every source here returns the same thing: a BGR
    NumPy image whose pixels are 2D camera coordinates (origin top-left).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RGBSource(ABC):
    """One colour-frame producer. start() -> get_frame() loop -> stop()."""

    @abstractmethod
    def start(self) -> None:
        """Open the camera or file. Call once before get_frame()."""

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """Return the next BGR image, or None if this tick has no colour frame.

        After the file is finished, implementations set `ended` to True and
        may keep returning None.
        """

    @abstractmethod
    def stop(self) -> None:
        """Release the camera or file handle."""

    @property
    @abstractmethod
    def ended(self) -> bool:
        """True when an uploaded file has no more frames. Live is always False."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Short name for the HUD, e.g. 'live', 'mp4', 'bag'."""
