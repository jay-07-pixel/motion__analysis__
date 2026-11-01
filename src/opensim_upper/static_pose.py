"""Stable upper-body reference positions in camera metres.

Uses observed shoulders, elbows, and wrists, plus the eight upper LSTM
study markers. A frame counts only when every one of those points has
finite XYZ. Positions are the coordinate-wise median. No Y flip, no
invented landmarks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.pose2sim_live.upper_lstm import RESPONSE_MARKERS

OBSERVED = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
PREDICTED = tuple(RESPONSE_MARKERS)
REQUIRED = OBSERVED + PREDICTED


def _finite_xyz(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if hasattr(value, "x_m"):
        raw = (value.x_m, value.y_m, value.z_m)
    elif isinstance(value, dict):
        raw = (value.get("x_m"), value.get("y_m"), value.get("z_m"))
    else:
        raw = tuple(value)
    if len(raw) != 3 or any(item is None or not math.isfinite(float(item)) for item in raw):
        return None
    return float(raw[0]), float(raw[1]), float(raw[2])


class StaticPoseCapture:
    """Collect valid frames. Does not open a camera."""

    def __init__(self, min_frames: int = 8) -> None:
        if min_frames < 2:
            raise ValueError("min_frames must be at least 2")
        self.min_frames = int(min_frames)
        self.frames: list[dict] = []
        self.n_rejected = 0

    def add_frame(
        self,
        time_sec: float,
        observed: dict,
        predicted: dict,
    ) -> bool:
        """Store one frame. Returns False when any required point is missing."""
        points = {}
        missing = []
        for name in OBSERVED:
            xyz = _finite_xyz(observed.get(name))
            if xyz is None:
                missing.append(name)
            else:
                points[name] = xyz
        for name in PREDICTED:
            xyz = _finite_xyz(predicted.get(name))
            if xyz is None:
                missing.append(name)
            else:
                points[name] = xyz
        if missing:
            self.n_rejected += 1
            return False
        self.frames.append({"t_sec": float(time_sec), "points": points})
        return True

    def ready(self) -> bool:
        return len(self.frames) >= self.min_frames

    def summary(self, height_m: float, mass_kg: float) -> dict:
        """Median camera-metre positions. Raises if too few valid frames."""
        if not self.ready():
            raise RuntimeError(
                f"Need {self.min_frames} valid frames, have {len(self.frames)}. "
                f"Rejected {self.n_rejected}."
            )
        if not math.isfinite(height_m) or height_m <= 0:
            raise ValueError("height_m must be a positive finite height in metres")
        if not math.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError("mass_kg must be a positive finite mass in kilograms")

        landmarks = {}
        for name in REQUIRED:
            stack = np.array([frame["points"][name] for frame in self.frames], dtype=np.float64)
            median = np.median(stack, axis=0)
            std = np.std(stack, axis=0, ddof=0)
            kind = "observed" if name in OBSERVED else "estimated_lstm_study"
            landmarks[name] = {
                "x_m": float(median[0]),
                "y_m": float(median[1]),
                "z_m": float(median[2]),
                "std_m": {"x": float(std[0]), "y": float(std[1]), "z": float(std[2])},
                "source": kind,
                "coord_frame": "camera_3d_metre",
            }
        times = [frame["t_sec"] for frame in self.frames]
        return {
            "coord_frame": "camera_3d_metre",
            "axes": {"x": "right", "y": "down", "z": "forward"},
            "units": "metres",
            "participant_height_m": float(height_m),
            "participant_mass_kg": float(mass_kg),
            "n_valid_frames": len(self.frames),
            "n_rejected_frames": self.n_rejected,
            "t_start_sec": float(min(times)),
            "t_end_sec": float(max(times)),
            "timestamps_sec": times,
            "landmarks": landmarks,
            "model_registration": "not applied",
        }

    def save(self, path: Path, height_m: float, mass_kg: float) -> dict:
        payload = self.summary(height_m, mass_kg)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
