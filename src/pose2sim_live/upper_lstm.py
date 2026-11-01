"""Official Pose2Sim v0.3 upper LSTM preprocess / postprocess.

Reproduces Pose2Sim 0.10.49 markerAugmentation.py for the upper model only.
Input landmarks are OpenSim Y-up metres. Hip is the centering root (proxy).
Neck is a shoulder midpoint proxy. Neither is an anatomical landmark.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.pose.keypoints import Keypoint2D

FEATURE_MARKERS = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RElbow",
    "LElbow",
    "RWrist",
    "LWrist",
)
RESPONSE_MARKERS = (
    "r_lelbow_study",
    "r_melbow_study",
    "r_lwrist_study",
    "r_mwrist_study",
    "L_lelbow_study",
    "L_melbow_study",
    "L_lwrist_study",
    "L_mwrist_study",
)
# MediaPipe observed names -> OpenPose feature names. Hips only for Hip proxy.
MP_TO_OPENPOSE = {
    "left_shoulder": "LShoulder",
    "right_shoulder": "RShoulder",
    "left_elbow": "LElbow",
    "right_elbow": "RElbow",
    "left_wrist": "LWrist",
    "right_wrist": "RWrist",
    "left_hip": "LHip",
    "right_hip": "RHip",
}
N_FEATURE_XYZ = 21  # 7 markers * 3
N_INPUT = 23  # 21 + height + mass
N_OUTPUT = 24  # 8 markers * 3
# Previous built-in person. Live sessions must pass explicit height_m / mass_kg;
# these constants are not applied when the user starts Pose2Sim.
HEIGHT_M = 1.80
MASS_KG = 61.0
DEFAULT_WINDOW = 45
LIVE_WINDOW = 16
DEFAULT_MIN_FRAMES = 8
# A few frames can lose depth. That must not erase the valid window.
# Those frames are not filled in. A longer run still clears the window.
DEFAULT_MAX_GAP = 8
MODEL_DIR = Path(
    r"C:\Users\jayjo\anaconda3\envs\pose2sim-offline\Lib\site-packages"
    r"\Pose2Sim\MarkerAugmenter\LSTM\v0.3_upper"
)
ISOLATED_PYTHON = Path(r"C:\Users\jayjo\anaconda3\envs\pose2sim-offline\python.exe")


def camera_to_yup(x: float, y: float, z: float) -> np.ndarray:
    """Camera metres (Y down) -> OpenSim Y-up metres."""
    return np.array([x, -y, z], dtype=np.float64)


def yup_to_camera(x: float, y: float, z: float) -> tuple[float, float, float]:
    """OpenSim Y-up metres -> camera metres (Y down)."""
    return float(x), float(-y), float(z)


def _finite_xyz(kp: Keypoint2D | None) -> np.ndarray | None:
    if kp is None:
        return None
    if kp.x_m is None or kp.y_m is None or kp.z_m is None:
        return None
    if not (np.isfinite(kp.x_m) and np.isfinite(kp.y_m) and np.isfinite(kp.z_m)):
        return None
    return np.array([kp.x_m, kp.y_m, kp.z_m], dtype=np.float64)


def extract_upper_frame(
    keypoints: list[Keypoint2D],
    min_visibility: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (feature_xyz 21, hip_yup 3) or None if any required XYZ is missing.

    Does not interpolate, hold stale values, or write zeros.
    Neck and Hip are geometric proxies.
    """
    by_name = {kp.name: kp for kp in keypoints}
    mapped: dict[str, np.ndarray] = {}
    for src, dst in MP_TO_OPENPOSE.items():
        kp = by_name.get(src)
        if kp is None or kp.confidence < min_visibility:
            return None
        cam = _finite_xyz(kp)
        if cam is None:
            return None
        mapped[dst] = camera_to_yup(float(cam[0]), float(cam[1]), float(cam[2]))
    neck = 0.5 * (mapped["LShoulder"] + mapped["RShoulder"])  # proxy, not C7
    hip = 0.5 * (mapped["LHip"] + mapped["RHip"])  # proxy, not ASIS/HJC
    mapped["Neck"] = neck
    feat = np.concatenate([mapped[name] for name in FEATURE_MARKERS], axis=0)
    return feat, hip


def normalize_sequence(
    feature_xyz: np.ndarray,
    hip_yup: np.ndarray,
    height_m: float,
    mass_kg: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Official upper preprocess: Hip-center, /height, append h,m, (x-mean)/std.

    feature_xyz: (T, 21), hip_yup: (T, 3), mean/std: (23,)
    """
    if feature_xyz.ndim != 2 or feature_xyz.shape[1] != N_FEATURE_XYZ:
        raise ValueError(f"feature_xyz must be (T, 21), got {feature_xyz.shape}")
    if hip_yup.shape != (feature_xyz.shape[0], 3):
        raise ValueError(f"hip_yup must be (T, 3), got {hip_yup.shape}")
    centered = feature_xyz - np.tile(hip_yup, (1, N_FEATURE_XYZ // 3))
    scaled = centered / float(height_m)
    extras = np.column_stack(
        [
            np.full((scaled.shape[0], 1), float(height_m)),
            np.full((scaled.shape[0], 1), float(mass_kg)),
        ]
    )
    inputs = np.concatenate([scaled, extras], axis=1)
    return (inputs - mean) / std


def denorm_outputs(
    raw: np.ndarray,
    hip_yup: np.ndarray,
    height_m: float,
) -> np.ndarray:
    """Official postprocess: *height + Hip. raw (T, 24) or (24,), hip matching T or (3,)."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        hip = np.asarray(hip_yup, dtype=np.float64).reshape(1, 3)
    else:
        hip = np.asarray(hip_yup, dtype=np.float64)
    if arr.shape[-1] != N_OUTPUT:
        raise ValueError(f"raw last dim must be 24, got {arr.shape}")
    unnorm = arr * float(height_m)
    return unnorm + np.tile(hip, (1, N_OUTPUT // 3))


def load_norm_stats(model_dir: Path = MODEL_DIR) -> tuple[np.ndarray, np.ndarray]:
    mean = np.load(model_dir / "mean.npy", allow_pickle=True).astype(np.float64).reshape(-1)
    std = np.load(model_dir / "std.npy", allow_pickle=True).astype(np.float64).reshape(-1)
    if mean.shape != (N_INPUT,) or std.shape != (N_INPUT,):
        raise ValueError(f"mean/std must be (23,), got {mean.shape} {std.shape}")
    return mean, std


@dataclass
class UpperFrame:
    time_sec: float
    feature_xyz: np.ndarray  # (21,) Y-up
    hip_yup: np.ndarray  # (3,) Y-up
    frame_index: int | None = None


class UpperSequenceBuffer:
    """Rolling valid-frame buffer. Incomplete frames are dropped, never filled."""

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        min_frames: int = DEFAULT_MIN_FRAMES,
        max_gap: int = DEFAULT_MAX_GAP,
    ) -> None:
        self.window = int(window)
        self.min_frames = int(min_frames)
        self.max_gap = int(max_gap)
        self.frames: deque[UpperFrame] = deque(maxlen=self.window)
        self.gap = 0

    def clear(self) -> None:
        self.frames.clear()
        self.gap = 0

    def push(
        self,
        extracted: tuple[np.ndarray, np.ndarray] | None,
        time_sec: float,
        frame_index: int | None = None,
    ) -> str:
        """Append a valid frame or record a gap. Returns 'ready' or 'buffering'."""
        if extracted is None:
            self.gap += 1
            if self.gap > self.max_gap:
                self.clear()
            return "buffering"
        self.gap = 0
        feat, hip = extracted
        stored_frame = None if frame_index is None else int(frame_index)
        self.frames.append(
            UpperFrame(time_sec=float(time_sec), feature_xyz=feat, hip_yup=hip, frame_index=stored_frame)
        )
        if len(self.frames) < self.min_frames:
            return "buffering"
        return "ready"

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        feat = np.stack([f.feature_xyz for f in self.frames], axis=0)
        hip = np.stack([f.hip_yup for f in self.frames], axis=0)
        return feat, hip
