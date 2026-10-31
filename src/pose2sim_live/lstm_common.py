"""Shared Pose2Sim v0.3 LSTM preprocess helpers. Official hip-center recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def camera_to_yup(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, -y, z], dtype=np.float64)


def yup_to_camera(x: float, y: float, z: float) -> tuple[float, float, float]:
    return float(x), float(-y), float(z)


def normalize_lstm_sequence(
    feature_xyz: np.ndarray,
    hip_yup: np.ndarray,
    height_m: float,
    mass_kg: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    if feature_xyz.ndim != 2 or feature_xyz.shape[1] % 3 != 0:
        raise ValueError(f"feature_xyz must be (T, 3N), got {feature_xyz.shape}")
    if hip_yup.shape != (feature_xyz.shape[0], 3):
        raise ValueError(f"hip_yup must be (T, 3), got {hip_yup.shape}")
    n_xyz = feature_xyz.shape[1]
    centered = feature_xyz - np.tile(hip_yup, (1, n_xyz // 3))
    scaled = centered / float(height_m)
    extras = np.column_stack(
        [
            np.full((scaled.shape[0], 1), float(height_m)),
            np.full((scaled.shape[0], 1), float(mass_kg)),
        ]
    )
    inputs = np.concatenate([scaled, extras], axis=1)
    return (inputs - mean) / std


def denorm_lstm_outputs(
    raw: np.ndarray,
    hip_yup: np.ndarray,
    height_m: float,
    n_output: int,
) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        hip = np.asarray(hip_yup, dtype=np.float64).reshape(1, 3)
    else:
        hip = np.asarray(hip_yup, dtype=np.float64)
    if arr.shape[-1] != n_output:
        raise ValueError(f"raw last dim must be {n_output}, got {arr.shape}")
    return arr * float(height_m) + np.tile(hip, (1, n_output // 3))


def load_norm_stats(model_dir: Path, n_input: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.load(model_dir / "mean.npy", allow_pickle=True).astype(np.float64).reshape(-1)
    std = np.load(model_dir / "std.npy", allow_pickle=True).astype(np.float64).reshape(-1)
    if mean.shape != (n_input,) or std.shape != (n_input,):
        raise ValueError(f"mean/std must be ({n_input},), got {mean.shape} {std.shape}")
    return mean, std
