"""Side-pair twist scoring for IK fit checks.

Bone lengths are set before IK. These helpers only measure how far the
recorded lateral/medial pairs sit from the posed model pairs. They do not
change marker coordinates or joint ranges.
"""

from __future__ import annotations

import math

import numpy as np

_WRIST_PAIRS = (
    ("r_lwrist_study", "r_mwrist_study"),
    ("L_lwrist_study", "L_mwrist_study"),
)
_ELBOW_PAIRS = (
    ("r_lelbow_study", "r_melbow_study"),
    ("L_lelbow_study", "L_melbow_study"),
)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def _pair_twist_angles_deg(
    trc_frames: list[dict],
    model_rows: list[dict],
    pairs: tuple[tuple[str, str], ...],
) -> list[float]:
    if len(trc_frames) != len(model_rows):
        raise ValueError("TRC and model-marker tables do not have the same number of frames.")
    angles: list[float] = []
    for frame, row in zip(trc_frames, model_rows):
        for left, right in pairs:
            a = frame["points"].get(left)
            b = frame["points"].get(right)
            if a is None or b is None:
                continue
            if f"{left}_tx" not in row or f"{right}_tx" not in row:
                continue
            recorded = _unit(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
            model = _unit(
                np.array(
                    [
                        float(row[f"{left}_tx"]) - float(row[f"{right}_tx"]),
                        float(row[f"{left}_ty"]) - float(row[f"{right}_ty"]),
                        float(row[f"{left}_tz"]) - float(row[f"{right}_tz"]),
                    ],
                    dtype=float,
                )
            )
            if recorded is None or model is None:
                continue
            angles.append(float(np.degrees(np.arccos(np.clip(float(np.dot(recorded, model)), -1.0, 1.0)))))
    return angles


def wrist_pair_twist_angles_deg(trc_frames: list[dict], model_rows: list[dict]) -> list[float]:
    """Angle in degrees between each recorded wrist side-pair and the model pair."""
    return _pair_twist_angles_deg(trc_frames, model_rows, _WRIST_PAIRS)


def score_forearm_rotation(trc_frames: list[dict], model_rows: list[dict], marker_rms_mean_m: float | None) -> dict:
    """Lower is better. Wrist and elbow side-pair twist both count; RMS breaks ties."""
    wrist = wrist_pair_twist_angles_deg(trc_frames, model_rows)
    elbow = _pair_twist_angles_deg(trc_frames, model_rows, _ELBOW_PAIRS)
    median_wrist = 180.0 if not wrist else float(np.median(wrist))
    median_elbow = 180.0 if not elbow else float(np.median(elbow))
    combined = 0.6 * median_wrist + 0.4 * median_elbow
    rms = float(marker_rms_mean_m) if marker_rms_mean_m is not None and math.isfinite(marker_rms_mean_m) else 1.0
    return {
        "median_wrist_pair_twist_deg": round(median_wrist, 3),
        "median_elbow_pair_twist_deg": round(median_elbow, 3),
        "marker_rms_mean_m": None if marker_rms_mean_m is None else round(float(marker_rms_mean_m), 5),
        "score": round(combined + 100.0 * rms, 5),
        "n_pair_samples": len(wrist) + len(elbow),
    }
