"""Leave wrist study markers out of IK when spacing or forearm roll is wrong.

The recorded TRC is not edited. On a frame, both wrist estimate points of one
side are copied unchanged when:

1. their distance is within the same-parent mismatch flag of the model pair, and
2. the angle between the elbow ML pair and the wrist ML pair around the forearm
   is within the roll tolerance of the model.

Otherwise both are omitted on that frame. They are not moved. Elbow gating is
separate and runs first.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.opensim_upper.residuals import REPORT_MISMATCH_M
from src.opensim_upper.scaling import spacing_mapping

WRIST_STUDY_PAIRS = (
    ("r_lwrist_study", "r_mwrist_study"),
    ("L_lwrist_study", "L_mwrist_study"),
)

# Elbow ML and wrist ML are nearly aligned on the model (~7 deg). LSTM often
# places them near 90 deg apart; that cannot be fit with locked wrist DOFs.
WRIST_ELBOW_PAIRS = {
    ("r_lwrist_study", "r_mwrist_study"): ("r_lelbow_study", "r_melbow_study"),
    ("L_lwrist_study", "L_mwrist_study"): ("L_lelbow_study", "L_melbow_study"),
}
WRIST_ROLL_TOLERANCE_DEG = 30.0


def model_wrist_pair_distances(model_path) -> dict[tuple[str, str], float]:
    """Straight-line distance between each wrist study pair on this model."""
    locations = _model_locations(model_path)
    distances = {}
    for left, right in WRIST_STUDY_PAIRS:
        if left not in locations or right not in locations:
            raise ValueError(f"The model has no location for {left} and {right}.")
        distances[(left, right)] = float(np.linalg.norm(locations[left] - locations[right]))
    return distances


def model_wrist_elbow_roll_deg(model_path) -> dict[tuple[str, str], float]:
    """Unsigned angle (deg) between elbow ML and wrist ML around the forearm."""
    locations = _model_locations(model_path)
    rolls = {}
    for wrist_pair, elbow_pair in WRIST_ELBOW_PAIRS.items():
        roll = _ml_roll_deg(
            locations[elbow_pair[0]],
            locations[elbow_pair[1]],
            locations[wrist_pair[0]],
            locations[wrist_pair[1]],
        )
        if roll is None:
            raise ValueError(f"The model has no forearm roll for {wrist_pair}.")
        rolls[wrist_pair] = float(roll)
    return rolls


def gate_wrist_frames(
    frames: list[dict],
    pair_distance_m: dict[tuple[str, str], float],
    tolerance_m: float = REPORT_MISMATCH_M,
    pair_roll_deg: dict[tuple[str, str], float] | None = None,
    roll_tolerance_deg: float = WRIST_ROLL_TOLERANCE_DEG,
) -> tuple[list[dict], dict]:
    """Return IK frames and a summary. The input frames are not modified."""
    if tolerance_m < 0 or tolerance_m != tolerance_m:
        raise ValueError("Wrist spacing tolerance must be a finite number >= 0.")
    if roll_tolerance_deg < 0 or roll_tolerance_deg != roll_tolerance_deg:
        raise ValueError("Wrist roll tolerance must be a finite number >= 0.")
    expected = set(WRIST_STUDY_PAIRS)
    if set(pair_distance_m) != expected:
        raise ValueError("Wrist spacing gate requires both study pairs and no other pairs.")
    if pair_roll_deg is not None and set(pair_roll_deg) != expected:
        raise ValueError("Wrist roll gate requires both study pairs and no other pairs.")
    gated = []
    counts = {
        pair: {
            "both_present": 0,
            "kept": 0,
            "distances_m": [],
            "rolls_deg": [],
            "left_out_spacing": 0,
            "left_out_roll": 0,
        }
        for pair in WRIST_STUDY_PAIRS
    }
    for frame in frames:
        points = dict(frame["points"])
        for pair in WRIST_STUDY_PAIRS:
            left, right = pair
            a = points.get(left)
            b = points.get(right)
            if a is None or b is None:
                points[left] = None
                points[right] = None
                continue
            distance = float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
            counts[pair]["both_present"] += 1
            counts[pair]["distances_m"].append(distance)
            if abs(distance - float(pair_distance_m[pair])) > tolerance_m:
                counts[pair]["left_out_spacing"] += 1
                points[left] = None
                points[right] = None
                continue
            if pair_roll_deg is not None:
                elbow = WRIST_ELBOW_PAIRS[pair]
                e0 = points.get(elbow[0])
                e1 = points.get(elbow[1])
                roll = None
                if e0 is not None and e1 is not None:
                    roll = _ml_roll_deg(e0, e1, a, b)
                if roll is None or abs(float(roll) - float(pair_roll_deg[pair])) > roll_tolerance_deg:
                    if roll is not None:
                        counts[pair]["rolls_deg"].append(float(roll))
                    counts[pair]["left_out_roll"] += 1
                    points[left] = None
                    points[right] = None
                    continue
                counts[pair]["rolls_deg"].append(float(roll))
            counts[pair]["kept"] += 1
        gated.append({**frame, "points": points})
    pairs = []
    for pair in WRIST_STUDY_PAIRS:
        distances = counts[pair]["distances_m"]
        rolls = counts[pair]["rolls_deg"]
        both = counts[pair]["both_present"]
        kept = counts[pair]["kept"]
        pairs.append(
            {
                "markers": list(pair),
                "side": "right" if pair[0].startswith(("r_", "R")) else "left",
                "model_distance_m": float(pair_distance_m[pair]),
                "median_distance_m": float(np.median(distances)) if distances else None,
                "model_elbow_roll_deg": (
                    None if pair_roll_deg is None else float(pair_roll_deg[pair])
                ),
                "median_elbow_roll_deg": float(np.median(rolls)) if rolls else None,
                "n_both_present": both,
                "n_kept": kept,
                "n_left_out": both - kept,
                "n_left_out_spacing": counts[pair]["left_out_spacing"],
                "n_left_out_roll": counts[pair]["left_out_roll"],
                "tolerance_m": float(tolerance_m),
                "roll_tolerance_deg": float(roll_tolerance_deg),
                "coordinates_moved": False,
            }
        )
    summary = {
        "clinical_accuracy": False,
        "coordinates_moved": False,
        "wrist_markers_gated": True,
        "tolerance_m": float(tolerance_m),
        "roll_tolerance_deg": float(roll_tolerance_deg),
        "pairs": pairs,
        "statement": _statement(pairs, roll_checked=pair_roll_deg is not None),
    }
    return gated, summary


def _model_locations(model_path) -> dict[str, np.ndarray]:
    return {
        row["target"]: np.asarray(row["location_m"], dtype=float)
        for row in spacing_mapping(Path(model_path))["mapped"]
    }


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def _ml_roll_deg(elbow_a, elbow_b, wrist_a, wrist_b) -> float | None:
    """Unsigned angle between elbow ML and wrist ML in the forearm-normal plane."""
    e0 = np.asarray(elbow_a, dtype=float)
    e1 = np.asarray(elbow_b, dtype=float)
    w0 = np.asarray(wrist_a, dtype=float)
    w1 = np.asarray(wrist_b, dtype=float)
    e_mid = 0.5 * (e0 + e1)
    w_mid = 0.5 * (w0 + w1)
    forearm = _unit(w_mid - e_mid)
    if forearm is None:
        return None
    e_ml = e0 - e1
    w_ml = w0 - w1
    e_ml = e_ml - float(np.dot(e_ml, forearm)) * forearm
    w_ml = w_ml - float(np.dot(w_ml, forearm)) * forearm
    e_u = _unit(e_ml)
    w_u = _unit(w_ml)
    if e_u is None or w_u is None:
        return None
    cosine = float(np.clip(np.dot(e_u, w_u), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _statement(pairs: list[dict], roll_checked: bool) -> str:
    model_values = {round(float(row["model_distance_m"]), 5) for row in pairs}
    if len(model_values) == 1:
        spacing = f"{pairs[0]['model_distance_m']:.3f} m"
    else:
        spacing = "the skeleton spacing for that side"
    tolerance = pairs[0]["tolerance_m"]
    sentences = [
        "The two wrist estimate points on a side were left out of the fit on any frame "
        f"where their distance was more than {tolerance:.3f} m from the skeleton spacing of {spacing}."
    ]
    if roll_checked:
        roll_tol = pairs[0]["roll_tolerance_deg"]
        model_roll = pairs[0].get("model_elbow_roll_deg")
        roll_txt = (
            f"{model_roll:.1f} deg"
            if model_roll is not None
            else "the skeleton forearm roll"
        )
        sentences.append(
            f"They were also left out when the elbow-to-wrist ML roll differed by more than "
            f"{roll_tol:.0f} deg from {roll_txt}."
        )
    sentences.append("The saved points were not moved.")
    sentences.append("A narrow or twisted pair cannot sit on both sides of the radius at once.")
    for row in pairs:
        side = str(row["side"]).capitalize()
        if row["median_distance_m"] is None:
            sentences.append(
                f"{side} wrist: the two points were not available together, so that side was left out."
            )
            continue
        extra = ""
        if row.get("median_elbow_roll_deg") is not None:
            extra = f" roll typically {row['median_elbow_roll_deg']:.0f} deg,"
        sentences.append(
            f"{side} wrist: the points are typically {row['median_distance_m']:.3f} m apart,"
            f"{extra} and were used on {row['n_kept']} of {row['n_both_present']} frames."
        )
    return " ".join(sentences)
