"""Leave wrist study markers out of IK when spacing or forearm roll is wrong.

The recorded TRC is not edited. On a frame, both wrist estimate points of one
side are copied unchanged when:

1. their distance is within the same-parent mismatch flag of the model pair, and
2. (optional) their elbow-to-wrist ML roll in camera/ground frame lies inside
   the model's reachable ground-frame roll band over ``pro_sup`` (and a few
   ``elbow_flex`` samples).

Otherwise both are omitted on that frame. They are not moved. Elbow gating is
separate and runs first.

Phase 1 (docs/phase2_wrist_roll_math.md): comparing against a single ~7.5°
value from mixing humerus-local and radius-local marker XYZ was invalid. The
reachable band is computed in one frame via OpenSim ground marker positions.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from src.opensim_upper.environment import OPENSIM_PYTHON
from src.opensim_upper.residuals import REPORT_MISMATCH_M
from src.opensim_upper.scaling import spacing_mapping

WRIST_STUDY_PAIRS = (
    ("r_lwrist_study", "r_mwrist_study"),
    ("L_lwrist_study", "L_mwrist_study"),
)

WRIST_ELBOW_PAIRS = {
    ("r_lwrist_study", "r_mwrist_study"): ("r_lelbow_study", "r_melbow_study"),
    ("L_lwrist_study", "L_mwrist_study"): ("L_lelbow_study", "L_melbow_study"),
}
# Margin outside the reachable [min, max] band (numerical / mild pose mismatch).
WRIST_ROLL_MARGIN_DEG = 5.0

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ROLL_RANGE_TOOL = _PROJECT_ROOT / "tools" / "opensim_wrist_roll_range.py"


def model_wrist_pair_distances(model_path) -> dict[tuple[str, str], float]:
    """Straight-line distance between each wrist study pair on this model."""
    locations = _model_locations(model_path)
    distances = {}
    for left, right in WRIST_STUDY_PAIRS:
        if left not in locations or right not in locations:
            raise ValueError(f"The model has no location for {left} and {right}.")
        distances[(left, right)] = float(np.linalg.norm(locations[left] - locations[right]))
    return distances


def model_wrist_elbow_roll_range_report(model_path) -> dict:
    """OpenSim ground-frame reachable roll band plus coordinate verification.

    Left and right arms are swept independently. The returned ``pairs`` map
    uses ``left|right`` marker keys. ``default_pose_roll_deg`` is diagnostic
    only and must not be used as a universal gate target.
    """
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"OpenSim model was not found: {model}")
    if not OPENSIM_PYTHON.is_file():
        raise FileNotFoundError(f"Isolated OpenSim interpreter was not found: {OPENSIM_PYTHON}")
    if not _ROLL_RANGE_TOOL.is_file():
        raise FileNotFoundError(f"Wrist roll range tool was not found: {_ROLL_RANGE_TOOL}")
    completed = subprocess.run(
        [str(OPENSIM_PYTHON), str(_ROLL_RANGE_TOOL), str(model.resolve())],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Wrist roll range tool failed: "
            + ((completed.stderr or completed.stdout or "")[-500:])
        )
    # OpenSim may print [info]/[warning] lines before the JSON object.
    payload = _json_object_from_text(completed.stdout or "")
    pairs = payload.get("pairs") or {}
    for pair in WRIST_STUDY_PAIRS:
        key = f"{pair[0]}|{pair[1]}"
        row = pairs.get(key)
        if row is None:
            raise ValueError(f"Wrist roll range tool omitted {pair}.")
        low = float(row["min_deg"])
        high = float(row["max_deg"])
        if not (np.isfinite(low) and np.isfinite(high)) or high < low:
            raise ValueError(f"Invalid wrist roll range for {pair}: {low}..{high}")
    payload["clinical_accuracy"] = False
    payload["biomechanical_reliability"] = False
    return payload


def model_wrist_elbow_roll_range_deg(model_path) -> dict[tuple[str, str], tuple[float, float]]:
    """Reachable unsigned ground-frame elbow–wrist ML roll (deg) per wrist pair."""
    payload = model_wrist_elbow_roll_range_report(model_path)
    ranges: dict[tuple[str, str], tuple[float, float]] = {}
    for pair in WRIST_STUDY_PAIRS:
        key = f"{pair[0]}|{pair[1]}"
        row = payload["pairs"][key]
        ranges[pair] = (float(row["min_deg"]), float(row["max_deg"]))
    return ranges


def gate_wrist_frames(
    frames: list[dict],
    pair_distance_m: dict[tuple[str, str], float],
    tolerance_m: float = REPORT_MISMATCH_M,
    pair_roll_range_deg: dict[tuple[str, str], tuple[float, float]] | None = None,
    roll_margin_deg: float = WRIST_ROLL_MARGIN_DEG,
) -> tuple[list[dict], dict]:
    """Return IK frames and a summary. The input frames are not modified."""
    if tolerance_m < 0 or tolerance_m != tolerance_m:
        raise ValueError("Wrist spacing tolerance must be a finite number >= 0.")
    if roll_margin_deg < 0 or roll_margin_deg != roll_margin_deg:
        raise ValueError("Wrist roll margin must be a finite number >= 0.")
    expected = set(WRIST_STUDY_PAIRS)
    if set(pair_distance_m) != expected:
        raise ValueError("Wrist spacing gate requires both study pairs and no other pairs.")
    if pair_roll_range_deg is not None and set(pair_roll_range_deg) != expected:
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
            if pair_roll_range_deg is not None:
                elbow = WRIST_ELBOW_PAIRS[pair]
                e0 = points.get(elbow[0])
                e1 = points.get(elbow[1])
                roll = None
                if e0 is not None and e1 is not None:
                    roll = _ml_roll_deg(e0, e1, a, b)
                low, high = pair_roll_range_deg[pair]
                lo = float(low) - float(roll_margin_deg)
                hi = float(high) + float(roll_margin_deg)
                if roll is None or roll < lo or roll > hi:
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
        roll_range = None if pair_roll_range_deg is None else pair_roll_range_deg[pair]
        pairs.append(
            {
                "markers": list(pair),
                "side": "right" if pair[0].startswith(("r_", "R")) else "left",
                "model_distance_m": float(pair_distance_m[pair]),
                "median_distance_m": float(np.median(distances)) if distances else None,
                "model_roll_min_deg": None if roll_range is None else float(roll_range[0]),
                "model_roll_max_deg": None if roll_range is None else float(roll_range[1]),
                "median_elbow_roll_deg": float(np.median(rolls)) if rolls else None,
                "n_both_present": both,
                "n_kept": kept,
                "n_left_out": both - kept,
                "n_left_out_spacing": counts[pair]["left_out_spacing"],
                "n_left_out_roll": counts[pair]["left_out_roll"],
                "tolerance_m": float(tolerance_m),
                "roll_margin_deg": float(roll_margin_deg),
                "coordinates_moved": False,
            }
        )
    summary = {
        "clinical_accuracy": False,
        "biomechanical_reliability": False,
        "coordinates_moved": False,
        "wrist_markers_gated": True,
        "tolerance_m": float(tolerance_m),
        "roll_margin_deg": float(roll_margin_deg),
        "pairs": pairs,
        "statement": _statement(pairs, roll_checked=pair_roll_range_deg is not None),
    }
    return gated, summary


def _json_object_from_text(text: str) -> dict:
    """Parse the first top-level JSON object in OpenSim-noisy stdout."""
    start = text.find("{")
    if start < 0:
        raise RuntimeError("Wrist roll range tool printed no JSON object.")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text[start:])
    if not isinstance(payload, dict):
        raise RuntimeError("Wrist roll range tool JSON was not an object.")
    return payload


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
    """Unsigned angle between elbow ML and wrist ML in the forearm-normal plane.

    All four points must already be in the same Cartesian frame (camera or ground).
    """
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
        margin = pairs[0]["roll_margin_deg"]
        lo = pairs[0].get("model_roll_min_deg")
        hi = pairs[0].get("model_roll_max_deg")
        if lo is not None and hi is not None:
            band = f"{lo:.0f}–{hi:.0f} deg (±{margin:.0f} deg margin)"
        else:
            band = "the model's reachable forearm roll band"
        sentences.append(
            "They were also left out when the camera-frame elbow-to-wrist ML roll "
            f"fell outside {band}, from a ground-frame pro_sup sweep (not mixed body-local XYZ)."
        )
    sentences.append("The saved points were not moved.")
    sentences.append(
        "A narrow pair or a roll outside the reachable radioulnar band cannot sit "
        "on the radius stations used by this model."
    )
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
