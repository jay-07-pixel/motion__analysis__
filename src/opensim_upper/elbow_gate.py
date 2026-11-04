"""Leave elbow study markers out of IK when their spacing is not the model's.

The recorded TRC is not edited. A second TRC is written for inverse kinematics.
On a frame, the two estimate points of one elbow are copied unchanged when
their distance is within the same-parent mismatch flag of the model pair.
Otherwise both points are omitted on that frame. They are not moved closer
together or farther apart. Wrist study pairs use the same rule in wrist_gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.opensim_upper.residuals import REPORT_MISMATCH_M
from src.opensim_upper.scaling import spacing_mapping
from src.opensim_upper.step_gate import omit_impossible_steps
from src.opensim_upper.trc import read_trc, write_trc
from src.opensim_upper.wrist_gate import (
    gate_wrist_frames,
    model_wrist_elbow_roll_deg,
    model_wrist_pair_distances,
)

ELBOW_STUDY_PAIRS = (
    ("r_lelbow_study", "r_melbow_study"),
    ("L_lelbow_study", "L_melbow_study"),
)


def model_pair_distances(model_path: Path) -> dict[tuple[str, str], float]:
    """Straight-line distance between each elbow study pair on this model."""
    locations = {
        row["target"]: np.asarray(row["location_m"], dtype=float)
        for row in spacing_mapping(model_path)["mapped"]
    }
    distances = {}
    for left, right in ELBOW_STUDY_PAIRS:
        if left not in locations or right not in locations:
            raise ValueError(f"The model has no location for {left} and {right}.")
        distances[(left, right)] = float(np.linalg.norm(locations[left] - locations[right]))
    return distances


def gate_elbow_frames(
    frames: list[dict],
    pair_distance_m: dict[tuple[str, str], float],
    tolerance_m: float = REPORT_MISMATCH_M,
) -> tuple[list[dict], dict]:
    """Return IK frames and a summary. The input frames are not modified."""
    if tolerance_m < 0 or tolerance_m != tolerance_m:
        raise ValueError("Elbow spacing tolerance must be a finite number >= 0.")
    expected = set(ELBOW_STUDY_PAIRS)
    if set(pair_distance_m) != expected:
        raise ValueError("Elbow spacing gate requires both study pairs and no other pairs.")
    gated = []
    counts = {
        pair: {"both_present": 0, "kept": 0, "distances_m": []}
        for pair in ELBOW_STUDY_PAIRS
    }
    for frame in frames:
        points = dict(frame["points"])
        for pair in ELBOW_STUDY_PAIRS:
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
            if abs(distance - float(pair_distance_m[pair])) <= tolerance_m:
                counts[pair]["kept"] += 1
                continue
            points[left] = None
            points[right] = None
        gated.append({**frame, "points": points})
    pairs = []
    for pair in ELBOW_STUDY_PAIRS:
        distances = counts[pair]["distances_m"]
        median = float(np.median(distances)) if distances else None
        both = counts[pair]["both_present"]
        kept = counts[pair]["kept"]
        pairs.append(
            {
                "markers": list(pair),
                "side": "right" if pair[0].startswith("r_") else "left",
                "model_distance_m": float(pair_distance_m[pair]),
                "median_distance_m": median,
                "n_both_present": both,
                "n_kept": kept,
                "n_left_out": both - kept,
                "tolerance_m": float(tolerance_m),
                "coordinates_moved": False,
            }
        )
    summary = {
        "clinical_accuracy": False,
        "coordinates_moved": False,
        "wrist_markers_gated": False,
        "tolerance_m": float(tolerance_m),
        "pairs": pairs,
        "statement": _statement(pairs),
    }
    return gated, summary


def write_gated_ik_trc(source: Path, dest: Path, model_path: Path) -> dict:
    """Copy a TRC for IK, omitting elbow/wrist pairs that do not match model spacing."""
    source_path = Path(source)
    dest_path = Path(dest)
    if source_path.resolve() == dest_path.resolve():
        raise ValueError("The IK TRC must be a separate file. The recorded TRC is not edited.")
    before = source_path.read_bytes()
    parsed = read_trc(source_path)
    gated, elbow_summary = gate_elbow_frames(parsed["frames"], model_pair_distances(model_path))
    gated, wrist_summary = gate_wrist_frames(
        gated,
        model_wrist_pair_distances(model_path),
        pair_roll_deg=model_wrist_elbow_roll_deg(model_path),
    )
    gated, steps = omit_impossible_steps(gated)
    statement = (
        f"{elbow_summary['statement']} {wrist_summary['statement']} {steps['statement']}"
    )
    summary = {
        **elbow_summary,
        "wrist_markers_gated": True,
        "wrist_gate": wrist_summary,
        "step_gate": steps,
        "statement": statement,
    }
    names = list(parsed["markers"])
    samples = {
        name: [frame["points"].get(name) for frame in gated]
        for name in names
    }
    rows = [{"target": name} for name in names]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    write_trc(
        dest_path,
        [frame["time_sec"] for frame in gated],
        samples,
        float(parsed["data_rate_hz"]),
        rows,
    )
    if source_path.read_bytes() != before:
        raise RuntimeError("The recorded TRC changed while the IK copy was written.")
    report_path = dest_path.parent / "elbow_gate.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (dest_path.parent / "wrist_gate.json").write_text(
        json.dumps(wrist_summary, indent=2),
        encoding="utf-8",
    )
    return {
        "trc": str(dest_path),
        "report": str(report_path),
        "statement": statement,
        "summary": summary,
    }


def _statement(pairs: list[dict]) -> str:
    model_values = {round(float(row["model_distance_m"]), 5) for row in pairs}
    if len(model_values) == 1:
        spacing = f"{pairs[0]['model_distance_m']:.3f} m"
    else:
        spacing = "the skeleton spacing for that side"
    tolerance = pairs[0]["tolerance_m"]
    sentences = [
        "The two elbow estimate points on a side were left out of the fit on any frame "
        f"where their distance was more than {tolerance:.3f} m from the skeleton spacing of {spacing}. "
        "The saved points were not moved."
    ]
    for row in pairs:
        side = str(row["side"]).capitalize()
        if row["median_distance_m"] is None:
            sentences.append(
                f"{side}: the two points were not available together, so that side was left out."
            )
            continue
        sentences.append(
            f"{side}: the points are typically {row['median_distance_m']:.3f} m apart "
            f"and were used on {row['n_kept']} of {row['n_both_present']} frames."
        )
    return " ".join(sentences)
