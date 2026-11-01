"""Compare a TRC with the model markers written by inverse kinematics.

Distances between markers on the same body are fixed by the model. A rigid
camera-to-model registration does not change those distances, so a spacing
mismatch is a property of the marker cloud.
"""

from __future__ import annotations

import math

import numpy as np

from src.opensim_upper.marker_map import marker_mapping
from src.opensim_upper.residuals import same_body_residuals


def model_marker_rows(table: dict) -> list[dict]:
    """Read an OpenSim model-marker location table into one XYZ point per marker."""
    names = []
    for name in table["columns"]:
        if name.endswith("_tx"):
            names.append(name[:-3])
    rows = []
    for row in table["rows"]:
        points = {}
        for name in names:
            points[name] = (float(row[f"{name}_tx"]), float(row[f"{name}_ty"]), float(row[f"{name}_tz"]))
        rows.append({"time_sec": float(row["time"]), "points": points})
    return rows


def per_marker_residuals(trc_frames: list[dict], model_rows: list[dict]) -> dict:
    """Euclidean TRC-to-model residual for each marker. Missing TRC points are omitted."""
    if len(trc_frames) != len(model_rows):
        raise ValueError("TRC and model-marker tables do not have the same number of frames.")
    names = list(trc_frames[0]["points"]) if trc_frames else []
    series: dict[str, list[float]] = {name: [] for name in names}
    for frame, model in zip(trc_frames, model_rows):
        if abs(float(frame["time_sec"]) - float(model["time_sec"])) > 1e-4:
            raise ValueError("TRC and model-marker timestamps do not match.")
        for name in names:
            point = frame["points"].get(name)
            if point is None:
                continue
            if name not in model["points"]:
                raise ValueError(f"IK model-marker table has no column for {name}.")
            residual = float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(model["points"][name])))
            series[name].append(residual)
    markers = {}
    squared = []
    for name, values in series.items():
        if not values:
            markers[name] = {
                "n_frames": 0,
                "mean_m": None,
                "max_m": None,
                "rms_m": None,
            }
            continue
        square_mean = sum(value * value for value in values) / len(values)
        markers[name] = {
            "n_frames": len(values),
            "mean_m": sum(values) / len(values),
            "max_m": max(values),
            "rms_m": math.sqrt(square_mean),
        }
        squared.extend(value * value for value in values)
    ranked = sorted(
        (name for name, row in markers.items() if row["rms_m"] is not None),
        key=lambda name: markers[name]["rms_m"],
        reverse=True,
    )
    return {
        "definition": (
            "Euclidean distance from the TRC marker to the model marker location "
            "written by IK. Frames with a missing TRC point are omitted for that marker."
        ),
        "units": "metres",
        "markers": markers,
        "ranked_by_rms": ranked,
        "unweighted_rms_m": math.sqrt(sum(squared) / len(squared)) if squared else None,
    }


def marker_spacing(trc: dict, mapping: dict | None = None) -> dict:
    """Compare one TRC frame with the fixed distances between markers on the same body."""
    rows = list((mapping if mapping is not None else marker_mapping())["mapped"])

    def _finite_count(frame: dict) -> int:
        return sum(frame["points"].get(row["target"]) is not None for row in rows)

    chosen = max(trc["frames"], key=_finite_count)
    if _finite_count(chosen) < 2:
        raise ValueError("No TRC frame contains two mapped markers.")
    assignments = []
    for row in rows:
        point = chosen["points"].get(row["target"])
        if point is None:
            continue
        assignments.append({**row, "mapped": True, "model_m": list(point)})
    spacing = same_body_residuals(assignments)
    consistent = spacing["n_mismatch"] == 0
    worst = None
    if spacing["pairs"]:
        worst = max(spacing["pairs"], key=lambda pair: pair["abs_residual_m"])
    return {
        "consistent_with_model_marker_spacing": consistent,
        "registration_changes_these_distances": False,
        "frame": int(chosen["frame"]),
        "time_sec": float(chosen["time_sec"]),
        "n_pairs": spacing["n_pairs"],
        "n_mismatch": spacing["n_mismatch"],
        "mismatch_flag_m": spacing["mismatch_flag_m"],
        "largest_pair": worst,
        "pairs": spacing["pairs"],
        "reason": (
            "Every same-parent pair is within 0.02 m of the model marker spacing."
            if consistent
            else (
                "Same-parent marker distances in the TRC do not match the model. "
                "A proper camera-to-model rotation preserves distances, so this "
                "mismatch is in the marker cloud."
            )
        ),
    }
