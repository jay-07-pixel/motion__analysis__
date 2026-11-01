"""Compare registered landmarks with unchanged model marker locations.

Two markers on the same body have a parent-frame distance fixed by
``Markers_LSTM.xml``. The experimental distance is compared with that
value. The XML locations are not edited to reduce the residual.

A distance from a registered point to the generic model's default-pose
ground position is a separate figure. It mixes subject pose with marker
placement, so it is not used to move a marker.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

REPORT_MISMATCH_M = 0.02


def same_body_residuals(assignments: list[dict]) -> dict:
    """Pose-invariant spacing residuals for markers that share a parent body."""
    grouped: dict[str, list[dict]] = {}
    for row in assignments:
        if not row.get("mapped"):
            continue
        grouped.setdefault(str(row["body"]), []).append(row)
    pairs = []
    for body, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: str(item["target"]))
        for left, right in combinations(ordered, 2):
            local_left = np.asarray(left["location_m"], dtype=float)
            local_right = np.asarray(right["location_m"], dtype=float)
            experimental_left = np.asarray(left["model_m"], dtype=float)
            experimental_right = np.asarray(right["model_m"], dtype=float)
            model_distance = float(np.linalg.norm(local_left - local_right))
            experimental_distance = float(np.linalg.norm(experimental_left - experimental_right))
            residual = experimental_distance - model_distance
            pairs.append(
                {
                    "body": body,
                    "marker_a": left["target"],
                    "marker_b": right["target"],
                    "model_distance_m": model_distance,
                    "experimental_distance_m": experimental_distance,
                    "residual_m": residual,
                    "abs_residual_m": abs(residual),
                    "mismatch": abs(residual) > REPORT_MISMATCH_M,
                }
            )
    return {
        "method": (
            "Same-parent Euclidean distance. Model distances use location_m "
            "from Markers_LSTM.xml. Those locations are not changed."
        ),
        "units": "metres",
        "mismatch_flag_m": REPORT_MISMATCH_M,
        "mismatch_flag_moves_markers": False,
        "n_pairs": len(pairs),
        "n_mismatch": sum(1 for row in pairs if row["mismatch"]),
        "pairs": pairs,
    }


def default_pose_ground_distances(
    assignments: list[dict],
    ground_m: dict[str, list[float]],
) -> dict:
    """Distance to generic default-pose ground positions. Does not move markers."""
    rows = []
    for assignment in assignments:
        if not assignment.get("mapped"):
            continue
        target = str(assignment["target"])
        if target not in ground_m:
            raise KeyError(f"No default-pose ground position for {target}")
        experimental = np.asarray(assignment["model_m"], dtype=float)
        generic = np.asarray(ground_m[target], dtype=float)
        distance = float(np.linalg.norm(experimental - generic))
        rows.append(
            {
                "target": target,
                "distance_m": distance,
                "includes_subject_pose_and_scale": True,
                "location_m_unchanged": list(assignment["location_m"]),
            }
        )
    return {
        "method": (
            "Distance between the registered experimental point and the generic "
            "model marker in the default pose, both in ground metres. This is not "
            "a marker-placement update."
        ),
        "units": "metres",
        "markers_moved": False,
        "rows": rows,
    }
