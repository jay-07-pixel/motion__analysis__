"""Ground-frame reachable elbow–wrist ML roll vs pro_sup (opensim-upper only).

Left and right arms are swept independently. Roll is computed only from
getLocationInGround marker positions (one Cartesian frame). Does not edit the
model. Writes JSON to stdout (OpenSim may also print log lines).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import opensim as osim

# Independent side configs — never share samples across arms.
WRIST_ELBOW = {
    "right": {
        "wrist": ("r_lwrist_study", "r_mwrist_study"),
        "elbow": ("r_lelbow_study", "r_melbow_study"),
        "pro_sup": "pro_sup_r",
        "elbow_flex": "elbow_flex_r",
    },
    "left": {
        "wrist": ("L_lwrist_study", "L_mwrist_study"),
        "elbow": ("L_lelbow_study", "L_melbow_study"),
        "pro_sup": "pro_sup_l",
        "elbow_flex": "elbow_flex_l",
    },
}


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _ml_roll_deg(e0, e1, w0, w1) -> float | None:
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


def _ground(model: osim.Model, state: osim.State, name: str) -> np.ndarray:
    point = model.getMarkerSet().get(name).getLocationInGround(state)
    return np.array([point.get(0), point.get(1), point.get(2)], dtype=float)


def _side_roll(model: osim.Model, state: osim.State, elbow, wrist) -> float | None:
    return _ml_roll_deg(
        _ground(model, state, elbow[0]),
        _ground(model, state, elbow[1]),
        _ground(model, state, wrist[0]),
        _ground(model, state, wrist[1]),
    )


def reachable_rolls(model_path: Path) -> dict:
    model = osim.Model(str(model_path))
    state = model.initSystem()
    pairs = {}
    coordinates = {}
    for side, cfg in WRIST_ELBOW.items():
        wrist = cfg["wrist"]
        elbow = cfg["elbow"]
        pro = model.getCoordinateSet().get(cfg["pro_sup"])
        flex = model.getCoordinateSet().get(cfg["elbow_flex"])
        pro_min = float(pro.getRangeMin())
        pro_max = float(pro.getRangeMax())
        flex_min = float(flex.getRangeMin())
        flex_max = float(flex.getRangeMax())
        pro_default = float(pro.getDefaultValue())
        flex_default = float(flex.getDefaultValue())
        coordinates[side] = {
            "pro_sup": {
                "name": cfg["pro_sup"],
                "range_min_rad": pro_min,
                "range_max_rad": pro_max,
                "default_rad": pro_default,
                "unit": "radians",
            },
            "elbow_flex": {
                "name": cfg["elbow_flex"],
                "range_min_rad": flex_min,
                "range_max_rad": flex_max,
                "default_rad": flex_default,
                "unit": "radians",
            },
        }
        # Default-pose roll is diagnostic only — never used as the gate target.
        pro.setValue(state, pro_default)
        flex.setValue(state, flex_default)
        model.realizePosition(state)
        default_roll = _side_roll(model, state, elbow, wrist)

        samples: list[float] = []
        pro_values = np.linspace(pro_min, pro_max, 25)
        # Independent flex samples for this side only, clamped to that coord range.
        flex_candidates = [flex_default, flex_default + 0.5, flex_default + 1.0]
        flex_values = sorted(
            {
                max(flex_min, min(flex_max, float(value)))
                for value in flex_candidates
            }
        )
        for flex_value in flex_values:
            flex.setValue(state, float(flex_value))
            for pro_value in pro_values:
                pro.setValue(state, float(pro_value))
                model.realizePosition(state)
                roll = _side_roll(model, state, elbow, wrist)
                if roll is not None:
                    samples.append(float(roll))
        # Restore this side's defaults before the other arm's sweep.
        pro.setValue(state, pro_default)
        flex.setValue(state, flex_default)
        model.realizePosition(state)
        if not samples:
            raise RuntimeError(f"No reachable roll samples for {side}.")
        pairs[f"{wrist[0]}|{wrist[1]}"] = {
            "side": side,
            "markers": list(wrist),
            "elbow_markers": list(elbow),
            "min_deg": float(min(samples)),
            "max_deg": float(max(samples)),
            "n_samples": len(samples),
            "default_pose_roll_deg": None if default_roll is None else float(default_roll),
            "method": "ground_frame_independent_pro_sup_and_elbow_flex_sweep",
            "frame": "opensim_ground_via_getLocationInGround",
            "clinical_accuracy": False,
        }
    return {
        "model": str(Path(model_path).resolve()),
        "clinical_accuracy": False,
        "biomechanical_reliability": False,
        "coordinates": coordinates,
        "pairs": pairs,
        "note": (
            "min_deg/max_deg are the gate reference band. "
            "default_pose_roll_deg is diagnostic only and is not a universal target."
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: opensim_wrist_roll_range.py <model.osim>", file=sys.stderr)
        return 2
    payload = reachable_rolls(Path(sys.argv[1]))
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
