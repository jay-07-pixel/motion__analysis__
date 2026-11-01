"""Leave a one-frame jump out of IK.

A marker that moves 0.15 m or more from its previous sample, inside a tenth
of a second, is omitted on that frame. On this project's captures the camera
shoulder and elbow steps stayed under 0.13 m. The larger steps were wrist
estimates changing together. The point is left empty. It is not replaced by
the earlier point, and the recorded TRC is not edited.
"""

from __future__ import annotations

import numpy as np

STEP_GLITCH_M = 0.15
STEP_GLITCH_MAX_DT_S = 0.10


def omit_impossible_steps(
    frames: list[dict],
    limit_m: float = STEP_GLITCH_M,
    max_dt_s: float = STEP_GLITCH_MAX_DT_S,
) -> tuple[list[dict], dict]:
    """Return IK frames and a count of omitted points. Input frames stay as given."""
    if limit_m <= 0 or limit_m != limit_m:
        raise ValueError("The step limit must be a finite number greater than 0.")
    if max_dt_s <= 0 or max_dt_s != max_dt_s:
        raise ValueError("The step interval must be a finite number greater than 0.")
    previous: dict[str, tuple[float, tuple[float, float, float]]] = {}
    omitted: dict[str, int] = {}
    gated = []
    for frame in frames:
        points = dict(frame["points"])
        time_sec = float(frame["time_sec"])
        for name, point in list(points.items()):
            if point is None:
                continue
            prior = previous.get(name)
            previous[name] = (time_sec, tuple(float(item) for item in point))
            if prior is None:
                continue
            dt = time_sec - prior[0]
            if dt <= 0 or dt > max_dt_s:
                continue
            step = float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(prior[1], dtype=float)))
            if step < limit_m:
                continue
            points[name] = None
            omitted[name] = omitted.get(name, 0) + 1
        gated.append({**frame, "points": points})
    return gated, {
        "limit_m": float(limit_m),
        "max_dt_s": float(max_dt_s),
        "coordinates_moved": False,
        "replaced_with_previous": False,
        "omitted": omitted,
        "statement": _statement(limit_m, max_dt_s, omitted),
    }


def _statement(limit_m: float, max_dt_s: float, omitted: dict[str, int]) -> str:
    sentence = (
        f"A point that jumped {limit_m:.3f} m or more within {max_dt_s:.2f} s "
        "was left out of the fit. It was not replaced by the earlier point."
    )
    if not omitted:
        return sentence
    parts = [f"{name} {count}" for name, count in sorted(omitted.items())]
    return sentence + " Left out: " + ", ".join(parts) + "."
