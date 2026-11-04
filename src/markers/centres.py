"""Geometric joint centres from medial/lateral LSTM pairs. Not ISB joint centres."""

from __future__ import annotations

# Lateral/medial study pairs. The two side markers are the LSTM outputs.
STUDY_SIDE_PAIRS = (
    ("r_lelbow_study", "r_melbow_study"),
    ("L_lelbow_study", "L_melbow_study"),
    ("r_lwrist_study", "r_mwrist_study"),
    ("L_lwrist_study", "L_mwrist_study"),
    ("r_knee_study", "r_mknee_study"),
    ("L_knee_study", "L_mknee_study"),
    ("r_ankle_study", "r_mankle_study"),
    ("L_ankle_study", "L_mankle_study"),
)

PAIRS = (
    ("r_elbow_centre", "r_lelbow_study", "r_melbow_study"),
    ("L_elbow_centre", "L_lelbow_study", "L_melbow_study"),
    ("r_wrist_centre", "r_lwrist_study", "r_mwrist_study"),
    ("L_wrist_centre", "L_lwrist_study", "L_mwrist_study"),
    ("r_knee_centre", "r_knee_study", "r_mknee_study"),
    ("L_knee_centre", "L_knee_study", "L_mknee_study"),
    ("r_ankle_centre", "r_ankle_study", "r_mankle_study"),
    ("L_ankle_centre", "L_ankle_study", "L_mankle_study"),
)


def pair_centres(
    points: list[tuple[str, float, float, float]],
) -> list[tuple[str, float, float, float]]:
    by_name = {name: (x, y, z) for name, x, y, z in points}
    out = []
    for dest, a, b in PAIRS:
        if a not in by_name or b not in by_name:
            continue
        ax, ay, az = by_name[a]
        bx, by, bz = by_name[b]
        out.append((dest, 0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (az + bz)))
    return out
