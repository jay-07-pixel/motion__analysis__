"""Upper-body landmark quality from a diagnostics landmarks.csv.

SAT and GHJC stay unavailable. This does not invent their coordinates.
"""

from __future__ import annotations

import math

UPPER_BODY = (
    ("left_shoulder", "observed", "MediaPipe shoulder. Not the acromion and not the glenohumeral centre."),
    ("right_shoulder", "observed", "MediaPipe shoulder. Not the acromion and not the glenohumeral centre."),
    ("left_elbow", "observed", "MediaPipe elbow. Not a humeral epicondyle."),
    ("right_elbow", "observed", "MediaPipe elbow. Not a humeral epicondyle."),
    ("left_wrist", "observed", "MediaPipe wrist. Not a styloid process."),
    ("right_wrist", "observed", "MediaPipe wrist. Not a styloid process."),
    ("L_lelbow_study", "predicted", "Pose2Sim v0.3 lateral-elbow study marker. Not a palpated epicondyle."),
    ("L_melbow_study", "predicted", "Pose2Sim v0.3 medial-elbow study marker. Not a palpated epicondyle."),
    ("r_lelbow_study", "predicted", "Pose2Sim v0.3 lateral-elbow study marker. Not a palpated epicondyle."),
    ("r_melbow_study", "predicted", "Pose2Sim v0.3 medial-elbow study marker. Not a palpated epicondyle."),
    ("L_lwrist_study", "predicted", "Pose2Sim v0.3 lateral-wrist study marker. Not a palpated styloid."),
    ("L_mwrist_study", "predicted", "Pose2Sim v0.3 medial-wrist study marker. Not a palpated styloid."),
    ("r_lwrist_study", "predicted", "Pose2Sim v0.3 lateral-wrist study marker. Not a palpated styloid."),
    ("r_mwrist_study", "predicted", "Pose2Sim v0.3 medial-wrist study marker. Not a palpated styloid."),
    ("L_SAT", "unavailable", "Acromion is not estimated. No coordinate is fabricated."),
    ("R_SAT", "unavailable", "Acromion is not estimated. No coordinate is fabricated."),
    ("L_GHJC", "unavailable", "Glenohumeral centre is not estimated. No coordinate is fabricated."),
    ("R_GHJC", "unavailable", "Glenohumeral centre is not estimated. No coordinate is fabricated."),
)


def _num(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def upper_body_quality(rows: list[dict], *, min_valid_pixels: int = 13) -> list[dict]:
    """One row per upper-body marker. Counts come only from the supplied rows."""
    grouped: dict[str, list[dict]] = {name: [] for name, _kind, _note in UPPER_BODY}
    for row in rows:
        name = str(row.get("landmark") or "")
        if name in grouped:
            grouped[name].append(row)
    report = []
    for name, kind, note in UPPER_BODY:
        samples = grouped[name]
        valid = 0
        invalid = 0
        sparse = 0
        previous = None
        max_step = None
        for row in samples:
            point = (_num(row.get("raw_x_m")), _num(row.get("raw_y_m")), _num(row.get("raw_z_m")))
            present = all(value is not None for value in point)
            if present:
                valid += 1
                if previous is not None:
                    step = math.dist(previous, point)
                    max_step = step if max_step is None else max(max_step, step)
                previous = point
            else:
                invalid += 1
                previous = None
            count = _num(row.get("depth_n_valid_in_window"))
            if kind == "observed" and count is not None and count < min_valid_pixels:
                sparse += 1
        if kind == "unavailable":
            depth_quality = "unavailable"
        elif kind == "predicted":
            depth_quality = "not a depth sample"
        elif not samples:
            depth_quality = "no rows"
        else:
            depth_quality = f"{sparse} windows below {min_valid_pixels} valid pixels"
        report.append(
            {
                "name": name,
                "class": kind,
                "coord_frame": None if kind == "unavailable" else "camera_3d_metre",
                "valid": 0 if kind == "unavailable" else valid,
                "invalid": 0 if kind == "unavailable" else invalid,
                "depth_quality": depth_quality,
                "max_frame_to_frame_m": None if kind == "unavailable" else max_step,
                "anatomical": note,
            }
        )
    return report
