"""Build upper-body camera trajectories from diagnostics rows.

Only the 12 mapped sources are kept. Observed wrists, centres, toes, and
lower-body names are counted and then left out. Missing depth and missing
LSTM samples stay missing. Nothing is interpolated unless the caller asks,
and the original missing mask is always returned.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from src.opensim_upper.trc import ordered_marker_rows


def _blank(value: str | None) -> bool:
    return value is None or str(value).strip() == ""


def _finite_triplet(x: str, y: str, z: str) -> tuple[float, float, float] | None:
    if _blank(x) or _blank(y) or _blank(z):
        return None
    try:
        point = (float(x), float(y), float(z))
    except ValueError:
        return None
    if not all(math.isfinite(item) for item in point):
        return None
    return point


def load_diagnostics(folder: Path) -> tuple[list[dict], list[dict]]:
    """Read ``landmarks.csv`` and, when present, ``frames.csv``."""
    folder = Path(folder)
    landmarks_path = folder / "landmarks.csv"
    if not landmarks_path.is_file():
        raise FileNotFoundError(f"No landmarks.csv in {folder}")
    with landmarks_path.open(encoding="utf-8", newline="") as handle:
        landmarks = list(csv.DictReader(handle))
    frames_path = folder / "frames.csv"
    frames: list[dict] = []
    if frames_path.is_file():
        with frames_path.open(encoding="utf-8", newline="") as handle:
            frames = list(csv.DictReader(handle))
    return landmarks, frames


def assemble_camera_frames(
    landmark_rows: list[dict],
    frame_rows: list[dict] | None = None,
    mapping: dict | None = None,
) -> tuple[list[dict], dict]:
    """Group diagnostics rows into time-ordered camera-metre frames."""
    rows = ordered_marker_rows(mapping)
    by_source = {row["source"]: row["target"] for row in rows}
    dropped_before = {}
    fps_values = []
    for row in frame_rows or []:
        if _blank(row.get("frame")):
            continue
        frame_index = int(row["frame"])
        if not _blank(row.get("dropped_before")):
            dropped_before[frame_index] = int(float(row["dropped_before"]))
        if not _blank(row.get("fps")):
            fps = float(row["fps"])
            if math.isfinite(fps) and fps > 0:
                fps_values.append(fps)

    grouped: dict[int, dict] = {}
    ignored = 0
    invalid = 0
    duplicate_rows = 0
    for row in landmark_rows:
        name = str(row.get("landmark") or "")
        if name not in by_source:
            ignored += 1
            continue
        frame_index = int(row["frame"])
        time_sec = float(row["t_frame_sec"])
        slot = grouped.setdefault(
            frame_index,
            {"source_frame": frame_index, "t_sec": time_sec, "points": {source: None for source in by_source}},
        )
        if abs(slot["t_sec"] - time_sec) > 1e-6:
            invalid += 1
            continue
        if slot["points"][name] is not None or slot.get("_seen", set()).__contains__(name):
            duplicate_rows += 1
            continue
        slot.setdefault("_seen", set()).add(name)
        missing_flag = str(row.get("xyz_missing") or "").strip()
        point = _finite_triplet(row.get("raw_x_m"), row.get("raw_y_m"), row.get("raw_z_m"))
        if missing_flag == "1" or point is None:
            if missing_flag == "0" and point is None:
                invalid += 1
            slot["points"][name] = None
            continue
        slot["points"][name] = point

    ordered = [grouped[key] for key in sorted(grouped)]
    for frame in ordered:
        frame.pop("_seen", None)
        frame["dropped_before"] = dropped_before.get(frame["source_frame"])
    kept, duplicate_times, rewound = _monotonic(ordered)
    quality = {
        "n_input_frames": len(ordered),
        "n_kept_frames": len(kept),
        "ignored_unmapped_rows": ignored,
        "invalid_coordinate_rows": invalid,
        "duplicate_landmark_rows": duplicate_rows,
        "duplicate_timestamps_removed": duplicate_times,
        "non_increasing_timestamps_removed": rewound,
        "dropped_frames_recorded": max(dropped_before.values()) if dropped_before else 0,
        "dropped_frames_interpolated": False,
        "recorded_fps_median": float(np.median(fps_values)) if fps_values else None,
        "interpolation": "none",
    }
    return kept, quality


def _monotonic(frames: list[dict]) -> tuple[list[dict], list[int], list[int]]:
    kept: list[dict] = []
    duplicate_times: list[int] = []
    rewound: list[int] = []
    last_t = None
    for frame in frames:
        time_sec = float(frame["t_sec"])
        if last_t is not None and abs(time_sec - last_t) <= 1e-6:
            duplicate_times.append(int(frame["source_frame"]))
            continue
        if last_t is not None and time_sec <= last_t:
            rewound.append(int(frame["source_frame"]))
            continue
        kept.append(frame)
        last_t = time_sec
    return kept, duplicate_times, rewound


def nominal_rate_hz(frames: list[dict], recorded_fps: float | None) -> tuple[float, bool, bool]:
    """Return rate, whether it varied, and whether the interval was measured.

    The Time column remains the timestamp. The header rate is the median
    interval when at least two frames exist, otherwise the recorded fps.
    """
    if len(frames) >= 2:
        intervals = np.diff([float(frame["t_sec"]) for frame in frames])
        median = float(np.median(intervals))
        if median <= 0:
            raise ValueError("Frame intervals are not positive.")
        varied = bool(np.max(np.abs(intervals - median)) > max(1e-4, 0.05 * median))
        return 1.0 / median, varied, True
    if recorded_fps is not None and recorded_fps > 0:
        return float(recorded_fps), False, False
    return 1.0, False, False


def linear_fill(
    frames: list[dict],
    sources: list[str],
) -> tuple[list[dict], list[dict]]:
    """Optional linear fill of interior gaps. End gaps stay missing.

    The returned frames are a copy. ``filled`` lists the original gaps that
    were replaced. Callers that do not want interpolation must not call this.
    """
    filled_notes = []
    output = []
    for frame in frames:
        output.append(
            {
                "source_frame": frame["source_frame"],
                "t_sec": frame["t_sec"],
                "dropped_before": frame.get("dropped_before"),
                "points": dict(frame["points"]),
            }
        )
    times = np.array([frame["t_sec"] for frame in output], dtype=float)
    for source in sources:
        values = [frame["points"][source] for frame in output]
        known = [index for index, value in enumerate(values) if value is not None]
        if len(known) < 2:
            continue
        for left, right in zip(known, known[1:]):
            if right <= left + 1:
                continue
            start = np.asarray(values[left], dtype=float)
            stop = np.asarray(values[right], dtype=float)
            span = float(times[right] - times[left])
            for index in range(left + 1, right):
                weight = float((times[index] - times[left]) / span)
                output[index]["points"][source] = tuple(float(item) for item in (1.0 - weight) * start + weight * stop)
                filled_notes.append(
                    {
                        "source_frame": output[index]["source_frame"],
                        "source": source,
                        "method": "linear_between_nearest_finite_samples",
                    }
                )
    return output, filled_notes
