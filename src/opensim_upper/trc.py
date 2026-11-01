"""OpenSim 4.4 TRC writer.

The layout matches a file written by ``TRCFileAdapter`` in OpenSim 4.4:
tab-separated, ``PathFileType 4 (X/Y/Z)``, units token ``m`` (metres),
marker names followed by two blank header cells, ``X1 Y1 Z1`` column
order, frame numbers starting at 1, and the text ``nan`` for a missing
marker. A renamed CSV is not this format.

Missing markers stay missing. This module does not interpolate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.opensim_upper.marker_map import marker_mapping


def ordered_marker_rows(mapping: dict | None = None) -> list[dict]:
    """Mapped markers in the configuration order. Wrists are not included."""
    table = mapping if mapping is not None else marker_mapping()
    rows = list(table["mapped"])
    targets = [row["target"] for row in rows]
    if len(targets) != 12 or len(set(targets)) != 12:
        raise ValueError("The upper-body TRC mapping must contain 12 distinct markers.")
    if "RWrist" in targets or "LWrist" in targets:
        raise ValueError("RWrist and LWrist are not part of the TRC mapping.")
    return rows


def _number(value: float) -> str:
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        return "0"
    return text


def write_trc(
    path: Path,
    times_sec: list[float],
    samples: dict[str, list[tuple[float, float, float] | None]],
    data_rate_hz: float,
    marker_rows: list[dict],
) -> None:
    """Write one TRC. ``samples`` are already in model metres."""
    path = Path(path)
    if len(times_sec) < 1:
        raise ValueError("A TRC needs at least one frame.")
    if not np.isfinite(data_rate_hz) or data_rate_hz <= 0:
        raise ValueError("Data rate must be a positive finite number of frames per second.")
    names = [row["target"] for row in marker_rows]
    if any(name not in samples for name in names):
        raise ValueError("TRC samples are missing a mapped marker.")
    for name in names:
        if len(samples[name]) != len(times_sec):
            raise ValueError(f"{name} does not have one sample per frame.")
    times = [float(value) for value in times_sec]
    for earlier, later in zip(times, times[1:]):
        if later <= earlier:
            raise ValueError("TRC timestamps must be strictly increasing.")

    header_names = "\t".join(f"{name}\t\t" for name in names)
    xyz_header = "\t".join(f"X{index}\tY{index}\tZ{index}" for index in range(1, len(names) + 1))
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.resolve()}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        "\t".join(
            [
                _number(data_rate_hz),
                _number(data_rate_hz),
                str(len(times)),
                str(len(names)),
                "m",
                _number(data_rate_hz),
                "1",
                str(len(times)),
            ]
        ),
        f"Frame#\tTime\t{header_names}\t",
        f"\t\t{xyz_header}\t",
        "",
    ]
    for frame_number, time_sec in enumerate(times, start=1):
        cells = [str(frame_number), _number(time_sec)]
        for name in names:
            sample = samples[name][frame_number - 1]
            if sample is None:
                cells.extend(("nan", "nan", "nan"))
                continue
            point = np.asarray(sample, dtype=float).reshape(3)
            if not np.isfinite(point).all():
                cells.extend(("nan", "nan", "nan"))
                continue
            cells.extend(_number(float(value)) for value in point)
        lines.append("\t".join(cells) + "\t")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_trc(path: Path) -> dict:
    """Parse a TRC written by this module or by OpenSim 4.4."""
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    rows = text.split("\n")
    if not rows or not rows[0].startswith("PathFileType\t4\t(X/Y/Z)\t"):
        raise ValueError("TRC header is not PathFileType 4 (X/Y/Z).")
    labels = rows[1].split("\t")
    values = rows[2].split("\t")
    header = dict(zip(labels, values))
    if header.get("Units") != "m":
        raise ValueError(f"TRC units are {header.get('Units')!r}, expected m for metres.")
    name_cells = rows[3].split("\t")
    if name_cells[:2] != ["Frame#", "Time"]:
        raise ValueError("TRC marker header must start with Frame# and Time.")
    markers = [cell for cell in name_cells[2:] if cell]
    xyz_cells = [cell for cell in rows[4].split("\t") if cell]
    expected_xyz = [axis for index in range(1, len(markers) + 1) for axis in (f"X{index}", f"Y{index}", f"Z{index}")]
    if xyz_cells != expected_xyz:
        raise ValueError("TRC coordinate columns are not X, Y, Z for each marker.")
    data_rows = [row for row in rows[5:] if row.strip()]
    frames = []
    for row in data_rows:
        cells = row.split("\t")
        if cells and cells[-1] == "":
            cells = cells[:-1]
        frame_number = int(cells[0])
        time_sec = float(cells[1])
        points = {}
        for index, name in enumerate(markers):
            start = 2 + index * 3
            raw = cells[start : start + 3]
            if len(raw) != 3 or any(item.strip().lower() == "nan" or item.strip() == "" for item in raw):
                points[name] = None
                continue
            points[name] = (float(raw[0]), float(raw[1]), float(raw[2]))
        frames.append({"frame": frame_number, "time_sec": time_sec, "points": points})
    return {
        "units": header["Units"],
        "data_rate_hz": float(header["DataRate"]),
        "num_frames": int(header["NumFrames"]),
        "num_markers": int(header["NumMarkers"]),
        "orig_data_start_frame": int(header["OrigDataStartFrame"]),
        "markers": markers,
        "frames": frames,
    }
