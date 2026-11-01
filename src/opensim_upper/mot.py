"""Read an OpenSim motion file.

Rotational columns are degrees when the header says ``inDegrees=yes``.
Translational columns stay in metres either way. This reader does not convert.
"""

from __future__ import annotations

from pathlib import Path


def read_mot(path: Path) -> dict:
    """Parse a ``.mot`` or ``.sto`` table written by OpenSim 4.4."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = {}
    start = None
    for index, line in enumerate(lines):
        if "=" in line and not line.lower().startswith("endheader"):
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
        if line.strip().lower() == "endheader":
            start = index + 1
            break
    if start is None:
        raise ValueError("MOT file has no endheader.")
    while start < len(lines) and not lines[start].strip():
        start += 1
    names = lines[start].split()
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        cells = [float(item) for item in line.split()]
        if len(cells) != len(names):
            raise ValueError(f"MOT row has {len(cells)} values for {len(names)} columns.")
        rows.append(dict(zip(names, cells)))
    if not rows:
        raise ValueError("MOT file has no data rows.")
    times = [row["time"] for row in rows]
    for earlier, later in zip(times, times[1:]):
        if later <= earlier:
            raise ValueError("MOT timestamps are not strictly increasing.")
    return {
        "header": header,
        "columns": names,
        "rows": rows,
        "in_degrees": header.get("inDegrees", "").lower() == "yes",
    }
