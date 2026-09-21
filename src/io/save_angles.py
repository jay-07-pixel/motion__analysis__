"""CSV writer for 2D angles (one row per named angle per frame)."""

from __future__ import annotations

import csv
from pathlib import Path

from src.analysis.angles_2d import Angle2D

ANGLE_FIELDNAMES = (
    "frame",
    "time_sec",
    "angle",
    "degrees",
    "vertex_u_px",
    "vertex_v_px",
    "min_confidence",
    "coord_frame",
    "source",
)


class AngleCsvWriter:
    """Appends 2D angle rows. Same camera_2d_pixel frame as the keypoints CSV."""

    def __init__(self, csv_path: Path) -> None:
        """Create the file and write the header."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = csv_path
        self._row_count = 0
        self._file = csv_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=list(ANGLE_FIELDNAMES))
        self._writer.writeheader()

    def write_frame(
        self,
        frame_index: int,
        time_sec: float,
        angles: list[Angle2D],
        source_label: str,
    ) -> None:
        """Write each computed angle for this frame (skipped angles are absent)."""
        for angle in angles:
            self._writer.writerow(
                {
                    "frame": frame_index,
                    "time_sec": f"{time_sec:.4f}",
                    "angle": angle.name,
                    "degrees": f"{angle.degrees:.2f}",
                    "vertex_u_px": f"{angle.vertex_u_px:.2f}",
                    "vertex_v_px": f"{angle.vertex_v_px:.2f}",
                    "min_confidence": f"{angle.min_confidence:.4f}",
                    "coord_frame": "camera_2d_pixel",
                    "source": source_label,
                }
            )
            self._row_count += 1

    @property
    def row_count(self) -> int:
        """How many angle rows were written."""
        return self._row_count

    def close(self) -> None:
        """Close the CSV so it can be opened in Excel."""
        self._file.close()
