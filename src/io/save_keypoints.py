"""Write 2D keypoints to CSV (camera pixel coordinates).

Why:
    The on-screen skeleton disappears when you press q. A CSV is the
    handover file: every joint, every frame, so Step 5 can compute
    angles without re-running the camera.

Coordinate system (do not change):
    coord_frame = camera_2d_pixel
    origin      = top-left of the RGB image
    u_px, v_px  = pixels
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.pose.keypoints import Keypoint2D

# Column order is part of the handover contract. Step 5 will read these names.
CSV_FIELDNAMES = (
    "frame",
    "time_sec",
    "joint",
    "u_px",
    "v_px",
    "confidence",
    "coord_frame",
    "source",
)


class KeypointCsvWriter:
    """Appends one CSV row per joint per frame."""

    def __init__(self, csv_path: Path) -> None:
        """Create the file and write the header row.

        Args:
            csv_path: Full path, e.g. data/output/20260921-143000/keypoints_2d.csv
        """
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = csv_path
        self._row_count = 0
        # newline='' is required by csv on Windows so rows are not blank.
        self._file = csv_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=list(CSV_FIELDNAMES))
        self._writer.writeheader()

    def write_frame(
        self,
        frame_index: int,
        time_sec: float,
        keypoints: list[Keypoint2D],
        source_label: str,
    ) -> None:
        """Write every joint for this frame. Skip the frame if no person.

        Args:
            frame_index: 0-based frame number in this run.
            time_sec: Seconds since the start of this run.
            keypoints: Output of PoseExtractor2D.extract (may be empty).
            source_label: 'live', 'mp4', 'bag', ... from the RGB source.
        """
        for keypoint in keypoints:
            self._writer.writerow(
                {
                    "frame": frame_index,
                    "time_sec": f"{time_sec:.4f}",
                    "joint": keypoint.name,
                    "u_px": f"{keypoint.u_px:.2f}",
                    "v_px": f"{keypoint.v_px:.2f}",
                    "confidence": f"{keypoint.confidence:.4f}",
                    "coord_frame": keypoint.coord_frame,
                    "source": source_label,
                }
            )
            self._row_count += 1

    @property
    def row_count(self) -> int:
        """How many joint rows have been written (not counting the header)."""
        return self._row_count

    def close(self) -> None:
        """Flush and close the CSV so Excel / the next script can open it."""
        self._file.close()
