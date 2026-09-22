"""One 2D capture+analysis session (live or file). Used by the GUI.

Keeps Start/Stop in the app; pose, angles, and CSV writers stay here so
the window code does not duplicate Step 5.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.analysis.angles_2d import Angle2D, compute_configured_angles
from src.analysis.draw_analysis import draw_angles_2d, draw_joint_coords_2d, draw_trail_2d
from src.analysis.trajectory_2d import trail_from_config
from src.capture.factory import create_rgb_source
from src.io.save_angles import AngleCsvWriter
from src.io.save_keypoints import KeypointCsvWriter
from src.io.save_video import OverlayVideoWriter
from src.pose.draw import draw_keypoints_2d
from src.pose.extractor_2d import PoseExtractor2D
from src.utils.config_loader import resolve_project_path


def bgr(values) -> tuple[int, int, int]:
    """YAML [B, G, R] list -> OpenCV colour tuple."""
    return (int(values[0]), int(values[1]), int(values[2]))


class MotionSession2D:
    """Open source, process frames, optionally save, then close everything."""

    def __init__(self, config: dict, save_files: bool) -> None:
        """Store config. Camera/file are opened in start().

        Args:
            config: Full YAML dict; source.mode / file_path must already be set.
            save_files: If False, only show overlay (no CSV / mp4).
        """
        self.config = config
        self.save_files = save_files
        self.source = None
        self.extractor = None
        self.trail = None
        self.csv_writer = None
        self.angle_writer = None
        self.video_writer = None
        self.run_dir: Path | None = None
        self.frame_index = 0
        self._t0 = 0.0
        self.last_angles: list[Angle2D] = []
        self.last_speed_px_s: float | None = None
        self.last_gauge_angles: dict[str, float | None] = {}
        self.last_time_sec: float = 0.0
        self.last_highlight_uv: dict[str, tuple[float, float] | None] = {}

    def start(self) -> None:
        """Open RGB source, pose model, and output files."""
        pose_cfg = self.config["pose"]
        analysis_cfg = self.config["analysis"]
        output_cfg = self.config["output"]

        self.source = create_rgb_source(self.config)
        self.source.start()
        self.extractor = PoseExtractor2D(
            model_complexity=int(pose_cfg["model_complexity"]),
            min_detection_confidence=float(pose_cfg["min_detection_confidence"]),
            min_tracking_confidence=float(pose_cfg["min_tracking_confidence"]),
        )
        self.trail = trail_from_config(analysis_cfg)
        if self.save_files:
            self.run_dir = _make_run_dir(output_cfg)
            if output_cfg.get("save_csv", True):
                self.csv_writer = KeypointCsvWriter(self.run_dir / str(output_cfg["csv_name"]))
            self.angle_writer = AngleCsvWriter(
                self.run_dir / str(output_cfg.get("angles_csv_name", "angles_2d.csv"))
            )
            if output_cfg.get("save_overlay_video", True):
                self.video_writer = OverlayVideoWriter(
                    self.run_dir / str(output_cfg["video_name"]),
                    fps=float(self.config["camera"]["fps"]),
                    codec=str(output_cfg.get("video_codec", "mp4v")),
                )
        self.frame_index = 0
        self._t0 = time.perf_counter()

    def process_frame(self) -> np.ndarray | None:
        """Grab one RGB frame, draw overlay, save rows. None = skip this tick.

        Returns:
            BGR overlay image, or None if no colour frame this wait.

        Raises:
            StopIteration: uploaded file has ended.
        """
        frame = self.source.get_frame()
        if self.source.ended:
            raise StopIteration("End of file")
        if frame is None:
            return None

        pose_cfg = self.config["pose"]
        overlay_cfg = self.config["overlay"]
        analysis_cfg = self.config["analysis"]
        min_vis = float(pose_cfg["min_visibility"])
        min_ang = float(analysis_cfg["min_confidence"])

        keypoints = self.extractor.extract(frame)
        time_sec = time.perf_counter() - self._t0
        angles = compute_configured_angles(keypoints, analysis_cfg["angles"], min_ang)
        self.trail.update(keypoints, time_sec, min_ang)
        by_name = {item.name: item.degrees for item in angles}
        gauges = analysis_cfg.get("gauge_joints") or {}
        names = [str(n) for side in ("left", "right") for n in gauges.get(side, [])]
        self.last_gauge_angles = {name: by_name.get(name) for name in names}
        self.last_angles = angles
        self.last_speed_px_s = self.trail.last_speed_px_s
        self.last_time_sec = time_sec

        if self.csv_writer is not None:
            self.csv_writer.write_frame(self.frame_index, time_sec, keypoints, self.source.label)
        if self.angle_writer is not None:
            self.angle_writer.write_frame(self.frame_index, time_sec, angles, self.source.label)

        canvas = draw_keypoints_2d(
            frame,
            keypoints,
            min_visibility=min_vis,
            point_radius=int(overlay_cfg["point_radius"]),
            line_thickness=int(overlay_cfg["line_thickness"]),
            point_color=bgr(overlay_cfg["point_color_bgr"]),
            line_color=bgr(overlay_cfg["line_color_bgr"]),
        )
        draw_trail_2d(
            canvas,
            self.trail.polyline(),
            bgr(analysis_cfg["trail_color_bgr"]),
            int(analysis_cfg["trail_thickness"]),
        )
        draw_angles_2d(canvas, angles, bgr(analysis_cfg["angle_text_bgr"]))
        self.last_highlight_uv = draw_joint_coords_2d(
            canvas,
            keypoints,
            list(analysis_cfg.get("highlight_joints", [])),
            min_ang,
        )
        _draw_hud(canvas, self.source.label)

        if self.video_writer is not None:
            self.video_writer.write(canvas)
        self.frame_index += 1
        return canvas

    def stop(self) -> dict:
        """Close camera/file and writers. Returns paths and counts for the UI."""
        if self.extractor is not None:
            try:
                self.extractor.close()
            except Exception:
                pass
            self.extractor = None
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass
            self.source = None
        angle_rows = 0
        if self.angle_writer is not None:
            angle_rows = self.angle_writer.row_count
            self.angle_writer.close()
            self.angle_writer = None
        key_rows = 0
        if self.csv_writer is not None:
            key_rows = self.csv_writer.row_count
            self.csv_writer.close()
            self.csv_writer = None
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if self.save_files and self.run_dir is not None:
            _write_run_json(
                self.run_dir / "run.json",
                self.config,
                self.frame_index,
                angle_rows,
            )
        return {
            "run_dir": str(self.run_dir) if self.run_dir else "",
            "frames": self.frame_index,
            "angle_rows": angle_rows,
            "keypoint_rows": key_rows,
        }


def _make_run_dir(output_cfg: dict) -> Path:
    """Create data/output/<timestamp>/ from config."""
    stamp = datetime.now().strftime(str(output_cfg["stamp_format"]))
    run_dir = resolve_project_path(output_cfg["folder"]) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_run_json(path: Path, config: dict, frames: int, angle_rows: int) -> None:
    """Write a small handover snapshot for this GUI run."""
    payload = {
        "step": 6,
        "coord_frame": "camera_2d_pixel",
        "units_angle": "degrees_2d_image_plane",
        "units_speed": "pixels_per_second",
        "frames": frames,
        "angle_rows": angle_rows,
        "source": config.get("source", {}),
        "analysis": config.get("analysis", {}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _draw_hud(frame, source_label) -> None:
    """Small green tag on the video. Live numbers live in the GUI side panel."""
    cv2.putText(
        frame,
        f"{source_label}  |  2D camera px",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
