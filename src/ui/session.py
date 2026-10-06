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
from src.analysis.angles_3d import compute_configured_angles_3d
from src.analysis.dof_angles_3d import compute_initial_dofs, unavailable_initial_dofs
from src.analysis.draw_analysis import draw_joint_frames, highlight_readout
from src.analysis.regions import region_id, region_label
from src.analysis.velocity import VelocityTracker
from src.capture.factory import create_rgb_source
from src.geometry.deproject import CameraXyzSmoother, attach_camera_xyz
from src.io.save_angles import AngleCsvWriter
from src.io.save_keypoints import KeypointCsvWriter
from src.io.save_video import OverlayVideoWriter
from src.pose.draw import draw_keypoints_2d
from src.pose.extractor_2d import PoseExtractor2D
from src.pose.hands import HandsExtractor2D, merge_pose_and_hands
from src.utils.config_loader import resolve_project_path


def bgr(values) -> tuple[int, int, int]:
    """YAML [B, G, R] list -> OpenCV colour tuple."""
    return (int(values[0]), int(values[1]), int(values[2]))


def _initial_dofs(space: str, keypoints, analysis_cfg: dict, min_confidence: float):
    """New DOFs beside the dials. 2D stays unavailable. Thresholds come from YAML."""
    if space != "3d":
        return unavailable_initial_dofs("needs_3d")
    dof_cfg = analysis_cfg.get("dof") or {}
    if not bool(dof_cfg.get("enabled", True)):
        return unavailable_initial_dofs("disabled")
    return compute_initial_dofs(
        keypoints,
        min_confidence,
        plane_singular_deg=float(dof_cfg["plane_singular_deg"]),
        elbow_sign_deadband_deg=float(dof_cfg["elbow_sign_deadband_deg"]),
    )


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
        self.hands_extractor = None
        self.csv_writer = None
        self.angle_writer = None
        self.video_writer = None
        self.run_dir: Path | None = None
        self.frame_index = 0
        self._t0 = 0.0
        self.last_angles: list[Angle2D] = []
        self.last_gauge_angles: dict[str, float | None] = {}
        self.last_angular_dps: dict[str, float | None] = {}
        self.last_linear_mps: dict[str, float | None] = {}
        self.last_time_sec: float = 0.0
        self.last_highlight_readout: list[dict] = []
        self.last_joint_frames: list = []
        self.last_dof_angles = unavailable_initial_dofs("needs_3d")
        self._xyz_smoother: CameraXyzSmoother | None = None

    def start(self) -> None:
        """Open RGB source, Pose + Hands models, and output files."""
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
        self.hands_extractor = None
        hands_cfg = pose_cfg.get("hands") or {}
        use_hands = bool(hands_cfg.get("enabled", True)) and region_id(self.config) != "face"
        if use_hands:
            self.hands_extractor = HandsExtractor2D(
                max_num_hands=int(hands_cfg.get("max_num_hands", 2)),
                model_complexity=int(hands_cfg.get("model_complexity", 1)),
                min_detection_confidence=float(
                    hands_cfg.get("min_detection_confidence", pose_cfg["min_detection_confidence"])
                ),
                min_tracking_confidence=float(
                    hands_cfg.get("min_tracking_confidence", pose_cfg["min_tracking_confidence"])
                ),
                swap_handedness=bool(hands_cfg.get("swap_handedness", True)),
                match_to_pose_wrists=bool(hands_cfg.get("match_to_pose_wrists", True)),
            )
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
        depth_cfg = (self.config.get("camera") or {}).get("depth") or {}
        if str(analysis_cfg.get("space", "2d")).lower() == "3d":
            self._xyz_smoother = CameraXyzSmoother(
                smooth=float(depth_cfg.get("smooth", 0.35)),
                deadband_m=float(depth_cfg.get("deadband_m", 0.008)),
                hold_frames=int(depth_cfg.get("hold_frames", 5)),
            )
        else:
            self._xyz_smoother = None
        vel_cfg = analysis_cfg.get("velocity") or {}
        self._linear_names = [str(name) for name in vel_cfg.get("linear_joints") or []]
        self._velocity = VelocityTracker(
            smooth=float(vel_cfg.get("smooth", 0.45)),
            deadband_deg=float(vel_cfg.get("deadband_deg", 1.5)),
            deadband_m=float(vel_cfg.get("deadband_m", 0.004)),
            max_gap_sec=float(vel_cfg.get("max_gap_sec", 0.4)),
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
        if self.hands_extractor is not None:
            keypoints = merge_pose_and_hands(
                keypoints,
                self.hands_extractor.extract(frame, pose_keypoints=keypoints),
            )
        space = str(analysis_cfg.get("space", "2d")).lower()
        if space == "3d":
            depth_cfg = (self.config.get("camera") or {}).get("depth") or {}
            keypoints = attach_camera_xyz(
                keypoints,
                self.source,
                min_depth_m=float(depth_cfg.get("min_m", 0.3)),
                max_depth_m=float(depth_cfg.get("max_m", 6.0)),
                sample_window=int(depth_cfg.get("sample_window", 5)),
            )
            if self._xyz_smoother is not None:
                keypoints = self._xyz_smoother.apply(keypoints)
        time_sec = time.perf_counter() - self._t0
        if space == "3d":
            angles, self.last_joint_frames = compute_configured_angles_3d(
                keypoints,
                analysis_cfg["angles"],
                min_ang,
                frame_specs=list(analysis_cfg.get("joint_frames") or []),
            )
        else:
            angles = compute_configured_angles(keypoints, analysis_cfg["angles"], min_ang)
            self.last_joint_frames = []
        self.last_dof_angles = _initial_dofs(space, keypoints, analysis_cfg, min_ang)
        by_name = {item.name: item.degrees for item in angles}
        gauges = analysis_cfg.get("gauge_joints") or {}
        names = [str(n) for side in ("left", "right") for n in gauges.get(side, [])]
        self.last_gauge_angles = {name: by_name.get(name) for name in names}
        self.last_angular_dps = self._velocity.angular(time_sec, self.last_gauge_angles)
        if space == "3d":
            by_kp = {kp.name: kp for kp in keypoints}
            points = {}
            for name in self._linear_names:
                kp = by_kp.get(name)
                if kp is not None and kp.x_m is not None and kp.y_m is not None and kp.z_m is not None:
                    points[name] = (float(kp.x_m), float(kp.y_m), float(kp.z_m))
                else:
                    points[name] = None
            self.last_linear_mps = self._velocity.linear(time_sec, points)
        else:
            self.last_linear_mps = {name: None for name in self._linear_names}
        self.last_angles = angles
        self.last_time_sec = time_sec

        draw_names = analysis_cfg.get("draw_joints") or None
        allowed = set(draw_names) if draw_names else None
        extra_bones = [
            (str(a), str(b))
            for pair in (analysis_cfg.get("extra_bones") or [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
            for a, b in (pair,)
        ]
        csv_points = [kp for kp in keypoints if allowed is None or kp.name in allowed]

        if self.csv_writer is not None:
            self.csv_writer.write_frame(self.frame_index, time_sec, csv_points, self.source.label)
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
            allowed_names=allowed,
            extra_bones=extra_bones,
        )
        if space == "3d" and self.last_joint_frames:
            draw_joint_frames(
                canvas,
                self.last_joint_frames,
                self.source.color_intrinsics(),
                float(analysis_cfg.get("joint_frame_axis_m", 0.12)),
                dict(analysis_cfg.get("joint_frame_colors_bgr") or {}),
                allowed_names=allowed,
            )
        highlight_specs = list(analysis_cfg.get("highlight_joints", []))
        display_unit = str(analysis_cfg.get("coord_3d_display", "cm"))
        self.last_highlight_readout = highlight_readout(
            keypoints,
            highlight_specs,
            min_ang,
            space=space,
            display_unit=display_unit,
        )
        _draw_hud(canvas, self.source.label, region_label(self.config), space)

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
        if self.hands_extractor is not None:
            try:
                self.hands_extractor.close()
            except Exception:
                pass
            self.hands_extractor = None
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
        "coord_frame": (
            "camera_3d_metre"
            if str((config.get("analysis") or {}).get("space", "2d")).lower() == "3d"
            else "camera_2d_pixel"
        ),
        "units_angle": (
            "degrees_3d_camera"
            if str((config.get("analysis") or {}).get("space", "2d")).lower() == "3d"
            else "degrees_2d_image_plane"
        ),
        "frames": frames,
        "angle_rows": angle_rows,
        "source": config.get("source", {}),
        "analysis": config.get("analysis", {}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _draw_hud(frame, source_label, region: str = "Full body", space: str = "2d") -> None:
    """Small green tag on the video: source, region, 2D pixels or 3D metres."""
    if str(space).lower() == "3d":
        coord = "3D camera m  (origin=optical)"
    else:
        coord = "2D camera px"
    cv2.putText(
        frame,
        f"{source_label}  |  {region}  |  {coord}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
