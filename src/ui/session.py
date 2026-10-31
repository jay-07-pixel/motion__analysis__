"""One 2D capture+analysis session (live or file). Used by the GUI.

Keeps Start/Stop in the app; pose, angles, and CSV writers stay here so
the window code does not duplicate Step 5.
"""

from __future__ import annotations

import json
import queue
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
from src.diagnostics.recorder import LiveDiagnosticsRecorder
from src.geometry.deproject import CameraXyzSmoother, attach_camera_xyz
from src.io.save_angles import AngleCsvWriter
from src.io.save_keypoints import KeypointCsvWriter
from src.io.save_video import OverlayVideoWriter
from src.pose.draw import draw_keypoints_2d
from src.pose.extractor_2d import PoseExtractor2D
from src.pose.hands import HandsExtractor2D, merge_pose_and_hands
from src.pose.keypoints import Keypoint2D
from src.pose2sim_live.participant import require_participant_metres_kg
from src.ui.foot_safety import draw_foot_safety_guide
from src.ui.live_markers import draw_live_landmarks
from src.utils.config_loader import resolve_project_path


def _project_lstm_points(intrinsics, points: list[tuple[str, float, float, float]]) -> list[Keypoint2D]:
    """Camera-metre LSTM XYZ -> Keypoint2D with projected colour pixels."""
    import pyrealsense2 as rs

    out: list[Keypoint2D] = []
    if intrinsics is None:
        return out
    for name, x, y, z in points:
        u_px, v_px = -1.0, -1.0
        if z > 1e-4:
            try:
                pixel = rs.rs2_project_point_to_pixel(intrinsics, [float(x), float(y), float(z)])
                u_px, v_px = float(pixel[0]), float(pixel[1])
            except Exception:
                u_px, v_px = -1.0, -1.0
        out.append(
            Keypoint2D(
                name=name,
                u_px=u_px,
                v_px=v_px,
                confidence=1.0,
                coord_frame="camera_3d_metre",
                x_m=float(x),
                y_m=float(y),
                z_m=float(z),
            )
        )
    return out


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

    def __init__(self, config: dict, save_files: bool, events: queue.Queue | None = None) -> None:
        """Store config. Camera/file are opened in start().

        Args:
            config: Full YAML dict; source.mode / file_path must already be set.
            save_files: If False, only show overlay (no CSV / mp4).
            events: UI-thread clicks (toe calibration) consumed on the worker.
        """
        self.config = config
        self.save_files = save_files
        self.events = events
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
        self.last_keypoints = []
        self.last_fps: float = 0.0
        self._fps_stamp: float | None = None
        self._xyz_smoother: CameraXyzSmoother | None = None
        self._lstm = None
        self._body = None
        self.last_lstm_estimates: list[Keypoint2D] = []
        self.last_lstm_banner = ""
        self.last_lstm_banners: list[str] = []
        self._diag: LiveDiagnosticsRecorder | None = None
        self.last_diag_dir = ""
        self.last_diag_status = ""

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
        self._lstm = None
        self._body = None
        self.last_lstm_estimates = []
        self.last_lstm_banner = ""
        self.last_lstm_banners = []
        self.participant_height_m: float | None = None
        self.participant_mass_kg: float | None = None
        want_lstm = bool(analysis_cfg.get("live_pose2sim_upper", False))
        space = str(analysis_cfg.get("space", "2d")).lower()
        if space == "3d" and want_lstm:
            from src.markers.coordinator import LiveBodyCoordinator

            project = self.config.get("project") or {}
            height_m, mass_kg = require_participant_metres_kg(
                project.get("participant_height_m"),
                project.get("participant_mass_kg"),
            )
            self.participant_height_m = height_m
            self.participant_mass_kg = mass_kg
            print(
                f"Starting live full-body marker coordinator... "
                f"participant {height_m:.2f} m  {mass_kg:.1f} kg",
                flush=True,
            )
            self._body = LiveBodyCoordinator(enabled=True, height_m=height_m, mass_kg=mass_kg)
            self._lstm = self._body
            self.last_lstm_banners = self._body.banners()
            self.last_lstm_banner = self.last_lstm_banners[0] if self.last_lstm_banners else ""
        elif want_lstm:
            self.last_lstm_banner = "LSTM off — this session is 2D. Stop, click 3D, Start."
            self.last_lstm_banners = [self.last_lstm_banner]
        else:
            self.last_lstm_banner = "Live Pose2Sim unchecked — no LSTM estimates"
            self.last_lstm_banners = [self.last_lstm_banner]
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
        self._diag = None
        self.last_diag_dir = ""
        self.last_diag_status = ""
        if bool(analysis_cfg.get("live_diagnostics", False)):
            self._open_diag()

    def _deproject_uv(self, u_px: float, v_px: float) -> tuple[float, float, float] | None:
        """Single-pixel RealSense deprojection. None if depth is invalid."""
        import pyrealsense2 as rs

        if self.source is None:
            return None
        depth_cfg = (self.config.get("camera") or {}).get("depth") or {}
        z_m = self.source.depth_distance_m(u_px, v_px, int(depth_cfg.get("sample_window", 5)))
        min_m = float(depth_cfg.get("min_m", 0.3))
        max_m = float(depth_cfg.get("max_m", 6.0))
        if z_m < min_m or z_m > max_m:
            return None
        intrinsics = self.source.color_intrinsics()
        if intrinsics is None:
            return None
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u_px), float(v_px)], float(z_m))
        return float(point[0]), float(point[1]), float(point[2])

    def _drain_marker_events(self, keypoints: list[Keypoint2D], min_visibility: float) -> None:
        if self.events is None:
            return
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            kind = event.get("kind") if isinstance(event, dict) else None
            if kind == "diag_start":
                self._open_diag()
                continue
            if kind == "diag_stop":
                self._close_diag()
                continue
            if kind == "toe_click":
                if self._body is None:
                    continue
                xyz = self._deproject_uv(float(event["u"]), float(event["v"]))
                if xyz is None:
                    self.last_lstm_banner = "Toe click has no valid depth. Aim at the toe and click again."
                    continue
                self.last_lstm_banner = self._body.add_toe_click(
                    float(event["u"]), float(event["v"]), xyz, keypoints
                )

    def _open_diag(self) -> None:
        if self._diag is not None:
            return
        self._diag = LiveDiagnosticsRecorder.create_session(
            participant_height_m=self.participant_height_m,
            participant_mass_kg=self.participant_mass_kg,
        )
        self.last_diag_dir = str(self._diag.dir)
        self.last_diag_status = f"DIAG ON  {self._diag.dir.name}"
        print(f"Live diagnostics -> {self._diag.dir}", flush=True)

    def _close_diag(self) -> dict | None:
        if self._diag is None:
            return None
        summary = self._diag.close()
        self.last_diag_dir = str(summary.get("dir") or self.last_diag_dir)
        self.last_diag_status = (
            f"DIAG saved {summary.get('frames', 0)} frames  {Path(self.last_diag_dir).name}"
        )
        print(self.last_diag_status, flush=True)
        self._diag = None
        return summary

    def process_frame(self) -> np.ndarray | None:
        """Grab one RGB frame, draw overlay, save rows. None = skip this tick.

        Returns:
            BGR overlay image, or None if no colour frame this wait.

        Raises:
            StopIteration: uploaded file has ended.
        """
        t_loop = time.perf_counter()
        frame = self.source.get_frame()
        if self.source.ended:
            raise StopIteration("End of file")
        if frame is None:
            if self._diag is not None:
                self._diag.note_dropped()
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
            time_sec = time.perf_counter() - self._t0
            self._drain_marker_events(keypoints, min_vis)
            if self._body is not None:
                self._body.observe(keypoints, time_sec, min_vis)
            raw_for_diag = list(keypoints)
            if self._xyz_smoother is not None:
                keypoints = self._xyz_smoother.apply(keypoints)
        else:
            time_sec = time.perf_counter() - self._t0
            self._drain_marker_events(keypoints, min_vis)
            raw_for_diag = list(keypoints)
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
        self.last_keypoints = list(keypoints)
        now = time.perf_counter()
        if self._fps_stamp is not None:
            dt = now - self._fps_stamp
            if dt > 1e-4:
                inst = 1.0 / dt
                self.last_fps = inst if self.last_fps <= 0 else (0.25 * inst + 0.75 * self.last_fps)
        self._fps_stamp = now

        self.last_lstm_estimates = []
        if self._body is not None:
            self.last_lstm_banners = self._body.banners()
            self.last_lstm_banner = self.last_lstm_banners[0] if self.last_lstm_banners else ""
            self.last_lstm_estimates = _project_lstm_points(
                self.source.color_intrinsics(),
                self._body.overlay_points(),
            )
        elif self.last_lstm_banner.startswith("LSTM unavailable"):
            pass
        elif bool(analysis_cfg.get("live_pose2sim_upper", False)) and space != "3d":
            self.last_lstm_banner = "LSTM off — this session is 2D. Stop, click 3D, Start."
            self.last_lstm_banners = [self.last_lstm_banner]
        else:
            self.last_lstm_banners = [self.last_lstm_banner] if self.last_lstm_banner else []

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

        save_canvas = draw_keypoints_2d(
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
                save_canvas,
                self.last_joint_frames,
                self.source.color_intrinsics(),
                float(analysis_cfg.get("joint_frame_axis_m", 0.12)),
                dict(analysis_cfg.get("joint_frame_colors_bgr") or {}),
                allowed_names=allowed,
            )
        canvas = draw_live_landmarks(
            frame,
            keypoints,
            min_visibility=min_vis,
            fps=self.last_fps,
            space=space,
            estimates=self.last_lstm_estimates,
            banner=self.last_lstm_banner or None,
            banners=self.last_lstm_banners,
        )
        display_cfg = self.config.get("display") or {}
        if bool(display_cfg.get("foot_safety_line_enabled", True)):
            draw_foot_safety_guide(
                canvas,
                keypoints,
                min_vis,
                float(display_cfg.get("foot_safety_line_v_px", 640)),
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
        _draw_hud(save_canvas, self.source.label, region_label(self.config), space)

        if self._diag is not None:
            predicted = []
            if self._body is not None:
                for rec in self._body.overlay_records():
                    item = dict(rec)
                    done = item.get("t_pred_done_perf")
                    item["t_pred_done_sec"] = (float(done) - self._t0) if done is not None else None
                    predicted.append(item)
            self.last_lstm_banners = [f"DIAG ON  {self._diag.dir.name}"] + [
                b for b in self.last_lstm_banners if not str(b).startswith("DIAG ON")
            ]
            depth_cfg = (self.config.get("camera") or {}).get("depth") or {}
            sample_window = int(depth_cfg.get("sample_window", 5))
            depth_inspect = {
                kp.name: self.source.inspect_depth_sample(kp.u_px, kp.v_px, sample_window)
                for kp in raw_for_diag
            }
            self._diag.write_frame(
                self.frame_index,
                time_sec,
                (time.perf_counter() - t_loop) * 1000.0,
                self.last_fps,
                raw_for_diag,
                predicted,
                depth_inspect=depth_inspect,
            )
            self.last_diag_status = f"DIAG {self._diag.n_frames} frames  {self._diag.dir.name}"

        if self.video_writer is not None:
            self.video_writer.write(save_canvas)
        self.frame_index += 1
        return canvas

    def stop(self) -> dict:
        """Close camera/file and writers. Returns paths and counts for the UI."""
        self._close_diag()
        if self._body is not None:
            try:
                self._body.close()
            except Exception:
                pass
            self._body = None
            self._lstm = None
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
            "diag_dir": self.last_diag_dir,
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
        "participant_height_m": (config.get("project") or {}).get("participant_height_m"),
        "participant_mass_kg": (config.get("project") or {}).get("participant_mass_kg"),
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
