"""Step 5 — 2D motion analysis: joint angles (degrees) + wrist trail (pixels).

How to run (from the project folder):
    python step5_analyze_2d.py

Same live / file switch as Step 4. Angles are in the camera image (2D).
Press q to quit. Saves keypoints CSV, angles CSV, overlay video.

Not 3D. Face the camera. Weak joints (legs if sitting close) are skipped
when confidence is below analysis.min_confidence in config.yaml.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.angles_2d import compute_configured_angles
from src.analysis.draw_analysis import draw_angles_2d, draw_trail_2d
from src.analysis.trajectory_2d import JointTrail2D
from src.capture.factory import create_rgb_source
from src.capture.video_file import VideoFileRGB
from src.io.save_angles import AngleCsvWriter
from src.io.save_keypoints import KeypointCsvWriter
from src.io.save_video import OverlayVideoWriter
from src.pose.draw import draw_keypoints_2d
from src.pose.extractor_2d import PoseExtractor2D
from src.utils.config_loader import load_config, resolve_project_path


def main() -> None:
    """Pose + 2D angles + trail, then save CSVs like Step 4."""
    config = load_config()
    display_cfg = config["display"]
    pose_cfg = config["pose"]
    overlay_cfg = config["overlay"]
    output_cfg = config["output"]
    analysis_cfg = config["analysis"]
    quit_key = str(display_cfg["quit_key"])
    window_title = str(display_cfg["window_title"])
    min_vis = float(pose_cfg["min_visibility"])
    min_ang = float(analysis_cfg["min_confidence"])

    try:
        source = create_rgb_source(config)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print("Could not open the RGB source.")
        print(str(error))
        sys.exit(1)

    run_dir = _make_run_dir(output_cfg)
    csv_writer = KeypointCsvWriter(run_dir / str(output_cfg["csv_name"])) if output_cfg.get("save_csv", True) else None
    angle_writer = AngleCsvWriter(run_dir / str(output_cfg.get("angles_csv_name", "angles_2d.csv")))
    video_writer = None
    if output_cfg.get("save_overlay_video", True):
        video_writer = OverlayVideoWriter(
            run_dir / str(output_cfg["video_name"]),
            fps=float(config["camera"]["fps"]),
            codec=str(output_cfg.get("video_codec", "mp4v")),
        )

    extractor = PoseExtractor2D(
        model_complexity=int(pose_cfg["model_complexity"]),
        min_detection_confidence=float(pose_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(pose_cfg["min_tracking_confidence"]),
    )
    trail = JointTrail2D(
        joint_name=str(analysis_cfg["trail_joint"]),
        max_points=int(analysis_cfg["trail_max_points"]),
    )
    point_color = _bgr(overlay_cfg["point_color_bgr"])
    line_color = _bgr(overlay_cfg["line_color_bgr"])
    angle_color = _bgr(analysis_cfg["angle_text_bgr"])
    trail_color = _bgr(analysis_cfg["trail_color_bgr"])

    print("Starting Step 5 (2D angles + path)...")
    print(f"  Source: {source.label}")
    print(f"  Output: {run_dir}")
    print("  Angles are 2D (image plane). Face the camera.")
    print(f"  Press '{quit_key}' to quit.")

    try:
        source.start()
    except RuntimeError as error:
        print("Could not start. Close Viewer if using live.")
        print(str(error))
        extractor.close()
        sys.exit(1)

    delay_ms = _playback_delay_ms(config, source)
    frame_index = 0
    t0 = time.perf_counter()

    try:
        while True:
            frame = source.get_frame()
            if source.ended:
                print("End of file.")
                break
            if frame is None:
                continue

            keypoints = extractor.extract(frame)
            time_sec = time.perf_counter() - t0
            angles = compute_configured_angles(
                keypoints,
                analysis_cfg["angles"],
                min_confidence=min_ang,
            )
            trail.update(keypoints, time_sec, min_confidence=min_ang)

            if csv_writer is not None:
                csv_writer.write_frame(frame_index, time_sec, keypoints, source.label)
            angle_writer.write_frame(frame_index, time_sec, angles, source.label)

            canvas = draw_keypoints_2d(
                frame,
                keypoints,
                min_visibility=min_vis,
                point_radius=int(overlay_cfg["point_radius"]),
                line_thickness=int(overlay_cfg["line_thickness"]),
                point_color=point_color,
                line_color=line_color,
            )
            draw_trail_2d(
                canvas,
                trail.polyline(),
                trail_color,
                int(analysis_cfg["trail_thickness"]),
            )
            draw_angles_2d(canvas, angles, angle_color)
            _draw_hud(canvas, source.label, angles, trail.last_speed_px_s, quit_key)
            if video_writer is not None:
                video_writer.write(canvas)
            cv2.imshow(window_title, canvas)

            frame_index += 1
            if cv2.waitKey(delay_ms) & 0xFF == ord(quit_key):
                break
    finally:
        extractor.close()
        source.stop()
        if csv_writer is not None:
            csv_writer.close()
        angle_writer.close()
        if video_writer is not None:
            video_writer.close()
        _write_run_json(
            run_dir / "run.json",
            config,
            source.label,
            frame_index,
            angle_writer.row_count,
        )
        cv2.destroyAllWindows()
        print("Stopped.")
        print(f"  Frames: {frame_index}")
        print(f"  Angle rows: {angle_writer.row_count}  ->  {angle_writer.path}")
        if csv_writer is not None:
            print(f"  Keypoints: {csv_writer.row_count}  ->  {csv_writer.path}")


def _draw_hud(frame, source_label, angles, speed_px_s, quit_key) -> None:
    """Top line: source, live angle list, optional wrist speed in px/s."""
    bits = [source_label]
    if angles:
        bits.append("  ".join(f"{a.name} {a.degrees:.0f}deg" for a in angles))
    else:
        bits.append("no 2D angles (low conf. or no person)")
    if speed_px_s is not None:
        bits.append(f"{speed_px_s:.0f} px/s")
    bits.append(f"{quit_key}=quit")
    cv2.putText(
        frame,
        "  |  ".join(bits),
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _make_run_dir(output_cfg: dict) -> Path:
    """Timestamped folder under data/output (path from config)."""
    stamp = datetime.now().strftime(str(output_cfg["stamp_format"]))
    run_dir = resolve_project_path(output_cfg["folder"]) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_run_json(path: Path, config: dict, source_label: str, frames: int, angle_rows: int) -> None:
    """Snapshot of settings for this analysis run."""
    payload = {
        "step": 5,
        "coord_frame": "camera_2d_pixel",
        "units_angle": "degrees_2d_image_plane",
        "units_speed": "pixels_per_second",
        "source_label": source_label,
        "frames": frames,
        "angle_rows": angle_rows,
        "analysis": config.get("analysis", {}),
        "pose": config.get("pose", {}),
        "source": config.get("source", {}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bgr(values) -> tuple[int, int, int]:
    """YAML [B,G,R] -> OpenCV tuple."""
    return (int(values[0]), int(values[1]), int(values[2]))


def _playback_delay_ms(config: dict, source: object) -> int:
    """Movies wait by FPS; live/bag already time themselves."""
    if source.label in {"bag", "live"}:
        return 1
    fps = source.playback_fps() if isinstance(source, VideoFileRGB) else 0.0
    if fps <= 1.0:
        fps = float(config["camera"]["fps"])
    return max(1, int(1000.0 / fps))


if __name__ == "__main__":
    main()
