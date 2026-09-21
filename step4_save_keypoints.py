"""Step 4 — save 2D keypoints (camera pixels) to CSV + optional overlay video.

How to run (from the project folder):
    python step4_save_keypoints.py

Same live / file switch as Step 3 (config.yaml -> source.mode).
Press q to quit. Files land in a timestamped folder under data/output/.

No angles yet (that is Step 5). This step only keeps the numbers.
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

from src.capture.factory import create_rgb_source
from src.capture.video_file import VideoFileRGB
from src.io.save_keypoints import KeypointCsvWriter
from src.io.save_video import OverlayVideoWriter
from src.pose.draw import draw_keypoints_2d
from src.pose.extractor_2d import PoseExtractor2D
from src.utils.config_loader import load_config, resolve_project_path


def main() -> None:
    """Run pose like Step 3, and write CSV / overlay video while it runs."""
    config = load_config()
    display_cfg = config["display"]
    pose_cfg = config["pose"]
    overlay_cfg = config["overlay"]
    output_cfg = config["output"]
    quit_key = str(display_cfg["quit_key"])
    window_title = str(display_cfg["window_title"])

    try:
        source = create_rgb_source(config)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print("Could not open the RGB source.")
        print(str(error))
        sys.exit(1)

    run_dir = _make_run_dir(output_cfg)
    csv_writer = None
    video_writer = None
    if bool(output_cfg.get("save_csv", True)):
        csv_writer = KeypointCsvWriter(run_dir / str(output_cfg["csv_name"]))
    if bool(output_cfg.get("save_overlay_video", True)):
        video_fps = float(config["camera"]["fps"])
        video_writer = OverlayVideoWriter(
            run_dir / str(output_cfg["video_name"]),
            fps=video_fps,
            codec=str(output_cfg.get("video_codec", "mp4v")),
        )

    extractor = PoseExtractor2D(
        model_complexity=int(pose_cfg["model_complexity"]),
        min_detection_confidence=float(pose_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(pose_cfg["min_tracking_confidence"]),
    )
    min_visibility = float(pose_cfg["min_visibility"])
    point_color = _bgr(overlay_cfg["point_color_bgr"])
    line_color = _bgr(overlay_cfg["line_color_bgr"])

    print("Starting Step 4 (save keypoints)...")
    print(f"  Source: {source.label}")
    print(f"  Output: {run_dir}")
    print("  Coordinates: camera_2d_pixel  (origin = top-left)")
    print(f"  Press '{quit_key}' to quit and finish saving.")

    try:
        source.start()
    except RuntimeError as error:
        print("Could not start the source. Close RealSense Viewer if using live.")
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
            if csv_writer is not None:
                csv_writer.write_frame(frame_index, time_sec, keypoints, source.label)

            canvas = draw_keypoints_2d(
                frame,
                keypoints,
                min_visibility=min_visibility,
                point_radius=int(overlay_cfg["point_radius"]),
                line_thickness=int(overlay_cfg["line_thickness"]),
                point_color=point_color,
                line_color=line_color,
            )
            _draw_hud(canvas, source.label, len(keypoints), quit_key, run_dir.name)
            if video_writer is not None:
                video_writer.write(canvas)
            cv2.imshow(window_title, canvas)

            frame_index += 1
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord(quit_key):
                break
    finally:
        extractor.close()
        source.stop()
        if csv_writer is not None:
            csv_writer.close()
        if video_writer is not None:
            video_writer.close()
        if bool(output_cfg.get("save_run_json", True)):
            _write_run_json(
                run_dir / "run.json",
                config=config,
                source_label=source.label,
                frames=frame_index,
                csv_rows=csv_writer.row_count if csv_writer else 0,
            )
        cv2.destroyAllWindows()
        print("Stopped.")
        print(f"  Frames: {frame_index}")
        if csv_writer is not None:
            print(f"  CSV rows: {csv_writer.row_count}  ->  {csv_writer.path}")
        if video_writer is not None:
            print(f"  Overlay: {video_writer.frame_count} frames  ->  {video_writer.path}")


def _make_run_dir(output_cfg: dict) -> Path:
    """Create data/output/<timestamp>/ for this run (path from config)."""
    base = resolve_project_path(output_cfg["folder"])
    stamp = datetime.now().strftime(str(output_cfg["stamp_format"]))
    run_dir = base / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_run_json(path: Path, config: dict, source_label: str, frames: int, csv_rows: int) -> None:
    """Snapshot settings used for this run so handover can reproduce it."""
    payload = {
        "coord_frame": "camera_2d_pixel",
        "source_label": source_label,
        "frames": frames,
        "csv_rows": csv_rows,
        "source": config.get("source", {}),
        "camera": config.get("camera", {}),
        "pose": config.get("pose", {}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _draw_hud(frame, source_label: str, joint_count: int, quit_key: str, run_name: str) -> None:
    """Show source, joint count, and that this run is being saved."""
    status = f"{joint_count} joints" if joint_count else "no person"
    text = f"{source_label}  {status}  |  saving {run_name}  |  {quit_key}=quit"
    cv2.putText(
        frame,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _bgr(values) -> tuple[int, int, int]:
    """Turn a YAML list [B, G, R] into an OpenCV colour tuple."""
    return (int(values[0]), int(values[1]), int(values[2]))


def _playback_delay_ms(config: dict, source: object) -> int:
    """Wait between frames for movies; bags/live already time themselves."""
    if source.label in {"bag", "live"}:
        return 1
    fps = 0.0
    if isinstance(source, VideoFileRGB):
        fps = source.playback_fps()
    if fps <= 1.0:
        fps = float(config["camera"]["fps"])
    return max(1, int(1000.0 / fps))


if __name__ == "__main__":
    main()
