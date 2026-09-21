"""Step 3 — extract 2D keypoints in camera pixel coordinates and overlay them.

How to run (from the project folder):
    python step3_show_keypoints.py

Uses source.mode in config.yaml:
    live  = USB D455f (close Viewer first)
    file  = mp4 / avi / bag  (needs a PERSON in the clip; sample.mp4 has none)

Press q to quit.

This step does NOT save CSV yet (that is Step 4). It only shows the skeleton
and prints how many joints were found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.factory import create_rgb_source
from src.capture.video_file import VideoFileRGB
from src.pose.draw import draw_keypoints_2d
from src.pose.extractor_2d import PoseExtractor2D
from src.utils.config_loader import load_config


def main() -> None:
    """Open live or file RGB, run MediaPipe Pose, draw the 2D skeleton."""
    config = load_config()
    display_cfg = config["display"]
    pose_cfg = config["pose"]
    overlay_cfg = config["overlay"]
    quit_key = str(display_cfg["quit_key"])
    window_title = str(display_cfg["window_title"])

    try:
        source = create_rgb_source(config)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print("Could not open the RGB source.")
        print(str(error))
        sys.exit(1)

    extractor = PoseExtractor2D(
        model_complexity=int(pose_cfg["model_complexity"]),
        min_detection_confidence=float(pose_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(pose_cfg["min_tracking_confidence"]),
    )
    min_visibility = float(pose_cfg["min_visibility"])
    point_color = _bgr(overlay_cfg["point_color_bgr"])
    line_color = _bgr(overlay_cfg["line_color_bgr"])

    print("Starting 2D keypoint overlay...")
    print(f"  Source: {source.label}  (config source.mode)")
    print("  Coordinates: camera_2d_pixel  (origin = top-left of RGB)")
    print(f"  Press '{quit_key}' in the video window to quit.")
    if source.label != "live":
        print("  Tip: sample.mp4 has no person. Use live or a clip of a person.")

    try:
        source.start()
    except RuntimeError as error:
        print("Could not start the source. Close RealSense Viewer if using live.")
        print(str(error))
        extractor.close()
        sys.exit(1)

    delay_ms = _playback_delay_ms(config, source)

    try:
        while True:
            frame = source.get_frame()
            if source.ended:
                print("End of file.")
                break
            if frame is None:
                continue

            keypoints = extractor.extract(frame)
            canvas = draw_keypoints_2d(
                frame,
                keypoints,
                min_visibility=min_visibility,
                point_radius=int(overlay_cfg["point_radius"]),
                line_thickness=int(overlay_cfg["line_thickness"]),
                point_color=point_color,
                line_color=line_color,
            )
            _draw_hud(canvas, source.label, len(keypoints), quit_key)
            cv2.imshow(window_title, canvas)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord(quit_key):
                break
    finally:
        extractor.close()
        source.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


def _draw_hud(frame, source_label: str, joint_count: int, quit_key: str) -> None:
    """Write source, joint count, and coordinate frame on the image."""
    status = f"{joint_count} joints" if joint_count else "no person"
    text = (
        f"{source_label}  {status}  |  camera 2D pixels (u,v)  |  {quit_key}=quit"
    )
    cv2.putText(
        frame,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
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
