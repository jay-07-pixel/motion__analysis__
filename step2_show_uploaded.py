"""Step 2 — show RGB from an uploaded file (.mp4 / .avi / .bag).

How to run (from the project folder):
    1. Put a clip in data/input/  (or any path)
    2. Set source.file_path in config.yaml
    3. python step2_show_uploaded.py

Keys:
    q  quit (from config.yaml -> display.quit_key)

.mp4/.avi = colour movie only (2D ok).
.bag      = RealSense recording (RGB now; depth is in the file for later 3D).

No body joints yet. This step only proves: uploaded file -> same RGB frames
as live, in camera 2D pixel coordinates.
"""

import sys
from pathlib import Path

import cv2

# Make sure "src" can be imported when we run this file from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.factory import create_rgb_source
from src.capture.video_file import VideoFileRGB
from src.utils.config_loader import load_config


def main() -> None:
    """Load config, open the uploaded file, show RGB until quit or end of file."""
    config = load_config()
    display_cfg = config["display"]
    source_cfg = config["source"]
    quit_key = str(display_cfg["quit_key"])
    window_title = str(display_cfg["window_title"])

    # Step 2 is the file path. If someone left mode on live, still run file
    # from this script so notes stay clear: this file = uploaded.
    if str(source_cfg.get("mode", "")).lower() != "file":
        print("Note: this script plays an uploaded file.")
        print("      Set source.mode: file in config.yaml (doing that for this run).")
        source_cfg["mode"] = "file"

    try:
        source = create_rgb_source(config)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print("Could not open the uploaded file.")
        print(str(error))
        print()
        print("Put a clip in data/input/ and set source.file_path in config.yaml")
        print("  .mp4 / .avi  = colour video")
        print("  .bag         = RealSense Viewer recording")
        sys.exit(1)

    print("Starting uploaded RGB playback...")
    print(f"  Source: {source.label}")
    print(f"  Path:   {source_cfg.get('file_path')}")
    print(f"  Press '{quit_key}' in the video window to quit.")

    try:
        source.start()
    except (FileNotFoundError, RuntimeError) as error:
        print("Could not start playback.")
        print(str(error))
        print()
        print("Put a clip in data/input/ and set source.file_path in config.yaml")
        sys.exit(1)

    # FPS is known only after the file is open.
    delay_ms = _playback_delay_ms(config, source)

    try:
        while True:
            frame = source.get_frame()
            if source.ended:
                print("End of file.")
                break
            if frame is None:
                continue

            cv2.putText(
                frame,
                f"{source.label}  {frame.shape[1]}x{frame.shape[0]}  |  camera 2D pixels  |  {quit_key}=quit",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_title, frame)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord(quit_key):
                break
    finally:
        source.stop()
        cv2.destroyAllWindows()
        print("Playback stopped.")


def _playback_delay_ms(config: dict, source: object) -> int:
    """How long to wait between frames so an mp4 plays at real speed.

    Bags already wait inside the RealSense SDK (realtime playback), so we
    use 1 ms there. Movies use the file FPS, or camera.fps from config.
    """
    if source.label == "bag":
        return 1

    fps = 0.0
    if isinstance(source, VideoFileRGB):
        fps = source.playback_fps()
    if fps <= 1.0:
        fps = float(config["camera"]["fps"])
    return max(1, int(1000.0 / fps))


if __name__ == "__main__":
    main()
