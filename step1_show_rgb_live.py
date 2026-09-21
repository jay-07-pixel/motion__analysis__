"""Step 1 — show live RGB from the D455f (same colour stream as RealSense Viewer).

How to run (from the project folder, camera plugged into USB 3):
    python step1_show_rgb_live.py

Close RealSense Viewer first. Only one program can own the camera.

Keys:
    q  quit (key comes from config.yaml -> display.quit_key)

This step does NOT detect body joints yet. It only proves: our Python code
can read RGB frames in camera image coordinates.
"""

import sys
from pathlib import Path

import cv2

# Make sure "src" can be imported when we run this file from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.live_realsense import LiveRealSenseRGB
from src.utils.config_loader import load_config


def main() -> None:
    """Load config, open the D455f RGB stream, show frames until the user quits."""
    config = load_config()
    camera_cfg = config["camera"]
    display_cfg = config["display"]

    width = int(camera_cfg["width"])
    height = int(camera_cfg["height"])
    fps = int(camera_cfg["fps"])
    window_title = str(display_cfg["window_title"])
    quit_key = str(display_cfg["quit_key"])

    camera = LiveRealSenseRGB(width=width, height=height, fps=fps)

    print("Starting D455f RGB stream...")
    print(f"  Requested: {width}x{height} @ {fps} fps")
    print("  Close RealSense Viewer if the camera does not start.")
    print(f"  Press '{quit_key}' in the video window to quit.")

    try:
        camera.start()
    except RuntimeError as error:
        # Typical causes: Viewer still open, USB 2 port, or bad resolution.
        print("Could not start the camera.")
        print(str(error))
        sys.exit(1)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            # HUD: remind us these pixels are 2D camera coordinates (origin top-left).
            cv2.putText(
                frame,
                f"RGB  {frame.shape[1]}x{frame.shape[0]}  |  camera 2D pixels  |  {quit_key}=quit",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_title, frame)

            # waitKey(1) is required for OpenCV to actually paint the window.
            key = cv2.waitKey(1) & 0xFF
            if key == ord(quit_key):
                break
    finally:
        # Always stop the pipeline, even if the window is closed with X or an error.
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
