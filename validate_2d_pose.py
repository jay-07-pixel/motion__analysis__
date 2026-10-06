"""2D pose validation for the D455f colour stream.

Separate from the live 3D session. Depth is not opened and is not used.
Angles here are image-plane angles from pixel coordinates, not 3D DOFs.

Run from the project folder, with RealSense Viewer and the live app closed:

    python validate_2d_pose.py

Press q in the window to quit. A short summary is printed on exit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.angles_2d import angle_at_vertex_deg
from src.capture.live_realsense import LiveRealSenseRGB
from src.pose.extractor_2d import PoseExtractor2D
from src.pose.keypoints import Keypoint2D
from src.pose.skeleton import LANDMARK_NAMES
from src.utils.config_loader import load_config

# Joints the presenter needs to see by name. Index is the MediaPipe id.
IMPORTANT = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

# Image-plane angles. Vertex is the middle name. Not 3D biomechanical DOFs.
ANGLE_SPECS = (
    ("left elbow", "left_shoulder", "left_elbow", "left_wrist"),
    ("right elbow", "right_shoulder", "right_elbow", "right_wrist"),
    ("left knee", "left_hip", "left_knee", "left_ankle"),
    ("right knee", "right_hip", "right_knee", "right_ankle"),
    ("left shoulder", "left_hip", "left_shoulder", "left_elbow"),
    ("right shoulder", "right_hip", "right_shoulder", "right_elbow"),
)

# Both sides must be present for that row of the exit summary.
SUMMARY_PAIRS = (
    ("left/right shoulder", "left_shoulder", "right_shoulder"),
    ("left/right elbow", "left_elbow", "right_elbow"),
    ("left/right wrist", "left_wrist", "right_wrist"),
    ("left/right hip", "left_hip", "right_hip"),
    ("left/right knee", "left_knee", "right_knee"),
    ("left/right ankle", "left_ankle", "right_ankle"),
)

PANEL_WIDTH = 460
KNEE_SQUAT_DEG = 140.0
LATERAL_LEAN_RATIO = 0.22
TORSO_SHORT_RATIO = 0.75


def main() -> None:
    config = load_config()
    camera_cfg = config["camera"]
    pose_cfg = config["pose"]
    overlay_cfg = config["overlay"]
    min_confidence = float(pose_cfg["min_visibility"])

    camera = LiveRealSenseRGB(
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        fps=int(camera_cfg["fps"]),
        enable_depth=False,
    )
    extractor = PoseExtractor2D(
        model_complexity=int(pose_cfg["model_complexity"]),
        min_detection_confidence=float(pose_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(pose_cfg["min_tracking_confidence"]),
    )

    stats = {
        "frames": 0,
        "low_confidence": 0,
        "pairs": {label: 0 for label, _a, _b in SUMMARY_PAIRS},
    }
    upright_torso_px: list[float] = []
    started = time.perf_counter()
    last_tick = started
    fps_smooth = 0.0

    print("2D pose validation. Depth is off.")
    print("Close the live app and RealSense Viewer first.")
    print("Press q in the window to quit.")

    try:
        camera.start()
    except RuntimeError as error:
        print("Could not start the camera.")
        print(str(error))
        extractor.close()
        sys.exit(1)

    window = "2D pose validation (image plane only)"
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            now = time.perf_counter()
            dt = now - last_tick
            last_tick = now
            if dt > 0:
                instant = 1.0 / dt
                fps_smooth = instant if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * instant

            keypoints = extractor.extract(frame)
            by_name = {kp.name: kp for kp in keypoints}
            stats["frames"] += 1
            status = tracking_status(by_name, min_confidence)
            if status != "GOOD":
                stats["low_confidence"] += 1
            for label, left, right in SUMMARY_PAIRS:
                if valid(by_name, left, min_confidence) and valid(by_name, right, min_confidence):
                    stats["pairs"][label] += 1

            angles = image_plane_angles(by_name, min_confidence)
            indicators = movement_indicators(
                by_name, angles, frame.shape[1], min_confidence, upright_torso_px, status
            )
            canvas = draw_pose(frame, by_name, overlay_cfg, min_confidence)
            view = compose(
                canvas,
                by_name,
                angles,
                indicators,
                status,
                stats["frames"],
                fps_smooth,
                min_confidence,
            )
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        elapsed = max(time.perf_counter() - started, 1e-6)
        camera.stop()
        extractor.close()
        cv2.destroyAllWindows()
        print_summary(stats, elapsed)


def valid(by_name: dict[str, Keypoint2D], name: str, min_confidence: float) -> bool:
    kp = by_name.get(name)
    return kp is not None and kp.confidence >= min_confidence


def tracking_status(by_name: dict[str, Keypoint2D], min_confidence: float) -> str:
    if not by_name:
        return "POOR"
    if all(valid(by_name, name, min_confidence) for name in IMPORTANT):
        return "GOOD"
    return "POOR"


def image_plane_angles(
    by_name: dict[str, Keypoint2D], min_confidence: float
) -> dict[str, float | None]:
    angles: dict[str, float | None] = {}
    for label, a_name, b_name, c_name in ANGLE_SPECS:
        if not all(valid(by_name, name, min_confidence) for name in (a_name, b_name, c_name)):
            angles[label] = None
            continue
        a, b, c = by_name[a_name], by_name[b_name], by_name[c_name]
        angles[label] = angle_at_vertex_deg((a.u_px, a.v_px), (b.u_px, b.v_px), (c.u_px, c.v_px))
    return angles


def movement_indicators(
    by_name: dict[str, Keypoint2D],
    angles: dict[str, float | None],
    width: int,
    min_confidence: float,
    upright_torso_px: list[float],
    status: str,
) -> list[str]:
    lines = ["Validation indicators (not game logic)"]

    arm_bits = []
    for side in ("left", "right"):
        shoulder = by_name.get(f"{side}_shoulder")
        wrist = by_name.get(f"{side}_wrist")
        if not valid(by_name, f"{side}_shoulder", min_confidence) or not valid(
            by_name, f"{side}_wrist", min_confidence
        ):
            arm_bits.append(f"{side} arm n/a")
            continue
        if wrist.v_px < shoulder.v_px - 20:
            arm_bits.append(f"{side} arm raised")
        elif wrist.v_px > shoulder.v_px + 20:
            arm_bits.append(f"{side} arm lowered")
        else:
            arm_bits.append(f"{side} arm mid")
    lines.append("  " + ", ".join(arm_bits))

    if valid(by_name, "left_shoulder", min_confidence) and valid(
        by_name, "right_shoulder", min_confidence
    ):
        mid_u = 0.5 * (by_name["left_shoulder"].u_px + by_name["right_shoulder"].u_px)
        margin = 0.08 * width
        if mid_u < width / 2 - margin:
            place = "image-left"
        elif mid_u > width / 2 + margin:
            place = "image-right"
        else:
            place = "centered"
        lines.append(f"  body in frame: {place} (shoulder mid u={mid_u:.0f})")
    else:
        lines.append("  body in frame: n/a")

    knee_vals = [angles["left knee"], angles["right knee"]]
    if all(value is not None for value in knee_vals):
        mean_knee = 0.5 * (knee_vals[0] + knee_vals[1])
        squat = "SQUAT" if mean_knee < KNEE_SQUAT_DEG else "not squat"
        lines.append(f"  squat check: {squat} (mean knee {mean_knee:.0f} deg)")
    else:
        lines.append("  squat check: n/a")

    lines.append("  " + lean_text(by_name, min_confidence, upright_torso_px, status))
    return lines


def lean_text(
    by_name: dict[str, Keypoint2D],
    min_confidence: float,
    upright_torso_px: list[float],
    status: str,
) -> str:
    needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(valid(by_name, name, min_confidence) for name in needed):
        return "lean: n/a"
    shoulder_u = 0.5 * (by_name["left_shoulder"].u_px + by_name["right_shoulder"].u_px)
    shoulder_v = 0.5 * (by_name["left_shoulder"].v_px + by_name["right_shoulder"].v_px)
    hip_u = 0.5 * (by_name["left_hip"].u_px + by_name["right_hip"].u_px)
    hip_v = 0.5 * (by_name["left_hip"].v_px + by_name["right_hip"].v_px)
    shoulder_width = abs(by_name["left_shoulder"].u_px - by_name["right_shoulder"].u_px)
    torso_px = hip_v - shoulder_v
    if status == "GOOD" and torso_px > 40 and len(upright_torso_px) < 30:
        upright_torso_px.append(torso_px)

    side = "side lean none"
    if shoulder_width > 10:
        ratio = (shoulder_u - hip_u) / shoulder_width
        if ratio > LATERAL_LEAN_RATIO:
            side = "side lean image-right"
        elif ratio < -LATERAL_LEAN_RATIO:
            side = "side lean image-left"

    if len(upright_torso_px) < 10:
        sag = "toward/away: collecting upright baseline"
    else:
        baseline = float(np.median(upright_torso_px))
        if baseline > 1 and torso_px < TORSO_SHORT_RATIO * baseline:
            sag = "toward/away: torso shortened in image"
        else:
            sag = "toward/away: torso length upright"
    return f"lean: {side}; {sag}"


def draw_pose(
    frame: np.ndarray,
    by_name: dict[str, Keypoint2D],
    overlay_cfg: dict,
    min_confidence: float,
) -> np.ndarray:
    canvas = frame.copy()
    line_color = tuple(int(v) for v in overlay_cfg["line_color_bgr"])
    point_color = tuple(int(v) for v in overlay_cfg["point_color_bgr"])
    thickness = int(overlay_cfg["line_thickness"])
    radius = int(overlay_cfg["point_radius"])
    index_of = {name: i for i, name in enumerate(LANDMARK_NAMES)}

    for start, end in mp.solutions.pose.POSE_CONNECTIONS:
        a = by_name.get(LANDMARK_NAMES[int(start)])
        b = by_name.get(LANDMARK_NAMES[int(end)])
        if a is None or b is None:
            continue
        strong = a.confidence >= min_confidence and b.confidence >= min_confidence
        color = line_color if strong else (80, 80, 80)
        cv2.line(
            canvas,
            (int(round(a.u_px)), int(round(a.v_px))),
            (int(round(b.u_px)), int(round(b.v_px))),
            color,
            thickness,
            cv2.LINE_AA,
        )

    for name in LANDMARK_NAMES:
        kp = by_name.get(name)
        if kp is None:
            continue
        strong = kp.confidence >= min_confidence
        cv2.circle(
            canvas,
            (int(round(kp.u_px)), int(round(kp.v_px))),
            radius,
            point_color if strong else (90, 90, 90),
            -1,
            cv2.LINE_AA,
        )
        if name in IMPORTANT:
            cv2.putText(
                canvas,
                f"{index_of[name]} {name}",
                (int(round(kp.u_px)) + 6, int(round(kp.v_px)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def compose(
    frame: np.ndarray,
    by_name: dict[str, Keypoint2D],
    angles: dict[str, float | None],
    indicators: list[str],
    status: str,
    frame_index: int,
    fps: float,
    min_confidence: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    view = np.zeros((height, width + PANEL_WIDTH, 3), dtype=np.uint8)
    view[:, :width] = frame
    panel = view[:, width:]
    panel[:] = (24, 24, 24)

    lines = [
        f"frame {frame_index}    FPS {fps:.1f}",
        f"2D Tracking Status: {status}",
        f"confidence gate {min_confidence:.2f}",
        "",
        "2D image-plane angle",
    ]
    for label, _a, _b, _c in ANGLE_SPECS:
        value = angles[label]
        shown = "n/a" if value is None else f"{value:.0f} deg"
        lines.append(f"  {label}: {shown}")
    lines.append("")
    lines.extend(indicators)
    lines.append("")
    lines.append("name  id  u_px  v_px  conf")
    index_of = {name: i for i, name in enumerate(LANDMARK_NAMES)}
    for name in IMPORTANT:
        kp = by_name.get(name)
        if kp is None:
            lines.append(f"{name}  {index_of[name]}  missing")
        else:
            lines.append(
                f"{name}  {index_of[name]}  {kp.u_px:.0f}  {kp.v_px:.0f}  {kp.confidence:.2f}"
            )

    y = 22
    for line in lines:
        color = (0, 220, 0) if line.startswith("2D Tracking Status: GOOD") else (230, 230, 230)
        if line.startswith("2D Tracking Status: POOR"):
            color = (0, 80, 255)
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y += 16
        if y > height - 8:
            break
    return view


def print_summary(stats: dict, elapsed: float) -> None:
    total = int(stats["frames"])
    print()
    print("2D validation summary")
    print(f"  total frames: {total}")
    print(f"  average FPS: {total / elapsed:.2f}")
    if total == 0:
        print("  no frames captured")
        return
    for label, _a, _b in SUMMARY_PAIRS:
        count = stats["pairs"][label]
        print(f"  valid {label}: {100.0 * count / total:.1f}%")
    print(f"  low-confidence frames: {stats['low_confidence']}")


if __name__ == "__main__":
    main()
