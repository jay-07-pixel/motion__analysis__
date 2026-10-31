"""Standalone LIVE Marker Enhancer demo. Not used by the game or app.py.

    python tools/live_marker_enhancer.py

Opens the D455f (one RealSense pipeline, colour + aligned depth), runs the
existing MediaPipe Pose extractor, feeds the existing arm LSTM with the same
preprocessing as tools/marker_enhancer_offline_inference.py, and draws the
eight predicted augmenter markers next to the measured shoulder/elbow/wrist.

Does not write over inference_predictions.csv. Does not change pair lengths.
Predicted points are not treated as epicondyles or styloids.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pyrealsense2 as rs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import marker_enhancer_offline_inference as me
from src.capture.live_realsense import LiveRealSenseRGB
from src.geometry.deproject import attach_camera_xyz
from src.pose.extractor_2d import PoseExtractor2D
from src.utils.config_loader import load_config

REQUIRED_MP = (
    "right_shoulder",
    "left_shoulder",
    "right_elbow",
    "left_elbow",
    "right_wrist",
    "left_wrist",
    "right_hip",
    "left_hip",
)
ARM_MP = REQUIRED_MP[:6]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "marker_enhancer_20261005"
    / "reference_models"
    / "lstm"
    / "arm"
)
ISOLATED_PYTHON = Path(r"C:\Users\jayjo\anaconda3\envs\marker-enhancer-offline\python.exe")
WORKER = PROJECT_ROOT / "tools" / "marker_enhancer_predict_worker.py"

# Same stature/mass as the 20261005 offline run of this subject.
DEFAULT_HEIGHT_M = 1.8
DEFAULT_WEIGHT_KG = 61.2
# Marker Enhancer training is 60 Hz. D455f colour cannot do 1280x720 at 60;
# 640x480 colour+depth is a 60 fps mode on this camera.
LIVE_WIDTH = 640
LIVE_HEIGHT = 480
LIVE_FPS = 60
LIVE_LOG_DIR = PROJECT_ROOT / "data" / "experiments" / "marker_enhancer_live"
MP_OVERLAY = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
ME_CSV_STEMS = (
    "RElbow",
    "RMElbow",
    "RWrist",
    "RMWrist",
    "LElbow",
    "LMElbow",
    "LWrist",
    "LMWrist",
)
MP_CSV_JOINTS = (
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
)
PAIR_STAT_LABELS = (
    ("RIGHT ELBOW PAIR", "right_elbow_pair_mm"),
    ("LEFT ELBOW PAIR", "left_elbow_pair_mm"),
    ("RIGHT WRIST PAIR", "right_wrist_pair_mm"),
    ("LEFT WRIST PAIR", "left_wrist_pair_mm"),
)
# OpenCV BGR + matplotlib hex. One colour per predicted marker, no text on the image.
ME_POINT_STYLE = {
    "RElbow_augmenter": ((0, 0, 255), "#FF0000"),
    "RMElbow_augmenter": ((0, 140, 255), "#FF8C00"),
    "RWrist_augmenter": ((0, 255, 255), "#FFFF00"),
    "RMWrist_augmenter": ((0, 220, 80), "#50DC00"),
    "LElbow_augmenter": ((255, 80, 80), "#5050FF"),
    "LMElbow_augmenter": ((255, 0, 255), "#FF00FF"),
    "LWrist_augmenter": ((255, 255, 0), "#00FFFF"),
    "LMWrist_augmenter": ((180, 80, 255), "#FF50B4"),
}


class Predictor:
    """Existing Keras arm LSTM, in-process or via the isolated TF 2.12 env."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._model = None
        self._proc: subprocess.Popen | None = None
        self.backend = ""
        self.runtime: dict = {}

    def start(self) -> None:
        python = Path(os.environ.get("MARKER_ENHANCER_PYTHON", ISOLATED_PYTHON))
        if python.is_file():
            self._start_worker(python)
            return
        try:
            tf = me._import_tensorflow()
            self._model = me.load_arm_model(self.model_dir, tf)
            self.backend = f"in-process TensorFlow {tf.__version__}"
            self.runtime = {
                "tensorflow": tf.__version__,
                "input_shape": list(self._model.input_shape),
                "output_shape": list(self._model.output_shape),
            }
            print(f"Marker Enhancer model loaded ({self.backend}).")
            return
        except (Exception, SystemExit) as exc:
            raise SystemExit(
                "Cannot load the Marker Enhancer model in this interpreter, and "
                f"the isolated runtime was not found at {ISOLATED_PYTHON}. {exc}"
            ) from exc

    def _start_worker(self, python: Path) -> None:
        if not python.is_file():
            raise SystemExit(
                "Cannot load the Marker Enhancer model in this interpreter, and "
                f"the isolated runtime was not found at {python}. "
                "Set MARKER_ENHANCER_PYTHON to the env that has TensorFlow 2.12."
            )
        self._proc = subprocess.Popen(
            [str(python), str(WORKER), str(self.model_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        info = None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise SystemExit("Predict worker died before ready.")
            try:
                info = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
        if not info.get("ok"):
            raise SystemExit(f"Predict worker failed: {info.get('error')}")
        self.backend = f"worker {python}  TensorFlow {info.get('tensorflow')}"
        self.runtime = info
        print(f"Marker Enhancer model loaded ({self.backend}).")
        print(f"  input_shape={info.get('input_shape')}  output_shape={info.get('output_shape')}")

    def predict_last(self, features_t23: np.ndarray) -> np.ndarray:
        if self._model is not None:
            raw = self._model.predict(features_t23.reshape(1, *features_t23.shape), verbose=0)
            if raw.ndim == 3:
                raw = raw[0]
            last = np.asarray(raw[-1], dtype=float)
            if last.shape != (24,) or not np.all(np.isfinite(last)):
                raise RuntimeError("Model returned a non-finite or wrong-shaped last step.")
            return last
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Predict worker is not running.")
        payload = json.dumps({"features": features_t23.astype(float).tolist()})
        self._proc.stdin.write(payload + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Predict worker closed.")
        info = json.loads(line)
        if not info.get("ok"):
            raise RuntimeError(str(info.get("error") or "predict failed"))
        last = np.asarray(info["raw_last"], dtype=float)
        if last.shape != (24,) or not np.all(np.isfinite(last)):
            raise RuntimeError("Worker returned a non-finite last step.")
        return last

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


def joints_from_keypoints(keypoints, min_confidence: float) -> dict[str, np.ndarray] | None:
    by_name = {kp.name: kp for kp in keypoints}
    joints = {}
    for name in REQUIRED_MP:
        kp = by_name.get(name)
        if kp is None or kp.confidence < min_confidence:
            return None
        if kp.x_m is None or kp.y_m is None or kp.z_m is None:
            return None
        point = np.array([kp.x_m, kp.y_m, kp.z_m], dtype=float)
        if not np.all(np.isfinite(point)):
            return None
        joints[name] = point
    return joints


def decode_last_step(raw_last: np.ndarray, midhip_me: np.ndarray, height_m: float) -> dict[str, np.ndarray]:
    predicted_camera = {}
    for index, name in enumerate(me.OUTPUT_MARKERS):
        local = raw_last[index * 3 : index * 3 + 3] * height_m + midhip_me
        predicted_camera[name] = me._to_camera(local.reshape(1, 3))[0]
    return predicted_camera


def pair_mm(pred: dict[str, np.ndarray], a: str, b: str) -> float:
    return float(np.linalg.norm(pred[a] - pred[b]) * 1000.0)


def live_csv_path() -> Path:
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LIVE_LOG_DIR / f"live_predictions_{stamp}.csv"


def live_csv_fields() -> list[str]:
    fields = ["timestamp", "frame"]
    for stem in ME_CSV_STEMS:
        fields.extend((f"{stem}_x", f"{stem}_y", f"{stem}_z"))
    fields.extend(
        (
            "right_elbow_pair_mm",
            "left_elbow_pair_mm",
            "right_wrist_pair_mm",
            "left_wrist_pair_mm",
        )
    )
    for joint in MP_CSV_JOINTS:
        fields.extend((f"{joint}_x", f"{joint}_y", f"{joint}_z"))
    return fields


def prediction_row(frame: int, pred: dict[str, np.ndarray], mp_joints: dict[str, np.ndarray]) -> dict:
    row = {"timestamp": datetime.now().isoformat(timespec="milliseconds"), "frame": frame}
    for marker, stem in zip(me.OUTPUT_MARKERS, ME_CSV_STEMS):
        xyz = pred[marker]
        row[f"{stem}_x"] = float(xyz[0])
        row[f"{stem}_y"] = float(xyz[1])
        row[f"{stem}_z"] = float(xyz[2])
    row["right_elbow_pair_mm"] = pair_mm(pred, "RElbow_augmenter", "RMElbow_augmenter")
    row["left_elbow_pair_mm"] = pair_mm(pred, "LElbow_augmenter", "LMElbow_augmenter")
    row["right_wrist_pair_mm"] = pair_mm(pred, "RWrist_augmenter", "RMWrist_augmenter")
    row["left_wrist_pair_mm"] = pair_mm(pred, "LWrist_augmenter", "LMWrist_augmenter")
    for joint in MP_CSV_JOINTS:
        xyz = mp_joints[joint]
        row[f"{joint}_x"] = float(xyz[0])
        row[f"{joint}_y"] = float(xyz[1])
        row[f"{joint}_z"] = float(xyz[2])
    return row


def print_live_summary(frames_seen: int, valid_me: int, pred_ok: int, pair_series: dict[str, list[float]]) -> None:
    print("")
    print("=== Marker Enhancer LIVE summary ===")
    print(f"Total camera frames: {frames_seen}")
    print(f"Valid ME frames: {valid_me}")
    print(f"Successful predictions: {pred_ok}")
    for title, key in PAIR_STAT_LABELS:
        values = np.asarray(pair_series[key], dtype=float)
        print("")
        print(title)
        if values.size == 0:
            print("  min: -- mm")
            print("  max: -- mm")
            print("  mean: -- mm")
            print("  median: -- mm")
            print("  std: -- mm")
            continue
        print(f"  min: {float(np.min(values)):.1f} mm")
        print(f"  max: {float(np.max(values)):.1f} mm")
        print(f"  mean: {float(np.mean(values)):.1f} mm")
        print(f"  median: {float(np.median(values)):.1f} mm")
        print(f"  std: {float(np.std(values)):.1f} mm")


def project_point(xyz: np.ndarray, intrinsics) -> tuple[int, int] | None:
    if intrinsics is None or xyz[2] <= 0.05:
        return None
    try:
        u, v = rs.rs2_project_point_to_pixel(intrinsics, xyz.astype(float).tolist())
    except Exception:
        return None
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return int(round(u)), int(round(v))


def _pixel_from_keypoint(kp) -> tuple[int, int] | None:
    if kp is None:
        return None
    u, v = int(round(kp.u_px)), int(round(kp.v_px))
    if not (math.isfinite(kp.u_px) and math.isfinite(kp.v_px)):
        return None
    return u, v


def draw_camera_overlay(bgr, keypoints, pred, pair_text, status, intrinsics) -> np.ndarray:
    """RGB overlay: MediaPipe pixels + projected Marker Enhancer camera metres."""
    canvas = bgr.copy()
    h, w = canvas.shape[:2]
    by_name = {kp.name: kp for kp in keypoints or []}
    grey_fill = (220, 220, 220)
    grey_edge = (80, 80, 80)
    bone = (190, 190, 190)
    for a, b in (
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
    ):
        pa, pb = _pixel_from_keypoint(by_name.get(a)), _pixel_from_keypoint(by_name.get(b))
        if pa and pb:
            cv2.line(canvas, pa, pb, bone, 3, cv2.LINE_AA)
    for name in MP_OVERLAY:
        uv = _pixel_from_keypoint(by_name.get(name))
        if uv is None:
            continue
        cv2.circle(canvas, uv, 5, grey_fill, -1, cv2.LINE_AA)
        cv2.circle(canvas, uv, 6, grey_edge, 1, cv2.LINE_AA)
    if pred:
        for name in me.OUTPUT_MARKERS:
            pt = project_point(pred[name], intrinsics)
            if pt is None or not (0 <= pt[0] < w and 0 <= pt[1] < h):
                continue
            bgr_color = ME_POINT_STYLE[name][0]
            cv2.circle(canvas, pt, 3, bgr_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)
    panel = np.zeros((220, w, 3), dtype=np.uint8)
    panel[:] = (28, 24, 20)
    lines = [
        "Marker Enhancer  LIVE   (points only, not epicondyles/styloids)",
        status,
        *pair_text,
        "Grey = MediaPipe    Coloured dots = Marker Enhancer",
        "Q or ESC to quit. Pair millimetres are raw model output (not forced).",
    ]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (12, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    x0 = 12
    y0 = 168
    for name in me.OUTPUT_MARKERS:
        bgr_color = ME_POINT_STYLE[name][0]
        cv2.circle(panel, (x0 + 6, y0), 5, bgr_color, -1, cv2.LINE_AA)
        cv2.circle(panel, (x0 + 6, y0), 6, (255, 255, 255), 1, cv2.LINE_AA)
        x0 += 22
    return np.vstack([canvas, panel])


def render_3d(mp_joints, pred, pair_text, status) -> np.ndarray:
    fig = plt.figure(figsize=(8.4, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    def xyz(p):
        return p[0], p[2], -p[1]

    if mp_joints:
        pts = np.vstack([mp_joints[n] for n in ARM_MP])
        xs, zs, ys = pts[:, 0], pts[:, 2], -pts[:, 1]
        ax.scatter(xs, zs, ys, c="#94A3B8", s=28, label="MediaPipe shoulder/elbow/wrist")
        for a, b in (
            ("left_shoulder", "right_shoulder"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
        ):
            xa, za, ya = xyz(mp_joints[a])
            xb, zb, yb = xyz(mp_joints[b])
            ax.plot([xa, xb], [za, zb], [ya, yb], color="#64748B", lw=1.6)
    if pred:
        for name in me.OUTPUT_MARKERS:
            x, z, y = xyz(pred[name])
            ax.scatter([x], [z], [y], c=ME_POINT_STYLE[name][1], s=20, marker="o")
    ax.set_xlabel("X camera m (right)", color="#E5E7EB")
    ax.set_ylabel("Z camera m (forward)", color="#E5E7EB")
    ax.set_zlabel("up m  (= -Y camera)", color="#E5E7EB")
    ax.tick_params(colors="#9CA3AF")
    title = "Marker Enhancer — LIVE\n" + status + "\n" + "  |  ".join(pair_text)
    ax.set_title(title, color="#F9FAFB", fontsize=10)
    if mp_joints or pred:
        ax.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", labelcolor="#E5E7EB")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Marker Enhancer visual demo (D455f).")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--height-m", type=float, default=DEFAULT_HEIGHT_M)
    parser.add_argument("--weight-kg", type=float, default=DEFAULT_WEIGHT_KG)
    parser.add_argument(
        "--history",
        type=int,
        default=60,
        help="LSTM sequence length in valid frames (rolling). Offline used the full take.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many camera frames (0 = run until Q/ESC).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    height_m, weight_kg = me._require_anthropometrics(args.height_m, args.weight_kg)
    history = max(1, int(args.history))
    mean = np.load(args.model_dir / "mean.npy")
    std = np.load(args.model_dir / "std.npy")
    if mean.shape != (23,) or std.shape != (23,):
        raise SystemExit(f"mean/std must be (23,). Got {mean.shape} {std.shape}.")

    print("Standalone Marker Enhancer LIVE demo.")
    print("Not PatientBalanceAI. Not app.py. Does not rewrite offline CSVs.")
    print(f"Model dir: {args.model_dir}")
    print(f"Anthropometrics: height_m={height_m}  weight_kg={weight_kg}")
    print("  (defaults match data/experiments/marker_enhancer_20261005)")
    print(f"LSTM rolling history: {history} valid frames. Training rate was 60 Hz.")
    print(f"Live D455f request: {LIVE_WIDTH}x{LIVE_HEIGHT} colour+depth @ {LIVE_FPS} fps")
    print("  (1280x720 is 30 fps max on this camera; 640x480 is used for 60 fps.)")
    print("Camera map: x_me = R_CAM_TO_ME @ x_camera, then midHip-centre / height, then (x-mean)/std.")
    print("Predictions: last LSTM step * height + midHip, then back through R_CAM_TO_ME.T.")
    print("Close RealSense Viewer, python app.py, and pose_game_bridge.py first.")

    predictor = Predictor(args.model_dir)
    predictor.start()

    config = load_config()
    camera_cfg = config["camera"]
    depth_cfg = camera_cfg.get("depth") or {}
    pose_cfg = config["pose"]
    camera = LiveRealSenseRGB(
        width=LIVE_WIDTH,
        height=LIVE_HEIGHT,
        fps=LIVE_FPS,
        enable_depth=True,
        depth_width=LIVE_WIDTH,
        depth_height=LIVE_HEIGHT,
        depth_fps=LIVE_FPS,
    )
    extractor = PoseExtractor2D(
        model_complexity=int(pose_cfg["model_complexity"]),
        min_detection_confidence=float(pose_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(pose_cfg["min_tracking_confidence"]),
    )

    try:
        camera.start()
    except RuntimeError as error:
        predictor.close()
        extractor.close()
        print("Could not start the D455f.")
        print(str(error))
        print("True live inference is blocked until the camera is free and plugged in (USB 3).")
        raise SystemExit(1)

    print("D455f started (colour + aligned depth). Stand in view. Q/ESC quits.")
    print("Windows: 'ME live camera' and 'ME live 3D'.")

    csv_path = live_csv_path()
    csv_fields = live_csv_fields()
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    csv_file.flush()
    print(f"Live CSV: {csv_path}")

    frame_slots: deque[dict] = deque(maxlen=history)
    frames_seen = 0
    valid_me = 0
    pred_ok = 0
    pair_series = {key: [] for _, key in PAIR_STAT_LABELS}
    t0 = time.perf_counter()
    fps = 0.0
    last_pred = None
    unavailable = "Prediction unavailable (need Pose + valid depth on 8 joints)"
    cam_win = "ME live camera"
    viz_win = "ME live 3D"

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            frames_seen += 1
            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                fps = frames_seen / elapsed
            keypoints = extractor.extract(frame)
            keypoints = attach_camera_xyz(
                keypoints,
                camera,
                min_depth_m=float(depth_cfg.get("min_m", 0.3)),
                max_depth_m=float(depth_cfg.get("max_m", 6.0)),
                sample_window=int(depth_cfg.get("sample_window", 5)),
            )
            joints = joints_from_keypoints(keypoints, me.MIN_CONFIDENCE)
            pair_text = [
                "Right Elbow Pair: -- mm",
                "Left Elbow Pair: -- mm",
                "Right Wrist Pair: -- mm",
                "Left Wrist Pair: -- mm",
            ]
            status = f"FPS: {fps:.1f}    Frames: {frames_seen}    LSTM window: {len(frame_slots)}/{history}"
            pred = None
            if joints is None:
                status = status + "    " + unavailable
            else:
                valid_me += 1
                frame_slots.append({"frame": frames_seen, "time_sec": elapsed, "joints": joints})
                try:
                    features, _measured, midhip_me = me.build_features(list(frame_slots), height_m, weight_kg, mean, std)
                    raw_last = predictor.predict_last(features[0])
                    pred = decode_last_step(raw_last, midhip_me[-1], height_m)
                    last_pred = pred
                    pred_ok += 1
                    pair_vals = {
                        "right_elbow_pair_mm": pair_mm(pred, "RElbow_augmenter", "RMElbow_augmenter"),
                        "left_elbow_pair_mm": pair_mm(pred, "LElbow_augmenter", "LMElbow_augmenter"),
                        "right_wrist_pair_mm": pair_mm(pred, "RWrist_augmenter", "RMWrist_augmenter"),
                        "left_wrist_pair_mm": pair_mm(pred, "LWrist_augmenter", "LMWrist_augmenter"),
                    }
                    for key, value in pair_vals.items():
                        pair_series[key].append(value)
                    pair_text = [
                        f"Right Elbow Pair: {pair_vals['right_elbow_pair_mm']:.1f} mm",
                        f"Left Elbow Pair: {pair_vals['left_elbow_pair_mm']:.1f} mm",
                        f"Right Wrist Pair: {pair_vals['right_wrist_pair_mm']:.1f} mm",
                        f"Left Wrist Pair: {pair_vals['left_wrist_pair_mm']:.1f} mm",
                    ]
                    csv_writer.writerow(prediction_row(frames_seen, pred, joints))
                    csv_file.flush()
                    status = (
                        f"FPS: {fps:.1f}    Frames: {frames_seen}    "
                        f"LSTM window: {len(frame_slots)}/{history}    predictions: {pred_ok}"
                    )
                except Exception as exc:
                    pred = last_pred
                    status = status + f"    Prediction unavailable ({exc})"
            overlay = draw_camera_overlay(frame, keypoints, pred, pair_text, status, camera.color_intrinsics())
            view3d = render_3d(joints, pred, pair_text, status)
            cv2.imshow(cam_win, overlay)
            cv2.imshow(viz_win, view3d)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(cam_win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.max_frames > 0 and frames_seen >= args.max_frames:
                break
    finally:
        cv2.destroyAllWindows()
        extractor.close()
        camera.stop()
        predictor.close()
        csv_file.close()
        print_live_summary(frames_seen, valid_me, pred_ok, pair_series)
        print("")
        print(f"Live CSV: {csv_path}")
        print(f"CSV rows: {pred_ok}")
        print("Camera released.")


if __name__ == "__main__":
    main()
