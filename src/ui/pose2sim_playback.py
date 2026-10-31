"""Offline playback of recorded Pose2Sim upper-LSTM estimates.

The eight `*_study` markers in augmented_markers_yup.csv belong to recording
20261011-144812. They are not live camera predictions. Official v0.3 upper
ONNX takes the full trial as one sequence (1, T, 23), not one frame.

This module replays that recorded sequence with observed MediaPipe+depth and
the saved estimates together. It does not open a second RealSense pipeline
and does not run live LSTM inference.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.pose.keypoints import Keypoint2D
from src.ui.live_markers import draw_live_landmarks

PROJECT = Path(__file__).resolve().parents[2]
RECORDING = PROJECT / "data" / "output" / "20261011-144812"
OBS_CSV = RECORDING / "keypoints_2d.csv"
OVERLAY_MP4 = RECORDING / "overlay.mp4"
EST_CSV = (
    PROJECT
    / "experiments"
    / "anatomical_landmark_audit"
    / "output"
    / "20261011-144812_full_body_augmentation"
    / "augmented_markers_yup.csv"
)
UPPER_ESTIMATES = (
    "r_lelbow_study",
    "r_melbow_study",
    "r_lwrist_study",
    "r_mwrist_study",
    "L_lelbow_study",
    "L_melbow_study",
    "L_lwrist_study",
    "L_mwrist_study",
)
BANNER = "OFFLINE PLAYBACK  20261011-144812  —  Pose2Sim estimates are NOT live"


def yup_to_camera(x: float, y: float, z: float) -> tuple[float, float, float]:
    """OpenSim Y-up metres -> camera metres (X right, Y down, Z forward)."""
    return float(x), float(-y), float(z)


def fit_pinhole(samples: list[tuple[float, float, float, float, float]]) -> dict[str, float] | None:
    """Fit u = fx*X/Z + cx, v = fy*Y/Z + cy from observed (X,Y,Z,u,v)."""
    if len(samples) < 20:
        return None
    xz, u, yz, v = [], [], [], []
    for x, y, z, uu, vv in samples:
        if z <= 1e-4:
            continue
        xz.append([x / z, 1.0])
        u.append(uu)
        yz.append([y / z, 1.0])
        v.append(vv)
    if len(u) < 20:
        return None
    fx_cx, _, _, _ = np.linalg.lstsq(np.asarray(xz), np.asarray(u), rcond=None)
    fy_cy, _, _, _ = np.linalg.lstsq(np.asarray(yz), np.asarray(v), rcond=None)
    return {
        "fx": float(fx_cx[0]),
        "cx": float(fx_cx[1]),
        "fy": float(fy_cy[0]),
        "cy": float(fy_cy[1]),
    }


def project_camera_xyz(x: float, y: float, z: float, k: dict[str, float]) -> tuple[float, float] | None:
    if z <= 1e-4:
        return None
    u = k["fx"] * (x / z) + k["cx"]
    v = k["fy"] * (y / z) + k["cy"]
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return float(u), float(v)


def load_observed(path: Path) -> tuple[dict[int, list[Keypoint2D]], dict[int, float], list[tuple]]:
    by_frame: dict[int, list[Keypoint2D]] = {}
    times: dict[int, float] = {}
    samples: list[tuple] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            times[frame] = float(row["time_sec"])
            xs, ys, zs = row["x_m"].strip(), row["y_m"].strip(), row["z_m"].strip()
            xyz = None if xs == "" else (float(xs), float(ys), float(zs))
            kp = Keypoint2D(
                name=row["joint"],
                u_px=float(row["u_px"]),
                v_px=float(row["v_px"]),
                confidence=float(row["confidence"] or 0),
                coord_frame="camera_3d_metre" if xyz else "camera_2d_pixel",
                x_m=None if xyz is None else xyz[0],
                y_m=None if xyz is None else xyz[1],
                z_m=None if xyz is None else xyz[2],
            )
            by_frame.setdefault(frame, []).append(kp)
            if xyz is not None and xyz[2] > 1e-4:
                samples.append((xyz[0], xyz[1], xyz[2], kp.u_px, kp.v_px))
    return by_frame, times, samples


def load_estimates(path: Path) -> dict[int, list[tuple[str, float, float, float]]]:
    by_frame: dict[int, list[tuple[str, float, float, float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["marker"] not in UPPER_ESTIMATES:
                continue
            xs, ys, zs = row["x_m"].strip(), row["y_m"].strip(), row["z_m"].strip()
            if xs == "":
                continue
            x_c, y_c, z_c = yup_to_camera(float(xs), float(ys), float(zs))
            by_frame.setdefault(int(row["frame"]), []).append((row["marker"], x_c, y_c, z_c))
    return by_frame


def estimates_as_keypoints(
    triples: list[tuple[str, float, float, float]],
    pinhole: dict[str, float] | None,
) -> list[Keypoint2D]:
    out: list[Keypoint2D] = []
    for name, x, y, z in triples:
        uv = project_camera_xyz(x, y, z, pinhole) if pinhole else None
        out.append(
            Keypoint2D(
                name=name,
                u_px=-1.0 if uv is None else uv[0],
                v_px=-1.0 if uv is None else uv[1],
                confidence=1.0,
                coord_frame="camera_3d_metre",
                x_m=x,
                y_m=y,
                z_m=z,
            )
        )
    return out


class Pose2SimPlayback:
    """Frame-aligned observed CSV + saved LSTM estimates + overlay video."""

    def __init__(self) -> None:
        if not OBS_CSV.is_file():
            raise FileNotFoundError(OBS_CSV)
        if not EST_CSV.is_file():
            raise FileNotFoundError(EST_CSV)
        self.observed, self.times, samples = load_observed(OBS_CSV)
        self.estimates = load_estimates(EST_CSV)
        self.pinhole = fit_pinhole(samples)
        self.frames = sorted(set(self.observed) | set(self.estimates))
        self.overlay = cv2.VideoCapture(str(OVERLAY_MP4)) if OVERLAY_MP4.is_file() else None
        self.index = 0
        self.mode = "offline_playback"
        self.live_inference = False

    def n_frames(self) -> int:
        return len(self.frames)

    def current_id(self) -> int:
        return self.frames[self.index]

    def seek(self, index: int) -> None:
        self.index = max(0, min(int(index), self.n_frames() - 1))

    def step(self, delta: int = 1) -> None:
        self.seek(self.index + delta)

    def frame_overlay(self) -> tuple[np.ndarray, list[Keypoint2D], list[Keypoint2D], float]:
        frame_id = self.current_id()
        observed = self.observed.get(frame_id, [])
        est_xyz = self.estimates.get(frame_id, [])
        estimates = estimates_as_keypoints(est_xyz, self.pinhole)
        estimates_px = [kp for kp in estimates if kp.u_px >= 0 and kp.v_px >= 0]
        time_sec = self.times.get(frame_id, 0.0)
        if self.overlay is not None:
            self.overlay.set(cv2.CAP_PROP_POS_FRAMES, float(frame_id))
            ok, image = self.overlay.read()
            if not ok or image is None:
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas = draw_live_landmarks(
            image,
            observed,
            min_visibility=0.0,
            fps=7.3,
            space="3d",
            estimates=estimates_px,
            banner=BANNER,
        )
        return canvas, observed, estimates, time_sec

    def close(self) -> None:
        if self.overlay is not None:
            self.overlay.release()
            self.overlay = None


class Pose2SimPlaybackWindow:
    """Tk window: recorded RGB + camera-space 3D observed vs LSTM estimates."""

    def __init__(self, parent) -> None:
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from PIL import Image, ImageTk

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk
        self.player = Pose2SimPlayback()
        self.playing = False
        self.win = tk.Toplevel(parent)
        self.win.title("OFFLINE Pose2Sim playback  ·  not live inference")
        self.win.geometry("1280x780")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self._photo = None

        info = tk.Label(
            self.win,
            text=(
                "Mode: synchronized OFFLINE playback of recording 20261011-144812.\n"
                "Magenta diamonds = Stanford v0.3 upper LSTM estimates saved after that take.\n"
                "They are not predictions of the live camera. Official LSTM uses the full sequence, not frame-by-frame."
            ),
            justify="left",
            font=("Segoe UI", 10),
        )
        info.pack(anchor="w", padx=12, pady=8)

        body = tk.Frame(self.win)
        body.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(body, bg="#111")
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._fig = Figure(figsize=(5.2, 5.2), dpi=100)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._canvas = FigureCanvasTkAgg(self._fig, master=body)
        self._canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        controls = tk.Frame(self.win)
        controls.pack(fill=tk.X, padx=12, pady=8)
        tk.Button(controls, text=" Play ", command=self._play).pack(side=tk.LEFT)
        tk.Button(controls, text=" Pause ", command=self._pause).pack(side=tk.LEFT, padx=6)
        self._slider = tk.Scale(
            controls,
            from_=0,
            to=max(0, self.player.n_frames() - 1),
            orient=tk.HORIZONTAL,
            command=self._on_slider,
            length=480,
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self._status = tk.StringVar(value="")
        tk.Label(controls, textvariable=self._status).pack(side=tk.LEFT)
        self._show()

    def _on_slider(self, value) -> None:
        if self.playing:
            return
        self.player.seek(int(float(value)))
        self._show()

    def _play(self) -> None:
        self.playing = True
        self._tick()

    def _pause(self) -> None:
        self.playing = False

    def _tick(self) -> None:
        if not self.playing:
            return
        try:
            exists = bool(self.win.winfo_exists())
        except Exception:
            return
        if not exists:
            return
        self.player.step(1)
        self._slider.set(self.player.index)
        self._show()
        if self.player.index >= self.player.n_frames() - 1:
            self.playing = False
            return
        self.win.after(80, self._tick)

    def _show(self) -> None:
        canvas, observed, estimates, time_sec = self.player.frame_overlay()
        h, w = canvas.shape[:2]
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        avail_w = max(320, self.video_label.winfo_width() or 720)
        avail_h = max(240, self.video_label.winfo_height() or 480)
        scale = min(avail_w / w, avail_h / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        if (new_w, new_h) != (w, h):
            rgb = cv2.resize(rgb, (new_w, new_h))
        image = self._Image.fromarray(rgb)
        self._photo = self._ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self._photo)
        frame_id = self.player.current_id()
        n_est = len(estimates)
        self._status.set(f"frame {frame_id}   t={time_sec:.2f}s   estimates {n_est}/8")
        self._draw_3d(observed, estimates)

    def _draw_3d(self, observed: list[Keypoint2D], estimates: list[Keypoint2D]) -> None:
        ax = self._ax
        ax.cla()
        ax.set_xlabel("X m")
        ax.set_ylabel("Y m (down)")
        ax.set_zlabel("Z m")
        ox, oy, oz = [], [], []
        for kp in observed:
            if kp.x_m is None:
                continue
            ox.append(kp.x_m)
            oy.append(kp.y_m)
            oz.append(kp.z_m)
        if ox:
            ax.scatter(ox, oy, oz, c="#3ec6ff", s=18, label="observed")
        ex, ey, ez = [], [], []
        for kp in estimates:
            if kp.x_m is None:
                continue
            ex.append(kp.x_m)
            ey.append(kp.y_m)
            ez.append(kp.z_m)
        if ex:
            ax.scatter(ex, ey, ez, c="#e000ff", s=40, marker="D", label="LSTM estimate")
        pts = list(zip(ox, oy, oz)) + list(zip(ex, ey, ez))
        if pts:
            P = np.asarray(pts)
            c = P.mean(axis=0)
            r = max(0.5, float(np.max(np.linalg.norm(P - c, axis=1))) * 1.25)
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.invert_yaxis()
        ax.set_title("OFFLINE  camera metres")
        if ox or ex:
            ax.legend(loc="upper left", fontsize=7)
        self._canvas.draw_idle()

    def close(self) -> None:
        self.playing = False
        self.player.close()
        try:
            self.win.destroy()
        except Exception:
            pass
