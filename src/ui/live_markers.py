"""Live inventory of observed MediaPipe + RealSense landmarks.

Does not invent anatomy. Invalid depth is never drawn as a 3D point.
RGB overlay and the 3D window reuse keypoints already produced by MotionSession2D.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.markers.catalog import GEOMETRIC_CENTRES, TOE_INPUTS
from src.pose.hands import HAND_SKELETON_BONES, HAND_STEMS, POSE_SIMPLE_HAND_BONES, hand_joint_name
from src.pose.keypoints import Keypoint2D
from src.pose.skeleton import LANDMARK_NAMES, SKELETON_BONES

POSE_NAME_SET = frozenset(LANDMARK_NAMES)
HAND_NAME_SET = frozenset(
    hand_joint_name(side, stem) for side in ("left", "right") for stem in HAND_STEMS
)

# BGR: left cyan, right orange, midline yellow.
_LEFT = (255, 190, 40)
_RIGHT = (40, 110, 255)
_MID = (0, 220, 220)
_INVALID_RING = (40, 40, 220)
# Pose2Sim LSTM estimates (offline). Distinct from left/right observed colours.
_ESTIMATE = (180, 0, 255)
_CALIBRATED = (0, 255, 180)
_CENTRE = (80, 220, 255)

# Observed 3D joints the live view should emphasise. Not paper SAT / epicondyles.
_FOCUS_OBSERVED = frozenset(
    {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    }
)


def has_camera_xyz(kp: Keypoint2D) -> bool:
    """True only when RealSense deprojection filled all three metres."""
    return kp.x_m is not None and kp.y_m is not None and kp.z_m is not None


def short_label(name: str) -> str:
    """Compact on-image tag. MediaPipe model names, not ISB anatomy."""
    if name.startswith("left_"):
        return "L " + name[5:].replace("_", " ")
    if name.startswith("right_"):
        return "R " + name[6:].replace("_", " ")
    return name.replace("_", " ")


def _bgr_for(name: str) -> tuple[int, int, int]:
    if name.startswith("left_"):
        return _LEFT
    if name.startswith("right_"):
        return _RIGHT
    return _MID


def visible(kp: Keypoint2D, min_visibility: float) -> bool:
    return kp.confidence >= float(min_visibility)


@dataclass(frozen=True)
class MarkerInventory:
    """Counts for the live HUD. Model landmarks, not verified bones."""

    n_pose_detected: int
    n_pose_xyz: int
    n_hand_detected: int
    n_hand_xyz: int
    pose_names: tuple[str, ...]
    hand_names: tuple[str, ...]
    xyz_names: tuple[str, ...]
    fps: float
    space: str
    tracking: str


def inventory(
    keypoints: list[Keypoint2D],
    min_visibility: float,
    fps: float,
    space: str,
) -> MarkerInventory:
    """Split Pose-33 vs Hands-21×2. Wrist/tips can count in both graphs."""
    pose_det: list[str] = []
    pose_xyz: list[str] = []
    hand_det: list[str] = []
    hand_xyz: list[str] = []
    xyz_names: list[str] = []
    for kp in keypoints:
        if not visible(kp, min_visibility):
            continue
        xyz = has_camera_xyz(kp)
        if xyz:
            xyz_names.append(kp.name)
        if kp.name in POSE_NAME_SET:
            pose_det.append(kp.name)
            if xyz:
                pose_xyz.append(kp.name)
        if kp.name in HAND_NAME_SET:
            hand_det.append(kp.name)
            if xyz:
                hand_xyz.append(kp.name)
    if pose_det:
        tracking = "person"
        if hand_det:
            sides = []
            if any(n.startswith("left_") for n in hand_det):
                sides.append("L hand")
            if any(n.startswith("right_") for n in hand_det):
                sides.append("R hand")
            tracking = "person + " + " + ".join(sides)
    else:
        tracking = "no person"
    return MarkerInventory(
        n_pose_detected=len(pose_det),
        n_pose_xyz=len(pose_xyz),
        n_hand_detected=len(hand_det),
        n_hand_xyz=len(hand_xyz),
        pose_names=tuple(pose_det),
        hand_names=tuple(hand_det),
        xyz_names=tuple(xyz_names),
        fps=float(fps),
        space=str(space),
        tracking=tracking,
    )


def draw_live_landmarks(
    frame: np.ndarray,
    keypoints: list[Keypoint2D],
    min_visibility: float,
    fps: float,
    space: str,
    estimates: list[Keypoint2D] | None = None,
    banner: str | None = None,
    banners: list[str] | None = None,
) -> np.ndarray:
    """All currently detected model landmarks on a copy of the RGB frame.

    Optional `estimates` are Pose2Sim LSTM points already converted to pixels.
    They are drawn as magenta diamonds. No text, names, or tags on the video.
    """
    _ = fps, space, banner, banners
    canvas = frame.copy()
    by_name = {kp.name: kp for kp in keypoints}
    bones = list(SKELETON_BONES) + list(HAND_SKELETON_BONES)
    for start_name, end_name in bones:
        if (start_name, end_name) in POSE_SIMPLE_HAND_BONES:
            knuckle = by_name.get(
                f"{'left' if start_name.startswith('left_') else 'right'}_index_mcp"
            )
            if knuckle is not None and visible(knuckle, min_visibility):
                continue
        start = by_name.get(start_name)
        end = by_name.get(end_name)
        if start is None or end is None:
            continue
        if not visible(start, min_visibility) or not visible(end, min_visibility):
            continue
        colour = _bgr_for(start_name if start_name.startswith(("left_", "right_")) else end_name)
        cv2.line(
            canvas,
            (int(round(start.u_px)), int(round(start.v_px))),
            (int(round(end.u_px)), int(round(end.v_px))),
            colour,
            2,
            cv2.LINE_AA,
        )

    for kp in keypoints:
        if not visible(kp, min_visibility):
            continue
        colour = _bgr_for(kp.name)
        centre = (int(round(kp.u_px)), int(round(kp.v_px)))
        radius = 4 if kp.name in _FOCUS_OBSERVED else 3
        if has_camera_xyz(kp):
            cv2.circle(canvas, centre, radius, colour, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(canvas, centre, radius, _INVALID_RING, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, centre, max(1, radius // 2), colour, thickness=-1, lineType=cv2.LINE_AA)

    h, w = canvas.shape[:2]
    for kp in estimates or []:
        u, v = int(round(kp.u_px)), int(round(kp.v_px))
        if u < 0 or v < 0 or u >= w or v >= h:
            continue
        if kp.name in TOE_INPUTS:
            colour, marker = _CALIBRATED, cv2.MARKER_SQUARE
        elif kp.name in GEOMETRIC_CENTRES:
            colour, marker = _CENTRE, cv2.MARKER_TILTED_CROSS
        else:
            colour, marker = _ESTIMATE, cv2.MARKER_DIAMOND
        cv2.circle(canvas, (u, v), 8, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 5, colour, thickness=-1, lineType=cv2.LINE_AA)
        cv2.drawMarker(
            canvas,
            (u, v),
            (255, 255, 255),
            markerType=marker,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    return canvas


def _xyz_points(keypoints: list[Keypoint2D], min_visibility: float) -> list[Keypoint2D]:
    return [kp for kp in keypoints if visible(kp, min_visibility) and has_camera_xyz(kp)]


class LiveMarkerWindow:
    """Tk Toplevel: 3D camera-space plot + inventory. UI thread only."""

    def __init__(self, parent) -> None:
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self._tk = tk
        self.win = tk.Toplevel(parent)
        self.win.title("Live landmarks  ·  camera metres (X right, Y down, Z forward)")
        self.win.geometry("1100x720")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._closed = False
        self._hover = tk.StringVar(value="Hover a 3D point or pick a name in the list.")
        self._inventory_text = tk.StringVar(value="Waiting for frames…")

        top = tk.Frame(self.win)
        top.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(top, textvariable=self._inventory_text, justify="left", font=("Segoe UI", 10)).pack(
            anchor="w"
        )
        tk.Label(top, textvariable=self._hover, justify="left", font=("Consolas", 10)).pack(anchor="w")

        body = tk.Frame(self.win)
        body.pack(fill=tk.BOTH, expand=True)
        self._fig = Figure(figsize=(7.2, 6.2), dpi=100)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._canvas = FigureCanvasTkAgg(self._fig, master=body)
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._list = tk.Listbox(body, width=36, font=("Consolas", 9))
        self._list.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self._list.bind("<<ListboxSelect>>", self._on_list)
        self._pick_names: list[str] = []
        self._by_name: dict[str, Keypoint2D] = {}
        self._last_draw = 0.0
        self._canvas.mpl_connect("pick_event", self._on_pick)

    def _on_close(self) -> None:
        self._closed = True
        try:
            self.win.destroy()
        except Exception:
            pass

    def alive(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self.win.winfo_exists())
        except Exception:
            return False

    def update(
        self,
        keypoints: list[Keypoint2D],
        min_visibility: float,
        fps: float,
        space: str,
        estimates: list[Keypoint2D] | None = None,
    ) -> None:
        if not self.alive():
            return
        stats = inventory(keypoints, min_visibility, fps, space)
        names = ", ".join(short_label(n) for n in stats.pose_names[:12])
        extra = "" if len(stats.pose_names) <= 12 else "…"
        xyz_line = (
            f"Pose XYZ {stats.n_pose_xyz}/{stats.n_pose_detected}   "
            f"Hand XYZ {stats.n_hand_xyz}/{stats.n_hand_detected}"
            if str(space).lower() == "3d"
            else "3D view needs the 3D button + Live (or a .bag). 2D mode has pixels only."
        )
        self._inventory_text.set(
            f"{stats.fps:.1f} fps   {stats.tracking}   space={stats.space}\n"
            f"Pose detected {stats.n_pose_detected}/33    Hands detected {stats.n_hand_detected}/42\n"
            f"{xyz_line}\n"
            f"Visible pose: {names}{extra}"
        )
        self._by_name = {kp.name: kp for kp in keypoints}
        now = time.perf_counter()
        if now - self._last_draw < 0.08:
            return
        self._last_draw = now
        self._redraw_3d(
            _xyz_points(keypoints, min_visibility),
            str(space).lower() == "3d",
            estimates or [],
        )

    def _redraw_3d(
        self,
        points: list[Keypoint2D],
        is_3d: bool,
        estimates: list[Keypoint2D] | None = None,
    ) -> None:
        ax = self._ax
        ax.cla()
        ax.set_xlabel("X m (right)")
        ax.set_ylabel("Y m (down)")
        ax.set_zlabel("Z m (forward)")
        self._list.delete(0, self._tk.END)
        self._pick_names = []
        if not is_3d:
            ax.set_title("No camera-space XYZ in 2D mode")
            self._canvas.draw_idle()
            return
        if not points:
            ax.set_title("No valid depth this frame")
            self._canvas.draw_idle()
            return

        by_name = {kp.name: kp for kp in points}
        for a, b in list(SKELETON_BONES) + list(HAND_SKELETON_BONES):
            if (a, b) in POSE_SIMPLE_HAND_BONES:
                side = "left" if a.startswith("left_") else "right"
                if f"{side}_index_mcp" in by_name:
                    continue
            pa, pb = by_name.get(a), by_name.get(b)
            if pa is None or pb is None:
                continue
            ax.plot(
                [pa.x_m, pb.x_m],
                [pa.y_m, pb.y_m],
                [pa.z_m, pb.z_m],
                color="#4aa3ff" if a.startswith("left_") else ("#e67e22" if a.startswith("right_") else "#cccc00"),
                lw=1.2,
            )
        xs = [kp.x_m for kp in points]
        ys = [kp.y_m for kp in points]
        zs = [kp.z_m for kp in points]
        colours = []
        sizes = []
        for kp in points:
            if kp.name.startswith("left_"):
                colours.append("#3ec6ff")
            elif kp.name.startswith("right_"):
                colours.append("#e67e22")
            else:
                colours.append("#f1c40f")
            sizes.append(36 if kp.name in _FOCUS_OBSERVED else 16)
        ax.scatter(xs, ys, zs, c=colours, s=sizes, picker=6, depthshade=False)
        est = [kp for kp in (estimates or []) if has_camera_xyz(kp)]
        if est:
            est_colours = []
            for kp in est:
                if kp.name in TOE_INPUTS:
                    est_colours.append("#00e0a0")
                elif kp.name in GEOMETRIC_CENTRES:
                    est_colours.append("#50dcff")
                else:
                    est_colours.append("#e000ff")
            ax.scatter(
                [kp.x_m for kp in est],
                [kp.y_m for kp in est],
                [kp.z_m for kp in est],
                c=est_colours,
                s=120,
                marker="D",
                depthshade=False,
            )
            xs.extend(kp.x_m for kp in est)
            ys.extend(kp.y_m for kp in est)
            zs.extend(kp.z_m for kp in est)
        self._pick_names = [kp.name for kp in points]
        for kp in sorted(points, key=lambda item: item.name):
            self._list.insert(
                self._tk.END,
                f"{short_label(kp.name):18s}  {kp.x_m:6.3f} {kp.y_m:6.3f} {kp.z_m:6.3f}",
            )
        for kp in est:
            kind = "TOE" if kp.name in TOE_INPUTS else ("CTR" if kp.name in GEOMETRIC_CENTRES else "EST")
            self._list.insert(
                self._tk.END,
                f"{kind} {kp.name:16s}  {kp.x_m:6.3f} {kp.y_m:6.3f} {kp.z_m:6.3f}",
            )
        c = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
        span = max(0.45, float(np.max(np.linalg.norm(np.column_stack([xs, ys, zs]) - c, axis=1))) * 1.25)
        ax.set_xlim(c[0] - span, c[0] + span)
        ax.set_ylim(c[1] - span, c[1] + span)
        ax.set_zlim(c[2] - span, c[2] + span)
        ax.invert_yaxis()
        ax.set_title("Valid depth only  ·  origin = colour optical centre")
        self._canvas.draw_idle()

    def _on_pick(self, event) -> None:
        if not getattr(event, "ind", None):
            return
        index = int(event.ind[0])
        if 0 <= index < len(self._pick_names):
            self._set_hover(self._pick_names[index])

    def _on_list(self, _event) -> None:
        selection = self._list.curselection()
        if not selection:
            return
        line = self._list.get(selection[0])
        label = line[:18].strip()
        for name in self._by_name:
            if short_label(name) == label:
                self._set_hover(name)
                return

    def _set_hover(self, name: str) -> None:
        kp = self._by_name.get(name)
        if kp is None or not has_camera_xyz(kp):
            self._hover.set(f"{short_label(name)}  ·  no valid depth")
            return
        self._hover.set(
            f"{short_label(name)}  ({name})   "
            f"X={kp.x_m:.4f} m  Y={kp.y_m:.4f} m  Z={kp.z_m:.4f} m   conf={kp.confidence:.2f}"
        )
