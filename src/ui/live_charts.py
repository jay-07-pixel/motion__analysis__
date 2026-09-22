"""Live Left/Right body analog dials (2D joint angles).

While the camera runs the needle is live image-plane degrees.
After Stop the same dial shows min (cyan), max (coral) and mode (amber).
~180° ≈ straight in the photo. Not metres, not px/s.
"""

from __future__ import annotations

import math
import statistics
import tkinter as tk

BG = "#101418"
PANEL = "#171e28"
CARD = "#1e2734"
LINE = "#2a3545"
TEXT = "#eef3f8"
MUTED = "#8b98a8"
NEEDLE = "#4c9aff"
MIN_C = "#3dd6c8"
MAX_C = "#e06c5c"
MODE_C = "#e8a338"
ARC = "#3a4658"

JOINT_LABELS = {
    "left_shoulder": "Shoulder",
    "left_elbow": "Elbow",
    "left_wrist": "Wrist",
    "right_shoulder": "Shoulder",
    "right_elbow": "Elbow",
    "right_wrist": "Wrist",
}


class TakeAngleRecorder:
    """Stores this take's gauge angles so Stop can compute min / max / mode."""

    def __init__(self, joint_names: list[str], bin_deg: float) -> None:
        """One list per joint. bin_deg is the histogram width for mode."""
        self.joint_names = list(joint_names)
        self.bin_deg = max(1.0, float(bin_deg))
        self.samples: dict[str, list[float]] = {name: [] for name in self.joint_names}

    def reset(self) -> None:
        """Forget the previous take."""
        for name in self.joint_names:
            self.samples[name] = []

    def add(self, angles: dict[str, float | None]) -> None:
        """Keep one frame. Skip a joint if it was not trusted this frame."""
        for name in self.joint_names:
            value = angles.get(name)
            if value is None:
                continue
            self.samples[name].append(float(value))

    def summary(self) -> dict[str, dict[str, float | None]]:
        """min, max, mode (bin start) per joint. None if no samples."""
        out: dict[str, dict[str, float | None]] = {}
        for name in self.joint_names:
            vals = self.samples[name]
            if not vals:
                out[name] = {"min": None, "max": None, "mode": None}
                continue
            out[name] = {
                "min": min(vals),
                "max": max(vals),
                "mode": _binned_mode(vals, self.bin_deg),
            }
        return out


def _binned_mode(values: list[float], bin_width: float) -> float:
    """Most common angle bin (degrees). Ties go to the lower bin."""
    if not values:
        return 0.0
    keys = [int(v // bin_width) for v in values]
    winner = statistics.multimode(keys)[0]
    return winner * bin_width


class Speedometer(tk.Canvas):
    """One analog dial: live needle, optional min/max/mode ticks after Stop."""

    def __init__(self, parent: tk.Widget, title: str, max_deg: float) -> None:
        """Build an empty dial. Call set_live() from the UI thread."""
        super().__init__(parent, bg=CARD, highlightthickness=0, height=168)
        self.title = title
        self.max_deg = max(1.0, float(max_deg))
        self._live: float | None = None
        self._min: float | None = None
        self._max: float | None = None
        self._mode: float | None = None
        self._stopped = False
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_live(self, value: float | None) -> None:
        """Move the needle while the camera is running."""
        self._live = value
        self._stopped = False
        self._min = self._max = self._mode = None
        self._redraw()

    def set_stopped(
        self,
        last: float | None,
        min_v: float | None,
        max_v: float | None,
        mode_v: float | None,
    ) -> None:
        """Freeze last needle; draw min / max / mode with other marks."""
        self._live = last
        self._min = min_v
        self._max = max_v
        self._mode = mode_v
        self._stopped = True
        self._redraw()

    def reset(self) -> None:
        """Idle / new take: needle off, no summary ticks."""
        self._live = None
        self._min = self._max = self._mode = None
        self._stopped = False
        self._redraw()

    def _angle(self, value: float) -> float:
        """Map 0..max onto a semicircle: left = 0, right = full scale."""
        t = min(1.0, max(0.0, float(value) / self.max_deg))
        return math.pi - t * math.pi

    def _redraw(self) -> None:
        """Paint the dial. Cheap enough to run from the 30 ms UI tick."""
        self.delete("all")
        w = max(int(self.winfo_width()), 180)
        h = max(int(self.winfo_height()), 150)
        cx, cy = w / 2, h * 0.72
        radius = min(w * 0.42, h * 0.58)
        x0, y0 = cx - radius, cy - radius
        x1, y1 = cx + radius, cy + radius

        # Grey track (empty progress bar).
        self.create_arc(x0, y0, x1, y1, start=0, extent=180, style=tk.ARC, outline=ARC, width=16)

        fill_frac = 0.0
        if self._live is not None:
            fill_frac = min(1.0, max(0.0, float(self._live) / self.max_deg))
        if fill_frac > 0.005:
            # Tk: start 180 = left (0°). Negative extent sweeps clockwise toward 180°.
            # Fill only the rim track so 45/90/135 stay readable in the middle.
            extent = -fill_frac * 180.0
            rim = NEEDLE if not self._stopped else MUTED
            self.create_arc(
                x0, y0, x1, y1, start=180, extent=extent, style=tk.ARC, outline=rim, width=16
            )

        for i in range(0, 5):
            frac = i / 4
            ang = math.pi - frac * math.pi
            inner = radius - 10
            outer = radius + 2
            self.create_line(
                cx + inner * math.cos(ang),
                cy - inner * math.sin(ang),
                cx + outer * math.cos(ang),
                cy - outer * math.sin(ang),
                fill=MUTED,
                width=2,
            )
            label = int(round(frac * self.max_deg))
            self.create_text(
                cx + (radius - 38) * math.cos(ang),
                cy - (radius - 38) * math.sin(ang),
                text=str(label),
                fill=MUTED,
                font=("Segoe UI", 8),
            )

        self.create_text(cx, 14, text=self.title.upper(), fill=MUTED, font=("Segoe UI", 9, "bold"))

        if self._stopped:
            self._draw_mark(cx, cy, radius, self._min, MIN_C, 3)
            self._draw_mark(cx, cy, radius, self._max, MAX_C, 3)
            self._draw_mode_mark(cx, cy, radius, self._mode)

        if self._live is not None:
            ang = self._angle(self._live)
            color = MUTED if self._stopped else NEEDLE
            self.create_line(
                cx,
                cy,
                cx + (radius - 16) * math.cos(ang),
                cy - (radius - 16) * math.sin(ang),
                fill=color,
                width=4,
                capstyle=tk.ROUND,
            )
        self.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=TEXT, outline="")

        if self._stopped:
            caption = (
                f"min {self._fmt(self._min)}   max {self._fmt(self._max)}   "
                f"mode {self._fmt(self._mode)}"
            )
            self.create_text(cx, h - 14, text=caption, fill=MUTED, font=("Segoe UI", 8))
        elif self._live is None:
            self.create_text(cx, h - 14, text="—  deg", fill=MUTED, font=("Segoe UI", 9))
        else:
            self.create_text(
                cx,
                h - 14,
                text=f"{self._live:.0f}°",
                fill=TEXT,
                font=("Segoe UI", 10, "bold"),
            )

    def _draw_mark(self, cx: float, cy: float, radius: float, value: float | None, color: str, width: int) -> None:
        """Radial tick from hub to rim (min or max after Stop)."""
        if value is None:
            return
        ang = self._angle(value)
        self.create_line(
            cx + 12 * math.cos(ang),
            cy - 12 * math.sin(ang),
            cx + (radius - 6) * math.cos(ang),
            cy - (radius - 6) * math.sin(ang),
            fill=color,
            width=width,
        )

    def _draw_mode_mark(self, cx: float, cy: float, radius: float, value: float | None) -> None:
        """Triangle on the rim — different from the min/max lines."""
        if value is None:
            return
        ang = self._angle(value)
        rim_x = cx + radius * math.cos(ang)
        rim_y = cy - radius * math.sin(ang)
        tx = math.cos(ang)
        ty = -math.sin(ang)
        px, py = -ty, tx
        size = 8
        self.create_polygon(
            rim_x + tx * 4,
            rim_y + ty * 4,
            rim_x - tx * size + px * size,
            rim_y - ty * size + py * size,
            rim_x - tx * size - px * size,
            rim_y - ty * size - py * size,
            fill=MODE_C,
            outline=MODE_C,
        )

    @staticmethod
    def _fmt(value: float | None) -> str:
        """Integer degrees or dash."""
        if value is None:
            return "—"
        return f"{value:.0f}°"


class BodyGauges(tk.Frame):
    """One column: Left body or Right body, three speedometers."""

    def __init__(
        self,
        parent: tk.Widget,
        heading: str,
        joint_names: list[str],
        max_deg: float,
        accent: str,
    ) -> None:
        """heading is 'Left body' / 'Right body'. joint_names from YAML."""
        super().__init__(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        tk.Label(
            self,
            text=heading.upper(),
            bg=PANEL,
            fg=accent,
            font=("Segoe UI", 11, "bold"),
        ).pack(fill=tk.X, pady=(10, 4))
        self.gauges: dict[str, Speedometer] = {}
        for name in joint_names:
            title = JOINT_LABELS.get(name, name.replace("_", " "))
            gauge = Speedometer(self, title, max_deg)
            gauge.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
            self.gauges[name] = gauge

    def set_live(self, angles: dict[str, float | None]) -> None:
        """Update needles for this side."""
        for name, gauge in self.gauges.items():
            gauge.set_live(angles.get(name))

    def set_stopped(self, last: dict[str, float | None], stats: dict[str, dict[str, float | None]]) -> None:
        """Draw min/max/mode on each dial."""
        for name, gauge in self.gauges.items():
            row = stats.get(name) or {}
            gauge.set_stopped(last.get(name), row.get("min"), row.get("max"), row.get("mode"))

    def reset(self) -> None:
        """Clear needles before a new take."""
        for gauge in self.gauges.values():
            gauge.reset()


class MotionDashboard(tk.Frame):
    """Split-panel speedometers: left body | right body."""

    def __init__(self, parent: tk.Widget, analysis_cfg: dict) -> None:
        """Joints, full-scale and mode bin all come from config.yaml."""
        super().__init__(parent, bg=BG)
        gauges_cfg = analysis_cfg.get("gauge_joints") or {}
        left_names = [str(n) for n in gauges_cfg.get("left", [])]
        right_names = [str(n) for n in gauges_cfg.get("right", [])]
        max_deg = float(analysis_cfg.get("angle_gauge_max_deg", 180))
        bin_deg = float(analysis_cfg.get("angle_mode_bin_deg", 10))
        self.joint_names = left_names + right_names
        self.recorder = TakeAngleRecorder(self.joint_names, bin_deg)

        region = str(analysis_cfg.get("region", "full_body")).replace("_", " ")
        tk.Label(
            self,
            text=f"LIVE ANGLES  ·  {region}  ·  2D image degrees  ·  ~180° = straight",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 6))

        if not self.joint_names:
            tk.Label(
                self,
                text="This region tracks landmarks only.\nNo joint-angle dials (face has no shoulder/elbow/wrist angle).",
                bg=CARD,
                fg=TEXT,
                font=("Segoe UI", 11),
                justify=tk.LEFT,
                anchor="nw",
            ).pack(fill=tk.BOTH, expand=True, padx=8, pady=16)
            self.left = None
            self.right = None
            return

        cols = tk.Frame(self, bg=BG)
        cols.pack(fill=tk.BOTH, expand=True)
        cols.grid_columnconfigure(0, weight=1)
        cols.grid_columnconfigure(1, weight=1)
        cols.grid_rowconfigure(0, weight=1)
        self.left = BodyGauges(cols, "Left body", left_names, max_deg, "#c8e050")
        self.right = BodyGauges(cols, "Right body", right_names, max_deg, "#4ca6ff")
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.legend = tk.Label(
            self,
            text="Live needle  ·  after Stop:  cyan = min    coral = max    amber ▲ = mode",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.legend.pack(fill=tk.X, pady=(6, 0))

    def on_start(self) -> None:
        """New take: empty samples, needles at rest until the first frame."""
        self.recorder.reset()
        if self.left is not None:
            self.left.reset()
        if self.right is not None:
            self.right.reset()

    def on_frame(self, angles: dict[str, float | None]) -> None:
        """UI thread: record sample and move needles."""
        self.recorder.add(angles)
        if self.left is not None:
            self.left.set_live(angles)
        if self.right is not None:
            self.right.set_live(angles)

    def on_stop(self, last: dict[str, float | None]) -> None:
        """Keep last needle; add min / max / mode marks."""
        stats = self.recorder.summary()
        if self.left is not None:
            self.left.set_stopped(last, stats)
        if self.right is not None:
            self.right.set_stopped(last, stats)
