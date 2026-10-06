"""Live Left/Right body analog dials (2D joint angles).

While the camera runs the needle is live image-plane degrees.
After Stop the same dial shows min (cyan), max (coral) and mode (amber).
~180° ≈ straight in the photo. Not metres, not px/s.
"""

from __future__ import annotations

import math
import statistics
import tkinter as tk

from src.ui.theme import PAL

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
        """min, median, mode (bin start), max per joint. None if no samples."""
        out: dict[str, dict[str, float | None]] = {}
        for name in self.joint_names:
            vals = self.samples[name]
            if not vals:
                out[name] = {"min": None, "median": None, "mode": None, "max": None}
                continue
            out[name] = {
                "min": min(vals),
                "median": statistics.median(vals),
                "mode": _binned_mode(vals, self.bin_deg),
                "max": max(vals),
            }
        return out


def _binned_mode(values: list[float], bin_width: float) -> float:
    """Most common angle bin (degrees). Ties go to the lower bin."""
    if not values:
        return 0.0
    keys = [int(v // bin_width) for v in values]
    winner = statistics.multimode(keys)[0]
    return winner * bin_width


class Speedometer(tk.Frame):
    """One analog dial: live needle, optional min/max/mode ticks after Stop.

    The joint name sits in the open bowl. Scale numbers sit outside the rim.
    The live degree stays under the hub.
    """

    def __init__(self, parent: tk.Widget, title: str, max_deg: float, step_deg: float = 45) -> None:
        """Build an empty dial. Call set_live() from the UI thread."""
        super().__init__(parent, bg=PAL.card)
        self.title = title
        self.max_deg = max(1.0, float(max_deg))
        self.step_deg = max(1.0, float(step_deg))
        self._live: float | None = None
        self._rate: float | None = None
        self._min: float | None = None
        self._max: float | None = None
        self._mode: float | None = None
        self._stopped = False

        self.canvas = tk.Canvas(self, bg=PAL.card, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))
        self._value = tk.Label(
            self,
            text="—",
            bg=PAL.card,
            fg=PAL.muted,
            font=("Segoe UI", 13, "bold"),
        )
        self._value.pack(side=tk.TOP, fill=tk.X, pady=(0, 0))
        self._stats = tk.Label(
            self,
            text=" ",
            bg=PAL.card,
            fg=PAL.muted,
            font=("Segoe UI", 8),
            justify=tk.CENTER,
        )
        self._stats.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        self.canvas.bind("<Configure>", self._on_configure)
        self.bind("<Configure>", self._on_configure)

    def set_live(self, value: float | None, rate: float | None = None) -> None:
        """Move the needle. rate is this joint's angular speed in degrees per second."""
        self._live = value
        self._rate = rate
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
        self._rate = None
        self._min = min_v
        self._max = max_v
        self._mode = mode_v
        self._stopped = True
        self._redraw()

    def reset(self) -> None:
        """Idle / new take: needle off, no summary ticks."""
        self._live = None
        self._rate = None
        self._min = self._max = self._mode = None
        self._stopped = False
        self._redraw()

    def set_step(self, step_deg: float) -> None:
        """Change the printed scale (45 gives 0, 45, 90… and 20 gives 0, 20, 40…)."""
        self.step_deg = max(1.0, float(step_deg))
        self._redraw()

    def _scale_marks(self) -> list[float]:
        """Degree labels from 0 to full scale, spaced by step_deg."""
        marks = [0.0]
        value = self.step_deg
        while value < self.max_deg - 0.5:
            marks.append(value)
            value += self.step_deg
        if marks[-1] != self.max_deg:
            marks.append(self.max_deg)
        return marks

    def _angle(self, value: float) -> float:
        """Map 0..max onto a semicircle: left = 0, right = full scale."""
        t = min(1.0, max(0.0, float(value) / self.max_deg))
        return math.pi - t * math.pi

    def _on_configure(self, _event=None) -> None:
        """Refit the arc when the panel is resized, and wrap the caption."""
        width = max(int(self.winfo_width()) - 16, 80)
        self._stats.configure(wraplength=width)
        self._redraw()

    def _redraw(self) -> None:
        """Paint the dial. Name inside the bowl, scale numbers outside the rim."""
        canvas = self.canvas
        canvas.delete("all")
        w = max(int(canvas.winfo_width()), 40)
        h = max(int(canvas.winfo_height()), 40)
        # Room outside the rim for the scale numbers, and a little under the hub.
        text_pad = 26
        hub_margin = 12
        stroke = 11
        radius = min((w / 2) - text_pad, h - text_pad - hub_margin)
        radius = max(radius, 28)
        cx = w / 2
        cy = h - hub_margin
        x0, y0 = cx - radius, cy - radius
        x1, y1 = cx + radius, cy + radius

        canvas.create_arc(
            x0, y0, x1, y1, start=0, extent=180, style=tk.ARC, outline=PAL.arc, width=stroke
        )

        fill_frac = 0.0
        if self._live is not None:
            fill_frac = min(1.0, max(0.0, float(self._live) / self.max_deg))
        if fill_frac > 0.005:
            # Tk: start 180 = left (0°). Negative extent sweeps clockwise toward 180°.
            extent = -fill_frac * 180.0
            rim = PAL.needle if not self._stopped else PAL.muted
            canvas.create_arc(
                x0, y0, x1, y1, start=180, extent=extent, style=tk.ARC, outline=rim, width=stroke
            )

        # Ticks on the rim. Numbers sit just outside so they do not crowd the name.
        marks = self._scale_marks()
        label_r = radius + stroke / 2 + 11
        font_size = 8 if len(marks) <= 6 else 7
        for index, value in enumerate(marks):
            frac = value / self.max_deg
            ang = math.pi - frac * math.pi
            inner = radius - stroke / 2 - 1
            outer = radius + 2
            canvas.create_line(
                cx + inner * math.cos(ang),
                cy - inner * math.sin(ang),
                cx + outer * math.cos(ang),
                cy - outer * math.sin(ang),
                fill=PAL.muted,
                width=2,
            )
            label = int(round(value))
            lx = cx + label_r * math.cos(ang)
            ly = cy - label_r * math.sin(ang)
            if index == 0 or index == len(marks) - 1:
                ly -= 4
            canvas.create_text(lx, ly, text=str(label), fill=PAL.muted, font=("Segoe UI", font_size))

        name_size = 10 if radius >= 64 else 8
        canvas.create_text(
            cx,
            cy - radius * 0.46,
            text=self.title.upper(),
            fill=PAL.muted,
            font=("Segoe UI", name_size, "bold"),
        )

        if self._stopped:
            self._draw_mark(cx, cy, radius, self._min, PAL.min_c, 3)
            self._draw_mark(cx, cy, radius, self._max, PAL.max_c, 3)
            self._draw_mode_mark(cx, cy, radius, self._mode)

        if self._live is not None:
            ang = self._angle(self._live)
            color = PAL.muted if self._stopped else PAL.needle
            canvas.create_line(
                cx,
                cy,
                cx + (radius - stroke - 6) * math.cos(ang),
                cy - (radius - stroke - 6) * math.sin(ang),
                fill=color,
                width=3,
                capstyle=tk.ROUND,
            )
        canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=PAL.text, outline="")
        self._paint_caption()

    def _paint_caption(self) -> None:
        """Degree under the dial; min / max / mode on the line below after Stop."""
        if self._stopped:
            self._value.configure(text=self._fmt(self._live), fg=PAL.text)
            self._stats.configure(
                text=(
                    f"min {self._fmt(self._min)}    max {self._fmt(self._max)}    "
                    f"mode {self._fmt(self._mode)}"
                )
            )
        elif self._live is None:
            self._value.configure(text="—", fg=PAL.muted)
            self._stats.configure(text=" ")
        else:
            self._value.configure(text=f"{self._live:.0f}°", fg=PAL.text)
            if self._rate is None:
                self._stats.configure(text=" ")
            else:
                self._stats.configure(text=f"{self._rate:.0f} °/s")

    def _draw_mark(self, cx: float, cy: float, radius: float, value: float | None, color: str, width: int) -> None:
        """Radial tick from hub to rim (min or max after Stop)."""
        if value is None:
            return
        ang = self._angle(value)
        self.canvas.create_line(
            cx + 12 * math.cos(ang),
            cy - 12 * math.sin(ang),
            cx + (radius - 8) * math.cos(ang),
            cy - (radius - 8) * math.sin(ang),
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
        self.canvas.create_polygon(
            rim_x + tx * 4,
            rim_y + ty * 4,
            rim_x - tx * size + px * size,
            rim_y - ty * size + py * size,
            rim_x - tx * size - px * size,
            rim_y - ty * size - py * size,
            fill=PAL.mode_c,
            outline=PAL.mode_c,
        )

    @staticmethod
    def _fmt(value: float | None) -> str:
        """Integer degrees or dash."""
        if value is None:
            return "—"
        return f"{value:.0f}°"


class BodyGauges(tk.Frame):
    """One column: Left or Right, three speedometers."""

    def __init__(
        self,
        parent: tk.Widget,
        heading: str,
        joint_names: list[str],
        max_deg: float,
        accent: str,
        step_deg: float = 45,
    ) -> None:
        """heading is 'Left' / 'Right'. joint_names from YAML."""
        super().__init__(parent, bg=PAL.panel, highlightbackground=PAL.line, highlightthickness=1)
        tk.Label(
            self,
            text=heading.upper(),
            bg=PAL.panel,
            fg=accent,
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="ew", pady=(10, 4))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.gauges: dict[str, Speedometer] = {}
        for index, name in enumerate(joint_names):
            title = JOINT_LABELS.get(name, name.replace("_", " "))
            gauge = Speedometer(self, title, max_deg, step_deg)
            gauge.grid(row=index + 1, column=0, sticky="nsew", padx=8, pady=(2, 6))
            self.rowconfigure(index + 1, weight=1, uniform="dial")
            self.gauges[name] = gauge

    def set_step(self, step_deg: float) -> None:
        """Reprint every dial on this side with the chosen scale."""
        for gauge in self.gauges.values():
            gauge.set_step(step_deg)

    def set_live(self, angles: dict[str, float | None], angular: dict[str, float | None] | None = None) -> None:
        """Update needles for this side. angular is degrees per second."""
        rates = angular or {}
        for name, gauge in self.gauges.items():
            gauge.set_live(angles.get(name), rates.get(name))

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
    """Split-panel speedometers: left | right."""

    def __init__(self, parent: tk.Widget, analysis_cfg: dict, on_step=None) -> None:
        """Joints, full-scale and mode bin all come from config.yaml."""
        super().__init__(parent, bg=PAL.bg)
        self._on_step = on_step
        gauges_cfg = analysis_cfg.get("gauge_joints") or {}
        left_names = [str(n) for n in gauges_cfg.get("left", [])]
        right_names = [str(n) for n in gauges_cfg.get("right", [])]
        max_deg = float(analysis_cfg.get("angle_gauge_max_deg", 180))
        self._step = float(analysis_cfg.get("angle_gauge_step_deg", 45))
        self._steps = [float(s) for s in (analysis_cfg.get("angle_gauge_steps") or [45, 20])]
        bin_deg = float(analysis_cfg.get("angle_mode_bin_deg", 10))
        self.joint_names = left_names + right_names
        self.recorder = TakeAngleRecorder(self.joint_names, bin_deg)
        vel_cfg = analysis_cfg.get("velocity") or {}
        self.linear_names = [str(name) for name in vel_cfg.get("linear_joints") or []]
        self.space = str(analysis_cfg.get("space", "2d")).lower()
        self.angular_recorder = TakeAngleRecorder(
            self.joint_names, float(vel_cfg.get("angular_mode_bin", 10))
        )
        self.linear_recorder = TakeAngleRecorder(
            self.linear_names, float(vel_cfg.get("linear_mode_bin", 0.05))
        )

        region = str(analysis_cfg.get("region", "full_body")).replace("_", " ")
        space = self.space
        if space == "3d":
            angle_note = "3D degrees  ·  angle between the body axes"
        else:
            angle_note = "2D image degrees  ·  ~180° = straight"
        tk.Label(
            self,
            text=f"LIVE ANGLES  ·  {region}  ·  {angle_note}",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

        if not self.joint_names:
            tk.Label(
                self,
                text="This region tracks landmarks only.\nNo joint-angle dials (face has no shoulder/elbow/wrist angle).",
                bg=PAL.card,
                fg=PAL.text,
                font=("Segoe UI", 11),
                justify=tk.LEFT,
                anchor="nw",
            ).pack(fill=tk.BOTH, expand=True, padx=8, pady=16)
            self.left = None
            self.right = None
            return

        self._build_scale_choice()
        self._build_mark_key()
        cols = tk.Frame(self, bg=PAL.bg)
        cols.pack(fill=tk.BOTH, expand=True)
        cols.grid_columnconfigure(0, weight=1)
        cols.grid_columnconfigure(1, weight=1)
        cols.grid_rowconfigure(0, weight=1)
        self.left = BodyGauges(cols, "Left", left_names, max_deg, "#c8e050", self._step)
        self.right = BodyGauges(cols, "Right", right_names, max_deg, "#4ca6ff", self._step)
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.linear_label = tk.Label(
            self,
            text=self._linear_text({}),
            bg=PAL.bg,
            fg=PAL.text,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.linear_label.pack(fill=tk.X, pady=(8, 0))
        self.legend = tk.Label(
            self,
            text="Under each dial: angular speed (°/s). Marks appear after Stop.",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.legend.pack(fill=tk.X, pady=(6, 0))

    def _build_scale_choice(self) -> None:
        """Two scale buttons: 0, 45, 90… or 0, 20, 40…"""
        row = tk.Frame(self, bg=PAL.bg)
        row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(
            row,
            text="SCALE",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 8, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._step_btns: dict[float, tk.Button] = {}
        for step in self._steps:
            label = self._step_label(step)
            btn = tk.Button(
                row,
                text=label,
                command=lambda s=step: self._choose_step(s),
                bd=0,
                padx=8,
                pady=3,
                font=("Segoe UI", 8, "bold"),
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=(0, 6))
            self._step_btns[float(step)] = btn
        self._paint_step_buttons()

    def _step_label(self, step: float) -> str:
        """Short preview of the first marks, e.g. '0  45  90'."""
        marks = [0]
        value = int(round(step))
        while value < 180 and len(marks) < 3:
            marks.append(value)
            value += int(round(step))
        return "   ".join(str(mark) for mark in marks)

    def _choose_step(self, step: float) -> None:
        """Reprint the dials and remember the choice for the next rebuild."""
        self._step = float(step)
        if self.left is not None:
            self.left.set_step(self._step)
        if self.right is not None:
            self.right.set_step(self._step)
        self._paint_step_buttons()
        if self._on_step is not None:
            self._on_step(self._step)

    def _paint_step_buttons(self) -> None:
        """Highlight the scale that is on the dials."""
        for step, btn in self._step_btns.items():
            on = abs(step - self._step) < 0.1
            btn.configure(
                bg=PAL.accent if on else PAL.idle,
                fg="white" if on else PAL.text,
                activebackground=PAL.accent if on else PAL.line,
                activeforeground="white" if on else PAL.text,
            )

    def _build_mark_key(self) -> None:
        """Colour key for the marks drawn on the dials after Stop."""
        key = tk.Frame(self, bg=PAL.bg)
        key.pack(fill=tk.X, pady=(0, 6))
        items = (
            (PAL.min_c, "Min  ·  smallest angle"),
            (PAL.max_c, "Max  ·  largest angle"),
            (PAL.mode_c, "Mode ▲  ·  most common 10°"),
        )
        for color, text in items:
            tk.Label(key, text="    ", bg=color).pack(side=tk.LEFT, padx=(0, 6), pady=2)
            tk.Label(
                key,
                text=text,
                bg=PAL.bg,
                fg=PAL.text,
                font=("Segoe UI", 9),
            ).pack(side=tk.LEFT, padx=(0, 16))

    def _linear_text(self, speeds: dict[str, float | None]) -> str:
        """One line for wrist speed. 2D has no camera metres."""
        if self.space != "3d":
            return "Linear velocity  ·  needs 3D camera metres"
        parts = []
        for name in self.linear_names:
            label = "L wrist" if name.startswith("left") else "R wrist" if name.startswith("right") else name
            value = speeds.get(name)
            shown = "—" if value is None else f"{value:.2f} m/s"
            parts.append(f"{label}  {shown}")
        if not parts:
            return "Linear velocity  ·  no wrist configured"
        return "Linear velocity  ·  " + "    ".join(parts)

    def on_start(self) -> None:
        """New take: empty samples, needles at rest until the first frame."""
        self.recorder.reset()
        self.angular_recorder.reset()
        self.linear_recorder.reset()
        if self.left is not None:
            self.left.reset()
        if self.right is not None:
            self.right.reset()
        if getattr(self, "linear_label", None) is not None:
            self.linear_label.configure(text=self._linear_text({}))

    def on_frame(
        self,
        angles: dict[str, float | None],
        angular: dict[str, float | None] | None = None,
        linear: dict[str, float | None] | None = None,
    ) -> None:
        """UI thread: record samples, move needles, show speeds."""
        self.recorder.add(angles)
        if angular is not None:
            self.angular_recorder.add(angular)
        if linear is not None:
            self.linear_recorder.add(linear)
        if getattr(self, "linear_label", None) is not None:
            self.linear_label.configure(text=self._linear_text(linear or {}))
        if self.left is not None:
            self.left.set_live(angles, angular)
        if self.right is not None:
            self.right.set_live(angles, angular)

    def on_stop(self, last: dict[str, float | None]) -> None:
        """Keep last needle; add min / max / mode marks."""
        stats = self.recorder.summary()
        if self.left is not None:
            self.left.set_stopped(last, stats)
        if self.right is not None:
            self.right.set_stopped(last, stats)

    def capture_summary(self) -> tuple[bool, dict[str, float | None], dict[str, list[float]]] | None:
        """Needle, stopped marks, and samples so a theme change can rebuild the dials."""
        gauges: dict[str, Speedometer] = {}
        if self.left is not None:
            gauges.update(self.left.gauges)
        if self.right is not None:
            gauges.update(self.right.gauges)
        if not gauges:
            return None
        stopped = any(gauge._stopped for gauge in gauges.values())
        last = {name: gauge._live for name, gauge in gauges.items()}
        samples = {name: list(values) for name, values in self.recorder.samples.items()}
        return stopped, last, samples

    def restore_summary(self, held: tuple[bool, dict[str, float | None], dict[str, list[float]]]) -> None:
        """Put a captured take back on the new dials."""
        stopped, last, samples = held
        for name, values in samples.items():
            if name in self.recorder.samples:
                self.recorder.samples[name] = list(values)
        if stopped:
            self.on_stop(last)
        elif self.left is not None and self.right is not None:
            self.left.set_live(last)
            self.right.set_live(last)
