"""Collect 2D elbow samples during a take, then show a report after Stop.

The main window stays video-only while you move. Charts appear only
when the take ends. Still image-plane degrees, not metres.
"""

from __future__ import annotations

import math
import statistics
import tkinter as tk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

BG = "#171e28"
AX = "#101418"
TEXT = "#c5cdd8"
LEFT = "#4c9aff"
RIGHT = "#3dd68c"
GRID = "#2a3545"
INK = "#eef3f8"

BIN_LABELS = ("0-30", "30-60", "60-90", "90-120", "120-150", "150-180")


class TakeRecorder:
    """Stores this take's elbow samples. No drawing until Stop."""

    def __init__(self) -> None:
        """Empty take. Call reset() on Start, add() each new frame."""
        self.times: list[float] = []
        self.left: list[float] = []
        self.right: list[float] = []

    def reset(self) -> None:
        """Forget the previous take."""
        self.times.clear()
        self.left.clear()
        self.right.clear()

    def add(self, time_sec: float, left_deg: float | None, right_deg: float | None) -> None:
        """Keep one frame. Missing side is NaN so the line has a gap, not a fake 0."""
        self.times.append(float(time_sec))
        self.left.append(float("nan") if left_deg is None else float(left_deg))
        self.right.append(float("nan") if right_deg is None else float(right_deg))

    def has_data(self) -> bool:
        """True if at least one elbow value was trusted this take."""
        return any(not math.isnan(v) for v in self.left + self.right)


def show_analysis_report(parent: tk.Tk, recorder: TakeRecorder) -> None:
    """Open a separate window with this take's line + bent/straight bars."""
    if not recorder.has_data():
        return

    win = tk.Toplevel(parent)
    win.title("2D analysis report — this take")
    win.configure(bg=BG)
    win.geometry("1100x640")
    win.minsize(900, 520)

    summary = _summary_text(recorder)
    tk.Label(
        win,
        text="2D analysis report  ·  this take only  ·  image degrees, not metres",
        bg=BG,
        fg=INK,
        font=("Segoe UI", 14, "bold"),
        anchor="w",
    ).pack(fill=tk.X, padx=16, pady=(14, 4))
    tk.Label(
        win,
        text=summary,
        bg=BG,
        fg=TEXT,
        font=("Segoe UI", 10),
        justify=tk.LEFT,
        anchor="w",
    ).pack(fill=tk.X, padx=16, pady=(0, 8))

    fig = Figure(figsize=(11, 4.6), dpi=100, facecolor=BG)
    ax_line = fig.add_subplot(1, 2, 1)
    ax_bar = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.14, wspace=0.28)
    _style_axes(ax_line)
    _style_axes(ax_bar)

    ax_line.set_title("Elbow angle over time (2D deg)", color=TEXT, fontsize=11, loc="left")
    ax_line.set_xlabel("seconds", color=TEXT, fontsize=9)
    ax_line.set_ylim(0, 200)
    ax_line.axhline(180, color="#5a6678", linestyle="--", linewidth=0.8)
    ax_line.plot(recorder.times, recorder.left, color=LEFT, linewidth=1.6, label="Left")
    ax_line.plot(recorder.times, recorder.right, color=RIGHT, linewidth=1.6, label="Right")
    if recorder.times:
        ax_line.set_xlim(0, max(recorder.times[-1], 1.0))
    ax_line.legend(loc="upper right", fontsize=8, facecolor=AX, edgecolor=GRID, labelcolor=TEXT)

    left_pct = _to_percent(_bin_counts(recorder.left))
    right_pct = _to_percent(_bin_counts(recorder.right))
    xs = list(range(6))
    ax_bar.set_title("How often bent vs straight this take (%)", color=TEXT, fontsize=11, loc="left")
    ax_bar.bar([i - 0.18 for i in xs], left_pct, 0.36, color=LEFT, label="Left")
    ax_bar.bar([i + 0.18 for i in xs], right_pct, 0.36, color=RIGHT, label="Right")
    ax_bar.set_xticks(xs, BIN_LABELS, fontsize=8)
    ax_bar.set_ylim(0, 100)
    ax_bar.legend(loc="upper right", fontsize=8, facecolor=AX, edgecolor=GRID, labelcolor=TEXT)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
    canvas.draw()


def _style_axes(ax) -> None:
    """Dark axes to match the main window."""
    ax.set_facecolor(AX)
    ax.tick_params(colors=TEXT, labelsize=8)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _finite(values: list[float]) -> list[float]:
    """Drop NaNs so min/max/mean are honest."""
    return [v for v in values if not math.isnan(v)]


def _summary_text(recorder: TakeRecorder) -> str:
    """One line per elbow: n, min, max, mean."""
    bits = [
        "~180° = straight in the photo. Near 0° can be 2D collapse (not a medical number)."
    ]
    for name, series in (("Left elbow", recorder.left), ("Right elbow", recorder.right)):
        vals = _finite(series)
        if not vals:
            bits.append(f"{name}: no trusted samples")
            continue
        bits.append(
            f"{name}: n={len(vals)}   min {min(vals):.0f}°   max {max(vals):.0f}°   "
            f"mean {statistics.fmean(vals):.0f}°"
        )
    return "\n".join(bits)


def _bin_index(degrees: float) -> int:
    """Map 0..180 into one of six 30-degree bins."""
    if degrees >= 180:
        return 5
    if degrees < 0:
        return 0
    return min(5, int(degrees // 30))


def _bin_counts(values: list[float]) -> list[int]:
    """How many samples fell in each 30° bin."""
    counts = [0] * 6
    for value in _finite(values):
        counts[_bin_index(value)] += 1
    return counts


def _to_percent(counts: list[int]) -> list[float]:
    """Bin counts -> % of that elbow's samples this take."""
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [100.0 * n / total for n in counts]
