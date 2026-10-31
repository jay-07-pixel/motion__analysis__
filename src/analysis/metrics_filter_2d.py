"""Temporal filter for game 2D angles. Does not change how an angle is defined.

MediaPipe pixel jitter of a few degrees is expected on a still person. The
game and dials need a calmer number. This module sits AFTER pose_metrics_2d
and wrist_metrics_2d:

    raw angle (unchanged math) -> GameMetricsFilter -> WebSocket metrics

It follows the same pattern as VelocityTracker and CameraXyzSmoother: a
deadband for tiny steps, then EMA. It is NOT CameraXyzSmoother (that blends
3D metres) and is NOT applied in PatientBalanceAI.

Per metric, independently:

    |raw - filtered| < deadband_deg  -> hold the last filtered value
    |raw - last_raw| and |raw - filtered| both > jump_deg, first such frame
        -> treat as a one-frame glitch; do not step the output
    two large jumps in a row, or a moderate step -> EMA toward raw
        still_smooth when the step is small, move_smooth when it is large

still_smooth / move_smooth are the share of the new sample (1 = raw).
Torso lean is signed; nothing here clamps to 0..180.

Null is never turned into 0. After hold_frames missing samples the channel
resets and the next valid value starts the filter again, so a new person
does not inherit the previous one.
"""

from __future__ import annotations

from dataclasses import dataclass

# Game metrics that flicker on a still person. Knees are left raw (this game
# does not score them). Tracking status is never filtered.
FILTERED_METRIC_NAMES: tuple[str, ...] = (
    "right_arm_angle_deg",
    "left_arm_angle_deg",
    "right_elbow_angle_deg",
    "left_elbow_angle_deg",
    "right_wrist_angle_deg",
    "left_wrist_angle_deg",
    "torso_lean_deg",
)


@dataclass
class _Channel:
    filtered: float
    last_raw: float
    misses: int = 0
    glitch_streak: int = 0


class GameMetricsFilter:
    """One state machine per game metric. Create once per bridge process."""

    def __init__(
        self,
        still_smooth: float = 0.28,
        move_smooth: float = 0.70,
        deadband_deg: float = 1.5,
        move_deg: float = 6.0,
        jump_deg: float = 40.0,
        hold_frames: int = 4,
    ) -> None:
        """Defaults are for ~15–30 fps 2D angles.

        Args:
            still_smooth: Share of the new sample while nearly still (0.28).
            move_smooth: Share of the new sample during a real swing (0.70).
            deadband_deg: Hold the output when |raw-filtered| is below this.
            move_deg: Switch to move_smooth at or above this step.
            jump_deg: One-frame change larger than this is a glitch unless
                the next frame agrees. 40° is above a fast raise (~15–25°/frame)
                and below a typical MediaPipe collapse (100°+).
            hold_frames: Missing samples before the output becomes null.
                4 frames is about 0.2 s at 20 fps.
        """
        self.still_smooth = min(1.0, max(0.0, float(still_smooth)))
        self.move_smooth = min(1.0, max(0.0, float(move_smooth)))
        self.deadband_deg = max(0.0, float(deadband_deg))
        self.move_deg = max(self.deadband_deg, float(move_deg))
        self.jump_deg = max(self.move_deg, float(jump_deg))
        self.hold_frames = max(0, int(hold_frames))
        self._channels: dict[str, _Channel] = {}

    def reset(self) -> None:
        """Drop every channel. Next valid sample is used as-is."""
        self._channels.clear()

    def apply(self, raw_metrics: dict[str, float | None]) -> dict[str, float | None]:
        """Return a copy: listed metrics filtered, everything else unchanged."""
        out = dict(raw_metrics)
        for name in FILTERED_METRIC_NAMES:
            if name not in raw_metrics:
                continue
            out[name] = self._step(name, raw_metrics[name])
        return out

    def _step(self, name: str, raw: float | None) -> float | None:
        if raw is None or not _finite(raw):
            return self._missing(name)
        channel = self._channels.get(name)
        if channel is None:
            self._channels[name] = _Channel(filtered=float(raw), last_raw=float(raw))
            return float(raw)
        channel.misses = 0
        value = float(raw)
        step = abs(value - channel.filtered)
        raw_jump = abs(value - channel.last_raw)
        if raw_jump > self.jump_deg and step > self.jump_deg:
            channel.glitch_streak += 1
            channel.last_raw = value
            if channel.glitch_streak < 2:
                return channel.filtered
        else:
            channel.glitch_streak = 0
        channel.last_raw = value
        if step < self.deadband_deg:
            return channel.filtered
        alpha = self.move_smooth if step >= self.move_deg else self.still_smooth
        channel.filtered = alpha * value + (1.0 - alpha) * channel.filtered
        return channel.filtered

    def _missing(self, name: str) -> float | None:
        channel = self._channels.get(name)
        if channel is None:
            return None
        channel.misses += 1
        if channel.misses > self.hold_frames:
            self._channels.pop(name, None)
            return None
        return channel.filtered


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def filter_from_config(analysis_cfg: dict | None) -> GameMetricsFilter:
    """Read analysis.game_metrics_filter if present; otherwise the defaults."""
    cfg = (analysis_cfg or {}).get("game_metrics_filter") or {}
    return GameMetricsFilter(
        still_smooth=float(cfg.get("still_smooth", 0.28)),
        move_smooth=float(cfg.get("move_smooth", 0.70)),
        deadband_deg=float(cfg.get("deadband_deg", 1.5)),
        move_deg=float(cfg.get("move_deg", 6.0)),
        jump_deg=float(cfg.get("jump_deg", 40.0)),
        hold_frames=int(cfg.get("hold_frames", 4)),
    )
