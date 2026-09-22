"""2D joint trail and pixel speed (camera coordinates).

What:
    Keeps the last N positions of one joint (e.g. wrist) and a speed in px/s.

Why:
    A trail shows the path. Speed is 2D pixels per second, not metres.

How:
    Speed = pixel distance / dt, after ignoring MediaPipe jitter.
    A still wrist still wiggles 1-4 pixels every frame. At 30 fps that
    fake wiggle looks like 30-120 px/s. We treat small moves as 0.
"""

from __future__ import annotations

from collections import deque

from src.pose.keypoints import Keypoint2D


class JointTrail2D:
    """Rolling path of one named joint in camera pixels."""

    def __init__(
        self,
        joint_name: str,
        max_points: int,
        deadband_px: float = 4.0,
        smooth: float = 0.35,
    ) -> None:
        """Args:
            joint_name: MediaPipe name from config, e.g. 'left_wrist'.
            max_points: How many recent samples to keep (from config).
            deadband_px: Moves smaller than this are treated as still (jitter).
            smooth: 0..1. Blend of new speed vs previous (1 = jumpy, 0.3 = calm).
        """
        self.joint_name = joint_name
        self.deadband_px = float(deadband_px)
        self.smooth = min(1.0, max(0.0, float(smooth)))
        self._points: deque[tuple[float, float, float]] = deque(maxlen=max_points)
        self._last: tuple[float, float, float] | None = None
        self.last_speed_px_s: float | None = None

    def update(self, keypoints: list[Keypoint2D], time_sec: float, min_confidence: float) -> None:
        """Append this frame's joint if it is visible enough. Updates speed."""
        match = next((kp for kp in keypoints if kp.name == self.joint_name), None)
        if match is None or match.confidence < min_confidence:
            return

        if self._last is not None:
            prev_u, prev_v, prev_t = self._last
            dt = time_sec - prev_t
            if dt > 1e-6:
                dist = ((match.u_px - prev_u) ** 2 + (match.v_px - prev_v) ** 2) ** 0.5
                instant = 0.0 if dist < self.deadband_px else dist / dt
                if self.last_speed_px_s is None:
                    self.last_speed_px_s = instant
                else:
                    alpha = self.smooth
                    self.last_speed_px_s = alpha * instant + (1.0 - alpha) * self.last_speed_px_s
                # Snap leftover jitter to zero so a still wrist reads 0.
                if self.last_speed_px_s < 8.0 and instant == 0.0:
                    self.last_speed_px_s = 0.0

        self._last = (match.u_px, match.v_px, time_sec)

        # Do not grow the drawn trail on jitter (that looks like motion too).
        if not self._points:
            self._points.append((match.u_px, match.v_px, time_sec))
            return
        last_u, last_v, _ = self._points[-1]
        step = ((match.u_px - last_u) ** 2 + (match.v_px - last_v) ** 2) ** 0.5
        if step >= self.deadband_px:
            self._points.append((match.u_px, match.v_px, time_sec))

    def polyline(self) -> list[tuple[int, int]]:
        """Integer pixel points for cv2.polylines (oldest first)."""
        return [(int(round(u)), int(round(v))) for u, v, _ in self._points]


class JointSpeed2D:
    """Pixel speed of one named joint. No trail — used by the live gauges."""

    def __init__(self, joint_name: str, deadband_px: float, smooth: float) -> None:
        """Same jitter rules as the wrist trail so still joints read 0 px/s."""
        self.joint_name = joint_name
        self.deadband_px = float(deadband_px)
        self.smooth = min(1.0, max(0.0, float(smooth)))
        self._last: tuple[float, float, float] | None = None
        self.last_speed_px_s: float | None = None

    def update(self, keypoints: list[Keypoint2D], time_sec: float, min_confidence: float) -> None:
        """Refresh speed from this frame's joint, if it is visible enough."""
        match = next((kp for kp in keypoints if kp.name == self.joint_name), None)
        if match is None or match.confidence < min_confidence:
            return
        if self._last is not None:
            prev_u, prev_v, prev_t = self._last
            dt = time_sec - prev_t
            if dt > 1e-6:
                dist = ((match.u_px - prev_u) ** 2 + (match.v_px - prev_v) ** 2) ** 0.5
                instant = 0.0 if dist < self.deadband_px else dist / dt
                if self.last_speed_px_s is None:
                    self.last_speed_px_s = instant
                else:
                    alpha = self.smooth
                    self.last_speed_px_s = alpha * instant + (1.0 - alpha) * self.last_speed_px_s
                if self.last_speed_px_s < 8.0 and instant == 0.0:
                    self.last_speed_px_s = 0.0
        self._last = (match.u_px, match.v_px, time_sec)


def trail_from_config(analysis_cfg: dict) -> JointTrail2D:
    """Build the configured wrist trail, including jitter settings."""
    return JointTrail2D(
        joint_name=str(analysis_cfg["trail_joint"]),
        max_points=int(analysis_cfg["trail_max_points"]),
        deadband_px=float(analysis_cfg.get("speed_deadband_px", 4.0)),
        smooth=float(analysis_cfg.get("speed_smooth", 0.35)),
    )


def speeds_from_config(analysis_cfg: dict) -> dict[str, JointSpeed2D]:
    """One speed tracker per gauge joint listed in YAML."""
    deadband = float(analysis_cfg.get("speed_deadband_px", 4.0))
    smooth = float(analysis_cfg.get("speed_smooth", 0.35))
    names: list[str] = []
    gauges = analysis_cfg.get("gauge_joints") or {}
    for side in ("left", "right"):
        names.extend(str(n) for n in gauges.get(side, []))
    return {name: JointSpeed2D(name, deadband, smooth) for name in names}
