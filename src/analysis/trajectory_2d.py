"""2D joint trail and pixel speed (camera coordinates).

What:
    Keeps the last N positions of one joint (e.g. wrist) and the distance
    moved since the previous frame.

Why:
    A trail shows the path. Speed in px/s is honest 2D motion (not metres).

How:
    Store (u, v, time). Speed = pixel distance / dt.
"""

from __future__ import annotations

from collections import deque

from src.pose.keypoints import Keypoint2D


class JointTrail2D:
    """Rolling path of one named joint in camera pixels."""

    def __init__(self, joint_name: str, max_points: int) -> None:
        """Args:
            joint_name: MediaPipe name from config, e.g. 'left_wrist'.
            max_points: How many recent samples to keep (from config).
        """
        self.joint_name = joint_name
        self._points: deque[tuple[float, float, float]] = deque(maxlen=max_points)
        self.last_speed_px_s: float | None = None

    def update(self, keypoints: list[Keypoint2D], time_sec: float, min_confidence: float) -> None:
        """Append this frame's joint if it is visible enough. Updates speed."""
        match = next((kp for kp in keypoints if kp.name == self.joint_name), None)
        if match is None or match.confidence < min_confidence:
            return
        if self._points:
            prev_u, prev_v, prev_t = self._points[-1]
            dt = time_sec - prev_t
            if dt > 1e-6:
                dist = ((match.u_px - prev_u) ** 2 + (match.v_px - prev_v) ** 2) ** 0.5
                self.last_speed_px_s = dist / dt
        self._points.append((match.u_px, match.v_px, time_sec))

    def polyline(self) -> list[tuple[int, int]]:
        """Integer pixel points for cv2.polylines (oldest first)."""
        return [(int(round(u)), int(round(v))) for u, v, _ in self._points]
