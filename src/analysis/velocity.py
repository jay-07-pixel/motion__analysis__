"""Angular and linear speed from one frame to the next.

Angular speed is how fast a joint angle changes (degrees per second).
Linear speed is how fast a joint moves in camera metres (metres per second).

Both are magnitudes: direction is not stored. A still joint jitters, so a
step smaller than the deadband counts as zero, then the speed is blended
with the previous value so the number does not flicker.
"""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]


class VelocityTracker:
    """Per-joint speed. Create one tracker per take and call it every frame."""

    def __init__(
        self,
        smooth: float,
        deadband_deg: float,
        deadband_m: float,
        max_gap_sec: float,
    ) -> None:
        """Args come from analysis.velocity in config.yaml.

        Args:
            smooth: 0..1 share of the new speed (1 = raw, 0.3 = calmer).
            deadband_deg: Angle steps smaller than this count as 0 °/s.
            deadband_m: Position steps smaller than this count as 0 m/s.
            max_gap_sec: Ignore a step if the previous sample is older than this.
        """
        self.smooth = min(1.0, max(0.0, float(smooth)))
        self.deadband_deg = max(0.0, float(deadband_deg))
        self.deadband_m = max(0.0, float(deadband_m))
        self.max_gap_sec = max(1e-3, float(max_gap_sec))
        self._prev_angle: dict[str, tuple[float, float]] = {}
        self._prev_point: dict[str, tuple[float, Vec3]] = {}
        self._angular: dict[str, float] = {}
        self._linear: dict[str, float] = {}

    def angular(
        self,
        time_sec: float,
        angles_deg: dict[str, float | None],
    ) -> dict[str, float | None]:
        """Degrees per second for each joint. None until two close samples exist."""
        out: dict[str, float | None] = {}
        for name, value in angles_deg.items():
            if value is None:
                out[name] = None
                continue
            prev = self._prev_angle.get(name)
            self._prev_angle[name] = (time_sec, float(value))
            if prev is None:
                out[name] = None
                continue
            dt = time_sec - prev[0]
            if dt <= 1e-4 or dt > self.max_gap_sec:
                self._angular.pop(name, None)
                out[name] = None
                continue
            delta = abs(float(value) - prev[1])
            instant = 0.0 if delta < self.deadband_deg else delta / dt
            out[name] = self._blend(self._angular, name, instant)
        return out

    def linear(
        self,
        time_sec: float,
        points_m: dict[str, Vec3 | None],
    ) -> dict[str, float | None]:
        """Metres per second from camera X, Y, Z. None if depth or timing is missing."""
        out: dict[str, float | None] = {}
        for name, point in points_m.items():
            if point is None:
                out[name] = None
                continue
            prev = self._prev_point.get(name)
            current = (float(point[0]), float(point[1]), float(point[2]))
            self._prev_point[name] = (time_sec, current)
            if prev is None:
                out[name] = None
                continue
            dt = time_sec - prev[0]
            if dt <= 1e-4 or dt > self.max_gap_sec:
                self._linear.pop(name, None)
                out[name] = None
                continue
            dx = current[0] - prev[1][0]
            dy = current[1] - prev[1][1]
            dz = current[2] - prev[1][2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            instant = 0.0 if dist < self.deadband_m else dist / dt
            out[name] = self._blend(self._linear, name, instant)
        return out

    def _blend(self, store: dict[str, float], name: str, instant: float) -> float:
        """Blend this step with the previous speed. A held joint settles to zero."""
        previous = store.get(name)
        if previous is None:
            speed = instant
        else:
            speed = self.smooth * instant + (1.0 - self.smooth) * previous
        if instant == 0.0 and speed < 1e-3:
            speed = 0.0
        store[name] = speed
        return speed
