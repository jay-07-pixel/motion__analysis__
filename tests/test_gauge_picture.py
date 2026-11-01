"""Dial fallback when a joint is visible but the measured angle is missing."""

from __future__ import annotations

import unittest

from src.analysis.angles_2d import fill_missing_from_picture
from src.pose.keypoints import Keypoint2D


def _kp(name: str, u: float, v: float, conf: float = 0.9) -> Keypoint2D:
    return Keypoint2D(name=name, u_px=u, v_px=v, confidence=conf)


class PictureFillTest(unittest.TestCase):
    def test_keeps_a_measured_angle_and_fills_a_visible_gap(self) -> None:
        points = [
            _kp("left_shoulder", 100, 40),
            _kp("left_elbow", 100, 80),
            _kp("left_wrist", 140, 80),
        ]
        specs = [{"name": "left_elbow", "points": ["left_shoulder", "left_elbow", "left_wrist"]}]
        shown = fill_missing_from_picture(
            points,
            specs,
            {"left_elbow": None, "left_wrist": 90.0},
            0.5,
        )
        self.assertEqual(shown["left_wrist"], 90.0)
        self.assertAlmostEqual(shown["left_elbow"], 90.0, places=3)

    def test_stays_blank_when_a_dot_is_missing(self) -> None:
        points = [_kp("left_shoulder", 100, 40), _kp("left_elbow", 100, 80)]
        specs = [{"name": "left_elbow", "points": ["left_shoulder", "left_elbow", "left_wrist"]}]
        shown = fill_missing_from_picture(points, specs, {"left_elbow": None}, 0.5)
        self.assertIsNone(shown["left_elbow"])


if __name__ == "__main__":
    unittest.main()
