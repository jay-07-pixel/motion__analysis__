"""Foot-safety preview guide. Visual only; does not invent XYZ."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from src.pose.keypoints import Keypoint2D
from src.ui.foot_safety import (
    LINE_LABEL,
    WARNING_TEXT,
    draw_foot_safety_guide,
    feet_too_low,
)


def kp(name: str, u: float, v: float, conf: float = 0.9) -> Keypoint2D:
    return Keypoint2D(name=name, u_px=u, v_px=v, confidence=conf)


class FootSafetyLogicTest(unittest.TestCase):
    def test_below_guide_warns(self) -> None:
        points = [kp("left_ankle", 640, 650)]
        self.assertTrue(feet_too_low(points, 0.5, 640, 1280, 720))

    def test_on_or_above_guide_does_not_warn(self) -> None:
        points = [kp("left_ankle", 640, 640), kp("right_heel", 700, 500)]
        self.assertFalse(feet_too_low(points, 0.5, 640, 1280, 720))

    def test_outside_image_warns(self) -> None:
        points = [kp("right_foot_index", 640, 730)]
        self.assertTrue(feet_too_low(points, 0.5, 640, 1280, 720))

    def test_low_confidence_is_unknown_not_too_low(self) -> None:
        points = [kp("left_heel", 640, 700, conf=0.2)]
        self.assertFalse(feet_too_low(points, 0.5, 640, 1280, 720))

    def test_missing_foot_landmarks_is_unknown_not_safe_or_low(self) -> None:
        points = [kp("left_shoulder", 100, 80)]
        self.assertFalse(feet_too_low(points, 0.5, 640, 1280, 720))

    def test_guide_row_is_red_on_native_1280x720(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        before = frame.copy()
        out = draw_foot_safety_guide(frame, [], 0.5, 640)
        self.assertIs(out, frame)
        self.assertEqual(out.shape, (720, 1280, 3))
        row = out[640]
        red = (row[:, 2] >= 200) & (row[:, 1] <= 40) & (row[:, 0] <= 40)
        self.assertGreater(int(red.sum()), 40)
        self.assertEqual(int(out[100].sum()), 0)
        self.assertEqual(int(before[640].sum()), 0)
        self.assertEqual(LINE_LABEL, "FOOT SAFETY LINE")
        self.assertIn("FEET TOO LOW", WARNING_TEXT)

    def test_warning_does_not_change_keypoint_coordinates(self) -> None:
        point = kp("left_ankle", 100.25, 700.5)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_foot_safety_guide(frame, [point], 0.5, 640)
        self.assertEqual(point.u_px, 100.25)
        self.assertEqual(point.v_px, 700.5)
        self.assertIsNone(point.x_m)

    def test_session_draws_guide_on_preview_not_saved_video(self) -> None:
        from src.ui.session import MotionSession2D

        source = inspect.getsource(MotionSession2D.process_frame)
        self.assertIn("draw_foot_safety_guide(", source)
        after_canvas = source.split("canvas = draw_live_landmarks(")[1]
        self.assertIn("draw_foot_safety_guide(", after_canvas)
        self.assertNotIn("draw_foot_safety_guide(save_canvas", source)
        self.assertIn("self.video_writer.write(save_canvas)", source)


if __name__ == "__main__":
    unittest.main()
