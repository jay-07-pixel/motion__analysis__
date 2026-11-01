"""Pose and Hands overlap on one frame. Landmark names still come from Pose wrists."""

from __future__ import annotations

import threading
import time
import unittest

from src.pose.hands import HandsExtractor2D
from src.pose.keypoints import Keypoint2D
from src.pose.parallel import run_together


class _Hands:
    """Side assignment only. No MediaPipe graph."""

    def __init__(self) -> None:
        self._swap_handedness = True
        self._match_to_pose_wrists = True

    keypoints_from_inference = HandsExtractor2D.keypoints_from_inference


class SeparateGraphsTest(unittest.TestCase):
    def test_pose_and_hands_own_different_mediapipe_graphs(self) -> None:
        from src.pose.extractor_2d import PoseExtractor2D

        pose = PoseExtractor2D(model_complexity=1)
        hands = HandsExtractor2D(max_num_hands=2, model_complexity=1)
        try:
            self.assertIsNot(pose._pose, hands._hands)
            self.assertIsNot(type(pose._pose), type(hands._hands))
        finally:
            pose.close()
            hands.close()


class ParallelPoseHandsTest(unittest.TestCase):
    def test_hands_starts_before_pose_finishes(self) -> None:
        started = threading.Event()

        def pose_call():
            started.wait(timeout=1)
            time.sleep(0.02)
            return "pose"

        def hands_call():
            started.set()
            time.sleep(0.02)
            return "hands"

        pose_value, hands_value = run_together(pose_call, hands_call)
        self.assertEqual(pose_value, "pose")
        self.assertEqual(hands_value, "hands")

    def test_hand_is_named_from_the_same_frame_pose_wrist(self) -> None:
        pose_wrist = Keypoint2D("right_wrist", 100.0, 200.0, 0.9)
        raw = [("left", 0.8, [(110.0, 210.0), (120.0, 220.0)] + [(0.0, 0.0)] * 19)]
        named = _Hands().keypoints_from_inference(raw, [pose_wrist])
        self.assertTrue(named)
        self.assertTrue(all(kp.name.startswith("right_") for kp in named))
        self.assertAlmostEqual(named[0].u_px, 110.0)
        self.assertAlmostEqual(named[0].v_px, 210.0)

    def test_hands_error_is_raised_after_pose(self) -> None:
        def pose_call():
            return "pose"

        def hands_call():
            raise RuntimeError("hands failed")

        with self.assertRaises(RuntimeError):
            run_together(pose_call, hands_call)


if __name__ == "__main__":
    unittest.main()
