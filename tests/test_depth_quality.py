"""Depth windows, the optional XYZ filter, and prediction-versus-observed timing."""

from __future__ import annotations

import unittest

from src.capture.depth_util import accept_depth_change, robust_depth_from_samples, torso_gap_ok
from src.geometry.deproject import CameraXyzSmoother, DepthGate
from src.pose.keypoints import Keypoint2D
from src.pose2sim_live.sync import classify_prediction


def _point(name: str, x: float | None, y: float | None = 0.0, z: float | None = 1.0) -> Keypoint2D:
    return Keypoint2D(name, 10.0, 20.0, 0.9, x_m=x, y_m=y, z_m=z)


class DepthWindowTest(unittest.TestCase):
    def test_mixed_window_is_rejected_instead_of_the_background(self) -> None:
        samples = [0.70] * 7 + [2.10] * 11
        self.assertIsNone(robust_depth_from_samples(samples, min_valid=8, max_spread_m=0.15))

    def test_sparse_window_is_invalid(self) -> None:
        self.assertIsNone(robust_depth_from_samples([1.154, 1.160, 1.148], min_valid=13, max_spread_m=0.15))

    def test_consistent_window_keeps_its_median(self) -> None:
        samples = [1.10 + 0.001 * i for i in range(20)]
        chosen = robust_depth_from_samples(samples, min_valid=13, max_spread_m=0.15)
        self.assertAlmostEqual(chosen, sorted(samples)[10])

    def test_fast_image_motion_keeps_a_large_depth_change(self) -> None:
        self.assertTrue(accept_depth_change(1.15, 0.70, 120.0, max_step_m=0.35, fast_px=80.0))

    def test_small_image_motion_rejects_a_depth_jump_without_reusing_the_old_value(self) -> None:
        gate = DepthGate(max_step_m=0.35, fast_px=80.0, max_torso_gap_m=0.75)
        self.assertAlmostEqual(gate.consider("right_elbow", 1.154, 712.0, 338.0), 1.154)
        self.assertIsNone(gate.consider("right_elbow", 2.097, 705.0, 360.0))
        self.assertFalse(accept_depth_change(1.154, 2.097, 23.0, max_step_m=0.35, fast_px=80.0))
        self.assertAlmostEqual(gate.consider("right_elbow", 1.20, 710.0, 340.0), 1.20)

    def test_background_elbow_fails_the_shoulder_gap_and_a_near_elbow_does_not(self) -> None:
        self.assertFalse(torso_gap_ok(2.097, 0.665, 0.75))
        self.assertTrue(torso_gap_ok(0.507, 0.702, 0.75))
        gate = DepthGate(max_step_m=0.35, fast_px=80.0, max_torso_gap_m=0.75)
        self.assertIsNone(gate.consider("right_elbow", 2.097, 705.0, 360.0, reference_z_m=0.665))
        self.assertAlmostEqual(gate.consider("right_elbow", 0.507, 678.0, 362.0, reference_z_m=0.702), 0.507)


class XyzFilterTest(unittest.TestCase):
    def test_slow_motion_lags_and_a_fast_step_does_not(self) -> None:
        filt = CameraXyzSmoother(smooth=0.2, deadband_m=0.0, fast_m=0.05)
        first = filt.apply([_point("right_elbow", 0.0)])[0]
        slow = filt.apply([_point("right_elbow", 0.01)])[0]
        self.assertAlmostEqual(slow.x_m, 0.002)
        self.assertLess(slow.x_m, 0.01)
        fast = filt.apply([_point("right_elbow", 0.20)])[0]
        self.assertAlmostEqual(fast.x_m, 0.20)
        self.assertAlmostEqual(first.x_m, 0.0)

    def test_missing_depth_is_not_replaced_by_the_previous_point(self) -> None:
        filt = CameraXyzSmoother(smooth=0.5, deadband_m=0.0, fast_m=0.05)
        filt.apply([_point("right_elbow", 0.10)])
        missing = filt.apply([_point("right_elbow", None, None, None)])[0]
        self.assertIsNone(missing.x_m)
        self.assertIsNone(missing.y_m)
        self.assertIsNone(missing.z_m)
        nxt = filt.apply([_point("right_elbow", 0.40)])[0]
        self.assertAlmostEqual(nxt.x_m, 0.40)

    def test_raw_point_stays_separate_from_the_filtered_point(self) -> None:
        filt = CameraXyzSmoother(smooth=0.2, deadband_m=0.0, fast_m=0.05)
        filt.apply([_point("right_wrist", 0.0)])
        raw = _point("right_wrist", 0.01)
        filtered = filt.apply([raw])[0]
        self.assertAlmostEqual(raw.x_m, 0.01)
        self.assertIsNone(raw.filt_x_m)
        self.assertAlmostEqual(filtered.filt_x_m, filtered.x_m)
        self.assertNotAlmostEqual(filtered.x_m, raw.x_m)


class ObservedPredictionSyncTest(unittest.TestCase):
    def test_rejected_observed_point_is_not_filled_from_an_older_prediction(self) -> None:
        observed = _point("right_elbow", None, None, None)
        decision = classify_prediction(
            1.00,
            1.12,
            delay_ms=80,
            max_age_ms=200,
            source_frame=4,
            current_frame=6,
        )
        self.assertIsNone(observed.x_m)
        self.assertEqual(decision["status"], "delayed")
        self.assertEqual(decision["source_frame"], 4)
        self.assertEqual(decision["current_frame"], 6)
        self.assertNotEqual(decision["source_frame"], decision["current_frame"])


if __name__ == "__main__":
    unittest.main()
