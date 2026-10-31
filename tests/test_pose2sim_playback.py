"""Offline Pose2Sim playback helpers. No camera. No live LSTM."""

from __future__ import annotations

import unittest

import numpy as np

from src.pose.keypoints import Keypoint2D
from src.ui.live_markers import draw_live_landmarks
from src.ui.pose2sim_playback import (
    EST_CSV,
    OBS_CSV,
    UPPER_ESTIMATES,
    fit_pinhole,
    load_estimates,
    load_observed,
    project_camera_xyz,
    yup_to_camera,
)


class TransformTest(unittest.TestCase):
    def test_yup_flips_camera_y_only(self) -> None:
        x, y, z = yup_to_camera(0.2, 0.3, 1.1)
        self.assertAlmostEqual(x, 0.2)
        self.assertAlmostEqual(y, -0.3)
        self.assertAlmostEqual(z, 1.1)

    def test_pinhole_recovers_known_intrinsics(self) -> None:
        fx, fy, cx, cy = 900.0, 910.0, 640.0, 360.0
        samples = []
        rng = np.random.default_rng(0)
        for _ in range(40):
            z = 1.2 + float(rng.random())
            x = float(rng.uniform(-0.4, 0.4))
            y = float(rng.uniform(-0.3, 0.3))
            u = fx * x / z + cx
            v = fy * y / z + cy
            samples.append((x, y, z, u, v))
        k = fit_pinhole(samples)
        self.assertIsNotNone(k)
        self.assertAlmostEqual(k["fx"], fx, delta=1e-6)
        self.assertAlmostEqual(k["fy"], fy, delta=1e-6)
        uv = project_camera_xyz(0.1, -0.05, 1.5, k)
        self.assertIsNotNone(uv)
        self.assertAlmostEqual(uv[0], fx * 0.1 / 1.5 + cx, delta=1e-6)


class RecordingFilesTest(unittest.TestCase):
    def test_estimate_frames_are_a_subset_of_the_recording(self) -> None:
        if not OBS_CSV.is_file() or not EST_CSV.is_file():
            self.skipTest("recording or estimate CSV missing")
        observed, _times, samples = load_observed(OBS_CSV)
        estimates = load_estimates(EST_CSV)
        overlap = set(estimates) & set(observed)
        self.assertGreater(len(overlap), 500)
        first = estimates[next(iter(sorted(overlap)))]
        names = {row[0] for row in first}
        self.assertEqual(names, set(UPPER_ESTIMATES))
        self.assertGreater(len(samples), 20)
        self.assertIsNotNone(fit_pinhole(samples))


class DrawTest(unittest.TestCase):
    def test_estimate_diamonds_do_not_add_name_labels(self) -> None:
        import inspect

        from src.ui.live_markers import draw_live_landmarks as draw

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        obs = [Keypoint2D("left_elbow", 40, 50, 0.9, x_m=0.1, y_m=0.2, z_m=1.0)]
        est = [Keypoint2D("r_lelbow_study", 70, 55, 1.0, x_m=-0.1, y_m=0.1, z_m=1.0)]
        out = draw(frame, obs, 0.5, 7.0, "3d", estimates=est, banner="OFFLINE PLAYBACK")
        self.assertGreater(int(out.sum()), 0)
        source = inspect.getsource(draw)
        self.assertNotIn("short_label", source)
        self.assertIn("banner", source)


if __name__ == "__main__":
    unittest.main()
