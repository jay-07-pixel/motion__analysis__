"""Full-body live catalog, toes, and lower LSTM I/O. No camera."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from src.markers.catalog import (
    POSE2SIM_LOWER_IN,
    POSE2SIM_LOWER_OUT,
    POSE2SIM_UPPER_OUT,
    TOE_INPUTS,
    UNAVAILABLE,
    assert_no_foot_index_as_toe,
    specs_by_kind,
)
from src.markers.centres import pair_centres
from src.markers.foot_calib import FootCalibrator, foot_rotation
from src.pose.keypoints import Keypoint2D
from src.pose2sim_live.lower_lstm import extract_lower_frame, missing_lower_inputs
from src.pose2sim_live.upper_lstm import extract_upper_frame


def kp(name: str, xyz, conf: float = 0.9) -> Keypoint2D:
    x, y, z = xyz
    return Keypoint2D(
        name=name,
        u_px=100,
        v_px=100,
        confidence=conf,
        coord_frame="camera_3d_metre",
        x_m=x,
        y_m=y,
        z_m=z,
    )


def pose_body() -> list[Keypoint2D]:
    return [
        kp("left_shoulder", (0.2, 0.1, 1.0)),
        kp("right_shoulder", (-0.2, 0.1, 1.0)),
        kp("left_elbow", (0.25, 0.3, 1.05)),
        kp("right_elbow", (-0.25, 0.3, 1.05)),
        kp("left_wrist", (0.28, 0.5, 1.02)),
        kp("right_wrist", (-0.28, 0.5, 1.02)),
        kp("left_hip", (0.1, 0.6, 1.1)),
        kp("right_hip", (-0.1, 0.6, 1.1)),
        kp("left_knee", (0.12, 0.9, 1.12)),
        kp("right_knee", (-0.12, 0.9, 1.12)),
        kp("left_ankle", (0.13, 1.2, 1.14)),
        kp("right_ankle", (-0.13, 1.2, 1.14)),
        kp("left_heel", (0.12, 1.25, 1.18)),
        kp("right_heel", (-0.12, 1.25, 1.18)),
        kp("left_foot_index", (0.14, 1.22, 1.05)),
        kp("right_foot_index", (-0.14, 1.22, 1.05)),
    ]


class CatalogTest(unittest.TestCase):
    def test_model_outputs_match_installed_pose2sim(self) -> None:
        self.assertEqual(len(POSE2SIM_UPPER_OUT), 8)
        self.assertEqual(len(POSE2SIM_LOWER_OUT), 35)
        self.assertEqual(len(POSE2SIM_LOWER_IN), 15)
        self.assertIn("C7_study", POSE2SIM_LOWER_OUT)
        self.assertIn("RHJC_study", POSE2SIM_LOWER_OUT)
        self.assertNotIn("IJ", POSE2SIM_LOWER_OUT)
        self.assertNotIn("PX", POSE2SIM_LOWER_OUT)
        self.assertNotIn("T8", POSE2SIM_LOWER_OUT)
        for name in UNAVAILABLE:
            self.assertTrue(any(s.name == name and s.kind == "unavailable" for s in specs_by_kind("unavailable")))

    def test_foot_index_is_not_a_toe_input(self) -> None:
        assert_no_foot_index_as_toe()
        self.assertNotIn("left_foot_index", TOE_INPUTS)
        self.assertNotIn("right_foot_index", TOE_INPUTS)
        src = inspect.getsource(extract_lower_frame)
        self.assertNotIn("foot_index", src)


class LowerExtractTest(unittest.TestCase):
    def test_missing_toes_does_not_run_or_fake(self) -> None:
        body = pose_body()
        self.assertIsNone(extract_lower_frame(body, 0.5))
        missing = missing_lower_inputs(body, 0.5)
        self.assertEqual(set(missing), set(TOE_INPUTS))
        self.assertIsNotNone(extract_upper_frame(body, 0.5))

    def test_four_distinct_toes_enable_lower(self) -> None:
        body = pose_body() + [
            kp("LBigToe", (0.15, 1.21, 1.04)),
            kp("LSmallToe", (0.10, 1.21, 1.04)),
            kp("RBigToe", (-0.15, 1.21, 1.04)),
            kp("RSmallToe", (-0.10, 1.21, 1.04)),
        ]
        extracted = extract_lower_frame(body, 0.5)
        self.assertIsNotNone(extracted)
        feat, hip = extracted
        self.assertEqual(feat.shape, (45,))
        self.assertEqual(hip.shape, (3,))
        self.assertEqual(missing_lower_inputs(body, 0.5), ())


class FootCalibTest(unittest.TestCase):
    def test_roundtrip_rigid_foot(self) -> None:
        body = pose_body()
        by = {p.name: p for p in body}
        ankle = np.array([by["left_ankle"].x_m, by["left_ankle"].y_m, by["left_ankle"].z_m])
        heel = np.array([by["left_heel"].x_m, by["left_heel"].y_m, by["left_heel"].z_m])
        index = np.array([by["left_foot_index"].x_m, by["left_foot_index"].y_m, by["left_foot_index"].z_m])
        rot = foot_rotation(ankle, heel, index)
        self.assertIsNotNone(rot)
        big = ankle + rot @ np.array([0.02, 0.08, 0.01])
        cal = FootCalibrator()
        msg = cal.add_click(10, 10, (float(big[0]), float(big[1]), float(big[2])), body)
        self.assertIn("LEFT small", msg)
        recon = cal.reconstruct(body, 0.5)
        self.assertEqual(len(recon), 1)
        self.assertEqual(recon[0].name, "LBigToe")
        self.assertAlmostEqual(recon[0].x_m, big[0], places=6)
        self.assertAlmostEqual(recon[0].y_m, big[1], places=6)
        self.assertAlmostEqual(recon[0].z_m, big[2], places=6)

    def test_not_ready_until_four_clicks(self) -> None:
        self.assertFalse(FootCalibrator().ready())


class CentreTest(unittest.TestCase):
    def test_midpoint_only_when_both_exist(self) -> None:
        pts = [
            ("r_lelbow_study", 0.0, 0.0, 1.0),
            ("r_melbow_study", 0.2, 0.0, 1.0),
        ]
        out = pair_centres(pts)
        self.assertEqual(out[0][0], "r_elbow_centre")
        self.assertAlmostEqual(out[0][1], 0.1)


if __name__ == "__main__":
    unittest.main()
