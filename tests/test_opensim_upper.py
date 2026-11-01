"""Upper-body static pose, registration refusal, and segment lengths. No camera, no IK."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.opensim_upper.environment import POSE2SIM_MODEL
from src.opensim_upper.frame_calibration import FrameCalibration, apply_camera_to_model
from src.opensim_upper.leveling import LevelingError, assess_static_accel, rotate_motion_to_color
from src.opensim_upper.model_xml import EXCLUDED_FULL_BODY_MARKERS, inspect_model
from src.opensim_upper.registration import (
    RegistrationInsufficient,
    camera_to_model_transform,
    registration_report,
)
from src.opensim_upper.scaling import UNSCALED_BODIES, subject_lengths
from src.opensim_upper.static_pose import OBSERVED, PREDICTED, StaticPoseCapture
from src.opensim_upper.workflow import write_calibration


def _pose(shift: float = 0.0) -> tuple[dict, dict]:
    observed = {
        "left_shoulder": (0.20, -0.20 + shift, 1.50),
        "right_shoulder": (-0.20, -0.20, 1.50),
        "left_elbow": (0.22, 0.10, 1.55),
        "right_elbow": (-0.22, 0.10, 1.55),
        "left_wrist": (0.24, 0.35, 1.52),
        "right_wrist": (-0.24, 0.35, 1.52),
    }
    predicted = {}
    for name, base in (
        ("r_lelbow_study", (-0.26, 0.12, 1.56)),
        ("r_melbow_study", (-0.18, 0.12, 1.54)),
        ("r_lwrist_study", (-0.28, 0.36, 1.53)),
        ("r_mwrist_study", (-0.20, 0.36, 1.51)),
        ("L_lelbow_study", (0.26, 0.12, 1.56)),
        ("L_melbow_study", (0.18, 0.12, 1.54)),
        ("L_lwrist_study", (0.28, 0.36, 1.53)),
        ("L_mwrist_study", (0.20, 0.36, 1.51)),
    ):
        x, y, z = base
        predicted[name] = (x, y + shift, z)
    return observed, predicted


class RegistrationTest(unittest.TestCase):
    def test_full_transform_is_refused(self) -> None:
        report = registration_report()
        self.assertFalse(report["complete"])
        self.assertIsNone(report["rotation"])
        self.assertIsNone(report["translation_m"])
        self.assertFalse(report["y_flip_applied"])
        self.assertEqual(report["units"], "metres")
        self.assertEqual(report["camera_handedness"], "right")
        self.assertEqual(report["model_handedness"], "right")
        with self.assertRaises(RegistrationInsufficient):
            camera_to_model_transform()

    def test_y_only_flip_is_a_reflection_and_is_not_used(self) -> None:
        reflection = np.diag([1.0, -1.0, 1.0])
        self.assertLess(float(np.linalg.det(reflection)), 0.0)
        self.assertIsNone(registration_report()["rotation"])


class StaticPoseTest(unittest.TestCase):
    def test_median_and_round_trip_in_camera_metres(self) -> None:
        cap = StaticPoseCapture(min_frames=3)
        for index, shift in enumerate((0.0, 0.02, -0.02)):
            observed, predicted = _pose(shift if index else 0.0)
            self.assertTrue(cap.add_frame(float(index) * 0.1, observed, predicted))
        summary = cap.summary(1.78, 86.0)
        self.assertEqual(summary["coord_frame"], "camera_3d_metre")
        self.assertEqual(summary["units"], "metres")
        self.assertEqual(summary["n_valid_frames"], 3)
        self.assertAlmostEqual(summary["participant_height_m"], 1.78)
        self.assertAlmostEqual(summary["participant_mass_kg"], 86.0)
        shoulder = summary["landmarks"]["left_shoulder"]
        self.assertAlmostEqual(shoulder["y_m"], -0.20)
        self.assertEqual(shoulder["source"], "observed")
        self.assertEqual(summary["landmarks"]["r_lelbow_study"]["source"], "estimated_lstm_study")
        again = (shoulder["x_m"], shoulder["y_m"], shoulder["z_m"])
        self.assertEqual(again, (shoulder["x_m"], shoulder["y_m"], shoulder["z_m"]))

    def test_missing_lstm_marker_rejects_the_frame(self) -> None:
        cap = StaticPoseCapture(min_frames=2)
        observed, predicted = _pose()
        del predicted["L_lwrist_study"]
        self.assertFalse(cap.add_frame(0.0, observed, predicted))
        self.assertEqual(cap.n_rejected, 1)
        self.assertFalse(cap.ready())

    def test_does_not_accept_lower_body_stand_ins(self) -> None:
        self.assertNotIn("C7_study", OBSERVED + PREDICTED)
        self.assertNotIn("left_hip", OBSERVED)
        for name in EXCLUDED_FULL_BODY_MARKERS:
            self.assertNotIn(name, PREDICTED)


class ScalingTest(unittest.TestCase):
    def test_lengths_are_metres_and_not_applied(self) -> None:
        cap = StaticPoseCapture(min_frames=2)
        observed, predicted = _pose()
        cap.add_frame(0.0, observed, predicted)
        cap.add_frame(0.1, observed, predicted)
        summary = cap.summary(1.78, 86.0)
        lengths = subject_lengths(summary)
        humerus = lengths["lengths"]["humerus_r"]["distance_m"]
        expected = float(np.linalg.norm(np.array([-0.20, -0.20, 1.50]) - np.array([-0.22, 0.10, 1.55])))
        self.assertAlmostEqual(humerus, expected, places=6)
        self.assertIsNone(lengths["lengths"]["humerus_r"]["scale_factor"])
        self.assertFalse(lengths["lengths"]["humerus_r"]["applied_to_model"])
        self.assertFalse(lengths["height_used_for_scale"])
        self.assertIn("pelvis", lengths["bodies_left_at_model_default"])
        self.assertIn("torso", UNSCALED_BODIES)
        self.assertNotIn("humerus_r", lengths["bodies_left_at_model_default"])

    def test_calibration_folder_keeps_registration_incomplete(self) -> None:
        cap = StaticPoseCapture(min_frames=2)
        observed, predicted = _pose()
        cap.add_frame(0.0, observed, predicted)
        cap.add_frame(0.1, observed, predicted)
        folder = Path(tempfile.mkdtemp()) / "calib"
        payload = write_calibration(cap, folder, 1.78, 86.0)
        self.assertFalse(payload["registration"]["complete"])
        self.assertFalse(payload["inverse_kinematics"])
        self.assertFalse(payload["trc_written"])
        self.assertTrue((folder / "static_pose.json").is_file())
        self.assertTrue((folder / "static_pose_camera.png").is_file())
        saved = json.loads((folder / "static_pose.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["landmarks"]["left_wrist"]["coord_frame"], "camera_3d_metre")


class ModelXmlTest(unittest.TestCase):
    def test_installed_model_lists_upper_coordinates(self) -> None:
        if not POSE2SIM_MODEL.is_file():
            self.skipTest("Pose2Sim model is not installed on this machine")
        info = inspect_model()
        self.assertEqual(info["length_units"], "meters")
        self.assertAlmostEqual(info["gravity_m_s2"][1], -9.80665, places=3)
        self.assertIn("arm_flex_r", info["coordinates"])
        self.assertIn("elbow_flex_r", info["coordinates"])
        self.assertIn("humerus_r", info["upper_bodies_present"])
        self.assertIn("RElbow", info["upper_markers"])
        self.assertIn("r_lelbow_study", info["upper_markers"])
        self.assertIn("C7_study", info["excluded_present_in_xml"])
        self.assertFalse(info["wrist_joint_marker_defined"])
        self.assertGreater(info["n_meshes"], 0)


def _measured_level_camera() -> FrameCalibration:
    """Synthetic still accelerometer: colour Y is down, so up is -Y.

    This vector stands in for a measured reading. The library does not
    assume it when the record is empty.
    """
    samples = np.tile(np.array([0.0, -9.80665, 0.0]), (25, 1))
    assessment = assess_static_accel(samples)
    calibration = FrameCalibration()
    calibration.set_measured_up(assessment)
    calibration.set_anterior_point("anterior_from", (0.0, 0.0, 1.0))
    calibration.set_anterior_point("anterior_to", (0.0, 0.0, 0.0))
    return calibration


class FrameRegistrationTest(unittest.TestCase):
    def test_empty_record_does_not_invent_a_matrix(self) -> None:
        calibration = FrameCalibration()
        solved = calibration.solve()
        self.assertFalse(solved["complete"])
        self.assertIsNone(solved["rotation"])
        self.assertIsNone(solved["translation_m"])
        self.assertFalse(solved["assumed_facing_camera"])
        self.assertFalse(solved["assumed_image_y_is_gravity"])
        self.assertFalse(solved["origin_is_model_pelvis"])
        self.assertFalse(solved["applied_to_landmarks"])
        with self.assertRaises(RegistrationInsufficient):
            camera_to_model_transform(calibration)

    def test_rotation_without_a_surveyed_origin_is_not_applied(self) -> None:
        calibration = _measured_level_camera()
        solved = calibration.solve()
        self.assertIsNotNone(solved["rotation"])
        self.assertIsNone(solved["translation_m"])
        self.assertFalse(solved["complete"])
        self.assertIsNone(calibration.origin_model_m)
        with self.assertRaises(RegistrationInsufficient):
            camera_to_model_transform(calibration)

    def test_known_level_pose_is_a_proper_rigid_transform(self) -> None:
        calibration = _measured_level_camera()
        calibration.set_origin_camera((0.2, 0.4, 1.5))
        calibration.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
        solved = camera_to_model_transform(calibration)
        rotation = np.asarray(solved["rotation"])
        expected = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(rotation, expected))
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)
        self.assertTrue(np.allclose(solved["translation_m"], (2.5, 2.4, 3.2)))
        checks = solved["checks"]
        self.assertTrue(checks["orthogonal"])
        self.assertTrue(checks["proper_rotation"])
        self.assertEqual(checks["translation_units"], "metres")
        self.assertLess(checks["inverse_max_error_m"], 1e-8)
        self.assertLess(checks["reference_alignment_error_m"], 1e-8)
        self.assertLess(checks["anterior_axis_error_m"], 1e-8)
        mapped = apply_camera_to_model((0.2, 0.4, 1.5), solved)
        self.assertTrue(np.allclose(mapped, (1.0, 2.0, 3.0)))
        reflection = np.diag([1.0, -1.0, 1.0])
        self.assertLess(float(np.linalg.det(reflection)), 0.0)
        self.assertFalse(np.allclose(rotation, reflection))

    def test_tilted_up_still_maps_the_reference_and_inverts(self) -> None:
        samples = np.tile(np.array([0.2, -9.7, 0.15]), (25, 1))
        # Keep the samples still and near 1 g.
        samples += np.array([0.0, -0.05, 0.0])
        assessment = assess_static_accel(samples)
        calibration = FrameCalibration()
        calibration.set_measured_up(assessment)
        calibration.set_anterior_point("anterior_from", (0.0, 0.0, 1.2))
        calibration.set_anterior_point("anterior_to", (0.35, 0.05, 0.4))
        calibration.set_origin_camera((0.1, 0.2, 1.2))
        calibration.set_origin_model((0.4, 1.1, -0.2), "corner of the floor tape, surveyed")
        solved = camera_to_model_transform(calibration)
        rotation = np.asarray(solved["rotation"])
        translation = np.asarray(solved["translation_m"])
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8))
        point = np.array([0.55, -0.1, 0.8])
        model = rotation @ point + translation
        back = rotation.T @ (model - translation)
        self.assertTrue(np.allclose(back, point))
        self.assertTrue(np.allclose(apply_camera_to_model((0.1, 0.2, 1.2), solved), (0.4, 1.1, -0.2)))

    def test_facing_parallel_to_up_is_rejected(self) -> None:
        calibration = _measured_level_camera()
        calibration.set_anterior_point("anterior_from", (0.0, 0.0, 1.0))
        calibration.set_anterior_point("anterior_to", (0.0, -0.4, 1.0))
        calibration.set_origin_camera((0.0, 0.0, 1.0))
        calibration.set_origin_model((0.0, 0.0, 0.0), "end of the vertical rod, surveyed")
        solved = calibration.solve()
        self.assertIsNone(solved["rotation"])
        self.assertFalse(solved["complete"])
        with self.assertRaises(RegistrationInsufficient):
            camera_to_model_transform(calibration)

    def test_pelvis_is_not_accepted_as_the_origin_description(self) -> None:
        calibration = _measured_level_camera()
        with self.assertRaises(ValueError):
            calibration.set_origin_model((0.0, 0.0, 0.0), "pelvis")

    def test_static_accel_accepts_gravity_and_rejects_motion(self) -> None:
        still = assess_static_accel(np.tile([0.0, -9.80665, 0.0], (25, 1)))
        self.assertTrue(np.allclose(still["up_camera"], [0.0, -1.0, 0.0]))
        self.assertFalse(still["image_y_used_as_gravity"])
        moving = np.tile([0.0, -9.80665, 0.0], (25, 1))
        moving[::2, 0] = 3.0
        with self.assertRaises(LevelingError):
            assess_static_accel(moving)
        with self.assertRaises(LevelingError):
            assess_static_accel(np.tile([0.0, -2.0, 0.0], (25, 1)))
        with self.assertRaises(LevelingError):
            assess_static_accel(np.tile([0.0, -9.8, 0.0], (5, 1)))

    def test_extrinsic_rotation_is_column_major_and_translation_is_not_applied(self) -> None:
        # 90 degrees about Z. rs2_extrinsics stores the matrix column-major.
        column_major = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        rotated = rotate_motion_to_color([1.0, 0.0, 0.0], column_major)
        self.assertTrue(np.allclose(rotated, [0.0, 1.0, 0.0]))

    def test_save_writes_parameters_without_applying_them(self) -> None:
        calibration = _measured_level_camera()
        calibration.set_origin_camera((0.2, 0.4, 1.5))
        calibration.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
        folder = Path(tempfile.mkdtemp()) / "registration.json"
        payload = calibration.save(folder)
        saved = json.loads(folder.read_text(encoding="utf-8"))
        self.assertTrue(saved["complete"])
        self.assertFalse(saved["applied_to_landmarks"])
        self.assertEqual(saved["references"]["origin_description"], payload["references"]["origin_description"])
        self.assertEqual(saved["references"]["origin_model_m"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
