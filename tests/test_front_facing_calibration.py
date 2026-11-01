"""Front-facing camera registration. No camera and no inverse kinematics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.opensim_upper.frame_calibration import FrameCalibration, apply_camera_to_model
from src.opensim_upper.leveling import assess_static_accel
from src.opensim_upper.registration import RegistrationInsufficient
from src.opensim_upper.transform import load_registration
from src.opensim_upper.workflow import registration_for_capture, write_capture_registration
from src.ui.capture_analysis import analyze_capture


def _up(vector) -> dict:
    samples = np.tile(np.asarray(vector, dtype=float), (25, 1))
    return assess_static_accel(samples)


def _facing(up_vector) -> FrameCalibration:
    vector = np.asarray(up_vector, dtype=float)
    vector = vector / np.linalg.norm(vector) * 9.80665
    calibration = FrameCalibration()
    calibration.set_measured_up(_up(vector))
    calibration.confirm_facing_camera()
    return calibration


class FrontFacingRotationTest(unittest.TestCase):
    def test_level_camera_anterior_is_negative_z(self) -> None:
        calibration = _facing([0.0, -9.80665, 0.0])
        calibration.set_origin_camera((0.2, 0.4, 1.5))
        calibration.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
        solved = calibration.solve()
        rotation = np.asarray(solved["rotation"])
        expected = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
        self.assertTrue(solved["complete"])
        self.assertTrue(solved["assumed_facing_camera"])
        self.assertEqual(solved["heading_source"], "camera_optical_axis_negative_z")
        self.assertFalse(solved["y_flip_applied"])
        self.assertTrue(np.allclose(rotation, expected))
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)
        self.assertTrue(np.allclose(apply_camera_to_model((0.2, 0.4, 1.5), solved), (1.0, 2.0, 3.0)))

    def test_tilted_camera_keeps_anterior_horizontal(self) -> None:
        up = np.array([0.0, -0.80, 0.45])
        calibration = _facing(up)
        calibration.set_origin_camera((0.1, -0.2, 1.4))
        calibration.set_origin_model((0.3, 0.9, -0.2), "corner of the floor tape, surveyed")
        solved = calibration.solve()
        rotation = np.asarray(solved["rotation"])
        measured_up = np.asarray(calibration.up_camera)
        self.assertTrue(solved["complete"])
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)
        self.assertAlmostEqual(float(np.dot(rotation[0], measured_up)), 0.0, places=8)
        self.assertGreater(float(np.dot(rotation[0], np.array([0.0, 0.0, -1.0]))), 0.5)
        self.assertTrue(np.allclose(apply_camera_to_model((0.1, -0.2, 1.4), solved), (0.3, 0.9, -0.2)))

    def test_nearly_vertical_camera_is_rejected(self) -> None:
        for vector in ((0.0, -0.02, -1.0), (0.0, 0.0, -9.80665), (0.0, 0.0, 9.80665)):
            calibration = _facing(vector)
            calibration.set_origin_camera((0.0, 0.0, 1.0))
            calibration.set_origin_model((0.2, 0.8, 0.1), "mark on the wall tape, surveyed")
            solved = calibration.solve()
            self.assertFalse(solved["complete"], vector)
            self.assertIsNone(solved["rotation"])
            with self.assertRaises(RegistrationInsufficient):
                apply_camera_to_model((0.0, 0.0, 1.0), solved)

    def test_missing_reference_or_up_blocks_completion(self) -> None:
        no_point = _facing([0.0, -9.80665, 0.0])
        self.assertFalse(no_point.solve()["complete"])
        no_up = FrameCalibration()
        no_up.confirm_facing_camera()
        no_up.set_origin_camera((0.2, 0.4, 1.5))
        no_up.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
        solved = no_up.solve()
        self.assertFalse(solved["complete"])
        self.assertIsNone(solved["rotation"])
        self.assertTrue(solved["assumed_facing_camera"])


class RegistrationFileTest(unittest.TestCase):
    def test_front_facing_file_reloads(self) -> None:
        calibration = _facing([0.0, -9.80665, 0.0])
        calibration.set_origin_camera((0.2, 0.4, 1.5))
        calibration.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
        path = Path(tempfile.mkdtemp()) / "registration.json"
        calibration.save(path)
        loaded = load_registration(path)
        self.assertTrue(loaded["assumed_facing_camera"])
        self.assertEqual(loaded["heading_source"], "camera_optical_axis_negative_z")
        self.assertTrue(loaded["checks"]["accepted"])

    def test_rod_file_still_loads(self) -> None:
        calibration = FrameCalibration()
        calibration.set_measured_up(_up([0.0, -9.80665, 0.0]))
        calibration.set_anterior_point("anterior_from", (0.0, 0.0, 1.0))
        calibration.set_anterior_point("anterior_to", (0.0, 0.0, 0.0))
        calibration.set_origin_camera((0.2, 0.4, 1.5))
        calibration.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
        path = Path(tempfile.mkdtemp()) / "registration.json"
        calibration.save(path)
        loaded = load_registration(path)
        self.assertFalse(loaded["assumed_facing_camera"])
        self.assertEqual(loaded["heading_source"], "facing_rod")
        self.assertTrue(np.allclose(loaded["rotation"], [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]))

    def test_invalid_registration_does_not_reach_ik(self) -> None:
        folder = Path(tempfile.mkdtemp())
        other = folder / "output" / "opensim_frame" / "old"
        other.mkdir(parents=True)
        capture = folder / "capture-a"
        capture.mkdir()
        (capture / "landmarks.csv").write_text("frame,landmark\n", encoding="utf-8")
        calibration = _facing([0.0, -9.80665, 0.0])
        calibration.set_origin_camera((0.2, 0.4, 1.5))
        calibration.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
        calibration.save(other / "registration.json")
        (capture / "registration.json").write_text(json.dumps({"complete": False}), encoding="utf-8")
        result = analyze_capture(capture, folder)
        self.assertFalse(result["ok"])
        self.assertFalse(result["opensim_ran"])
        self.assertIsNone(result.get("registration_json"))
        self.assertFalse((folder / "output" / "opensim_upper").exists())

    def test_another_captures_registration_is_not_reused(self) -> None:
        folder = Path(tempfile.mkdtemp())
        first = folder / "capture-a"
        second = folder / "capture-b"
        first.mkdir()
        second.mkdir()
        (second / "landmarks.csv").write_text("frame,landmark\n", encoding="utf-8")
        calibration = _facing([0.0, -9.80665, 0.0])
        calibration.set_origin_camera((0.2, 0.4, 1.5))
        calibration.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
        saved = calibration.save(Path(tempfile.mkdtemp()) / "registration.json")
        bound = write_capture_registration(first, saved)
        archive = folder / "output" / "opensim_frame" / "earlier"
        archive.mkdir(parents=True)
        (archive / "registration.json").write_text(Path(bound).read_text(encoding="utf-8"), encoding="utf-8")
        self.assertEqual(registration_for_capture(first), bound)
        self.assertIsNone(registration_for_capture(second))
        result = analyze_capture(second, folder)
        self.assertFalse(result["opensim_ran"])
        self.assertNotEqual(result.get("registration_json"), str(bound))
        self.assertNotIn("registration.json", result["message"])
        self.assertIn("No frames", result["message"])


if __name__ == "__main__":
    unittest.main()
