"""Size a copy of the camera-frame model. The generic files stay untouched."""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.opensim_upper.camera_model import CAMERA_MODEL, SOURCE_MODEL
from src.opensim_upper.model_xml import UPPER_MODEL_MARKERS
from src.opensim_upper.scaling import (
    _arm_geometry,
    forearm_reference_distance,
    write_scaled_capture_model,
)
from src.opensim_upper.trc import write_trc


def _location(path: Path, marker: str) -> np.ndarray:
    root = ET.parse(path).getroot()
    for element in root.iter():
        if element.tag.endswith("Marker") and element.get("name") == marker:
            for child in element:
                if child.tag.endswith("location") and child.text:
                    return np.array([float(part) for part in child.text.split()], dtype=float)
    raise AssertionError(marker)


def _cloud(model: Path, folder: Path) -> tuple[Path, float]:
    geometry = _arm_geometry(ET.parse(model).getroot())
    width_model = float(np.linalg.norm(geometry["shoulder_right"] - geometry["shoulder_left"]))
    toward_left = geometry["shoulder_left"] - geometry["shoulder_right"]
    toward_left = toward_left / np.linalg.norm(toward_left)
    medial_generic = float(np.dot(geometry["glenohumeral_right"] - geometry["shoulder_right"], toward_left))
    subject_width = 0.30
    factor = subject_width / width_model
    medial = medial_generic * factor
    right = np.zeros(3)
    left = np.array([subject_width, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    elbow_r = right + direction * medial + np.array([0.0, geometry["humerus_length_m"], 0.0])
    elbow_l = left - direction * medial + np.array([0.0, geometry["humerus_length_m"], 0.0])
    wrist_r = elbow_r + np.array([0.0, 0.28, 0.0])
    wrist_l = elbow_l + np.array([0.0, 0.28, 0.0])
    points = {
        "RShoulder": right,
        "LShoulder": left,
        "RElbow": elbow_r,
        "LElbow": elbow_l,
        "r_lelbow_study": elbow_r,
        "r_melbow_study": elbow_r,
        "L_lelbow_study": elbow_l,
        "L_melbow_study": elbow_l,
        "r_lwrist_study": wrist_r + np.array([0.01, 0.0, 0.0]),
        "r_mwrist_study": wrist_r - np.array([0.01, 0.0, 0.0]),
        "L_lwrist_study": wrist_l + np.array([0.01, 0.0, 0.0]),
        "L_mwrist_study": wrist_l - np.array([0.01, 0.0, 0.0]),
    }
    times = [index * 0.05 for index in range(40)]
    samples = {
        name: [tuple(float(value) for value in points[name]) for _ in times] for name in UPPER_MODEL_MARKERS
    }
    path = folder / "markers.trc"
    write_trc(path, times, samples, 20.0, [{"target": name} for name in UPPER_MODEL_MARKERS])
    return path, factor


class BodyScaleTest(unittest.TestCase):
    def test_generic_forearm_distance_matches_the_checked_pose(self) -> None:
        if not CAMERA_MODEL.is_file():
            self.skipTest("camera model is not present")
        distance = forearm_reference_distance(CAMERA_MODEL)
        self.assertAlmostEqual(distance, 0.2288, places=3)

    def test_sized_copy_keeps_the_chest_and_lengthens_the_forearm(self) -> None:
        if not CAMERA_MODEL.is_file():
            self.skipTest("camera model is not present")
        before_camera = CAMERA_MODEL.read_bytes()
        before_source = SOURCE_MODEL.read_bytes() if SOURCE_MODEL.is_file() else None
        folder = Path(tempfile.mkdtemp())
        trc, torso_factor = _cloud(CAMERA_MODEL, folder)
        original_shoulder = _location(CAMERA_MODEL, "RShoulder")
        original_elbow = _location(CAMERA_MODEL, "RElbow")
        original_wrist = _location(CAMERA_MODEL, "r_lwrist_study")
        report = write_scaled_capture_model(CAMERA_MODEL, folder / "Model_upper_scaled.osim", trc)
        self.assertFalse(report["torso"]["applied"])
        self.assertAlmostEqual(report["torso"]["factor"], 1.0, places=4)
        self.assertAlmostEqual(report["torso"]["measured_factor"], torso_factor, places=4)
        self.assertFalse(report["humerus"]["applied"])
        self.assertTrue(report["forearm"]["applied"])
        self.assertFalse(report["clinical_accuracy"])
        self.assertFalse(report["mass_changed"])
        sized = Path(report["model_path"])
        self.assertNotEqual(sized.resolve(), CAMERA_MODEL.resolve())
        shoulder = _location(sized, "RShoulder")
        self.assertAlmostEqual(shoulder[0], original_shoulder[0], places=6)
        self.assertAlmostEqual(shoulder[1], original_shoulder[1], places=6)
        self.assertAlmostEqual(shoulder[2], original_shoulder[2], places=4)
        self.assertTrue(np.allclose(_location(sized, "RElbow"), original_elbow))
        wrist = _location(sized, "r_lwrist_study")
        self.assertAlmostEqual(wrist[0], original_wrist[0], places=6)
        self.assertAlmostEqual(wrist[2], original_wrist[2], places=6)
        self.assertLess(wrist[1], original_wrist[1])
        self.assertAlmostEqual(forearm_reference_distance(sized), 0.28, delta=0.005)
        self.assertEqual(CAMERA_MODEL.read_bytes(), before_camera)
        if before_source is not None:
            self.assertEqual(SOURCE_MODEL.read_bytes(), before_source)

    def test_ik_still_receives_a_sized_copy_when_lengths_stay_generic(self) -> None:
        if not CAMERA_MODEL.is_file():
            self.skipTest("camera model is not present")
        folder = Path(tempfile.mkdtemp())
        source = folder / "markers.trc"
        samples = {name: [(0.0, 0.0, 1.0)] for name in UPPER_MODEL_MARKERS}
        write_trc(source, [0.0], samples, 30.0, [{"target": name} for name in UPPER_MODEL_MARKERS])
        before = CAMERA_MODEL.read_bytes()
        report = write_scaled_capture_model(CAMERA_MODEL, folder / "Model_upper_scaled.osim", source)
        self.assertTrue(report["sized_copy_written"])
        self.assertFalse(report["ik_uses_generic_model"])
        self.assertEqual(Path(report["model_path"]).resolve(), (folder / "Model_upper_scaled.osim").resolve())
        self.assertIn('name="UpperBody_CameraFrame_Scaled"', Path(report["model_path"]).read_text(encoding="utf-8"))
        self.assertEqual(CAMERA_MODEL.read_bytes(), before)
        self.assertIn("generic model file is not loaded", report["statement"])

    def test_generic_models_cannot_be_overwritten(self) -> None:
        if not CAMERA_MODEL.is_file():
            self.skipTest("camera model is not present")
        before = CAMERA_MODEL.read_bytes()
        with self.assertRaises(ValueError):
            write_scaled_capture_model(CAMERA_MODEL, CAMERA_MODEL, Path("missing.trc"))
        self.assertEqual(CAMERA_MODEL.read_bytes(), before)
        if SOURCE_MODEL.is_file():
            source_before = SOURCE_MODEL.read_bytes()
            with self.assertRaises(ValueError):
                write_scaled_capture_model(CAMERA_MODEL, SOURCE_MODEL, Path("missing.trc"))
            self.assertEqual(SOURCE_MODEL.read_bytes(), source_before)


if __name__ == "__main__":
    unittest.main()
