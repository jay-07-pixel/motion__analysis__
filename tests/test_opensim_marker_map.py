"""Registration application and upper-body marker mapping. No IK and no TRC."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.opensim_upper.environment import OPENSIM_PYTHON, POSE2SIM_GEOMETRY, POSE2SIM_MODEL
from src.opensim_upper.frame_calibration import FrameCalibration
from src.opensim_upper.leveling import assess_static_accel
from src.opensim_upper.marker_map import (
    UnknownMarkerError,
    map_static_landmarks,
    marker_mapping,
    require_known_target,
)
from src.opensim_upper.model_copy import write_upper_model
from src.opensim_upper.model_xml import EXCLUDED_FULL_BODY_MARKERS, UPPER_MODEL_MARKERS
from src.opensim_upper.registration import RegistrationInsufficient
from src.opensim_upper.residuals import default_pose_ground_distances, same_body_residuals
from src.opensim_upper.static_pose import StaticPoseCapture
from src.opensim_upper.transform import (
    camera_to_model_points,
    load_registration,
    model_to_camera_points,
    transform_static_pose,
)
from src.opensim_upper.workflow import apply_registered_markers, device_registration_status
from src.pose2sim_live.upper_lstm import RESPONSE_MARKERS


def _registration(folder: Path, complete: bool = True) -> Path:
    samples = np.tile(np.array([0.0, -9.80665, 0.0]), (25, 1))
    calibration = FrameCalibration()
    calibration.set_measured_up(assess_static_accel(samples))
    calibration.set_anterior_point("anterior_from", (0.0, 0.0, 1.0))
    calibration.set_anterior_point("anterior_to", (0.0, 0.0, 0.0))
    if complete:
        calibration.set_origin_camera((0.2, 0.4, 1.5))
        calibration.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
    path = folder / "registration.json"
    calibration.save(path)
    return path


def _summary() -> dict:
    observed = {
        "left_shoulder": (0.20, -0.20, 1.50),
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
        predicted[name] = base
    capture = StaticPoseCapture(min_frames=2)
    capture.add_frame(0.0, observed, predicted)
    capture.add_frame(0.1, observed, predicted)
    return capture.summary(1.78, 80.0)


def _coordinates(path: Path) -> dict[str, tuple[str | None, str | None]]:
    table = {}
    for element in ET.parse(path).getroot().iter():
        if element.tag.endswith("Coordinate") and element.get("name"):
            table[element.get("name")] = (element.findtext("default_value"), element.findtext("range"))
    return table


class RegistrationLoadTest(unittest.TestCase):
    def test_saved_registration_round_trips_observed_and_predicted_points(self) -> None:
        folder = Path(tempfile.mkdtemp())
        path = _registration(folder)
        registration = load_registration(path)
        rotation = np.asarray(registration["rotation"])
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)
        self.assertEqual(registration["units"], "metres")
        self.assertEqual(registration["checks"]["translation_units"], "metres")
        summary = _summary()
        transformed = transform_static_pose(summary, registration)
        self.assertEqual(transformed["coord_frame"], "opensim_ground_metre")
        for name in ("left_shoulder", "right_wrist", "r_lelbow_study", "L_mwrist_study"):
            row = summary["landmarks"][name]
            camera = np.array([row["x_m"], row["y_m"], row["z_m"]])
            model = camera_to_model_points(camera, registration)[0]
            stored = transformed["landmarks"][name]
            self.assertTrue(np.allclose(model, [stored["x_m"], stored["y_m"], stored["z_m"]]))
            back = model_to_camera_points(model, registration)[0]
            self.assertTrue(np.allclose(back, camera, atol=1e-8))
        self.assertEqual(transformed["landmarks"]["left_shoulder"]["source"], "observed")
        self.assertEqual(transformed["landmarks"]["r_lelbow_study"]["source"], "estimated_lstm_study")
        self.assertTrue(transformed["landmarks"]["r_lelbow_study"]["estimated_not_palpated"])
        self.assertFalse(transformed["inverse_kinematics"])
        self.assertFalse(transformed["trc_written"])

    def test_incomplete_and_reflection_are_rejected_without_an_identity(self) -> None:
        folder = Path(tempfile.mkdtemp())
        partial = _registration(folder, complete=False)
        summary = _summary()
        before = summary["landmarks"]["left_elbow"]["x_m"]
        with self.assertRaises(RegistrationInsufficient):
            load_registration(partial)
        with self.assertRaises(RegistrationInsufficient):
            transform_static_pose(summary, json.loads(partial.read_text(encoding="utf-8")))
        self.assertEqual(summary["landmarks"]["left_elbow"]["x_m"], before)
        lying = {
            "complete": True,
            "units": "metres",
            "rotation": np.diag([1.0, -1.0, 1.0]).tolist(),
            "translation_m": [0.0, 0.0, 0.0],
            "y_flip_applied": True,
            "checks": {
                "accepted": True,
                "proper_rotation": True,
                "translation_units": "metres",
            },
        }
        with self.assertRaises(RegistrationInsufficient):
            camera_to_model_points([[0.0, 0.1, 0.2]], lying)
        missing = folder / "absent.json"
        with self.assertRaises(RegistrationInsufficient):
            load_registration(missing)


class MarkerMapTest(unittest.TestCase):
    def setUp(self) -> None:
        if not POSE2SIM_MODEL.is_file():
            self.skipTest("Pose2Sim model is not installed on this machine")

    def test_mapping_uses_existing_names_and_leaves_wrists_unmapped(self) -> None:
        table = marker_mapping()
        targets = [row["target"] for row in table["mapped"]]
        self.assertEqual(set(targets), set(UPPER_MODEL_MARKERS))
        self.assertEqual(len(targets), 12)
        sources = {row["source"] for row in table["mapped"]}
        self.assertTrue(set(RESPONSE_MARKERS).issubset(sources))
        self.assertNotIn("right_wrist", sources)
        self.assertNotIn("left_wrist", sources)
        unmapped = {row["source"]: row["target"] for row in table["unmapped_observed"]}
        self.assertIsNone(unmapped["right_wrist"])
        self.assertIsNone(unmapped["left_wrist"])
        for name in EXCLUDED_FULL_BODY_MARKERS:
            self.assertNotIn(name, targets)
            self.assertIn(name, table["not_added"])
        self.assertIn("RWrist", table["not_added"])
        self.assertFalse(table["placement_adjusted"])
        self.assertFalse(table["full_body_ik_configuration_used"])
        for row in table["mapped"]:
            self.assertEqual(row["placement_method"], "copied_unchanged_from_Markers_LSTM_xml")
            self.assertEqual(len(row["location_m"]), 3)
            if row["source_type"] == "predicted":
                self.assertTrue(row["estimated_not_palpated"])
                self.assertEqual(row["marker_weight"], 5.0)
            else:
                self.assertIsNone(row["marker_weight"])
                self.assertFalse(row["official_ik_task_present"])
        elbow = require_known_target("r_lelbow_study", table)
        self.assertEqual(elbow["body"], "humerus_r")
        self.assertEqual(elbow["source"], "r_lelbow_study")
        with self.assertRaises(UnknownMarkerError):
            require_known_target("RWrist", table)
        with self.assertRaises(UnknownMarkerError):
            require_known_target("C7_study", table)

    def test_transformed_pose_maps_estimates_and_rejects_unknown_names(self) -> None:
        folder = Path(tempfile.mkdtemp())
        registration = load_registration(_registration(folder))
        transformed = transform_static_pose(_summary(), registration)
        mapped = map_static_landmarks(transformed)
        by_source = {row["source"]: row for row in mapped["assignments"]}
        self.assertEqual(by_source["right_shoulder"]["target"], "RShoulder")
        self.assertEqual(by_source["r_lwrist_study"]["target"], "r_lwrist_study")
        self.assertFalse(by_source["right_wrist"]["mapped"])
        self.assertNotEqual(by_source["right_wrist"]["model_m"], by_source["r_lwrist_study"]["model_m"])
        self.assertEqual(mapped["n_mapped"], 12)
        self.assertEqual(mapped["n_unmapped"], 2)
        self.assertFalse(mapped["inverse_kinematics"])
        extra = json.loads(json.dumps(transformed))
        extra["landmarks"]["C7_study"] = {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "source": "observed",
        }
        with self.assertRaises(UnknownMarkerError):
            map_static_landmarks(extra)

    def test_model_copy_preserves_the_original_and_its_joints(self) -> None:
        folder = Path(tempfile.mkdtemp())
        dest = folder / "Model_Pose2Sim_upper.osim"
        before = hashlib.sha256(POSE2SIM_MODEL.read_bytes()).hexdigest()
        geometry_before = len(list(POSE2SIM_GEOMETRY.glob("*")))
        note = write_upper_model(dest)
        self.assertEqual(hashlib.sha256(POSE2SIM_MODEL.read_bytes()).hexdigest(), before)
        self.assertEqual(len(list(POSE2SIM_GEOMETRY.glob("*"))), geometry_before)
        self.assertFalse(note["geometry_copied"])
        self.assertFalse(note["joints_modified"])
        self.assertFalse(note["locations_adjusted"])
        self.assertEqual(_coordinates(POSE2SIM_MODEL), _coordinates(dest))
        copy_markers = {}
        for element in ET.parse(dest).getroot().iter():
            if element.tag.endswith("Marker") and element.get("name"):
                location = (element.findtext("location") or "").split()
                copy_markers[element.get("name")] = location
        self.assertEqual(set(copy_markers), set(UPPER_MODEL_MARKERS))
        source_markers = {}
        markers_path = POSE2SIM_MODEL.parent / "Markers_LSTM.xml"
        for element in ET.parse(markers_path).getroot().iter():
            if element.tag.endswith("Marker") and element.get("name") in copy_markers:
                source_markers[element.get("name")] = (element.findtext("location") or "").split()
        self.assertEqual(copy_markers, source_markers)
        original_markers = [
            element.get("name")
            for element in ET.parse(POSE2SIM_MODEL).getroot().iter()
            if element.tag.endswith("Marker") and element.get("name")
        ]
        self.assertEqual(original_markers, [])
        with self.assertRaises(ValueError):
            write_upper_model(POSE2SIM_MODEL)

    def test_same_body_residual_does_not_move_marker_locations(self) -> None:
        table = marker_mapping()
        assignments = []
        for row in table["mapped"]:
            location = list(row["location_m"])
            assignments.append({**row, "mapped": True, "model_m": list(location)})
        before = [list(row["location_m"]) for row in assignments]
        report = same_body_residuals(assignments)
        self.assertEqual([list(row["location_m"]) for row in assignments], before)
        self.assertGreater(report["n_pairs"], 0)
        self.assertEqual(report["n_mismatch"], 0)
        self.assertFalse(report["mismatch_flag_moves_markers"])
        for pair in report["pairs"]:
            self.assertAlmostEqual(pair["residual_m"], 0.0, places=8)
        shifted = json.loads(json.dumps(assignments))
        for row in shifted:
            if row["target"] == "r_melbow_study":
                origin = next(item for item in shifted if item["target"] == "r_lelbow_study")
                direction = np.asarray(row["model_m"]) - np.asarray(origin["model_m"])
                direction = direction / np.linalg.norm(direction)
                row["model_m"] = (np.asarray(row["model_m"]) + 0.05 * direction).tolist()
        mismatched = same_body_residuals(shifted)
        elbow = next(
            pair
            for pair in mismatched["pairs"]
            if pair["marker_a"] == "r_lelbow_study" and pair["marker_b"] == "r_melbow_study"
        )
        self.assertGreater(elbow["abs_residual_m"], 0.0)
        self.assertTrue(elbow["mismatch"])
        self.assertEqual(shifted[0]["location_m"], assignments[0]["location_m"])
        ground = {
            row["target"]: [value + 0.1 for value in row["model_m"]]
            for row in assignments
        }
        distances = default_pose_ground_distances(assignments, ground)
        self.assertTrue(all(abs(row["distance_m"] - (0.1 * 3 ** 0.5)) < 1e-8 for row in distances["rows"]))
        self.assertFalse(distances["markers_moved"])
        self.assertEqual(assignments[0]["location_m"], before[0])

    def test_workflow_writes_the_mapped_pose_without_ik(self) -> None:
        folder = Path(tempfile.mkdtemp())
        registration = _registration(folder)
        dest = folder / "model" / "Model_Pose2Sim_upper.osim"
        payload = apply_registered_markers(
            _summary(),
            registration,
            folder / "out",
            dest,
            project_root=folder,
        )
        self.assertFalse(payload["device_calibration"]["end_to_end_device_calibration"])
        self.assertFalse(payload["inverse_kinematics"])
        self.assertFalse(payload["trc_written"])
        self.assertFalse(payload["placement_adjusted"])
        self.assertEqual(payload["assignments"]["n_mapped"], 12)
        self.assertTrue((folder / "out" / "registered_markers.json").is_file())
        self.assertTrue((folder / "out" / "registered_landmarks.png").is_file())
        self.assertTrue(dest.is_file())
        status = device_registration_status(folder)
        self.assertFalse(status["end_to_end_device_calibration"])


class OpenSimCopyLoadTest(unittest.TestCase):
    def test_isolated_opensim_reads_twelve_markers(self) -> None:
        if not OPENSIM_PYTHON.is_file() or not POSE2SIM_MODEL.is_file():
            self.skipTest("Isolated OpenSim or the Pose2Sim model is not installed")
        import subprocess

        folder = Path(tempfile.mkdtemp())
        dest = folder / "Model_Pose2Sim_upper.osim"
        write_upper_model(dest)
        result_path = folder / "load.json"
        script = """
import json, sys
import opensim
copy_path, original, result_path = sys.argv[1], sys.argv[2], sys.argv[3]
model = opensim.Model(copy_path)
model.initSystem()
names = [model.getMarkerSet().get(i).getName() for i in range(model.getMarkerSet().getSize())]
original_model = opensim.Model(original)
original_model.initSystem()
Path = __import__("pathlib").Path
Path(result_path).write_text(json.dumps({
    "copy_markers": names,
    "original_n_markers": original_model.getMarkerSet().getSize(),
    "n_coordinates": model.getCoordinateSet().getSize(),
    "n_bodies": model.getBodySet().getSize(),
}), encoding="utf-8")
"""
        completed = subprocess.run(
            [str(OPENSIM_PYTHON), "-c", script, str(dest), str(POSE2SIM_MODEL), str(result_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["original_n_markers"], 0)
        self.assertEqual(set(payload["copy_markers"]), set(UPPER_MODEL_MARKERS))
        self.assertEqual(payload["n_bodies"], 22)
        self.assertEqual(payload["n_coordinates"], 40)


if __name__ == "__main__":
    unittest.main()
