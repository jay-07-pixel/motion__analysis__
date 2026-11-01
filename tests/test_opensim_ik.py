"""Upper-body IK configuration and a labelled synthetic solve. Not a clinical result."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.opensim_upper.environment import OPENSIM_PYTHON, POSE2SIM_MODEL
from src.opensim_upper.frame_calibration import FrameCalibration
from src.opensim_upper.ik import UPPER_MODEL, run_upper_body_ik
from src.opensim_upper.ik_fit import marker_spacing, per_marker_residuals
from src.opensim_upper.ik_tasks import (
    FORBIDDEN_MARKERS,
    LOCKED,
    SOLVED,
    coordinate_decisions,
    ik_tasks,
    marker_weights,
    validate_trc_for_ik,
)
from src.opensim_upper.leveling import assess_static_accel
from src.opensim_upper.trial import write_synthetic_trial


def _registration(folder: Path) -> Path:
    samples = np.tile(np.array([0.0, -9.80665, 0.0]), (25, 1))
    calibration = FrameCalibration()
    calibration.set_measured_up(assess_static_accel(samples))
    calibration.set_anterior_point("anterior_from", (0.0, 0.0, 1.0))
    calibration.set_anterior_point("anterior_to", (0.0, 0.0, 0.0))
    calibration.set_origin_camera((0.2, 0.4, 1.5))
    calibration.set_origin_model((1.0, 2.0, 3.0), "tape cross on the stool, surveyed")
    path = folder / "registration.json"
    calibration.save(path)
    return path


class IkPlanTest(unittest.TestCase):
    def test_every_coordinate_is_classified_and_unsupported_ones_are_locked(self) -> None:
        decisions = coordinate_decisions()
        self.assertEqual(len(decisions), 40)
        self.assertEqual(set(LOCKED) | set(SOLVED), set(decisions))
        for name in ("pelvis_ty", "knee_angle_r", "L5_S1_Flex_Ext", "neck_flexion", "wrist_flex_r"):
            self.assertTrue(decisions[name]["locked"])
            self.assertEqual(decisions[name]["value_source"], "model_default")
            self.assertNotIn("0.94", decisions[name]["reason"])
        for name in ("arm_flex_r", "arm_add_l", "arm_rot_r", "elbow_flex_r", "elbow_flex_l", "pro_sup_l"):
            self.assertTrue(decisions[name]["solved"])
            self.assertFalse(decisions[name]["locked"])
            self.assertNotIn("0.94", decisions[name]["reason"])
        with self.assertRaises(ValueError):
            coordinate_decisions(["not_a_coordinate"])

    def test_tasks_use_the_twelve_markers_and_configurable_weights(self) -> None:
        if not POSE2SIM_MODEL.is_file():
            self.skipTest("Pose2Sim model is not installed on this machine")
        plan = ik_tasks()
        names = [task["name"] for task in plan["tasks"]]
        self.assertEqual(len(names), 12)
        self.assertNotIn("RWrist", names)
        self.assertNotIn("C7_study", names)
        self.assertTrue(all(not task["glenohumeral_centre"] for task in plan["tasks"]))
        self.assertFalse(plan["clinical_accuracy"])
        self.assertFalse(plan["full_body_ik_configuration_used"])
        study = next(task for task in plan["tasks"] if task["name"] == "r_lelbow_study")
        shoulder = next(task for task in plan["tasks"] if task["name"] == "RShoulder")
        self.assertEqual(study["weight"], 5.0)
        self.assertTrue(study["official_reference"])
        self.assertEqual(shoulder["weight"], 5.0)
        self.assertFalse(shoulder["official_reference"])
        overridden = marker_weights({"RShoulder": 1.5, "r_lelbow_study": 2.0})
        self.assertEqual(overridden["RShoulder"]["weight"], 1.5)
        self.assertEqual(overridden["RShoulder"]["source"], "caller override")
        self.assertEqual(overridden["r_lelbow_study"]["weight"], 2.0)
        with self.assertRaises(ValueError):
            marker_weights({"C7_study": 5.0})
        for name in FORBIDDEN_MARKERS:
            self.assertNotIn(name, names)

    def test_trc_must_match_the_marker_map(self) -> None:
        if not POSE2SIM_MODEL.is_file():
            self.skipTest("Pose2Sim model is not installed on this machine")
        folder = Path(tempfile.mkdtemp())
        trial = folder / "trial"
        write_synthetic_trial(_registration(folder), trial, "synthetic-ik", n_frames=4, project_root=folder)
        parsed = validate_trc_for_ik(trial / "markers.trc")
        self.assertEqual(parsed["num_markers"], 12)
        text = (trial / "markers.trc").read_text(encoding="utf-8").replace("RShoulder", "C7_study", 1)
        bad = folder / "bad.trc"
        bad.write_text(text, encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_trc_for_ik(bad)


class IkExecutionTest(unittest.TestCase):
    def test_synthetic_ik_runs_without_claiming_a_biomechanical_result(self) -> None:
        if not OPENSIM_PYTHON.is_file() or not UPPER_MODEL.is_file() or not POSE2SIM_MODEL.is_file():
            self.skipTest("OpenSim or the upper-body model copy is not installed")
        folder = Path(tempfile.mkdtemp())
        original = UPPER_MODEL.read_bytes()
        pose2sim = POSE2SIM_MODEL.read_bytes()
        trial = folder / "trial"
        write_synthetic_trial(_registration(folder), trial, "synthetic-ik", n_frames=4, project_root=folder)
        # One marker missing on every frame must be dropped from the solve, not filled in.
        trc = trial / "markers.trc"
        lines = trc.read_text(encoding="utf-8").split("\n")
        rewritten = []
        for line in lines:
            cells = line.split("\t")
            if cells and cells[0].isdigit() and len(cells) >= 38:
                # LElbow is marker index 3: columns 2+9, 2+10, 2+11.
                cells[11] = "nan"
                cells[12] = "nan"
                cells[13] = "nan"
                line = "\t".join(cells)
            rewritten.append(line)
        trc.write_text("\n".join(rewritten), encoding="utf-8")
        report = run_upper_body_ik(
            trc,
            folder / "ik",
            trial_metadata=trial / "trial_metadata.json",
        )
        self.assertTrue((folder / "ik" / "ik_results.mot").is_file())
        self.assertTrue((folder / "ik" / "ik_setup.xml").is_file())
        self.assertTrue((folder / "ik" / "ik_report.json").is_file())
        self.assertTrue((folder / "ik" / "ik_logs.txt").is_file())
        self.assertTrue(report["execution_success"])
        self.assertFalse(report["biomechanical_reliability"])
        self.assertFalse(report["clinical_accuracy"])
        self.assertTrue(report["synthetic"])
        self.assertFalse(report["participant_capture"])
        self.assertEqual(report["n_frames"], 4)
        self.assertTrue(report["timestamps_match_trc"])
        self.assertEqual(report["n_coordinates"], 40)
        self.assertEqual(report["mot_file_angular_unit"], "degrees")
        self.assertEqual(report["reported_angular_unit"], "radians")
        self.assertIn("LElbow", report["markers_absent_on_every_frame"])
        self.assertFalse(next(task["apply"] for task in report["marker_tasks"] if task["name"] == "LElbow"))
        self.assertNotIn("C7_study", [task["name"] for task in report["marker_tasks"]])
        for name in ("pelvis_ty", "wrist_flex_r", "knee_angle_l"):
            row = report["coordinates"][name]
            self.assertTrue(row["locked"])
            self.assertTrue(row["unchanged_from_model_default"])
            self.assertEqual(row["value_source"], "model_default")
        for name in ("arm_rot_r", "pro_sup_r", "arm_flex_r"):
            row = report["coordinates"][name]
            self.assertTrue(row["solved"])
            self.assertTrue(row["clamped_to_model_range"])
            self.assertFalse(row["outside_model_range"])
        elbow = report["coordinates"]["elbow_flex_r"]
        self.assertTrue(elbow["solved"])
        self.assertEqual(elbow["unit"], "radians")
        self.assertFalse(report["model_file_modified"])
        self.assertEqual(UPPER_MODEL.read_bytes(), original)
        self.assertEqual(POSE2SIM_MODEL.read_bytes(), pose2sim)
        setup = (folder / "ik" / "ik_setup.xml").read_text(encoding="utf-8")
        self.assertIn("r_lelbow_study", setup)
        self.assertNotIn("C7_study", setup)
        saved = json.loads((folder / "ik" / "ik_report.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["biomechanical_reliability"])
        self.assertTrue(any("marker_error_RMS" in reason for reason in report["reliability_reasons"]))
        self.assertFalse(report["sampled_from_opensim_model"])
        self.assertFalse(report["geometry"]["consistent_with_model_marker_spacing"])
        self.assertEqual(report["geometry"]["n_mismatch"], report["geometry"]["n_pairs"])
        self.assertGreater(report["geometry"]["n_pairs"], 0)
        self.assertEqual(len(report["marker_fit"]["markers"]), 12)
        self.assertGreater(report["marker_fit"]["markers"]["RShoulder"]["rms_m"], 1.0)
        self.assertIn("registration does not change these distances", " ".join(report["reliability_reasons"]))
        arm_add = report["coordinates"]["arm_add_r"]
        self.assertAlmostEqual(arm_add["min_result"], arm_add["range"][0], places=3)
        self.assertAlmostEqual(arm_add["max_result"], arm_add["range"][0], places=3)
        self.assertIn("arm_add_r remains at the model range minimum", " ".join(report["reliability_reasons"]))
        self.assertIn("The range was not widened", " ".join(report["reliability_reasons"]))


class ModelPoseRecoveryTest(unittest.TestCase):
    def test_model_sampled_markers_recover_shoulder_elbow_and_forearm(self) -> None:
        if not OPENSIM_PYTHON.is_file() or not UPPER_MODEL.is_file():
            self.skipTest("OpenSim or the upper-body model copy is not installed")
        folder = Path(tempfile.mkdtemp())
        tool = Path(__file__).resolve().parents[1] / "tools" / "opensim_ik_observe.py"
        completed = subprocess.run(
            [str(OPENSIM_PYTHON), str(tool), str(folder)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(tool.parents[1]),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        summary = json.loads((folder / "observability.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["synthetic"])
        self.assertFalse(summary["participant_capture"])
        self.assertFalse(summary["clinical_accuracy"])
        self.assertEqual(summary["wrist_coordinate_max_marker_shift_m"], 0.0)
        for name, row in summary["arm_only_recovery_rad"].items():
            self.assertLess(row["max_abs_rad"], 0.01, name)
        for name in ("arm_rot_r", "arm_rot_l", "pro_sup_r", "pro_sup_l"):
            self.assertLess(summary["axial_free_recovery_rad"][name]["max_abs_rad"], 0.02, name)
        self.assertGreater(summary["axial_locked_shoulder_elbow_rad"]["arm_flex_r"]["max_abs_rad"], 0.05)
        self.assertGreater(summary["axial_locked_rms_m"], 0.01)
        self.assertLess(summary["arm_only_rms_m"], 0.001)
        self.assertLess(summary["axial_free_rms_m"], 0.001)
        self.assertTrue(summary["arm_only_spacing_consistent"])
        self.assertTrue(summary["axial_free_spacing_consistent"])
        self.assertLess(summary["arm_only_largest_marker_rms_m"], summary["documented_tolerances"]["per_marker_rms_m"])
        self.assertLess(summary["axial_free_largest_marker_rms_m"], summary["documented_tolerances"]["per_marker_rms_m"])
        self.assertIn("Not a participant or clinical tolerance", summary["documented_tolerances"]["meaning"])


class CameraCloudSpacingTest(unittest.TestCase):
    def test_camera_line_is_not_a_model_pose(self) -> None:
        if not POSE2SIM_MODEL.is_file():
            self.skipTest("Pose2Sim model is not installed on this machine")
        folder = Path(tempfile.mkdtemp())
        trial = folder / "trial"
        write_synthetic_trial(_registration(folder), trial, "synthetic-ik", n_frames=4, project_root=folder)
        from src.opensim_upper.trc import read_trc

        spacing = marker_spacing(read_trc(trial / "markers.trc"))
        self.assertFalse(spacing["consistent_with_model_marker_spacing"])
        self.assertFalse(spacing["registration_changes_these_distances"])
        self.assertEqual(spacing["n_mismatch"], spacing["n_pairs"])
        pair = spacing["largest_pair"]
        self.assertGreater(pair["abs_residual_m"], 0.5)
        fit = per_marker_residuals(
            [
                {"time_sec": 0.0, "points": {"RShoulder": (1.0, 2.0, 3.0)}},
                {"time_sec": 0.1, "points": {"RShoulder": None}},
            ],
            [
                {"time_sec": 0.0, "points": {"RShoulder": (1.0, 2.0, 4.0)}},
                {"time_sec": 0.1, "points": {"RShoulder": (0.0, 0.0, 0.0)}},
            ],
        )
        self.assertEqual(fit["markers"]["RShoulder"]["n_frames"], 1)
        self.assertAlmostEqual(fit["markers"]["RShoulder"]["rms_m"], 1.0)


if __name__ == "__main__":
    unittest.main()
