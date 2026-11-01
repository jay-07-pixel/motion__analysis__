"""Joint-angle extraction from an IK motion file. Not a clinical test."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.opensim_upper.angles import (
    extract_angles,
    radians_from_mot,
    validate_against_known,
    write_angle_analysis,
)
from src.opensim_upper.model_poses import FRAME_TIMES_SEC, with_axial_free_poses
from src.opensim_upper.mot import read_mot

PROJECT = Path(__file__).resolve().parents[1]
MODEL_MOT = PROJECT / "output" / "opensim_upper" / "ik" / "model-pose-check" / "with_axial_free" / "ik_results.mot"
MODEL_REPORT = PROJECT / "output" / "opensim_upper" / "ik" / "model-pose-check" / "with_axial_free" / "ik_report.json"


def _report(locked: tuple[str, ...] = ()) -> dict:
    coordinates = {}
    for name in (
        "arm_flex_r",
        "arm_add_r",
        "arm_rot_r",
        "elbow_flex_r",
        "pro_sup_r",
        "arm_flex_l",
        "arm_add_l",
        "arm_rot_l",
        "elbow_flex_l",
        "pro_sup_l",
    ):
        coordinates[name] = {
            "locked": name in locked,
            "solved": name not in locked,
            "identifiable": "model_cluster_recovered",
            "reason": "held for this test" if name in locked else "solved for this test",
            "motion_type": "rotational",
            "default_value": 0.0,
        }
    coordinates["wrist_flex_r"] = {
        "locked": True,
        "solved": False,
        "identifiable": "not_observable",
        "reason": "No hand marker.",
        "motion_type": "rotational",
        "default_value": 0.0,
    }
    coordinates["pelvis_ty"] = {
        "locked": True,
        "solved": False,
        "identifiable": "not_observable",
        "reason": "No pelvis marker.",
        "motion_type": "translational",
        "default_value": 0.94,
    }
    return {
        "sampled_from_opensim_model": True,
        "geometry": {"consistent_with_model_marker_spacing": True},
        "coordinates": coordinates,
        "clinical_accuracy": False,
    }


def _mot(path: Path, rows: list[str]) -> None:
    header = "\n".join(
        [
            "Coordinates",
            "inDegrees=yes",
            "endheader",
            "time\tarm_flex_r\tarm_add_r\tarm_rot_r\telbow_flex_r\tpro_sup_r\tarm_flex_l\tarm_add_l\tarm_rot_l\telbow_flex_l\tpro_sup_l\twrist_flex_r\tpelvis_ty",
            *rows,
        ]
    )
    path.write_text(header + "\n", encoding="utf-8")


class MotAndAngleTest(unittest.TestCase):
    def test_degrees_convert_and_translations_do_not(self) -> None:
        self.assertAlmostEqual(radians_from_mot(180.0, True, True), math.pi)
        self.assertAlmostEqual(radians_from_mot(0.5, False, True), 0.5)
        self.assertAlmostEqual(radians_from_mot(0.94, True, False), 0.94)

    def test_extraction_keeps_time_and_splits_sides(self) -> None:
        folder = Path(tempfile.mkdtemp())
        path = folder / "sample.mot"
        _mot(
            path,
            [
                "0\t90\t0\t0\t0\t0\t-45\t0\t0\t0\t0\t0\t0.94",
                "0.1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t12\t0.94",
            ],
        )
        motion = read_mot(path)
        self.assertTrue(motion["in_degrees"])
        self.assertEqual(motion["rows"][0]["time"], 0.0)
        self.assertEqual(motion["rows"][1]["time"], 0.1)
        extracted = extract_angles(motion, _report(("arm_rot_r",)))
        self.assertEqual(extracted["timestamps_sec"], [0.0, 0.1])
        self.assertAlmostEqual(extracted["frames"][0]["solved"]["arm_flex_r"]["radians"], math.pi / 2)
        self.assertEqual(extracted["frames"][0]["solved"]["arm_flex_r"]["side"], "right")
        self.assertEqual(extracted["frames"][0]["solved"]["arm_flex_l"]["side"], "left")
        self.assertNotIn("arm_rot_r", extracted["frames"][0]["solved"])
        self.assertNotIn("wrist_flex_r", extracted["frames"][0]["solved"])
        self.assertNotIn("pelvis_ty", extracted["frames"][0]["solved"])
        locked = {row["coordinate"]: row for row in extracted["locked_not_measured"]}
        self.assertFalse(locked["wrist_flex_r"]["reported_as_measurement"])
        self.assertEqual(locked["wrist_flex_r"]["status"], "unsupported")
        self.assertFalse(locked["arm_rot_r"]["reported_as_measurement"])
        self.assertEqual(locked["arm_rot_r"]["status"], "locked")

    def test_validation_metrics_and_refusal_of_a_non_model_trial(self) -> None:
        extracted = extract_angles(
            {
                "in_degrees": False,
                "rows": [
                    {"time": 0.0, "arm_flex_r": 0.0, "arm_flex_l": 1.0},
                    {"time": 0.1, "arm_flex_r": 0.0, "arm_flex_l": 1.2},
                ],
            },
            _report(
                (
                    "arm_add_r",
                    "arm_rot_r",
                    "elbow_flex_r",
                    "pro_sup_r",
                    "arm_add_l",
                    "arm_rot_l",
                    "elbow_flex_l",
                    "pro_sup_l",
                )
            ),
        )
        # The second frame of arm_flex_l is 0.2 rad above the known value.
        known = [{"arm_flex_r": 0.0, "arm_flex_l": 1.0}, {"arm_flex_r": 0.0, "arm_flex_l": 1.0}]
        # extract only includes solved columns that exist. arm_flex_l is solved.
        # known must include every measured coordinate. measured are arm_flex_r and arm_flex_l.
        validation = validate_against_known(extracted, known, _report())
        self.assertEqual(validation["label"], "model_generated_numerical_validation")
        self.assertFalse(validation["clinical_accuracy"])
        self.assertFalse(validation["participant_accuracy"])
        self.assertAlmostEqual(validation["per_coordinate"]["arm_flex_l"]["mean_absolute_error_rad"], 0.1)
        self.assertAlmostEqual(validation["per_coordinate"]["arm_flex_l"]["maximum_absolute_error_rad"], 0.2)
        self.assertAlmostEqual(validation["per_coordinate"]["arm_flex_l"]["rmse_rad"], math.sqrt(0.02))
        refused = _report()
        refused["sampled_from_opensim_model"] = False
        with self.assertRaises(ValueError):
            validate_against_known(extracted, known, refused)

    def test_plots_and_files_omit_locked_coordinates(self) -> None:
        folder = Path(tempfile.mkdtemp())
        mot = folder / "sample.mot"
        _mot(
            mot,
            [
                "0\t10\t-57\t20\t60\t45\t-17\t-40\t-10\t45\t50\t5\t0.94",
                "0.1\t11\t-58\t21\t61\t46\t-16\t-41\t-9\t46\t51\t6\t0.94",
            ],
        )
        report = _report(("arm_rot_l",))
        report_path = folder / "ik_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        known = [
            {
                "arm_flex_r": math.radians(10),
                "arm_add_r": math.radians(-57),
                "arm_rot_r": math.radians(20),
                "elbow_flex_r": math.radians(60),
                "pro_sup_r": math.radians(45),
                "arm_flex_l": math.radians(-17),
                "arm_add_l": math.radians(-40),
                "elbow_flex_l": math.radians(45),
                "pro_sup_l": math.radians(50),
            },
            {
                "arm_flex_r": math.radians(11),
                "arm_add_r": math.radians(-58),
                "arm_rot_r": math.radians(21),
                "elbow_flex_r": math.radians(61),
                "pro_sup_r": math.radians(46),
                "arm_flex_l": math.radians(-16),
                "arm_add_l": math.radians(-41),
                "elbow_flex_l": math.radians(46),
                "pro_sup_l": math.radians(51),
            },
        ]
        summary = write_angle_analysis(mot, report_path, folder / "out", known)
        text = (folder / "out" / "joint_angles.csv").read_text(encoding="utf-8")
        rows = list(csv.DictReader(text.splitlines()))
        self.assertEqual(sorted(set(row["time_sec"] for row in rows)), ["0.00000000", "0.10000000"])
        self.assertNotIn("wrist_flex_r", text)
        self.assertNotIn("arm_rot_l", text)
        self.assertNotIn("pelvis_ty", text)
        self.assertTrue(any(row["side"] == "right" for row in rows))
        self.assertTrue(any(row["side"] == "left" for row in rows))
        self.assertFalse(summary["clinical_accuracy"])
        for name in (
            "shoulder_right.png",
            "shoulder_left.png",
            "elbow_right.png",
            "elbow_left.png",
            "forearm.png",
            "timeline.png",
        ):
            plot = folder / "out" / "plots" / name
            self.assertTrue(plot.is_file())
            self.assertGreater(plot.stat().st_size, 1000)
        self.assertTrue((folder / "out" / "joint_angles.json").is_file())
        self.assertTrue((folder / "out" / "angle_validation.json").is_file())
        self.assertTrue((folder / "out" / "analysis_report.md").is_file())
        report_text = (folder / "out" / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("not a clinical measurement", report_text)


class ModelSampledAngleTest(unittest.TestCase):
    def test_recovered_coordinates_match_the_poses_that_placed_the_markers(self) -> None:
        if not MODEL_MOT.is_file() or not MODEL_REPORT.is_file():
            self.skipTest("Model-sampled IK result is not on disk")
        folder = Path(tempfile.mkdtemp())
        summary = write_angle_analysis(MODEL_MOT, MODEL_REPORT, folder, with_axial_free_poses())
        self.assertEqual(summary["extraction"]["timestamps_sec"], list(FRAME_TIMES_SEC))
        self.assertFalse(summary["validation"]["participant_accuracy"])
        self.assertLess(summary["validation"]["pooled"]["maximum_absolute_error_rad"], 0.02)
        self.assertNotIn("wrist_flex_r", summary["extraction"]["measured_coordinates"])
        saved = json.loads((folder / "angle_validation.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["label"], "model_generated_numerical_validation")


if __name__ == "__main__":
    unittest.main()
