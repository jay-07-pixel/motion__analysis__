"""Upper-body TRC writer. Synthetic trials are not participant captures."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from src.diagnostics.recorder import FRAME_FIELDS, LANDMARK_FIELDS
from src.opensim_upper.environment import OPENSIM_PYTHON, POSE2SIM_MODEL
from src.opensim_upper.frame_calibration import FrameCalibration
from src.opensim_upper.leveling import assess_static_accel
from src.opensim_upper.registration import RegistrationInsufficient
from src.opensim_upper.trajectory import assemble_camera_frames, linear_fill
from src.opensim_upper.trc import ordered_marker_rows, read_trc
from src.opensim_upper.trial import write_synthetic_trial, write_trial, write_trial_from_diagnostics


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


def _row(frame: int, time_sec: float, name: str, xyz, missing: bool = False) -> dict:
    row = {field: "" for field in LANDMARK_FIELDS}
    row.update(
        {
            "frame": str(frame),
            "t_frame_sec": f"{time_sec:.6f}",
            "landmark": name,
            "class": "observed" if "shoulder" in name or "elbow" in name or "wrist" in name else "predicted_upper",
            "xyz_missing": "1" if missing or xyz is None else "0",
            "raw_x_m": "" if xyz is None else f"{xyz[0]:.6f}",
            "raw_y_m": "" if xyz is None else f"{xyz[1]:.6f}",
            "raw_z_m": "" if xyz is None else f"{xyz[2]:.6f}",
        }
    )
    return row


class TrcFormatTest(unittest.TestCase):
    def setUp(self) -> None:
        if not POSE2SIM_MODEL.is_file():
            self.skipTest("Pose2Sim model is not installed on this machine")
        self.folder = Path(tempfile.mkdtemp())
        self.registration = _registration(self.folder)

    def test_synthetic_trial_header_order_and_transform(self) -> None:
        trial = self.folder / "trial"
        result = write_synthetic_trial(self.registration, trial, "synthetic-001", n_frames=4, project_root=self.folder)
        parsed = read_trc(trial / "markers.trc")
        rows = ordered_marker_rows()
        self.assertEqual(parsed["markers"], [row["target"] for row in rows])
        self.assertEqual(parsed["num_markers"], 12)
        self.assertEqual(parsed["num_frames"], 4)
        self.assertEqual(len(parsed["frames"]), 4)
        self.assertEqual([frame["frame"] for frame in parsed["frames"]], [1, 2, 3, 4])
        self.assertEqual(parsed["units"], "m")
        self.assertEqual(parsed["orig_data_start_frame"], 1)
        self.assertAlmostEqual(parsed["frames"][0]["time_sec"], 0.0)
        self.assertAlmostEqual(parsed["frames"][1]["time_sec"], 0.1)
        self.assertAlmostEqual(parsed["data_rate_hz"], 10.0)
        self.assertNotIn("RWrist", parsed["markers"])
        self.assertNotIn("right_wrist", parsed["markers"])
        self.assertIsNone(parsed["frames"][1]["points"]["RShoulder"])
        rotation = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
        translation = np.array([2.5, 2.4, 3.2])
        expected = rotation @ np.array([0.15, -0.4, 1.4]) + translation
        self.assertTrue(np.allclose(parsed["frames"][0]["points"]["RShoulder"], expected))
        metadata = json.loads((trial / "trial_metadata.json").read_text(encoding="utf-8"))
        quality = json.loads((trial / "data_quality.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["synthetic"])
        self.assertFalse(metadata["participant_capture"])
        self.assertEqual(metadata["physical_validation"], "not_performed_synthetic_trial")
        self.assertFalse(metadata["sampled_from_opensim_model"])
        self.assertIn("not positions of the OpenSim model markers", metadata["point_cloud"])
        self.assertFalse(metadata["inverse_kinematics"])
        self.assertEqual(metadata["transform"], "p_model = R @ p_camera + t")
        self.assertEqual(quality["interpolation"], "none")
        self.assertGreater(quality["n_missing_samples"], 0)
        self.assertFalse(quality["compared_to_generic_default_pose"])
        self.assertFalse(quality["spacing_vs_marker_xml"]["participant_measurement"])
        self.assertIn("markers.trc", result["trc"])
        self.assertTrue((trial / "processing.log").is_file())

    def test_incomplete_registration_writes_no_trc(self) -> None:
        partial = _registration(self.folder / "partial", complete=False)
        trial = self.folder / "no-trial"
        with self.assertRaises(RegistrationInsufficient):
            write_synthetic_trial(partial, trial, "should-not-exist", project_root=self.folder)
        self.assertFalse((trial / "markers.trc").exists())

    def test_diagnostics_drop_wrists_duplicates_and_keep_gaps(self) -> None:
        diagnostics = self.folder / "diag"
        diagnostics.mkdir()
        rows = ordered_marker_rows()
        landmark_rows = []
        for frame, time_sec in ((0, 0.0), (1, 0.0), (2, 0.2), (3, 0.1)):
            for row in rows:
                xyz = (0.1, -0.2, 1.5) if not (frame == 2 and row["target"] == "RElbow") else None
                landmark_rows.append(_row(frame, time_sec, row["source"], xyz, missing=xyz is None))
            landmark_rows.append(_row(frame, time_sec, "right_wrist", (0.2, 0.3, 1.5)))
            landmark_rows.append(_row(frame, time_sec, "LBigToe", (0.0, 0.4, 1.2)))
        landmark_rows.append(_row(2, 0.2, rows[0]["source"], (9.0, 9.0, 9.0)))
        with (diagnostics / "landmarks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LANDMARK_FIELDS)
            writer.writeheader()
            writer.writerows(landmark_rows)
        with (diagnostics / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
            writer.writeheader()
            for frame, time_sec, dropped in ((0, 0.0, 0), (1, 0.0, 1), (2, 0.2, 1), (3, 0.1, 2)):
                writer.writerow({field: "" for field in FRAME_FIELDS} | {"frame": frame, "t_frame_sec": time_sec, "dropped_before": dropped, "fps": "10"})
        camera, quality = assemble_camera_frames(landmark_rows, [
            {"frame": "0", "t_frame_sec": "0", "dropped_before": "0", "fps": "10"},
            {"frame": "1", "t_frame_sec": "0", "dropped_before": "1", "fps": "10"},
            {"frame": "2", "t_frame_sec": "0.2", "dropped_before": "1", "fps": "10"},
            {"frame": "3", "t_frame_sec": "0.1", "dropped_before": "2", "fps": "10"},
        ])
        self.assertEqual([frame["source_frame"] for frame in camera], [0, 2])
        self.assertEqual(quality["duplicate_timestamps_removed"], [1])
        self.assertEqual(quality["non_increasing_timestamps_removed"], [3])
        self.assertEqual(quality["dropped_frames_recorded"], 2)
        self.assertGreater(quality["duplicate_landmark_rows"], 0)
        self.assertGreater(quality["ignored_unmapped_rows"], 0)
        self.assertEqual(camera[1]["points"]["right_shoulder"], (0.1, -0.2, 1.5))
        self.assertIsNone(camera[1]["points"]["right_elbow"])
        self.assertNotIn("right_wrist", camera[0]["points"])
        trial = self.folder / "from-diag"
        written = write_trial_from_diagnostics(
            diagnostics,
            self.registration,
            trial,
            "diag-001",
            project_root=self.folder,
        )
        parsed = read_trc(trial / "markers.trc")
        self.assertEqual([frame["frame"] for frame in parsed["frames"]], [1, 2])
        self.assertAlmostEqual(parsed["frames"][1]["time_sec"], 0.2)
        self.assertIsNone(parsed["frames"][1]["points"]["RElbow"])
        self.assertTrue(written["metadata"]["participant_capture"])
        self.assertFalse(written["metadata"]["synthetic"])
        self.assertEqual(written["metadata"]["physical_validation"], "not_performed")
        self.assertFalse(written["metadata"]["device_calibration"]["end_to_end_device_calibration"])

    def test_optional_interpolation_keeps_the_original_gap_list(self) -> None:
        rows = ordered_marker_rows()
        frames = []
        for index, time_sec in enumerate((0.0, 0.1, 0.2)):
            points = {row["source"]: (0.1, -0.2, 1.5) for row in rows}
            if index == 1:
                points[rows[0]["source"]] = None
            frames.append({"source_frame": index, "t_sec": time_sec, "points": points})
        filled, notes = linear_fill(frames, [row["source"] for row in rows])
        self.assertEqual(len(notes), 1)
        self.assertIsNotNone(filled[1]["points"][rows[0]["source"]])
        self.assertIsNone(frames[1]["points"][rows[0]["source"]])
        trial = self.folder / "interp"
        write_trial(
            frames,
            self.registration,
            trial,
            "interp-001",
            synthetic=True,
            source_description="synthetic_test_data",
            project_root=self.folder,
            interpolate=True,
            quality={"recorded_fps_median": None},
        )
        quality = json.loads((trial / "data_quality.json").read_text(encoding="utf-8"))
        parsed = read_trc(trial / "markers.trc")
        self.assertEqual(quality["interpolation"], "linear_between_nearest_finite_samples")
        self.assertEqual(len(quality["missing_before_interpolation"]), 1)
        self.assertIsNotNone(parsed["frames"][1]["points"]["RShoulder"])

    def test_variable_rate_keeps_the_time_column(self) -> None:
        rows = ordered_marker_rows()
        frames = []
        for index, time_sec in enumerate((0.0, 0.1, 0.5)):
            points = {row["source"]: (0.1 + index * 0.01, -0.2, 1.5) for row in rows}
            frames.append({"source_frame": index, "t_sec": time_sec, "points": points})
        trial = self.folder / "variable"
        write_trial(
            frames,
            self.registration,
            trial,
            "variable-001",
            synthetic=True,
            source_description="synthetic_test_data",
            project_root=self.folder,
            quality={"recorded_fps_median": None},
        )
        parsed = read_trc(trial / "markers.trc")
        quality = json.loads((trial / "data_quality.json").read_text(encoding="utf-8"))
        self.assertTrue(quality["variable_frame_rate"])
        self.assertAlmostEqual(parsed["frames"][2]["time_sec"], 0.5)
        self.assertEqual(parsed["num_frames"], 3)


class OpenSimTrcLoadTest(unittest.TestCase):
    def test_opensim_reads_labels_frames_units_and_missing_values(self) -> None:
        if not OPENSIM_PYTHON.is_file() or not POSE2SIM_MODEL.is_file():
            self.skipTest("Isolated OpenSim or the Pose2Sim model is not installed")
        model = ROOT / "models" / "opensim_upper" / "Model_Pose2Sim_upper.osim"
        if not model.is_file():
            self.skipTest("Upper-body model copy is not present")
        import subprocess

        folder = Path(tempfile.mkdtemp())
        before = POSE2SIM_MODEL.read_bytes()
        registration = _registration(folder)
        trial = folder / "trial"
        write_synthetic_trial(registration, trial, "synthetic-osim", n_frames=4, project_root=folder)
        result_path = folder / "load.json"
        completed = subprocess.run(
            [
                str(OPENSIM_PYTHON),
                "tools/opensim_trc_check.py",
                str(model),
                str(trial / "markers.trc"),
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["n_frames"], 4)
        self.assertEqual(payload["markerdata_n_frames"], 4)
        self.assertEqual(payload["markerdata_n_markers"], 12)
        self.assertEqual(payload["units"], "m")
        self.assertEqual(payload["unrecognised_markers"], [])
        self.assertEqual(payload["missing_model_markers"], [])
        self.assertEqual(payload["nan_counts"]["RShoulder"], 1)
        self.assertFalse(payload["inverse_kinematics"])
        self.assertEqual(POSE2SIM_MODEL.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
