"""Live marker diagnostics CSV and summary. No camera."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.capture.depth_util import DepthSampleInspect
from src.diagnostics.recorder import LiveDiagnosticsRecorder
from src.pose.keypoints import Keypoint2D


def obs(name: str, xyz=None, conf: float = 0.9) -> Keypoint2D:
    if xyz is None:
        return Keypoint2D(name=name, u_px=10, v_px=20, confidence=conf)
    x, y, z = xyz
    return Keypoint2D(
        name=name,
        u_px=10,
        v_px=20,
        confidence=conf,
        coord_frame="camera_3d_metre",
        x_m=x,
        y_m=y,
        z_m=z,
    )


class DiagnosticsRecorderTest(unittest.TestCase):
    def test_missing_xyz_is_blank_not_zero_and_filtered_empty(self) -> None:
        folder = Path(tempfile.mkdtemp()) / "diag"
        rec = LiveDiagnosticsRecorder(folder)
        rec.write_frame(
            0,
            0.0,
            12.5,
            30.0,
            [obs("left_wrist", (0.11, 0.22, 0.80)), obs("right_wrist")],
            [
                {
                    "name": "r_lelbow_study",
                    "x_m": 0.05,
                    "y_m": 0.06,
                    "z_m": 0.90,
                    "class": "predicted_upper",
                    "visibility": 1.0,
                    "t_pred_input_sec": 0.0,
                    "t_pred_done_sec": 0.018,
                    "infer_latency_ms": 18.0,
                }
            ],
        )
        rec.note_dropped()
        rec.write_frame(
            1,
            0.033,
            11.0,
            30.2,
            [obs("left_wrist", (0.12, 0.21, 0.81)), obs("right_wrist")],
            [
                {
                    "name": "r_lelbow_study",
                    "x_m": 0.05,
                    "y_m": 0.07,
                    "z_m": 0.91,
                    "class": "predicted_upper",
                    "visibility": 1.0,
                    "t_pred_input_sec": 0.033,
                    "t_pred_done_sec": 0.050,
                    "infer_latency_ms": 17.0,
                }
            ],
        )
        summary = rec.close()
        again = rec.close()
        self.assertEqual(again["frames"], 2)
        self.assertEqual(summary["dropped_frames"], 1)
        self.assertEqual(summary["units"], "metres")
        self.assertFalse(summary["filtered"])

        with rec.landmarks_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        missing = [r for r in rows if r["landmark"] == "right_wrist"]
        self.assertTrue(missing)
        for row in missing:
            self.assertEqual(row["raw_x_m"], "")
            self.assertEqual(row["raw_y_m"], "")
            self.assertEqual(row["raw_z_m"], "")
            self.assertEqual(row["xyz_missing"], "1")
            self.assertEqual(row["depth_valid"], "0")
            self.assertEqual(row["filt_x_m"], "")
            self.assertEqual(row["filt_y_m"], "")
            self.assertEqual(row["filt_z_m"], "")
            self.assertNotEqual(row["raw_x_m"], "0")
            self.assertNotEqual(row["raw_x_m"], "0.000000")

        valid = [r for r in rows if r["landmark"] == "left_wrist"]
        self.assertEqual(len(valid), 2)
        self.assertAlmostEqual(float(valid[0]["t_frame_sec"]), 0.0)
        self.assertAlmostEqual(float(valid[1]["t_frame_sec"]), 0.033)
        self.assertAlmostEqual(float(valid[0]["raw_x_m"]), 0.11)
        self.assertAlmostEqual(float(valid[0]["raw_z_m"]), 0.80)
        self.assertAlmostEqual(float(valid[0]["u_px"]), 10.0)
        self.assertAlmostEqual(float(valid[0]["v_px"]), 20.0)
        self.assertEqual(valid[0]["depth_sample_m"], "")
        self.assertEqual(valid[0]["depth_n_valid_in_window"], "")
        self.assertEqual(valid[0]["in_image"], "")

        pred = [r for r in rows if r["landmark"] == "r_lelbow_study"]
        self.assertEqual(pred[0]["u_px"], "")
        self.assertEqual(pred[0]["depth_sample_m"], "")
        self.assertEqual(pred[0]["t_pred_input_sec"], "0.000000")
        self.assertEqual(pred[0]["t_pred_done_sec"], "0.018000")
        self.assertNotEqual(pred[0]["t_frame_sec"], pred[0]["t_pred_done_sec"])
        self.assertTrue(pred[0]["pred_lag_ms"] != "")

        with rec.frames_path.open(encoding="utf-8", newline="") as fh:
            frames = list(csv.DictReader(fh))
        self.assertEqual(len(frames), 2)
        self.assertEqual(int(frames[1]["dropped_before"]), 1)
        self.assertAlmostEqual(float(frames[0]["process_ms"]), 12.5)
        times = [float(f["t_unix"]) for f in frames]
        self.assertGreaterEqual(times[1], times[0])

        payload = json.loads((rec.dir / "summary.json").read_text(encoding="utf-8"))
        wrist = payload["landmarks"]["left_wrist"]
        self.assertEqual(wrist["n_valid_xyz"], 2)
        self.assertIsNotNone(wrist["std_m"]["x"])
        self.assertIsNotNone(wrist["mean_frame_to_frame_m"])
        missing_stats = payload["landmarks"]["right_wrist"]
        self.assertEqual(missing_stats["n_valid_xyz"], 0)
        self.assertEqual(missing_stats["invalid_xyz_pct"], 100.0)
        self.assertTrue((rec.dir / "summary.txt").is_file())

    def test_depth_inspect_columns_in_bounds_zero_and_valid(self) -> None:
        folder = Path(tempfile.mkdtemp()) / "diag_depth"
        rec = LiveDiagnosticsRecorder(folder)
        rec.write_frame(
            0,
            0.0,
            10.0,
            8.0,
            [
                obs("left_ankle"),
                obs("left_knee", (0.01, 0.02, 1.40)),
                obs("right_ankle"),
            ],
            [],
            depth_inspect={
                "left_ankle": DepthSampleInspect(in_image=True, n_valid=0, depth_sample_m=0.0),
                "left_knee": DepthSampleInspect(in_image=True, n_valid=9, depth_sample_m=1.41),
                "right_ankle": DepthSampleInspect(in_image=False, n_valid=0, depth_sample_m=0.0),
            },
        )
        rec.close()
        with rec.landmarks_path.open(encoding="utf-8", newline="") as fh:
            rows = {r["landmark"]: r for r in csv.DictReader(fh)}
        hole = rows["left_ankle"]
        self.assertEqual(hole["xyz_missing"], "1")
        self.assertEqual(hole["raw_z_m"], "")
        self.assertEqual(hole["in_image"], "1")
        self.assertEqual(hole["depth_n_valid_in_window"], "0")
        self.assertAlmostEqual(float(hole["depth_sample_m"]), 0.0)
        self.assertAlmostEqual(float(hole["u_px"]), 10.0)
        knee = rows["left_knee"]
        self.assertEqual(knee["xyz_missing"], "0")
        self.assertEqual(knee["in_image"], "1")
        self.assertEqual(knee["depth_n_valid_in_window"], "9")
        self.assertAlmostEqual(float(knee["depth_sample_m"]), 1.41)
        oob = rows["right_ankle"]
        self.assertEqual(oob["in_image"], "0")
        self.assertEqual(oob["depth_n_valid_in_window"], "0")
        self.assertAlmostEqual(float(oob["depth_sample_m"]), 0.0)
        self.assertEqual(oob["raw_x_m"], "")

    def test_new_folder_does_not_overwrite(self) -> None:
        root = Path(tempfile.mkdtemp())
        a = LiveDiagnosticsRecorder.create_session(str(root))
        b = LiveDiagnosticsRecorder.create_session(str(root))
        self.assertNotEqual(a.dir, b.dir)
        a.close()
        b.close()
        self.assertTrue(a.dir.exists())
        self.assertTrue(b.dir.exists())


if __name__ == "__main__":
    unittest.main()
