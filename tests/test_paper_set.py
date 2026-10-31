"""Paper (Bousigues) landmark correspondences. No camera. No invented XYZ."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.diagnostics.recorder import LiveDiagnosticsRecorder
from src.markers.paper_set import (
    BY_SOURCE,
    NOT_PAPER_ALIASES,
    PAPER_LANDMARKS,
    annotate_overlay_records,
    assert_no_forbidden_aliases,
    paper_label_for_source,
    resolve_paper_landmarks,
)
from src.pose.keypoints import Keypoint2D


class MappingTest(unittest.TestCase):
    def test_no_mediapipe_or_toe_aliases(self) -> None:
        assert_no_forbidden_aliases()
        for spec in PAPER_LANDMARKS:
            if spec.source_name:
                self.assertNotIn(spec.source_name, NOT_PAPER_ALIASES)

    def test_sat_and_ghjc_are_unavailable(self) -> None:
        for name in ("L_SAT", "R_SAT", "L_GHJC", "R_GHJC"):
            spec = next(s for s in PAPER_LANDMARKS if s.paper_name == name)
            self.assertEqual(spec.method, "unavailable")
            self.assertIsNone(spec.source_name)

    def test_estimated_sources_are_lstm_study_names(self) -> None:
        self.assertEqual(paper_label_for_source("L_melbow_study"), "L_H-ME")
        self.assertEqual(paper_label_for_source("L_lelbow_study"), "L_H-LE")
        self.assertEqual(paper_label_for_source("r_lwrist_study"), "R_RS")
        self.assertEqual(paper_label_for_source("r_mwrist_study"), "R_US")
        self.assertEqual(paper_label_for_source("L.ASIS_study"), "L_ASIS")
        self.assertEqual(paper_label_for_source("r.PSIS_study"), "R_PSIS")
        self.assertEqual(paper_label_for_source("L_mknee_study"), "L_M-FE")
        self.assertEqual(paper_label_for_source("L_knee_study"), "L_L-FE")
        self.assertEqual(paper_label_for_source("r_calc_study"), "R_FCC")
        self.assertEqual(paper_label_for_source("L_5meta_study"), "L_FM5")
        self.assertEqual(paper_label_for_source("r_toe_study"), "R_FM1")
        self.assertIsNone(paper_label_for_source("left_elbow"))
        self.assertIsNone(paper_label_for_source("left_hip"))
        self.assertIsNone(paper_label_for_source("left_heel"))
        self.assertIsNone(paper_label_for_source("left_foot_index"))
        self.assertIsNone(paper_label_for_source("LBigToe"))
        self.assertIsNone(paper_label_for_source("L_shoulder_study"))

    def test_resolve_does_not_copy_mediapipe_joints(self) -> None:
        rows = resolve_paper_landmarks(
            {
                "left_elbow": (0.1, 0.2, 1.0),
                "left_shoulder": (0.0, 0.0, 1.0),
                "left_hip": (0.05, 0.4, 1.1),
                "left_heel": (0.1, 1.2, 1.1),
                "left_foot_index": (0.12, 1.18, 1.0),
                "LBigToe": (0.13, 1.17, 1.0),
                "L_melbow_study": (0.2, 0.25, 1.02),
            }
        )
        by = {r["paper_name"]: r for r in rows}
        self.assertFalse(by["L_SAT"]["present"])
        self.assertIsNone(by["L_SAT"]["x_m"])
        self.assertFalse(by["L_GHJC"]["present"])
        self.assertFalse(by["L_ASIS"]["present"])
        self.assertFalse(by["L_FCC"]["present"])
        self.assertFalse(by["L_FM1"]["present"])
        self.assertTrue(by["L_H-ME"]["present"])
        self.assertAlmostEqual(by["L_H-ME"]["x_m"], 0.2)
        self.assertEqual(by["L_H-ME"]["source_landmark"], "L_melbow_study")
        self.assertFalse(by["L_H-LE"]["present"])

    def test_annotate_does_not_duplicate_xyz_rows(self) -> None:
        recs = annotate_overlay_records(
            [
                {
                    "name": "L_melbow_study",
                    "x_m": 0.2,
                    "y_m": 0.1,
                    "z_m": 1.0,
                    "class": "predicted_upper",
                },
                {
                    "name": "LBigToe",
                    "x_m": 0.1,
                    "y_m": 1.2,
                    "z_m": 1.0,
                    "class": "calibrated_toe",
                },
            ]
        )
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["paper_name"], "L_H-ME")
        self.assertEqual(recs[0]["anatomical_status"], "estimated")
        self.assertEqual(recs[0]["name"], "L_melbow_study")
        self.assertEqual(recs[1]["paper_name"], "")
        names = {spec.source_name for spec in PAPER_LANDMARKS if spec.source_name}
        self.assertEqual(len(names), len(BY_SOURCE))


class DiagnosticsPaperTest(unittest.TestCase):
    def test_paper_csv_records_estimated_and_blank_unavailable(self) -> None:
        folder = Path(tempfile.mkdtemp()) / "diag"
        rec = LiveDiagnosticsRecorder(folder)
        rec.write_frame(
            0,
            1.0,
            10.0,
            8.0,
            [
                Keypoint2D(
                    name="left_elbow",
                    u_px=1,
                    v_px=2,
                    confidence=0.9,
                    coord_frame="camera_3d_metre",
                    x_m=0.1,
                    y_m=0.2,
                    z_m=1.0,
                )
            ],
            [
                {
                    "name": "L_lelbow_study",
                    "x_m": 0.3,
                    "y_m": 0.4,
                    "z_m": 1.1,
                    "class": "predicted_upper",
                    "paper_name": "L_H-LE",
                    "anatomical_status": "estimated",
                    "visibility": 1.0,
                    "t_pred_input_sec": 1.0,
                    "t_pred_done_sec": 1.02,
                    "infer_latency_ms": 20.0,
                }
            ],
        )
        rec.close()
        with rec.paper_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        by = {r["paper_name"]: r for r in rows}
        self.assertEqual(by["L_H-LE"]["present"], "1")
        self.assertAlmostEqual(float(by["L_H-LE"]["raw_x_m"]), 0.3)
        self.assertEqual(by["L_SAT"]["present"], "0")
        self.assertEqual(by["L_SAT"]["raw_x_m"], "")
        self.assertEqual(by["L_H-ME"]["present"], "0")
        payload = json.loads((rec.dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["paper_set"]["L_H-LE"]["n_frames_present"], 1)
        self.assertEqual(payload["paper_set"]["L_SAT"]["method"], "unavailable")
        self.assertEqual(payload["paper_set"]["L_SAT"]["n_frames_present"], 0)


if __name__ == "__main__":
    unittest.main()
