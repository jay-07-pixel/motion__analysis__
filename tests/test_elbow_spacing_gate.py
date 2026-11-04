"""Elbow study markers stay out of IK until their spacing matches the model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.opensim_upper.camera_model import CAMERA_MODEL
from src.opensim_upper.elbow_gate import (
    ELBOW_STUDY_PAIRS,
    gate_elbow_frames,
    model_pair_distances,
    write_gated_ik_trc,
)
from src.opensim_upper.model_xml import UPPER_MODEL_MARKERS
from src.opensim_upper.trc import read_trc, write_trc


def _frame(time_sec: float, points: dict) -> dict:
    return {"frame": 1, "time_sec": time_sec, "points": points}


class ElbowSpacingGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.distances = {
            ("r_lelbow_study", "r_melbow_study"): 0.091,
            ("L_lelbow_study", "L_melbow_study"): 0.091,
        }

    def test_a_pair_near_the_model_spacing_is_copied_unchanged(self) -> None:
        right = (0.10, 0.20, 1.50)
        medial = (0.10, 0.20, 1.50 - 0.091)
        original = _frame(
            1.0,
            {
                "r_lelbow_study": right,
                "r_melbow_study": medial,
                "L_lelbow_study": (0.0, 0.0, 1.0),
                "L_melbow_study": (0.024, 0.0, 1.0),
                "r_lwrist_study": (0.2, 0.2, 1.2),
                "r_mwrist_study": (0.2, 0.2, 1.21),
            },
        )
        gated, summary = gate_elbow_frames([original], self.distances)
        self.assertEqual(gated[0]["points"]["r_lelbow_study"], right)
        self.assertEqual(gated[0]["points"]["r_melbow_study"], medial)
        self.assertIsNone(gated[0]["points"]["L_lelbow_study"])
        self.assertIsNone(gated[0]["points"]["L_melbow_study"])
        # Elbow gate alone does not touch wrists.
        self.assertEqual(gated[0]["points"]["r_lwrist_study"], (0.2, 0.2, 1.2))
        self.assertEqual(gated[0]["points"]["r_mwrist_study"], (0.2, 0.2, 1.21))
        self.assertEqual(original["points"]["L_lelbow_study"], (0.0, 0.0, 1.0))
        right_row = summary["pairs"][0]
        left_row = summary["pairs"][1]
        self.assertEqual(right_row["n_kept"], 1)
        self.assertEqual(left_row["n_kept"], 0)
        self.assertEqual(left_row["n_left_out"], 1)
        self.assertFalse(summary["coordinates_moved"])
        self.assertFalse(summary["wrist_markers_gated"])
        self.assertFalse(summary["clinical_accuracy"])

    def test_a_missing_partner_removes_the_remaining_point(self) -> None:
        original = _frame(
            1.0,
            {
                "r_lelbow_study": (0.1, 0.2, 1.5),
                "r_melbow_study": None,
                "L_lelbow_study": None,
                "L_melbow_study": None,
            },
        )
        gated, summary = gate_elbow_frames([original], self.distances)
        self.assertIsNone(gated[0]["points"]["r_lelbow_study"])
        self.assertEqual(original["points"]["r_lelbow_study"], (0.1, 0.2, 1.5))
        self.assertEqual(summary["pairs"][0]["n_both_present"], 0)
        self.assertIsNone(summary["pairs"][0]["median_distance_m"])

    def test_camera_model_spacing_gates_a_narrow_pair_only(self) -> None:
        distances = model_pair_distances(CAMERA_MODEL)
        self.assertEqual(set(distances), set(ELBOW_STUDY_PAIRS))
        for distance in distances.values():
            self.assertAlmostEqual(distance, 0.0911, places=3)
        folder = Path(tempfile.mkdtemp())
        source = folder / "markers.trc"
        dest = folder / "markers_ik.trc"
        samples = {name: [(0.0, 0.0, 1.0)] for name in UPPER_MODEL_MARKERS}
        samples["r_lelbow_study"] = [(0.0, 0.0, 1.0)]
        samples["r_melbow_study"] = [(0.0, 0.0, 1.0 - distances[("r_lelbow_study", "r_melbow_study")])]
        samples["L_lelbow_study"] = [(0.1, 0.2, 1.2)]
        samples["L_melbow_study"] = [(0.1, 0.2, 1.224)]
        # Narrow wrists (~1 cm) left out by spacing. Good-width left wrists are
        # still left out here because the left elbow pair was already omitted,
        # so forearm roll cannot be checked.
        samples["r_lwrist_study"] = [(0.3, 0.4, 1.5)]
        samples["r_mwrist_study"] = [(0.3, 0.4, 1.51)]
        samples["L_lwrist_study"] = [(0.0, 0.0, 1.0)]
        samples["L_mwrist_study"] = [(0.0754, 0.0, 1.0)]
        write_trc(source, [0.5], samples, 30.0, [{"target": name} for name in UPPER_MODEL_MARKERS])
        before = source.read_bytes()
        written = write_gated_ik_trc(source, dest, CAMERA_MODEL)
        self.assertEqual(source.read_bytes(), before)
        parsed = read_trc(dest)
        self.assertIsNotNone(parsed["frames"][0]["points"]["r_lelbow_study"])
        self.assertIsNone(parsed["frames"][0]["points"]["L_lelbow_study"])
        self.assertIsNone(parsed["frames"][0]["points"]["L_melbow_study"])
        self.assertIsNone(parsed["frames"][0]["points"]["r_lwrist_study"])
        self.assertIsNone(parsed["frames"][0]["points"]["r_mwrist_study"])
        self.assertIsNone(parsed["frames"][0]["points"]["L_lwrist_study"])
        self.assertIsNone(parsed["frames"][0]["points"]["L_mwrist_study"])
        self.assertIn("were not moved", written["statement"])
        self.assertIn("wrist estimate", written["statement"].lower())
        self.assertIn("roll", written["statement"].lower())
        self.assertTrue((folder / "elbow_gate.json").is_file())
        self.assertTrue((folder / "wrist_gate.json").is_file())
        self.assertTrue(written["summary"]["wrist_markers_gated"])
        with self.assertRaises(ValueError):
            write_gated_ik_trc(source, source, CAMERA_MODEL)

    def test_the_boundary_of_two_centimetres_stays_in_the_fit(self) -> None:
        points = {
            "r_lelbow_study": (0.0, 0.0, 0.0),
            "r_melbow_study": (0.072, 0.0, 0.0),
            "L_lelbow_study": (0.0, 0.0, 0.0),
            "L_melbow_study": (0.069, 0.0, 0.0),
        }
        gated, summary = gate_elbow_frames([_frame(0.0, points)], self.distances)
        self.assertIsNotNone(gated[0]["points"]["r_lelbow_study"])
        self.assertEqual(summary["pairs"][0]["n_kept"], 1)
        self.assertIsNone(gated[0]["points"]["L_lelbow_study"])
        self.assertEqual(summary["pairs"][1]["n_kept"], 0)
        self.assertTrue(np.isfinite(summary["pairs"][1]["median_distance_m"]))
