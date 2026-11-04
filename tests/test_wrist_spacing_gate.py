"""Wrist study markers stay out of IK until spacing and forearm roll match."""

from __future__ import annotations

import unittest

from src.opensim_upper.camera_model import CAMERA_MODEL
from src.opensim_upper.wrist_gate import (
    WRIST_STUDY_PAIRS,
    gate_wrist_frames,
    model_wrist_elbow_roll_deg,
    model_wrist_pair_distances,
)


def _frame(time_sec: float, points: dict) -> dict:
    return {"frame": 1, "time_sec": time_sec, "points": points}


class WristSpacingGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.distances = {
            ("r_lwrist_study", "r_mwrist_study"): 0.0754,
            ("L_lwrist_study", "L_mwrist_study"): 0.0754,
        }
        self.rolls = {
            ("r_lwrist_study", "r_mwrist_study"): 7.5,
            ("L_lwrist_study", "L_mwrist_study"): 7.5,
        }

    def test_camera_model_has_both_wrist_pairs(self) -> None:
        distances = model_wrist_pair_distances(CAMERA_MODEL)
        rolls = model_wrist_elbow_roll_deg(CAMERA_MODEL)
        self.assertEqual(set(distances), set(WRIST_STUDY_PAIRS))
        self.assertEqual(set(rolls), set(WRIST_STUDY_PAIRS))
        for distance in distances.values():
            self.assertAlmostEqual(distance, 0.0754, places=3)
        for roll in rolls.values():
            self.assertLess(roll, 15.0)

    def test_a_pair_near_the_model_spacing_is_copied_unchanged(self) -> None:
        # Elbow and wrist ML nearly parallel (model-like roll), good spacing.
        right = (0.10, 0.20, 1.50)
        medial = (0.10, 0.20, 1.50 - 0.0754)
        original = _frame(
            1.0,
            {
                "r_lelbow_study": (0.10, 0.10, 1.20),
                "r_melbow_study": (0.10, 0.10, 1.20 - 0.091),
                "r_lwrist_study": right,
                "r_mwrist_study": medial,
                "L_lwrist_study": (0.0, 0.0, 1.0),
                "L_mwrist_study": (0.01, 0.0, 1.0),
            },
        )
        gated, summary = gate_wrist_frames(
            [original], self.distances, pair_roll_deg=self.rolls
        )
        self.assertEqual(gated[0]["points"]["r_lwrist_study"], right)
        self.assertEqual(gated[0]["points"]["r_mwrist_study"], medial)
        self.assertIsNone(gated[0]["points"]["L_lwrist_study"])
        self.assertIsNone(gated[0]["points"]["L_mwrist_study"])
        self.assertEqual(original["points"]["L_lwrist_study"], (0.0, 0.0, 1.0))
        self.assertEqual(summary["pairs"][0]["n_kept"], 1)
        self.assertEqual(summary["pairs"][1]["n_kept"], 0)
        self.assertTrue(summary["wrist_markers_gated"])
        self.assertFalse(summary["coordinates_moved"])
        self.assertFalse(summary["clinical_accuracy"])

    def test_a_missing_partner_removes_the_remaining_point(self) -> None:
        original = _frame(
            1.0,
            {
                "r_lwrist_study": (0.1, 0.2, 1.5),
                "r_mwrist_study": None,
                "L_lwrist_study": None,
                "L_mwrist_study": None,
            },
        )
        gated, summary = gate_wrist_frames([original], self.distances)
        self.assertIsNone(gated[0]["points"]["r_lwrist_study"])
        self.assertEqual(original["points"]["r_lwrist_study"], (0.1, 0.2, 1.5))
        self.assertEqual(summary["pairs"][0]["n_both_present"], 0)

    def test_the_boundary_of_two_centimetres_stays_in_the_fit(self) -> None:
        points = {
            "r_lelbow_study": (0.0, 0.1, 0.0),
            "r_melbow_study": (0.091, 0.1, 0.0),
            "r_lwrist_study": (0.0, 0.0, 0.0),
            "r_mwrist_study": (0.056, 0.0, 0.0),
            "L_lelbow_study": (0.0, 0.1, 0.0),
            "L_melbow_study": (0.091, 0.1, 0.0),
            "L_lwrist_study": (0.0, 0.0, 0.0),
            "L_mwrist_study": (0.054, 0.0, 0.0),
        }
        gated, summary = gate_wrist_frames(
            [_frame(0.0, points)], self.distances, pair_roll_deg=self.rolls
        )
        self.assertIsNotNone(gated[0]["points"]["r_lwrist_study"])
        self.assertEqual(summary["pairs"][0]["n_kept"], 1)
        self.assertIsNone(gated[0]["points"]["L_lwrist_study"])
        self.assertEqual(summary["pairs"][1]["n_kept"], 0)

    def test_a_ninety_degree_forearm_roll_is_left_out(self) -> None:
        # Good wrist width, but wrist ML is perpendicular to elbow ML around forearm.
        points = {
            "r_lelbow_study": (0.0, 0.0, 0.0),
            "r_melbow_study": (0.091, 0.0, 0.0),
            "r_lwrist_study": (0.0, 0.3, 0.0),
            "r_mwrist_study": (0.0, 0.3, 0.0754),
            "L_lelbow_study": None,
            "L_melbow_study": None,
            "L_lwrist_study": None,
            "L_mwrist_study": None,
        }
        gated, summary = gate_wrist_frames(
            [_frame(0.0, points)], self.distances, pair_roll_deg=self.rolls
        )
        self.assertIsNone(gated[0]["points"]["r_lwrist_study"])
        self.assertIsNone(gated[0]["points"]["r_mwrist_study"])
        self.assertEqual(summary["pairs"][0]["n_left_out_roll"], 1)
        self.assertEqual(summary["pairs"][0]["n_kept"], 0)
        self.assertIn("roll", summary["statement"].lower())
