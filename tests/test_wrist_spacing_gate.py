"""Wrist study markers stay out of IK until spacing and reachable roll match."""

from __future__ import annotations

import unittest

from src.opensim_upper.camera_model import CAMERA_MODEL
from src.opensim_upper.environment import OPENSIM_PYTHON
from src.opensim_upper.wrist_gate import (
    WRIST_STUDY_PAIRS,
    gate_wrist_frames,
    model_wrist_elbow_roll_range_deg,
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
        # Reachable ground-frame band (not the invalid mixed-local ~7.5°).
        self.roll_ranges = {
            ("r_lwrist_study", "r_mwrist_study"): (10.0, 120.0),
            ("L_lwrist_study", "L_mwrist_study"): (10.0, 120.0),
        }

    def test_camera_model_has_both_wrist_pairs(self) -> None:
        distances = model_wrist_pair_distances(CAMERA_MODEL)
        self.assertEqual(set(distances), set(WRIST_STUDY_PAIRS))
        for distance in distances.values():
            self.assertAlmostEqual(distance, 0.0754, places=3)

    @unittest.skipUnless(OPENSIM_PYTHON.is_file(), "opensim-upper interpreter required")
    def test_camera_model_reachable_roll_band_is_pose_dependent(self) -> None:
        from src.opensim_upper.wrist_gate import model_wrist_elbow_roll_range_report

        report = model_wrist_elbow_roll_range_report(CAMERA_MODEL)
        ranges = model_wrist_elbow_roll_range_deg(CAMERA_MODEL)
        self.assertEqual(set(ranges), set(WRIST_STUDY_PAIRS))
        # Left and right are independent entries with their own coordinates.
        self.assertIn("right", report["coordinates"])
        self.assertIn("left", report["coordinates"])
        self.assertNotEqual(
            report["coordinates"]["right"]["pro_sup"]["name"],
            report["coordinates"]["left"]["pro_sup"]["name"],
        )
        for pair, (low, high) in ranges.items():
            self.assertLess(low, high)
            self.assertGreater(high - low, 30.0)
            key = f"{pair[0]}|{pair[1]}"
            default_roll = report["pairs"][key]["default_pose_roll_deg"]
            # Diagnostic default roll must lie inside the swept band (not a hardcoded target).
            self.assertIsNotNone(default_roll)
            self.assertGreaterEqual(default_roll, low - 1e-6)
            self.assertLessEqual(default_roll, high + 1e-6)
            # Band endpoints must not collapse to the default alone.
            self.assertLess(low, float(default_roll) - 5.0)
            self.assertGreater(high, float(default_roll) + 5.0)

    def test_a_pair_near_the_model_spacing_is_copied_unchanged(self) -> None:
        # Good spacing; ML nearly parallel (~0°) is outside the reachable band.
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
            [original], self.distances, pair_roll_range_deg=self.roll_ranges
        )
        # Parallel ML (~0°) is left out by roll band; narrow left by spacing.
        self.assertIsNone(gated[0]["points"]["r_lwrist_study"])
        self.assertIsNone(gated[0]["points"]["L_lwrist_study"])
        self.assertEqual(original["points"]["r_lwrist_study"], right)
        self.assertEqual(summary["pairs"][0]["n_left_out_roll"], 1)
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

    def test_the_boundary_of_two_centimetres_stays_in_the_fit_with_reachable_roll(self) -> None:
        # ~90° wrist ML vs elbow ML (perpendicular) with good spacing — inside band.
        points = {
            "r_lelbow_study": (0.0, 0.0, 0.0),
            "r_melbow_study": (0.091, 0.0, 0.0),
            "r_lwrist_study": (0.0, 0.3, 0.0),
            "r_mwrist_study": (0.0, 0.3, 0.056),
            "L_lelbow_study": (0.0, 0.0, 0.0),
            "L_melbow_study": (0.091, 0.0, 0.0),
            "L_lwrist_study": (0.0, 0.3, 0.0),
            "L_mwrist_study": (0.0, 0.3, 0.054),
        }
        gated, summary = gate_wrist_frames(
            [_frame(0.0, points)], self.distances, pair_roll_range_deg=self.roll_ranges
        )
        self.assertIsNotNone(gated[0]["points"]["r_lwrist_study"])
        self.assertEqual(summary["pairs"][0]["n_kept"], 1)
        self.assertIsNone(gated[0]["points"]["L_lwrist_study"])
        self.assertEqual(summary["pairs"][1]["n_kept"], 0)

    def test_roll_outside_reachable_band_is_left_out(self) -> None:
        # Good width, ~0° roll (parallel ML) — outside 10–120° band.
        points = {
            "r_lelbow_study": (0.0, 0.0, 0.0),
            "r_melbow_study": (0.091, 0.0, 0.0),
            "r_lwrist_study": (0.0, 0.3, 0.0),
            "r_mwrist_study": (0.0754, 0.3, 0.0),
            "L_lelbow_study": None,
            "L_melbow_study": None,
            "L_lwrist_study": None,
            "L_mwrist_study": None,
        }
        gated, summary = gate_wrist_frames(
            [_frame(0.0, points)], self.distances, pair_roll_range_deg=self.roll_ranges
        )
        self.assertIsNone(gated[0]["points"]["r_lwrist_study"])
        self.assertEqual(summary["pairs"][0]["n_left_out_roll"], 1)
        self.assertEqual(summary["pairs"][0]["n_kept"], 0)
        self.assertIn("roll", summary["statement"].lower())
        self.assertIn("pro_sup", summary["statement"].lower())
