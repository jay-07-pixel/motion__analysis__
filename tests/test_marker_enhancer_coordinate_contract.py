"""Invertibility of camera <-> Y-up axis maps. No Marker Enhancer model."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tools.marker_enhancer_coordinate_contract import (
    PREFERRED_RECORDING,
    SYNTHETIC_HEIGHT_M,
    enumerate_y_up_maps,
    evaluate,
    height_round_trip,
    round_trip_errors,
    segment_length,
    apply_map,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CoordinateContractTests(unittest.TestCase):
    def test_exactly_four_right_handed_y_up_maps(self) -> None:
        maps = enumerate_y_up_maps()
        self.assertEqual(len(maps), 4)
        seen = set()
        for axis_map in maps:
            matrix = axis_map.matrix
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=9)
            self.assertTrue(np.allclose(matrix[1], [0.0, -1.0, 0.0]))
            seen.add(tuple(np.round(matrix, 6).ravel()))
        self.assertEqual(len(seen), 4)

    def test_round_trip_and_lengths_on_synthetic_points(self) -> None:
        points = np.array(
            [
                [0.10, -0.40, 1.20],
                [-0.25, -0.10, 0.80],
                [0.00, 0.20, 1.50],
                [0.40, -0.70, 0.90],
            ],
            dtype=float,
        )
        for axis_map in enumerate_y_up_maps():
            errors = round_trip_errors(points, axis_map.matrix)
            self.assertLess(float(np.max(errors)), 1e-12)
            mapped = apply_map(points, axis_map.matrix)
            self.assertAlmostEqual(
                segment_length(points[0], points[1]),
                segment_length(mapped[0], mapped[1]),
                places=12,
            )

    def test_height_normalization_recovers_points(self) -> None:
        points = np.array([[0.2, 0.1, 1.4], [-0.3, -0.2, 1.1]], dtype=float)
        mid_hip = np.array([0.05, 0.4, 1.0], dtype=float)
        errors = height_round_trip(points, mid_hip, SYNTHETIC_HEIGHT_M)
        self.assertLess(float(np.max(errors)), 1e-12)

    def test_recording_contract_is_invertible_but_not_identified(self) -> None:
        if not PREFERRED_RECORDING.is_file():
            self.skipTest(f"recording not present: {PREFERRED_RECORDING}")
        result = evaluate(PREFERRED_RECORDING)
        self.assertTrue(result["classification"].startswith("B."))
        for item in result["round_trips"]:
            self.assertLess(item["max"], 1e-9)
        for item in result["lengths"]:
            if item["frames"] == 0:
                continue
            self.assertLess(item["abs_error_m"], 1e-9)
        for item in result["angles"]:
            if item["frames"] == 0:
                continue
            self.assertLess(item["abs_diff_deg"], 1e-6)
        self.assertLess(result["height"]["max"], 1e-9)
        self.assertIn("Neck", result["missing_arm"])
        self.assertTrue(result["height_is_synthetic"])
        source = (
            PROJECT_ROOT / "tools" / "marker_enhancer_coordinate_contract.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("tensorflow", source.lower())
        self.assertNotIn("model.json", source)


if __name__ == "__main__":
    unittest.main()
