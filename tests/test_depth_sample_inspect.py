"""Diagnostic depth-window inspect. Does not change median_depth_m results."""

from __future__ import annotations

import unittest

from src.capture.depth_util import inspect_depth_sample, median_depth_m


class FakeDepthFrame:
    def __init__(self, width: int, height: int, distances: dict[tuple[int, int], float]) -> None:
        self._width = width
        self._height = height
        self._distances = distances

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height

    def get_distance(self, x: int, y: int) -> float:
        return float(self._distances.get((int(x), int(y)), 0.0))


class DepthSampleInspectTest(unittest.TestCase):
    def test_in_bounds_valid_window_matches_median(self) -> None:
        frame = FakeDepthFrame(
            10,
            8,
            {
                (4, 4): 1.10,
                (5, 4): 1.20,
                (6, 4): 1.30,
                (4, 5): 1.40,
                (5, 5): 1.50,
                (6, 5): 1.60,
                (4, 6): 1.70,
                (5, 6): 1.80,
                (6, 6): 1.90,
            },
        )
        info = inspect_depth_sample(frame, 5.4, 5.1, 3)
        median = median_depth_m(frame, 5.4, 5.1, 3)
        self.assertTrue(info.in_image)
        self.assertEqual(info.n_valid, 9)
        self.assertAlmostEqual(info.depth_sample_m, median)
        self.assertAlmostEqual(median, 1.50)

    def test_zero_depth_window_records_zero_not_invented(self) -> None:
        frame = FakeDepthFrame(8, 8, {})
        info = inspect_depth_sample(frame, 3.0, 3.0, 5)
        self.assertTrue(info.in_image)
        self.assertEqual(info.n_valid, 0)
        self.assertEqual(info.depth_sample_m, 0.0)
        self.assertEqual(median_depth_m(frame, 3.0, 3.0, 5), 0.0)

    def test_out_of_bounds_pixel(self) -> None:
        frame = FakeDepthFrame(8, 8, {(7, 7): 2.0})
        info = inspect_depth_sample(frame, 20.0, 40.0, 5)
        self.assertFalse(info.in_image)
        self.assertEqual(info.n_valid, 0)
        self.assertEqual(info.depth_sample_m, 0.0)
        self.assertEqual(median_depth_m(frame, 20.0, 40.0, 5), 0.0)

    def test_out_of_bounds_window_may_still_count_edge_samples(self) -> None:
        frame = FakeDepthFrame(8, 8, {(7, 7): 2.25})
        info = inspect_depth_sample(frame, 8.0, 8.0, 3)
        self.assertFalse(info.in_image)
        self.assertEqual(info.n_valid, 1)
        self.assertAlmostEqual(info.depth_sample_m, 2.25)
        self.assertAlmostEqual(median_depth_m(frame, 8.0, 8.0, 3), 2.25)

    def test_missing_frame_is_unavailable_not_zero(self) -> None:
        info = inspect_depth_sample(None, 1.0, 2.0, 5)
        self.assertIsNone(info.in_image)
        self.assertIsNone(info.n_valid)
        self.assertIsNone(info.depth_sample_m)
        self.assertEqual(median_depth_m(None, 1.0, 2.0, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
