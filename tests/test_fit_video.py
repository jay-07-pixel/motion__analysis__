"""The bone movie lasts as long as the capture and draws the skeleton."""

from __future__ import annotations

import unittest

import numpy as np

from src.opensim_upper.bone_figure import depth_raster, fit_view_camera
from src.opensim_upper.fit_video import _decimate, playback_fps, render_playback_frame


class FitVideoTest(unittest.TestCase):
    def test_the_file_lasts_as_long_as_the_capture(self) -> None:
        times = [10.0, 10.5, 11.0, 11.5]
        fps = playback_fps(times)
        self.assertAlmostEqual(len(times) / fps, 1.5)

    def test_a_long_mesh_is_thinned(self) -> None:
        faces = np.zeros((100, 3, 3))
        self.assertEqual(len(_decimate(faces, 10)), 10)
        self.assertEqual(len(_decimate(faces[:4], 10)), 4)

    def test_one_triangle_is_drawn_on_the_frame(self) -> None:
        triangles = np.array([[[-0.2, -0.4, 1.0], [0.2, -0.4, 1.0], [0.0, 0.5, 1.0]]], dtype=float)
        view = fit_view_camera(triangles.reshape(-1, 3), 80, 100)
        image = render_playback_frame(triangles, [], view, "Bone fit")
        self.assertEqual(image.shape, (100, 80, 3))
        self.assertTrue(bool(np.any(np.any(image != 247, axis=2))))

    def test_a_nearer_triangle_covers_the_one_behind_it(self) -> None:
        screen = np.array(
            [
                [[8.0, 8.0, 4.0], [42.0, 8.0, 4.0], [25.0, 42.0, 4.0]],
                [[10.0, 10.0, 1.0], [40.0, 10.0, 1.0], [25.0, 40.0, 1.0]],
            ],
            dtype=float,
        )
        colors = np.array([[10, 20, 30], [200, 210, 220]], dtype=np.uint8)
        image = depth_raster(screen, colors, 50, 50)
        self.assertEqual(tuple(int(value) for value in image[25, 25]), (200, 210, 220))
        self.assertEqual(tuple(int(value) for value in image[0, 0]), (247, 247, 247))
