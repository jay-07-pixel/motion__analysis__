"""The rotatable fit page names each marker and keeps the 3D gap."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.opensim_upper.fit_view import fit_markers, joint_label, write_fit_html


class FitViewTest(unittest.TestCase):
    def test_a_marker_gap_is_the_distance_in_millimetres(self) -> None:
        rows = fit_markers(
            {"RShoulder": (0.0, 0.0, 0.0), "LElbow": None},
            {"RShoulder": (0.03, 0.0, 0.0), "LElbow": (1.0, 0.0, 0.0)},
        )
        by_name = {row["name"]: row for row in rows}
        self.assertEqual(by_name["RShoulder"]["label"], "Right shoulder")
        self.assertAlmostEqual(by_name["RShoulder"]["error_mm"], 30.0)
        self.assertIsNone(by_name["LElbow"]["error_mm"])
        self.assertEqual(by_name["LElbow"]["label"], "Left elbow")
        self.assertTrue(by_name["RShoulder"]["used_in_ik"])

    def test_blank_ik_markers_are_marked_left_out(self) -> None:
        rows = fit_markers(
            {"RShoulder": (0.0, 0.0, 0.0), "r_lwrist_study": (0.1, 0.2, 1.0)},
            {"RShoulder": (0.0, 0.0, 0.0), "r_lwrist_study": (0.1, 0.2, 1.05)},
            used_in_ik={"RShoulder": (0.0, 0.0, 0.0), "r_lwrist_study": None},
        )
        by_name = {row["name"]: row for row in rows}
        self.assertTrue(by_name["RShoulder"]["used_in_ik"])
        self.assertFalse(by_name["r_lwrist_study"]["used_in_ik"])
        self.assertAlmostEqual(by_name["r_lwrist_study"]["error_mm"], 50.0)

    def test_a_body_origin_gets_a_side(self) -> None:
        self.assertEqual(joint_label("femur_r"), "Right thigh")
        self.assertEqual(joint_label("head"), "Head")

    def test_the_page_embeds_the_marker_and_can_be_opened_locally(self) -> None:
        scene = {
            "time_sec": 14.45,
            "triangles": np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=float),
            "joints": [{"name": "head", "label": "Head", "position": [0.0, 1.5, 0.0]}],
            "markers": [
                {
                    "name": "RElbow",
                    "label": "Right elbow",
                    "recorded": [0.1, 1.0, 0.2],
                    "model": [0.12, 1.0, 0.2],
                    "error_mm": 20.0,
                }
            ],
            "bones": [],
            "center": [0.0, 1.0, 0.0],
            "span": 1.8,
            "rms_mm": 20.0,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = write_fit_html(scene, Path(folder) / "skeleton_fit.html")
            text = path.read_text(encoding="utf-8")
        self.assertIn("Right elbow", text)
        self.assertIn("Drag to turn", text)
        self.assertIn("<canvas", text)
        self.assertNotIn("__FIT_JSON__", text)
