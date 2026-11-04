"""The fit picture can draw the full bone meshes and the marker points."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np

from src.opensim_upper.bone_figure import (
    apply_body_pose,
    read_vtp,
    render_triangles,
    stand_points,
    transforms_from_row,
)
from src.opensim_upper.visualize import _BONE_LINKS


class BoneFigureTest(unittest.TestCase):
    def test_a_body_pose_scales_then_rotates(self) -> None:
        points = np.array([[0.0, 1.0, 0.0]])
        rotation = np.eye(3)
        origin = np.array([1.0, 2.0, 3.0])
        world = apply_body_pose(points, (2.0, 1.0, 1.0), rotation, origin)
        self.assertTrue(np.allclose(world, [[1.0, 3.0, 3.0]]))

    def test_rotations_come_back_off_a_skeleton_row(self) -> None:
        row = {
            "torso_tx": 0.1,
            "torso_ty": 0.2,
            "torso_tz": 1.3,
            "torso_r00": 1.0,
            "torso_r01": 0.0,
            "torso_r02": 0.0,
            "torso_r10": 0.0,
            "torso_r11": 1.0,
            "torso_r12": 0.0,
            "torso_r20": 0.0,
            "torso_r21": 0.0,
            "torso_r22": 1.0,
        }
        transforms = transforms_from_row(row)
        self.assertIsNotNone(transforms)
        rotation, origin = transforms["torso"]
        self.assertTrue(np.allclose(rotation, np.eye(3)))
        self.assertTrue(np.allclose(origin, [0.1, 0.2, 1.3]))
        self.assertIsNone(transforms_from_row({"torso_tx": 0.0, "torso_ty": 0.0, "torso_tz": 0.0}))

    def test_a_triangle_mesh_round_trips(self) -> None:
        root = Element("VTKFile")
        poly = SubElement(root, "PolyData")
        piece = SubElement(poly, "Piece")
        points = SubElement(piece, "Points")
        point_array = SubElement(points, "DataArray")
        point_array.text = "0 0 0  1 0 0  0 1 0"
        polys = SubElement(piece, "Polys")
        connectivity = SubElement(polys, "DataArray", Name="connectivity")
        connectivity.text = "0 1 2"
        offsets = SubElement(polys, "DataArray", Name="offsets")
        offsets.text = "3"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "triangle.vtp"
            ElementTree(root).write(path, encoding="utf-8")
            read_points, triangles = read_vtp(str(path.resolve()))
        self.assertEqual(read_points.shape, (3, 3))
        self.assertTrue(np.array_equal(triangles, [[0, 1, 2]]))

    def test_one_triangle_paints_the_middle_of_the_picture(self) -> None:
        triangles = np.array([[[ -0.4, -0.5, 1.2], [0.4, -0.5, 1.2], [0.0, 0.7, 1.2]]], dtype=float)
        image = render_triangles(triangles, [], width=60, height=80)
        painted = np.any(image.reshape(-1, 3) != 247, axis=1)
        self.assertTrue(bool(np.any(painted)))

    def test_standing_keeps_length_and_points_body_axes_upright(self) -> None:
        rotation = np.eye(3)
        origin = np.array([0.2, -0.1, 1.4])
        transforms = {"torso": (rotation, origin)}
        stood_origin = stand_points(origin, transforms)
        up = stand_points(origin + np.array([0.0, 1.0, 0.0]), transforms) - stood_origin
        right = stand_points(origin + np.array([0.0, 0.0, 1.0]), transforms) - stood_origin
        front = stand_points(origin + np.array([1.0, 0.0, 0.0]), transforms) - stood_origin
        self.assertAlmostEqual(float(np.linalg.norm(up)), 1.0, places=6)
        self.assertGreater(float(up[1]), 0.9)
        self.assertGreater(float(right[0]), 0.9)
        self.assertLess(float(front[2]), -0.9)

    def test_the_stick_figure_includes_the_legs(self) -> None:
        links = set(_BONE_LINKS)
        self.assertIn(("pelvis", "femur_r"), links)
        self.assertIn(("pelvis", "femur_l"), links)
        self.assertIn(("calcn_r", "toes_r"), links)
        self.assertIn(("torso", "head"), links)
        self.assertIn(("radius_l", "hand_l"), links)
