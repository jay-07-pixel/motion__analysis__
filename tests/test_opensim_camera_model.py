"""Camera-frame upper-body model. Does not import OpenSim."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.opensim_upper.environment import OPENSIM_PYTHON
from src.opensim_upper.ik import run_upper_body_ik
from src.opensim_upper.ik_tasks import CAMERA_MODEL_LOCKED, CAMERA_SOLVED, SOLVED, coordinate_decisions
from src.opensim_upper.trial import write_camera_trial_from_diagnostics
from src.opensim_upper.trc import ordered_marker_rows, read_trc
from src.opensim_upper.camera_model import (
    CAMERA_MODEL,
    FACING_CAMERA_ROTATION,
    KEPT_BODIES,
    REMOVED_BODIES,
    ROOT_COORDINATES,
    ROOT_JOINT,
    SOURCE_MODEL,
    body_fixed_xyz,
    build_camera_model_tree,
    camera_pose_of_default_torso,
    default_torso_origin_yup,
    rotation_from_body_fixed_xyz,
    write_camera_model,
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for element in root.iter():
        if _local(element.tag) == tag and element.get("name") == name:
            return element
    raise AssertionError(f"missing {tag} {name}")


class CameraAxisTest(unittest.TestCase):
    def test_facing_camera_map_is_a_proper_rotation(self) -> None:
        rotation = FACING_CAMERA_ROTATION
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3)))
        # Anatomical anterior, up, and model-right, expressed in the camera.
        self.assertTrue(np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0]))
        self.assertTrue(np.allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.0, -1.0, 0.0]))
        self.assertTrue(np.allclose(rotation @ np.array([0.0, 0.0, 1.0]), [-1.0, 0.0, 0.0]))

    def test_body_fixed_xyz_round_trip(self) -> None:
        for matrix in (
            FACING_CAMERA_ROTATION,
            rotation_from_body_fixed_xyz(np.array([0.2, -0.4, 0.7])),
        ):
            recovered = rotation_from_body_fixed_xyz(body_fixed_xyz(matrix))
            self.assertTrue(np.allclose(recovered, matrix, atol=1e-10))

    def test_default_torso_origin_moves_with_the_axis_map(self) -> None:
        source = ET.parse(SOURCE_MODEL).getroot()
        origin = default_torso_origin_yup(source)
        self.assertAlmostEqual(origin[0], -0.1007, places=6)
        self.assertAlmostEqual(origin[1], 1.0215, places=6)
        self.assertAlmostEqual(origin[2], 0.0, places=6)
        rotation, camera_origin = camera_pose_of_default_torso(origin)
        self.assertTrue(np.allclose(camera_origin, rotation @ origin))
        self.assertAlmostEqual(camera_origin[0], 0.0, places=8)
        self.assertAlmostEqual(camera_origin[1], -1.0215, places=6)
        self.assertAlmostEqual(camera_origin[2], 0.1007, places=6)


class CameraModelFileTest(unittest.TestCase):
    def test_built_model_keeps_the_thorax_root_and_the_arms(self) -> None:
        before = hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest()
        tree = build_camera_model_tree()
        self.assertEqual(hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest(), before)
        model = tree.getroot().find("Model")
        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model.get("name"), "UpperBody_CameraFrame")
        gravity = model.find("gravity")
        self.assertIsNotNone(gravity)
        assert gravity is not None and gravity.text is not None
        gx, gy, gz = (float(part) for part in gravity.text.split())
        self.assertEqual((gx, gz), (0.0, 0.0))
        self.assertAlmostEqual(gy, 9.80665, places=5)
        bodies = [element.get("name") for element in model.iter() if _local(element.tag) == "Body"]
        self.assertEqual(bodies, list(KEPT_BODIES))
        for name in REMOVED_BODIES:
            self.assertNotIn(name, bodies)
        joints = [
            element.get("name")
            for element in model.iter()
            if _local(element.tag) in {"CustomJoint", "PinJoint", "WeldJoint", "UniversalJoint"}
        ]
        self.assertEqual(joints[0], ROOT_JOINT)
        self.assertNotIn("ground_pelvis", joints)
        self.assertNotIn("back", joints)
        self.assertIn("acromial_r", joints)
        self.assertIn("acromial_l", joints)
        root = _named(model, "CustomJoint", ROOT_JOINT)
        names = [element.get("name") for element in root.iter() if _local(element.tag) == "Coordinate"]
        self.assertEqual(names, list(ROOT_COORDINATES))
        child = _named(root, "PhysicalOffsetFrame", "torso_root_offset")
        parent_socket = child.find("socket_parent")
        self.assertIsNotNone(parent_socket)
        assert parent_socket is not None
        self.assertEqual(parent_socket.text, "/bodyset/torso")
        markers = [element.get("name") for element in model.iter() if _local(element.tag) == "Marker"]
        self.assertEqual(len(markers), 12)
        contacts = [
            element.get("name")
            for element in model.iter()
            if _local(element.tag) in {"ContactSphere", "ContactHalfSpace"}
        ]
        self.assertEqual(contacts, [])

    def test_write_refuses_the_source_path(self) -> None:
        with self.assertRaises(ValueError):
            write_camera_model(SOURCE_MODEL)

    def test_checked_in_model_matches_the_builder(self) -> None:
        built = build_camera_model_tree()
        existing = ET.parse(CAMERA_MODEL)
        built_xml = ET.tostring(built.getroot())
        existing_xml = ET.tostring(existing.getroot())
        self.assertEqual(built_xml, existing_xml)


class OpenSimCameraLoadTest(unittest.TestCase):
    def test_opensim_reports_the_thorax_in_camera_axes(self) -> None:
        if not OPENSIM_PYTHON.is_file() or not CAMERA_MODEL.is_file():
            self.skipTest("OpenSim interpreter or camera model is not available.")
        tool = Path(__file__).resolve().parents[1] / "tools" / "opensim_camera_model_check.py"
        completed = subprocess.run(
            [str(OPENSIM_PYTHON), str(tool)],
            check=False,
            capture_output=True,
            text=True,
        )
        line = next(line for line in completed.stdout.splitlines() if line.startswith("{"))
        report = json.loads(line)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(report["ok"])
        self.assertEqual(report["root_parent"], "ground")
        self.assertEqual(report["root_child"], "torso")
        self.assertEqual(report["n_bodies"], 10)
        self.assertEqual(report["n_markers"], 12)
        self.assertGreater(report["gravity"][1], 0.0)
        self.assertLess(report["axis_mismatch"], 1e-8)
        self.assertAlmostEqual(report["torso_origin_m"][1], -1.0215, places=4)


class CameraTrialTest(unittest.TestCase):
    def test_thorax_root_is_solved_and_camera_metres_are_copied(self) -> None:
        names = list(CAMERA_SOLVED) + list(SOLVED) + list(CAMERA_MODEL_LOCKED)
        decisions = coordinate_decisions(names)
        self.assertEqual(len(decisions), 23)
        self.assertTrue(decisions["thorax_tz"]["solved"])
        self.assertFalse(decisions["thorax_tz"]["locked"])
        self.assertTrue(decisions["wrist_flex_r"]["locked"])
        self.assertNotIn("pelvis_ty", decisions)
        rows = ordered_marker_rows()
        folder = Path(tempfile.mkdtemp())
        diag = folder / "capture-01"
        diag.mkdir()
        header = "frame,t_frame_sec,landmark,raw_x_m,raw_y_m,raw_z_m,xyz_missing\n"
        lines = [header]
        expected = {}
        for index, row in enumerate(rows):
            point = (0.01 * index, -0.40 - 0.01 * index, 1.50)
            expected[row["target"]] = point
            lines.append(f"1,0.0,{row['source']},{point[0]},{point[1]},{point[2]},0\n")
            lines.append(f"2,0.05,{row['source']},{point[0]},{point[1]},{point[2]},0\n")
        (diag / "landmarks.csv").write_text("".join(lines), encoding="utf-8")
        written = write_camera_trial_from_diagnostics(diag, folder / "trial", "capture-01")
        self.assertIsNone(written["metadata"]["registration_file"])
        self.assertEqual(written["metadata"]["coord_frame"], "camera_3d_metre")
        self.assertIn("copied unchanged", written["metadata"]["transform"])
        parsed = read_trc(written["trc"])
        for name, point in expected.items():
            stored = parsed["frames"][0]["points"][name]
            self.assertTrue(np.allclose(stored, point))

    def test_a_camera_translation_moves_the_thorax_root(self) -> None:
        if not OPENSIM_PYTHON.is_file() or not CAMERA_MODEL.is_file():
            self.skipTest("OpenSim or the camera-frame model is not available.")
        folder = Path(tempfile.mkdtemp())
        sampler = folder / "sample_markers.py"
        sampler.write_text(
            "import json, sys\n"
            "import opensim\n"
            "model = opensim.Model(sys.argv[1])\n"
            "state = model.initSystem()\n"
            "model.realizePosition(state)\n"
            "markers = model.getMarkerSet()\n"
            "out = {}\n"
            "for index in range(markers.getSize()):\n"
            "    marker = markers.get(index)\n"
            "    point = marker.getLocationInGround(state)\n"
            "    out[marker.getName()] = [point.get(0), point.get(1), point.get(2)]\n"
            "print(json.dumps(out))\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [str(OPENSIM_PYTHON), str(sampler), str(CAMERA_MODEL)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        located = json.loads(completed.stdout.strip().splitlines()[-1])
        shift_z = 0.30
        by_target = {row["target"]: row["source"] for row in ordered_marker_rows()}
        diag = folder / "shifted"
        diag.mkdir()
        lines = ["frame,t_frame_sec,landmark,raw_x_m,raw_y_m,raw_z_m,xyz_missing\n"]
        for target, source in by_target.items():
            x_m, y_m, z_m = located[target]
            z_m += shift_z
            lines.append(f"1,0.0,{source},{x_m},{y_m},{z_m},0\n")
            lines.append(f"2,0.05,{source},{x_m},{y_m},{z_m},0\n")
        (diag / "landmarks.csv").write_text("".join(lines), encoding="utf-8")
        written = write_camera_trial_from_diagnostics(diag, folder / "trial", "shifted")
        report = run_upper_body_ik(
            written["trc"],
            folder / "ik",
            trial_metadata=folder / "trial" / "trial_metadata.json",
            model_path=CAMERA_MODEL,
        )
        self.assertTrue(report["execution_success"], report.get("error"))
        thorax = report["coordinates"]["thorax_tz"]
        self.assertTrue(thorax["solved"])
        self.assertLess(abs(thorax["min_result"] - shift_z), 0.02)
        self.assertLess(abs(thorax["max_result"] - shift_z), 0.02)
        self.assertLess(abs(report["coordinates"]["thorax_tx"]["max_result"]), 0.02)
        self.assertTrue(report["coordinates"]["wrist_flex_r"]["unchanged_from_model_default"])
        elbow = report["coordinates"]["elbow_flex_r"]
        self.assertLess(abs(elbow["max_result"] - elbow["default_value"]), 0.05)
        rms = (report.get("marker_errors") or {}).get("columns", {}).get("marker_error_RMS", {})
        if rms:
            self.assertLess(rms["mean_abs_m"], 0.02)


if __name__ == "__main__":
    unittest.main()
