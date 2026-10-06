"""Synthetic checks for the initial trunk-relative degrees of freedom.

Points are built in a subject frame (X anterior, Y up, Z to the subject's
right) and converted into the RealSense camera frame (X right, Y down,
Z forward) for a person facing the camera. The old one-angle elbow is
checked on the same points and must stay near 180° when the new flexion is 0°.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.analysis.angles_3d import compute_configured_angles_3d
from src.analysis.dof_angles_3d import DOF_NAMES, compute_initial_dofs, unavailable_initial_dofs
from src.geometry.anatomical_frames import build_trunk_frame
from src.pose.keypoints import Keypoint2D
from src.utils.config_loader import load_config

MIN_CONFIDENCE = 0.5
PLANE_SINGULAR_DEG = 15.0
ELBOW_DEADBAND_DEG = 5.0
DELTA_DEG = 2.0

# Old elbow path, copied from config.yaml so this test calls the real function.
_OLD_FRAMES = [
    {
        "name": "left_elbow",
        "origin": "left_elbow",
        "reference": ["left_elbow", "left_shoulder"],
        "measure": ["left_elbow", "left_wrist"],
        "plane_hints": [
            ["left_shoulder", "left_hip"],
            ["left_shoulder", "right_shoulder"],
        ],
    },
    {
        "name": "right_elbow",
        "origin": "right_elbow",
        "reference": ["right_elbow", "right_shoulder"],
        "measure": ["right_elbow", "right_wrist"],
        "plane_hints": [
            ["right_shoulder", "right_hip"],
            ["left_shoulder", "right_shoulder"],
        ],
    },
]
_OLD_ANGLES = [
    {"name": "left_elbow", "points": ["left_shoulder", "left_elbow", "left_wrist"]},
    {"name": "right_elbow", "points": ["right_shoulder", "right_elbow", "right_wrist"]},
]


def _cam(anterior: float, up: float, right: float) -> tuple[float, float, float]:
    """Subject metres -> camera metres. Person faces the camera, 2 m away."""
    return (-right, -up, -anterior + 2.0)


def _kp(name: str, xyz: tuple[float, float, float], confidence: float = 1.0) -> Keypoint2D:
    return Keypoint2D(
        name=name,
        u_px=0.0,
        v_px=0.0,
        confidence=confidence,
        coord_frame="camera_3d_metre",
        x_m=xyz[0],
        y_m=xyz[1],
        z_m=xyz[2],
    )


def _mirror_lateral(offset: tuple[float, float, float]) -> tuple[float, float, float]:
    """+Z in a side-local offset is away from the trunk. Left arm flips Z."""
    return (offset[0], offset[1], -offset[2])


def _pose(elbow_off: tuple[float, float, float], wrist_off: tuple[float, float, float], **kwargs):
    """Build both arms. Offsets are from the shoulder: +X anterior, +Y up, +Z lateral."""
    left_s = (0.0, 1.4, -0.2)
    right_s = (0.0, 1.4, 0.2)

    def add(origin, offset):
        return (origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2])

    left_elbow = _mirror_lateral(elbow_off)
    left_wrist = _mirror_lateral(wrist_off)
    bodies = {
        "left_shoulder": left_s,
        "right_shoulder": right_s,
        "left_hip": (0.0, 0.9, -0.15),
        "right_hip": (0.0, 0.9, 0.15),
        "left_elbow": add(left_s, left_elbow),
        "right_elbow": add(right_s, elbow_off),
        "left_wrist": add(left_s, left_wrist),
        "right_wrist": add(right_s, wrist_off),
    }
    elbow_confidence = float(kwargs.get("elbow_confidence", 1.0))
    wrist_confidence = float(kwargs.get("wrist_confidence", 1.0))
    keypoints = []
    for name, body in bodies.items():
        conf = 1.0
        if name.endswith("_elbow"):
            conf = elbow_confidence
        elif name.endswith("_wrist"):
            conf = wrist_confidence
        keypoints.append(_kp(name, _cam(*body), conf))
    if kwargs.get("drop_hip_xyz"):
        keypoints = [
            Keypoint2D(
                name=kp.name,
                u_px=kp.u_px,
                v_px=kp.v_px,
                confidence=kp.confidence,
                x_m=None if kp.name == "left_hip" else kp.x_m,
                y_m=None if kp.name == "left_hip" else kp.y_m,
                z_m=None if kp.name == "left_hip" else kp.z_m,
            )
            for kp in keypoints
        ]
    return keypoints


def _dofs(keypoints):
    return compute_initial_dofs(
        keypoints,
        MIN_CONFIDENCE,
        PLANE_SINGULAR_DEG,
        ELBOW_DEADBAND_DEG,
    )


class DofAngleTests(unittest.TestCase):
    def test_keys_and_trunk_handedness(self) -> None:
        keypoints = _pose((0.0, -0.3, 0.0), (0.0, -0.6, 0.0))
        dofs = _dofs(keypoints)
        self.assertEqual(tuple(dofs), DOF_NAMES)
        trunk = build_trunk_frame(keypoints, MIN_CONFIDENCE)
        self.assertTrue(trunk.available and trunk.frame is not None)
        frame = trunk.frame
        cross = np.cross(np.array(frame.x_axis), np.array(frame.y_axis))
        self.assertTrue(np.allclose(cross, np.array(frame.z_axis), atol=1e-6))

    def test_hanging_straight(self) -> None:
        dofs = _dofs(_pose((0.0, -0.3, 0.0), (0.0, -0.6, 0.0)))
        for side in ("left", "right"):
            plane = dofs[f"{side}_shoulder_plane"]
            elevation = dofs[f"{side}_shoulder_elevation"]
            flexion = dofs[f"{side}_elbow_flexion"]
            self.assertFalse(plane.available)
            self.assertEqual(plane.reason, "singular_plane")
            self.assertIsNone(plane.degrees)
            self.assertTrue(elevation.available)
            self.assertAlmostEqual(elevation.degrees, 0.0, delta=DELTA_DEG)
            self.assertTrue(flexion.available)
            self.assertAlmostEqual(flexion.degrees, 0.0, delta=DELTA_DEG)
            self.assertEqual(flexion.reason, "near_extension")

    def test_forward_horizontal_straight(self) -> None:
        dofs = _dofs(_pose((0.3, 0.0, 0.0), (0.6, 0.0, 0.0)))
        for side in ("left", "right"):
            plane = dofs[f"{side}_shoulder_plane"]
            elevation = dofs[f"{side}_shoulder_elevation"]
            flexion = dofs[f"{side}_elbow_flexion"]
            self.assertTrue(plane.available)
            self.assertAlmostEqual(plane.degrees, 90.0, delta=DELTA_DEG)
            self.assertAlmostEqual(elevation.degrees, -90.0, delta=DELTA_DEG)
            self.assertAlmostEqual(flexion.degrees, 0.0, delta=DELTA_DEG)

    def test_sideways_horizontal_straight(self) -> None:
        dofs = _dofs(_pose((0.0, 0.0, 0.3), (0.0, 0.0, 0.6)))
        for side in ("left", "right"):
            plane = dofs[f"{side}_shoulder_plane"]
            elevation = dofs[f"{side}_shoulder_elevation"]
            self.assertTrue(plane.available)
            self.assertAlmostEqual(plane.degrees, 0.0, delta=DELTA_DEG)
            self.assertAlmostEqual(elevation.degrees, -90.0, delta=DELTA_DEG)
            self.assertAlmostEqual(dofs[f"{side}_elbow_flexion"].degrees, 0.0, delta=DELTA_DEG)

    def test_elbow_flexed_90(self) -> None:
        # Arm hanging. Wrist moves anterior: +90° flexion, not the old interior angle.
        dofs = _dofs(_pose((0.0, -0.3, 0.0), (0.3, -0.3, 0.0)))
        for side in ("left", "right"):
            flexion = dofs[f"{side}_elbow_flexion"]
            self.assertTrue(flexion.available)
            self.assertIsNone(flexion.reason)
            self.assertAlmostEqual(flexion.degrees, 90.0, delta=DELTA_DEG)

    def test_elbow_hyperextension_10(self) -> None:
        bend = math.radians(10.0)
        elbow = (0.0, -0.3, 0.0)
        distal = (-math.sin(bend), -math.cos(bend), 0.0)
        wrist = (elbow[0] + 0.3 * distal[0], elbow[1] + 0.3 * distal[1], elbow[2])
        dofs = _dofs(_pose(elbow, wrist))
        for side in ("left", "right"):
            flexion = dofs[f"{side}_elbow_flexion"]
            self.assertTrue(flexion.available, flexion.reason)
            self.assertAlmostEqual(flexion.degrees, -10.0, delta=DELTA_DEG)

    def test_halfway_forward_and_side(self) -> None:
        direction = (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))
        elbow = tuple(0.3 * value for value in direction)
        wrist = tuple(0.6 * value for value in direction)
        dofs = _dofs(_pose(elbow, wrist))
        for side in ("left", "right"):
            plane = dofs[f"{side}_shoulder_plane"]
            elevation = dofs[f"{side}_shoulder_elevation"]
            self.assertTrue(plane.available)
            self.assertAlmostEqual(plane.degrees, 45.0, delta=DELTA_DEG)
            self.assertAlmostEqual(elevation.degrees, -90.0, delta=DELTA_DEG)

    def test_overhead(self) -> None:
        dofs = _dofs(_pose((0.0, 0.3, 0.0), (0.0, 0.6, 0.0)))
        for side in ("left", "right"):
            plane = dofs[f"{side}_shoulder_plane"]
            elevation = dofs[f"{side}_shoulder_elevation"]
            self.assertFalse(plane.available)
            self.assertEqual(plane.reason, "singular_plane")
            self.assertTrue(elevation.available)
            self.assertAlmostEqual(elevation.degrees, -180.0, delta=DELTA_DEG)

    def test_configured_gate_hides_near_vertical_only(self) -> None:
        """25° gate from config: a near-hang is singular; raised poses stay valid."""
        gate = float(load_config()["analysis"]["dof"]["plane_singular_deg"])
        deadband = float(load_config()["analysis"]["dof"]["elbow_sign_deadband_deg"])

        def run(elbow, wrist):
            return compute_initial_dofs(_pose(elbow, wrist), MIN_CONFIDENCE, gate, deadband)

        tilt = math.radians(18.0)
        near = (math.sin(tilt), -math.cos(tilt), 0.0)
        near_dofs = run(tuple(0.3 * v for v in near), tuple(0.6 * v for v in near))
        for side in ("left", "right"):
            plane = near_dofs[f"{side}_shoulder_plane"]
            elevation = near_dofs[f"{side}_shoulder_elevation"]
            self.assertFalse(plane.available)
            self.assertEqual(plane.reason, "singular_plane")
            self.assertIsNone(plane.degrees)
            self.assertTrue(elevation.available)
            self.assertAlmostEqual(elevation.degrees, -18.0, delta=DELTA_DEG)

        forward = run((0.3, 0.0, 0.0), (0.6, 0.0, 0.0))
        side_pose = run((0.0, 0.0, 0.3), (0.0, 0.0, 0.6))
        raised = math.radians(40.0)
        up = (math.sin(raised), math.cos(raised), 0.0)
        overhead = run(tuple(0.35 * v for v in up), tuple(0.7 * v for v in up))
        expect = (
            (forward, 90.0, -90.0),
            (side_pose, 0.0, -90.0),
            (overhead, 90.0, -140.0),
        )
        for dofs, plane_deg, elevation_deg in expect:
            for side in ("left", "right"):
                plane = dofs[f"{side}_shoulder_plane"]
                elevation = dofs[f"{side}_shoulder_elevation"]
                self.assertTrue(plane.available, plane.reason)
                self.assertIsNone(plane.reason)
                self.assertAlmostEqual(plane.degrees, plane_deg, delta=DELTA_DEG)
                self.assertTrue(elevation.available)
                self.assertAlmostEqual(elevation.degrees, elevation_deg, delta=DELTA_DEG)

    def test_missing_xyz(self) -> None:
        dofs = _dofs(_pose((0.3, 0.0, 0.0), (0.6, 0.0, 0.0), drop_hip_xyz=True))
        for name in DOF_NAMES:
            angle = dofs[name]
            self.assertFalse(angle.available)
            self.assertIsNone(angle.degrees)
            self.assertIn("invalid_geometry", angle.reason or "")
            self.assertIn("left_hip", angle.reason or "")

    def test_low_confidence_one_side(self) -> None:
        keypoints = _pose((0.3, 0.0, 0.0), (0.6, 0.0, 0.0))
        keypoints = [
            Keypoint2D(
                name=kp.name,
                u_px=kp.u_px,
                v_px=kp.v_px,
                confidence=0.1 if kp.name == "right_elbow" else kp.confidence,
                coord_frame=kp.coord_frame,
                x_m=kp.x_m,
                y_m=kp.y_m,
                z_m=kp.z_m,
            )
            for kp in keypoints
        ]
        dofs = _dofs(keypoints)
        self.assertAlmostEqual(dofs["left_shoulder_plane"].degrees, 90.0, delta=DELTA_DEG)
        self.assertTrue(dofs["left_shoulder_elevation"].available)
        self.assertTrue(dofs["left_elbow_flexion"].available)
        for name in ("right_shoulder_plane", "right_shoulder_elevation", "right_elbow_flexion"):
            self.assertFalse(dofs[name].available)
            self.assertIn("low_confidence", dofs[name].reason or "")
            self.assertIn("right_elbow", dofs[name].reason or "")

    def test_old_elbow_unchanged_when_new_is_straight(self) -> None:
        keypoints = _pose((0.0, -0.3, 0.0), (0.0, -0.6, 0.0))
        dofs = _dofs(keypoints)
        old, _frames = compute_configured_angles_3d(
            keypoints,
            _OLD_ANGLES,
            MIN_CONFIDENCE,
            frame_specs=_OLD_FRAMES,
        )
        by_name = {angle.name: angle.degrees for angle in old}
        for side in ("left", "right"):
            self.assertAlmostEqual(dofs[f"{side}_elbow_flexion"].degrees, 0.0, delta=DELTA_DEG)
            self.assertAlmostEqual(by_name[f"{side}_elbow"], 180.0, delta=DELTA_DEG)

    def test_rigid_camera_transform(self) -> None:
        keypoints = _pose((0.3, 0.0, 0.0), (0.6, 0.0, 0.0))
        before = _dofs(keypoints)
        angle = 0.7
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array(
            [
                [cosine, 0.0, sine],
                [0.0, 1.0, 0.0],
                [-sine, 0.0, cosine],
            ]
        )
        translation = np.array([0.4, -0.25, 0.8])
        moved = []
        for kp in keypoints:
            xyz = rotation @ np.array([kp.x_m, kp.y_m, kp.z_m]) + translation
            moved.append(
                Keypoint2D(
                    name=kp.name,
                    u_px=kp.u_px,
                    v_px=kp.v_px,
                    confidence=kp.confidence,
                    coord_frame=kp.coord_frame,
                    x_m=float(xyz[0]),
                    y_m=float(xyz[1]),
                    z_m=float(xyz[2]),
                )
            )
        after = _dofs(moved)
        for name in DOF_NAMES:
            self.assertEqual(before[name].available, after[name].available)
            if before[name].degrees is None:
                self.assertIsNone(after[name].degrees)
            else:
                self.assertAlmostEqual(before[name].degrees, after[name].degrees, delta=0.5)

    def test_needs_3d_contract(self) -> None:
        dofs = unavailable_initial_dofs("needs_3d")
        self.assertEqual(tuple(dofs), DOF_NAMES)
        for angle in dofs.values():
            self.assertFalse(angle.available)
            self.assertEqual(angle.reason, "needs_3d")
            self.assertIsNone(angle.degrees)


if __name__ == "__main__":
    unittest.main()
