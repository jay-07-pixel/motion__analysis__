"""Practical wrist flexion and ulnar deviation from the forearm and hand frames."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.analysis.dof_angles_3d import DOF_NAMES, compute_initial_dofs, compute_wrist_dofs
from src.pose.keypoints import Keypoint2D

MIN_CONFIDENCE = 0.5


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


def _unit(v):
    arr = np.asarray(v, dtype=float)
    return arr / np.linalg.norm(arr)


def _cross(a, b):
    return np.cross(np.asarray(a, float), np.asarray(b, float))


def _hand(side: str, flexion_deg: float, deviation_deg: float) -> list[Keypoint2D]:
    distal = np.array([0.0, 0.0, -1.0])
    ulnar = np.array([1.0, 0.0, 0.0])
    frame_x = _unit(_cross(distal, ulnar))
    palmar = frame_x if side == "right" else -frame_x
    flex = math.radians(flexion_deg)
    deviate = math.radians(deviation_deg)
    y1 = _unit(math.cos(flex) * distal + math.sin(flex) * palmar)
    hand_y = _unit(math.cos(deviate) * y1 + math.sin(deviate) * ulnar)
    z_axis = _unit(math.cos(deviate) * ulnar - math.sin(deviate) * y1)
    wrist = np.array([0.12 if side == "right" else -0.12, 0.02, 1.0])
    elbow = wrist - 0.28 * distal

    def point(direction, scale):
        xyz = wrist + scale * np.asarray(direction, float)
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    return [
        _kp(f"{side}_elbow", (float(elbow[0]), float(elbow[1]), float(elbow[2]))),
        _kp(f"{side}_wrist", (float(wrist[0]), float(wrist[1]), float(wrist[2]))),
        _kp(f"{side}_middle_mcp", point(hand_y, 0.08)),
        _kp(f"{side}_index_mcp", point(hand_y * 0.03 - z_axis * 0.025, 1.0)),
        _kp(f"{side}_pinky_mcp", point(hand_y * 0.03 + z_axis * 0.025, 1.0)),
    ]


class WristDofTests(unittest.TestCase):
    def test_neutral_is_near_zero(self) -> None:
        for side in ("right", "left"):
            dofs = compute_wrist_dofs(_hand(side, 0.0, 0.0), MIN_CONFIDENCE)
            flexion = dofs[f"{side}_wrist_flexion"]
            deviation = dofs[f"{side}_wrist_deviation"]
            self.assertTrue(flexion.available, flexion.reason)
            self.assertTrue(deviation.available, deviation.reason)
            self.assertAlmostEqual(flexion.degrees, 0.0, delta=1.0)
            self.assertAlmostEqual(deviation.degrees, 0.0, delta=1.0)

    def test_flexion_and_extension_signs(self) -> None:
        flexed = compute_wrist_dofs(_hand("right", 30.0, 0.0), MIN_CONFIDENCE)
        extended = compute_wrist_dofs(_hand("right", -25.0, 0.0), MIN_CONFIDENCE)
        self.assertAlmostEqual(flexed["right_wrist_flexion"].degrees, 30.0, delta=2.0)
        self.assertAlmostEqual(flexed["right_wrist_deviation"].degrees, 0.0, delta=2.0)
        self.assertAlmostEqual(extended["right_wrist_flexion"].degrees, -25.0, delta=2.0)
        self.assertLess(extended["right_wrist_flexion"].degrees, 0.0)

    def test_radial_and_ulnar_signs(self) -> None:
        ulnar = compute_wrist_dofs(_hand("right", 0.0, 20.0), MIN_CONFIDENCE)
        radial = compute_wrist_dofs(_hand("right", 0.0, -20.0), MIN_CONFIDENCE)
        self.assertAlmostEqual(ulnar["right_wrist_deviation"].degrees, 20.0, delta=2.0)
        self.assertAlmostEqual(ulnar["right_wrist_flexion"].degrees, 0.0, delta=2.0)
        self.assertAlmostEqual(radial["right_wrist_deviation"].degrees, -20.0, delta=2.0)

    def test_combined_motion_stays_separable(self) -> None:
        dofs = compute_wrist_dofs(_hand("right", 20.0, 15.0), MIN_CONFIDENCE)
        flexion = dofs["right_wrist_flexion"]
        deviation = dofs["right_wrist_deviation"]
        self.assertTrue(flexion.available and deviation.available)
        self.assertGreater(flexion.degrees, 10.0)
        self.assertGreater(deviation.degrees, 8.0)
        self.assertNotAlmostEqual(flexion.degrees, deviation.degrees, delta=3.0)

    def test_left_hand_uses_the_same_anatomical_signs(self) -> None:
        flexed = compute_wrist_dofs(_hand("left", 30.0, 0.0), MIN_CONFIDENCE)
        radial = compute_wrist_dofs(_hand("left", 0.0, -15.0), MIN_CONFIDENCE)
        self.assertAlmostEqual(flexed["left_wrist_flexion"].degrees, 30.0, delta=2.0)
        self.assertAlmostEqual(flexed["left_wrist_deviation"].degrees, 0.0, delta=2.0)
        self.assertAlmostEqual(radial["left_wrist_deviation"].degrees, -15.0, delta=2.0)
        self.assertAlmostEqual(radial["left_wrist_flexion"].degrees, 0.0, delta=2.0)

    def test_missing_wrist_xyz(self) -> None:
        keypoints = _hand("right", 0.0, 0.0)
        keypoints[1] = Keypoint2D(
            name="right_wrist", u_px=0.0, v_px=0.0, confidence=1.0
        )
        dofs = compute_wrist_dofs(keypoints, MIN_CONFIDENCE)
        for name in ("right_wrist_flexion", "right_wrist_deviation"):
            self.assertFalse(dofs[name].available)
            self.assertIsNone(dofs[name].degrees)
            self.assertEqual(dofs[name].reason, "missing_xyz:right_wrist")

    def test_missing_elbow_xyz(self) -> None:
        keypoints = [kp for kp in _hand("right", 10.0, 0.0) if kp.name != "right_elbow"]
        dofs = compute_wrist_dofs(keypoints, MIN_CONFIDENCE)
        self.assertFalse(dofs["right_wrist_flexion"].available)
        self.assertIsNone(dofs["right_wrist_flexion"].degrees)
        self.assertEqual(dofs["right_wrist_flexion"].reason, "missing_landmark:right_elbow")
        self.assertEqual(dofs["right_wrist_deviation"].reason, "missing_landmark:right_elbow")

    def test_invalid_hand_frame(self) -> None:
        narrow = _hand("right", 0.0, 0.0)
        # Pull the knuckles onto the finger axis so the usable palm width collapses.
        wrist = next(kp for kp in narrow if kp.name == "right_wrist")
        middle = next(kp for kp in narrow if kp.name == "right_middle_mcp")
        replaced = []
        for kp in narrow:
            if kp.name in ("right_index_mcp", "right_pinky_mcp"):
                replaced.append(
                    Keypoint2D(
                        name=kp.name,
                        u_px=0.0,
                        v_px=0.0,
                        confidence=1.0,
                        coord_frame="camera_3d_metre",
                        x_m=wrist.x_m,
                        y_m=(wrist.y_m + middle.y_m) / 2.0,
                        z_m=(wrist.z_m + middle.z_m) / 2.0,
                    )
                )
            else:
                replaced.append(kp)
        dofs = compute_wrist_dofs(replaced, MIN_CONFIDENCE)
        self.assertFalse(dofs["right_wrist_flexion"].available)
        self.assertIsNone(dofs["right_wrist_flexion"].degrees)
        self.assertIn(dofs["right_wrist_flexion"].reason, ("degenerate_palm_width", "insufficient_palm_width"))

    def test_degenerate_forearm(self) -> None:
        keypoints = _hand("right", 0.0, 0.0)
        wrist = next(kp for kp in keypoints if kp.name == "right_wrist")
        keypoints = [
            Keypoint2D(
                name=kp.name,
                u_px=kp.u_px,
                v_px=kp.v_px,
                confidence=kp.confidence,
                coord_frame=kp.coord_frame,
                x_m=wrist.x_m if kp.name == "right_elbow" else kp.x_m,
                y_m=wrist.y_m if kp.name == "right_elbow" else kp.y_m,
                z_m=wrist.z_m if kp.name == "right_elbow" else kp.z_m,
            )
            for kp in keypoints
        ]
        dofs = compute_wrist_dofs(keypoints, MIN_CONFIDENCE)
        self.assertFalse(dofs["right_wrist_flexion"].available)
        self.assertIsNone(dofs["right_wrist_flexion"].degrees)
        self.assertEqual(dofs["right_wrist_flexion"].reason, "degenerate_forearm")

    def test_rigid_rotation_and_translation(self) -> None:
        original = _hand("right", 18.0, -12.0)
        before = compute_wrist_dofs(original, MIN_CONFIDENCE)
        angle = 0.6
        rotation = np.array(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        translation = np.array([0.3, -0.2, 0.5])
        moved = []
        shifted = []
        for kp in original:
            xyz = np.array([kp.x_m, kp.y_m, kp.z_m])
            turned = rotation @ xyz
            moved.append(_kp(kp.name, (float(turned[0]), float(turned[1]), float(turned[2]))))
            shifted.append(
                _kp(kp.name, (float(xyz[0] + translation[0]), float(xyz[1] + translation[1]), float(xyz[2] + translation[2])))
            )
        after_rotation = compute_wrist_dofs(moved, MIN_CONFIDENCE)
        after_translation = compute_wrist_dofs(shifted, MIN_CONFIDENCE)
        for name in ("right_wrist_flexion", "right_wrist_deviation"):
            self.assertAlmostEqual(before[name].degrees, after_rotation[name].degrees, delta=0.5)
            self.assertAlmostEqual(before[name].degrees, after_translation[name].degrees, delta=0.5)

    def test_nonfinite_input_has_no_angle(self) -> None:
        keypoints = _hand("left", 10.0, 5.0)
        keypoints[0] = Keypoint2D(
            name="left_elbow",
            u_px=0.0,
            v_px=0.0,
            confidence=float("nan"),
            coord_frame="camera_3d_metre",
            x_m=float("nan"),
            y_m=0.0,
            z_m=float("inf"),
        )
        dofs = compute_wrist_dofs(keypoints, MIN_CONFIDENCE)
        for name in ("left_wrist_flexion", "left_wrist_deviation"):
            self.assertFalse(dofs[name].available)
            self.assertIsNone(dofs[name].degrees)
            self.assertTrue(dofs[name].reason)

    def test_shoulder_and_elbow_names_are_unchanged(self) -> None:
        self.assertEqual(
            DOF_NAMES,
            (
                "left_shoulder_plane",
                "left_shoulder_elevation",
                "left_elbow_flexion",
                "right_shoulder_plane",
                "right_shoulder_elevation",
                "right_elbow_flexion",
            ),
        )
        dofs = compute_initial_dofs(_hand("right", 0.0, 0.0), MIN_CONFIDENCE, 25.0, 5.0)
        self.assertEqual(tuple(dofs), DOF_NAMES)
        for angle in dofs.values():
            self.assertFalse(angle.available)


if __name__ == "__main__":
    unittest.main()
