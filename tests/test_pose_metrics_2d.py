"""2D game metrics from synthetic pixel keypoints. No camera.

Person faces the camera in a 1280x720 frame, so the person's LEFT side is on
the image RIGHT (larger u).
"""

from __future__ import annotations

import json
import unittest

import validate_2d_pose
from pose_game_bridge import build_pose_message
from src.analysis import pose_metrics_2d
from src.analysis.angles_2d import angle_at_vertex_deg
from src.analysis.pose_metrics_2d import IMPORTANT, game_metrics_2d, tracking_status
from src.pose.keypoints import Keypoint2D
from src.pose.skeleton import LANDMARK_NAMES

GATE = 0.5

STANDING = {
    "nose": (640, 120),
    "left_shoulder": (700, 200),
    "right_shoulder": (580, 200),
    "left_elbow": (710, 300),
    "right_elbow": (570, 300),
    "left_wrist": (715, 390),
    "right_wrist": (565, 390),
    "left_hip": (680, 400),
    "right_hip": (600, 400),
    "left_knee": (680, 530),
    "right_knee": (600, 530),
    "left_ankle": (680, 660),
    "right_ankle": (600, 660),
    "left_heel": (675, 675),
    "right_heel": (605, 675),
    "left_foot_index": (690, 690),
    "right_foot_index": (590, 690),
}

RIGHT_ARM_UP = {"right_elbow": (575, 100), "right_wrist": (570, 10)}
LEFT_ARM_UP = {"left_elbow": (705, 100), "left_wrist": (710, 10)}
RIGHT_ELBOW_BENT = {"right_elbow": (480, 200), "right_wrist": (480, 100)}
SQUAT = {
    "left_hip": (680, 450),
    "right_hip": (600, 450),
    "left_knee": (730, 540),
    "right_knee": (550, 540),
}
LEAN_LEFT = {"left_shoulder": (740, 200), "right_shoulder": (620, 200)}
LEAN_RIGHT = {"left_shoulder": (660, 200), "right_shoulder": (540, 200)}


def body(overrides=None, low=(), missing=()) -> list[Keypoint2D]:
    points = {**STANDING, **(overrides or {})}
    out = []
    for name in LANDMARK_NAMES:
        if name in missing:
            continue
        u, v = points.get(name, (640, 300))
        out.append(Keypoint2D(name=name, u_px=float(u), v_px=float(v), confidence=0.2 if name in low else 0.95))
    return out


def metrics(overrides=None, low=(), missing=()) -> dict:
    return game_metrics_2d({kp.name: kp for kp in body(overrides, low, missing)}, GATE)


class ArmAndElbowTest(unittest.TestCase):
    def test_straight_right_elbow(self) -> None:
        self.assertGreater(metrics()["right_elbow_angle_deg"], 170)

    def test_bent_right_elbow(self) -> None:
        self.assertAlmostEqual(metrics(RIGHT_ELBOW_BENT)["right_elbow_angle_deg"], 90.0, delta=1.0)

    def test_straight_left_elbow(self) -> None:
        self.assertGreater(metrics()["left_elbow_angle_deg"], 170)

    def test_raised_right_arm(self) -> None:
        m = metrics(RIGHT_ARM_UP)
        self.assertGreater(m["right_arm_angle_deg"], 170)
        self.assertLess(m["left_arm_angle_deg"], 20)

    def test_raised_left_arm(self) -> None:
        m = metrics(LEFT_ARM_UP)
        self.assertGreater(m["left_arm_angle_deg"], 170)
        self.assertLess(m["right_arm_angle_deg"], 20)

    def test_arm_angle_is_the_existing_shoulder_angle(self) -> None:
        kp = {k.name: k for k in body(RIGHT_ARM_UP)}
        hip, shoulder, elbow = kp["right_hip"], kp["right_shoulder"], kp["right_elbow"]
        expected = angle_at_vertex_deg(
            (hip.u_px, hip.v_px), (shoulder.u_px, shoulder.v_px), (elbow.u_px, elbow.v_px)
        )
        self.assertEqual(metrics(RIGHT_ARM_UP)["right_arm_angle_deg"], expected)
        self.assertEqual(
            metrics(RIGHT_ARM_UP)["right_arm_angle_deg"],
            validate_2d_pose.image_plane_angles(kp, GATE)["right shoulder"],
        )


class KneeTest(unittest.TestCase):
    def test_standing_knees(self) -> None:
        m = metrics()
        self.assertAlmostEqual(m["left_knee_angle_deg"], 180.0, delta=0.5)
        self.assertAlmostEqual(m["right_knee_angle_deg"], 180.0, delta=0.5)

    def test_squat_knees(self) -> None:
        m = metrics(SQUAT)
        self.assertLess(m["left_knee_angle_deg"], 135)
        self.assertLess(m["right_knee_angle_deg"], 135)


class TorsoLeanTest(unittest.TestCase):
    def test_neutral_torso(self) -> None:
        self.assertAlmostEqual(metrics()["torso_lean_deg"], 0.0, delta=0.5)

    def test_lean_left_is_positive(self) -> None:
        self.assertGreater(metrics(LEAN_LEFT)["torso_lean_deg"], 8)

    def test_lean_right_is_negative(self) -> None:
        self.assertLess(metrics(LEAN_RIGHT)["torso_lean_deg"], -8)

    def test_mirrored_image_keeps_person_side(self) -> None:
        mirrored = {
            name: (1280 - u, v) for name, (u, v) in {**STANDING, **LEAN_LEFT}.items()
        }
        self.assertGreater(metrics(mirrored)["torso_lean_deg"], 8)


class MissingDataTest(unittest.TestCase):
    def test_missing_landmark_gives_null(self) -> None:
        m = metrics(missing=("right_wrist",))
        self.assertIsNone(m["right_elbow_angle_deg"])
        self.assertIsNotNone(m["left_elbow_angle_deg"])

    def test_low_confidence_landmark_gives_null(self) -> None:
        m = metrics(low=("left_ankle", "left_hip"))
        self.assertIsNone(m["left_knee_angle_deg"])
        self.assertIsNone(m["left_arm_angle_deg"])
        self.assertIsNone(m["torso_lean_deg"])
        self.assertIsNotNone(m["right_knee_angle_deg"])

    def test_no_person_gives_all_null_not_zero(self) -> None:
        self.assertTrue(all(value is None for value in game_metrics_2d({}, GATE).values()))


class MessageTest(unittest.TestCase):
    METRIC_KEYS = {
        "right_arm_angle_deg",
        "left_arm_angle_deg",
        "right_elbow_angle_deg",
        "left_elbow_angle_deg",
        "right_wrist_angle_deg",
        "left_wrist_angle_deg",
        "right_knee_angle_deg",
        "left_knee_angle_deg",
        "torso_lean_deg",
    }

    def test_message_contains_all_33_landmarks(self) -> None:
        message = build_pose_message(1, 1280, 720, body(RIGHT_ARM_UP), GATE, timestamp_ms=5)
        self.assertEqual(len(message["landmarks"]), 33)
        self.assertEqual([lm["index"] for lm in message["landmarks"]], list(range(33)))

    def test_message_contains_metrics(self) -> None:
        message = build_pose_message(1, 1280, 720, body(RIGHT_ARM_UP), GATE, timestamp_ms=5)
        self.assertEqual(set(message["metrics"]), self.METRIC_KEYS)
        self.assertGreater(message["metrics"]["right_arm_angle_deg"], 170)
        encoded = json.loads(json.dumps(message))
        self.assertEqual(encoded["metrics"], message["metrics"])

    def test_null_metric_is_json_null(self) -> None:
        message = build_pose_message(1, 1280, 720, body(missing=("right_wrist",)), GATE)
        self.assertIn('"right_elbow_angle_deg": null', json.dumps(message))

    def test_poor_tracking_still_sends_frame_and_measurable_metrics(self) -> None:
        message = build_pose_message(1, 1280, 720, body(low=("left_heel",)), GATE)
        self.assertEqual(message["tracking"], "POOR")
        self.assertEqual(len(message["landmarks"]), 33)
        self.assertIsNotNone(message["metrics"]["right_elbow_angle_deg"])


class TrackingStatusUnchangedTest(unittest.TestCase):
    def test_validator_uses_the_shared_functions(self) -> None:
        self.assertIs(validate_2d_pose.tracking_status, pose_metrics_2d.tracking_status)
        self.assertIs(validate_2d_pose.image_plane_angles, pose_metrics_2d.image_plane_angles)
        self.assertIs(validate_2d_pose.IMPORTANT, pose_metrics_2d.IMPORTANT)

    def test_tracking_status_behavior(self) -> None:
        by_name = {kp.name: kp for kp in body()}
        self.assertEqual(tracking_status(by_name, GATE), "GOOD")
        self.assertEqual(tracking_status({}, GATE), "POOR")
        for name in IMPORTANT:
            weak = {kp.name: kp for kp in body(low=(name,))}
            self.assertEqual(tracking_status(weak, GATE), "POOR", name)
        exact = {
            kp.name: Keypoint2D(kp.name, kp.u_px, kp.v_px, GATE) for kp in body()
        }
        self.assertEqual(tracking_status(exact, GATE), "GOOD")
        not_important = {kp.name: kp for kp in body(low=("left_eye", "mouth_left"))}
        self.assertEqual(tracking_status(not_important, GATE), "GOOD")


if __name__ == "__main__":
    unittest.main()
