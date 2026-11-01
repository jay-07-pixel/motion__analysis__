"""Wrist duplicate check after side assignment, before Hands overwrites Pose."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from src.diagnostics.recorder import LiveDiagnosticsRecorder
from src.pose.hands import HandsExtractor2D, merge_pose_and_hands
from src.pose.keypoints import Keypoint2D
from src.pose.wrist_identity import (
    CLASS_CROSSING,
    CLASS_DUPLICATE,
    CLASS_INSUFFICIENT,
    CLASS_NORMAL,
    CLASS_UNCERTAIN,
    validate_wrist_identity,
)

CAPTURE = Path("data/diagnostics/20261101-145709/landmarks.csv")
CAPTURE_154302 = Path("data/diagnostics/20261101-154302/wrist_identity.csv")


class _Hands:
    """Assignment and validation only. Does not load a MediaPipe graph."""

    def __init__(self) -> None:
        self._swap_handedness = True
        self._match_to_pose_wrists = True
        self._min_pose_confidence = 0.5
        self.last_wrist_identity = None

    keypoints_from_inference = HandsExtractor2D.keypoints_from_inference


def _kp(name: str, u_px: float, v_px: float, confidence: float = 0.99) -> Keypoint2D:
    return Keypoint2D(name, u_px, v_px, confidence)


def _pixels(u_px: float, v_px: float) -> list[tuple[float, float]]:
    return [(u_px + index * 0.01, v_px) for index in range(21)]


def _apply(raw, pose):
    hands = _Hands()
    named = hands.keypoints_from_inference(raw, pose)
    merged = merge_pose_and_hands(pose, named)
    return hands.last_wrist_identity, {kp.name: kp for kp in merged}


class WristIdentityTest(unittest.TestCase):
    def test_frame_2063_duplicate_near_left_elbow_rejects_right_overwrite(self) -> None:
        pose = [
            _kp("left_elbow", 692.261, 235.259, 0.9725),
            _kp("right_elbow", 493.732, 250.216, 0.9791),
            _kp("left_wrist", 733.355, 220.293, 0.97),
            _kp("right_wrist", 461.633, 221.871, 0.97),
            _kp("right_index", 443.934, 197.644, 0.96),
            _kp("right_thumb", 450.0, 210.0, 0.96),
            _kp("right_pinky", 447.986, 217.062, 0.96),
        ]
        left_pixels = _pixels(733.355, 220.293)
        right_pixels = _pixels(729.548, 226.548)
        raw = [("right", 0.9931, left_pixels), ("left", 0.9852, right_pixels)]
        decision, merged = _apply(raw, pose)

        self.assertEqual(decision.classification, CLASS_DUPLICATE)
        self.assertEqual(decision.reason, "both_detections_match_pose_left_wrist")
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "rejected")
        self.assertAlmostEqual(merged["left_wrist"].u_px, 733.355)
        self.assertAlmostEqual(merged["left_wrist"].v_px, 220.293)
        self.assertAlmostEqual(merged["right_wrist"].u_px, 461.633)
        self.assertAlmostEqual(merged["right_wrist"].v_px, 221.871)
        self.assertAlmostEqual(merged["right_index"].u_px, 443.934)
        self.assertAlmostEqual(merged["right_thumb"].u_px, 450.0)
        self.assertAlmostEqual(merged["right_pinky"].u_px, 447.986)
        self.assertNotIn("right_index_mcp", merged)
        self.assertEqual(decision.raw_hands[0].handedness, "right")
        self.assertEqual(decision.raw_hands[1].handedness, "left")
        self.assertAlmostEqual(decision.raw_hands[0].score, 0.9931)
        self.assertEqual(len(decision.raw_hands[0].landmarks_px), 21)
        self.assertEqual(len(decision.raw_hands[1].landmarks_px), 21)
        self.assertIsNot(decision.raw_hands[0].landmarks_px, left_pixels)
        left_pixels[0] = (0.0, 0.0)
        self.assertAlmostEqual(decision.raw_hands[0].wrist_px[0], 733.355)

    def test_frame_2772_duplicate_near_right_elbow_rejects_left_overwrite(self) -> None:
        pose = [
            _kp("left_elbow", 791.697, 152.713, 0.9952),
            _kp("right_elbow", 354.091, 159.480, 0.9934),
            _kp("left_wrist", 910.166, 186.644, 0.99),
            _kp("right_wrist", 207.895, 205.953, 0.99),
            _kp("left_index", 953.245, 193.470, 0.98),
            _kp("left_thumb", 940.0, 190.0, 0.98),
            _kp("left_pinky", 951.737, 195.636, 0.98),
        ]
        raw = [
            ("left", 0.6996, _pixels(231.951, 199.707)),
            ("right", 0.9875, _pixels(207.895, 205.953)),
        ]
        decision, merged = _apply(raw, pose)

        self.assertEqual(decision.classification, CLASS_DUPLICATE)
        self.assertEqual(decision.reason, "both_detections_match_pose_right_wrist")
        self.assertEqual(decision.overwrite("left"), "rejected")
        self.assertEqual(decision.overwrite("right"), "accepted")
        self.assertAlmostEqual(merged["left_wrist"].u_px, 910.166)
        self.assertAlmostEqual(merged["left_wrist"].v_px, 186.644)
        self.assertAlmostEqual(merged["left_index"].u_px, 953.245)
        self.assertAlmostEqual(merged["left_thumb"].u_px, 940.0)
        self.assertAlmostEqual(merged["left_pinky"].u_px, 951.737)
        self.assertAlmostEqual(merged["right_wrist"].u_px, 207.895)
        self.assertNotIn("left_index_mcp", merged)

    def test_fast_movement_toward_own_elbow_is_accepted_unchanged(self) -> None:
        pose = [
            _kp("left_elbow", 500.0, 300.0),
            _kp("right_elbow", 150.0, 300.0),
            _kp("left_wrist", 500.0, 300.0),
            _kp("right_wrist", 150.0, 320.0),
        ]
        far = _pixels(500.0, 100.0)
        raw = [("left", 0.99, far), ("right", 0.99, _pixels(150.0, 320.0))]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_NORMAL)
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "accepted")
        self.assertAlmostEqual(merged["left_wrist"].u_px, 500.0)
        self.assertAlmostEqual(merged["left_wrist"].v_px, 100.0)
        self.assertAlmostEqual(abs(merged["left_wrist"].v_px - 300.0), 200.0)

        nearer = [("left", 0.99, _pixels(500.0, 300.0)), ("right", 0.99, _pixels(150.0, 320.0))]
        second, merged_again = _apply(nearer, pose)
        self.assertEqual(second.classification, CLASS_NORMAL)
        self.assertAlmostEqual(merged_again["left_wrist"].u_px, 500.0)
        self.assertAlmostEqual(merged_again["left_wrist"].v_px, 300.0)

    def test_genuine_crossing_is_flagged_and_not_swapped(self) -> None:
        pose = [
            _kp("left_elbow", 700.0, 200.0),
            _kp("right_elbow", 300.0, 200.0),
            _kp("left_wrist", 320.0, 200.0),
            _kp("right_wrist", 680.0, 200.0),
        ]
        raw = [
            ("left", 0.91, _pixels(320.0, 210.0)),
            ("right", 0.92, _pixels(680.0, 190.0)),
        ]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_CROSSING)
        self.assertEqual(decision.reason, "each_wrist_nearer_opposite_elbow")
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "accepted")
        self.assertAlmostEqual(merged["left_wrist"].u_px, 320.0)
        self.assertAlmostEqual(merged["left_wrist"].v_px, 210.0)
        self.assertAlmostEqual(merged["right_wrist"].u_px, 680.0)
        self.assertAlmostEqual(merged["right_wrist"].v_px, 190.0)

    def test_single_hand_assignment_is_unchanged(self) -> None:
        wrist = _pixels(710.0, 230.0)
        pose = [
            _kp("left_elbow", 700.0, 250.0),
            _kp("right_elbow", 400.0, 250.0),
            _kp("left_wrist", 700.0, 240.0),
            _kp("right_wrist", 400.0, 240.0),
        ]
        decision, merged = _apply([("right", 0.88, wrist)], pose)
        self.assertEqual(decision.classification, CLASS_NORMAL)
        self.assertEqual(decision.reason, "single_wrist_nearer_own_elbow")
        self.assertIn("left_index_mcp", merged)
        self.assertNotIn("right_index_mcp", merged)
        self.assertAlmostEqual(merged["left_wrist"].u_px, 710.0)
        self.assertAlmostEqual(merged["left_wrist"].v_px, 230.0)
        self.assertAlmostEqual(merged["right_wrist"].u_px, 400.0)
        self.assertAlmostEqual(merged["right_wrist"].v_px, 240.0)
        self.assertNotIn("right_index_mcp", merged)

    def test_anatomical_left_hand_on_image_right_follows_elbow(self) -> None:
        pose = [
            _kp("left_elbow", 900.0, 200.0),
            _kp("right_elbow", 100.0, 200.0),
            _kp("left_wrist", 880.0, 200.0),
            _kp("right_wrist", 120.0, 200.0),
        ]
        raw = [
            ("left", 0.95, _pixels(120.0, 200.0)),
            ("right", 0.95, _pixels(880.0, 200.0)),
        ]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_NORMAL)
        self.assertGreater(merged["left_wrist"].u_px, merged["right_wrist"].u_px)
        self.assertAlmostEqual(merged["left_wrist"].u_px, 880.0)
        self.assertAlmostEqual(merged["right_wrist"].u_px, 120.0)

    def test_missing_elbows_still_reject_a_cluster_on_one_pose_wrist(self) -> None:
        pose = [
            _kp("left_wrist", 733.355, 220.293),
            _kp("right_wrist", 461.633, 221.871),
            _kp("right_index", 443.934, 197.644),
        ]
        raw = [
            ("right", 0.99, _pixels(733.355, 220.293)),
            ("left", 0.99, _pixels(729.548, 226.548)),
        ]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_DUPLICATE)
        self.assertEqual(decision.reason, "both_detections_match_pose_left_wrist")
        self.assertEqual(decision.overwrite("right"), "rejected")
        self.assertAlmostEqual(merged["right_wrist"].u_px, 461.633)
        self.assertAlmostEqual(merged["right_index"].u_px, 443.934)

    def test_missing_pose_wrists_and_elbows_do_not_reject(self) -> None:
        raw = [
            ("left", 0.9, _pixels(733.355, 220.293)),
            ("right", 0.9, _pixels(729.548, 226.548)),
        ]
        decision = validate_wrist_identity(raw, ["left", "right"], [])
        self.assertEqual(decision.classification, CLASS_INSUFFICIENT)
        self.assertEqual(decision.reason, "missing_pose_wrist")
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "accepted")

    def test_low_confidence_pose_wrist_does_not_reject_the_cluster(self) -> None:
        pose = [
            _kp("left_elbow", 692.261, 235.259, 0.97),
            _kp("right_elbow", 493.732, 250.216, 0.97),
            _kp("left_wrist", 733.355, 220.293, 0.2),
            _kp("right_wrist", 461.633, 221.871, 0.99),
        ]
        raw = [
            ("left", 0.99, _pixels(733.355, 220.293)),
            ("right", 0.99, _pixels(729.548, 226.548)),
        ]
        decision = validate_wrist_identity(raw, ["left", "right"], pose)
        self.assertEqual(decision.classification, CLASS_INSUFFICIENT)
        self.assertEqual(decision.reason, "low_confidence_pose_wrist")
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "accepted")

    def test_missing_wrist_detections_are_insufficient(self) -> None:
        pose = [
            _kp("left_elbow", 700.0, 200.0),
            _kp("right_elbow", 300.0, 200.0),
            _kp("left_wrist", 710.0, 210.0),
        ]
        decision, merged = _apply([], pose)
        self.assertEqual(decision.classification, CLASS_INSUFFICIENT)
        self.assertEqual(decision.reason, "no_hand_detections")
        self.assertEqual(decision.raw_hands, ())
        self.assertAlmostEqual(merged["left_wrist"].u_px, 710.0)

        empty = validate_wrist_identity(
            [("left", 0.4, [])],
            ["left"],
            pose,
        )
        self.assertEqual(empty.classification, CLASS_INSUFFICIENT)
        self.assertEqual(empty.reason, "missing_wrist_detection")
        self.assertEqual(empty.accepted_sides, ())
        self.assertEqual(len(empty.raw_hands[0].landmarks_px), 0)

    def test_low_confidence_pose_elbow_does_not_reject(self) -> None:
        pose = [
            _kp("left_elbow", 692.261, 235.259, 0.49),
            _kp("right_elbow", 493.732, 250.216, 0.99),
            _kp("left_wrist", 733.355, 220.293, 0.2),
            _kp("right_wrist", 461.633, 221.871, 0.99),
        ]
        raw = [
            ("right", 0.9931, _pixels(733.355, 220.293)),
            ("left", 0.9852, _pixels(729.548, 226.548)),
        ]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_INSUFFICIENT)
        self.assertEqual(decision.reason, "low_confidence_pose_elbow")
        self.assertEqual(decision.overwrite("right"), "accepted")
        self.assertAlmostEqual(merged["right_wrist"].u_px, 729.548)
        self.assertAlmostEqual(decision.pose_left_elbow.confidence, 0.49)
        self.assertAlmostEqual(decision.pose_left_wrist.confidence, 0.2)

    def test_frame_2599_keeps_the_pose_wrist_that_matches_the_cluster(self) -> None:
        pose = [
            _kp("left_elbow", 654.688, 218.980, 0.9723),
            _kp("right_elbow", 582.604, 213.555, 0.9904),
            _kp("left_wrist", 570.208, 123.151, 0.9655),
            _kp("right_wrist", 676.390, 155.147, 0.9601),
            _kp("right_index", 690.0, 140.0, 0.96),
            _kp("left_index", 560.0, 100.0, 0.96),
        ]
        raw = [
            ("right", 0.9353, _pixels(564.159, 119.889)),
            ("right", 0.9768, _pixels(556.420, 111.371)),
        ]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_DUPLICATE)
        self.assertEqual(decision.reason, "both_detections_match_pose_left_wrist")
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "rejected")
        self.assertAlmostEqual(merged["right_wrist"].u_px, 676.390)
        self.assertAlmostEqual(merged["right_wrist"].v_px, 155.147)
        self.assertAlmostEqual(merged["right_index"].u_px, 690.0)
        self.assertIn("left_index_mcp", merged)
        self.assertNotIn("right_index_mcp", merged)
        self.assertGreater(abs(merged["left_wrist"].u_px - merged["right_wrist"].u_px), 80.0)

    def test_frame_2603_matches_2599(self) -> None:
        pose = [
            _kp("left_elbow", 656.945, 224.394, 0.9784),
            _kp("right_elbow", 581.464, 227.879, 0.9908),
            _kp("left_wrist", 568.118, 127.366, 0.9728),
            _kp("right_wrist", 672.677, 155.208, 0.9604),
        ]
        raw = [
            ("right", 0.7518, _pixels(553.507, 111.727)),
            ("left", 0.8389, _pixels(554.047, 116.969)),
        ]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_DUPLICATE)
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "rejected")
        self.assertAlmostEqual(merged["right_wrist"].u_px, 672.677)
        self.assertLess(abs(merged["left_wrist"].u_px - 554.0), 5.0)

    def test_frames_2924_separated_hands_near_one_elbow_are_kept(self) -> None:
        pose = [
            _kp("left_elbow", 706.166, 281.573, 0.9829),
            _kp("right_elbow", 552.251, 249.572, 0.9921),
            _kp("left_wrist", 733.341, 408.632, 0.9793),
            _kp("right_wrist", 666.555, 185.061, 0.9760),
        ]
        raw = [
            ("left", 0.9676, _pixels(677.642, 179.692)),
            ("right", 0.9610, _pixels(724.140, 420.164)),
        ]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_NORMAL)
        self.assertEqual(decision.reason, "each_detection_matches_own_pose_wrist")
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "accepted")
        self.assertIn("left_index_mcp", merged)
        self.assertIn("right_index_mcp", merged)
        self.assertAlmostEqual(merged["right_wrist"].u_px, 677.642)
        self.assertAlmostEqual(merged["left_wrist"].u_px, 724.140)

    def test_frames_3156_separated_hands_are_not_a_duplicate(self) -> None:
        pose = [
            _kp("left_elbow", 735.3, 250.1, 0.9936),
            _kp("right_elbow", 556.4, 265.6, 0.9962),
            _kp("left_wrist", 720.7, 126.2, 0.9919),
            _kp("right_wrist", 641.6, 149.8, 0.99),
        ]
        raw = [
            ("left", 0.962, _pixels(652.2, 154.6)),
            ("right", 0.978, _pixels(714.5, 122.3)),
        ]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_NORMAL)
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "accepted")
        self.assertIn("right_index_mcp", merged)
        self.assertAlmostEqual(merged["right_wrist"].u_px, 652.2)
        self.assertAlmostEqual(merged["left_wrist"].u_px, 714.5)

    def test_tight_cluster_without_a_clear_pose_wrist_is_uncertain(self) -> None:
        pose = [
            _kp("left_elbow", 718.086, 257.074, 0.5837),
            _kp("right_elbow", 620.461, 265.214, 0.9955),
            _kp("left_wrist", 743.826, 362.012, 0.6463),
            _kp("right_wrist", 720.194, 369.046, 0.9813),
        ]
        raw = [
            ("left", 0.9907, _pixels(719.404, 383.438)),
            ("left", 0.9975, _pixels(705.442, 361.055)),
        ]
        decision, merged = _apply(raw, pose)
        self.assertEqual(decision.classification, CLASS_UNCERTAIN)
        self.assertEqual(decision.reason, "cluster_without_clear_pose_wrist")
        self.assertEqual(decision.overwrite("left"), "accepted")
        self.assertEqual(decision.overwrite("right"), "accepted")
        self.assertIn("left_index_mcp", merged)
        self.assertIn("right_index_mcp", merged)
        self.assertAlmostEqual(merged["left_wrist"].u_px, 719.404, places=2)

    def test_diagnostic_row_keeps_raw_detections_apart_from_accepted_landmarks(self) -> None:
        pose = [
            _kp("left_elbow", 692.261, 235.259),
            _kp("right_elbow", 493.732, 250.216),
            _kp("left_wrist", 733.355, 220.293),
            _kp("right_wrist", 461.633, 221.871),
        ]
        raw = [
            ("right", 0.9931, _pixels(733.355, 220.293)),
            ("left", 0.9852, _pixels(729.548, 226.548)),
        ]
        decision, merged = _apply(raw, pose)
        folder = Path(tempfile.mkdtemp()) / "diag"
        recorder = LiveDiagnosticsRecorder(folder)
        recorder.write_frame(2063, 1.0, 10.0, 20.0, list(merged.values()), [])
        recorder.write_wrist_identity(2063, 1.0, decision)
        summary = recorder.close()
        self.assertEqual(summary["wrist_identity"][CLASS_DUPLICATE], 1)
        with recorder.wrist_identity_path.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["classification"], CLASS_DUPLICATE)
        self.assertEqual(row["left_overwrite"], "accepted")
        self.assertEqual(row["right_overwrite"], "rejected")
        self.assertEqual(row["mp_label_0"], "right")
        self.assertEqual(row["mp_score_1"], "0.9852")
        raw_json = json.loads(row["raw_hands_json"])
        self.assertEqual(len(raw_json), 2)
        self.assertEqual(len(raw_json[1]["landmarks_px"]), 21)
        self.assertAlmostEqual(raw_json[1]["landmarks_px"][0][0], 729.548)
        with recorder.landmarks_path.open(encoding="utf-8", newline="") as handle:
            accepted = {item["landmark"]: item for item in csv.DictReader(handle)}
        self.assertEqual(accepted["right_wrist"]["u_px"], "461.633")
        self.assertNotEqual(accepted["right_wrist"]["u_px"], row["raw_wrist_u_1"])


def replay_capture(path: Path) -> dict[str, int]:
    """Classify recorded wrists against recorded elbows. Does not run MediaPipe."""
    grouped: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["frame"])][row["landmark"]] = row
    counts = {
        CLASS_NORMAL: 0,
        CLASS_DUPLICATE: 0,
        CLASS_CROSSING: 0,
        CLASS_UNCERTAIN: 0,
        CLASS_INSUFFICIENT: 0,
    }
    details = {}
    for frame, joints in grouped.items():
        pose = []
        for name in ("left_elbow", "right_elbow"):
            item = joints.get(name)
            if item and item["u_px"] != "":
                pose.append(
                    _kp(name, float(item["u_px"]), float(item["v_px"]), float(item["visibility"] or 0))
                )
        raw = []
        sides = []
        for side in ("left", "right"):
            item = joints.get(f"{side}_wrist")
            if item and item["u_px"] != "":
                raw.append(
                    (
                        "recorded",
                        float(item["visibility"] or 0),
                        _pixels(float(item["u_px"]), float(item["v_px"])),
                    )
                )
                sides.append(side)
        decision = validate_wrist_identity(raw, sides, pose, min_pose_confidence=0.5)
        counts[decision.classification] += 1
        if frame in (2063, 2064, 2772, 2773):
            details[frame] = decision
    return {"counts": counts, "details": details, "frames": len(grouped)}


def decision_from_identity_row(row: dict[str, str]):
    """Replay one saved wrist_identity.csv row through the current check."""
    hands = json.loads(row["raw_hands_json"])
    raw = []
    sides = []
    for index, hand in enumerate(hands):
        raw.append(
            (
                hand["handedness"],
                float(hand["score"]),
                [tuple(point) for point in hand["landmarks_px"]],
            )
        )
        side = row.get(f"assigned_side_{index}") or None
        sides.append(None if side == "" else side)
    pose = []
    for name in ("left_wrist", "right_wrist", "left_elbow", "right_elbow"):
        if row.get(f"pose_{name}_u", "") == "":
            continue
        pose.append(
            _kp(
                name,
                float(row[f"pose_{name}_u"]),
                float(row[f"pose_{name}_v"]),
                float(row[f"pose_{name}_confidence"] or 0.0),
            )
        )
    return validate_wrist_identity(raw, sides, pose)


class Capture154302ReplayTest(unittest.TestCase):
    def test_listed_frames_and_true_stacks(self) -> None:
        if not CAPTURE_154302.is_file():
            self.skipTest(f"missing {CAPTURE_154302}")
        with CAPTURE_154302.open(encoding="utf-8", newline="") as handle:
            rows = {int(row["frame"]): row for row in csv.DictReader(handle)}
        separated = list(range(2924, 2948)) + [2949, 3151] + list(range(3156, 3161))
        for frame in separated:
            decision = decision_from_identity_row(rows[frame])
            self.assertEqual(decision.classification, CLASS_NORMAL, frame)
            self.assertEqual(decision.overwrite("left"), "accepted", frame)
            self.assertEqual(decision.overwrite("right"), "accepted", frame)
        for frame in (2599, 2603):
            decision = decision_from_identity_row(rows[frame])
            self.assertEqual(decision.classification, CLASS_DUPLICATE, frame)
            self.assertEqual(decision.reason, "both_detections_match_pose_left_wrist", frame)
            self.assertEqual(decision.overwrite("left"), "accepted", frame)
            self.assertEqual(decision.overwrite("right"), "rejected", frame)
        for frame in (2436, 2489, 3190, 3205, 3283):
            decision = decision_from_identity_row(rows[frame])
            self.assertEqual(decision.classification, CLASS_DUPLICATE, frame)
            self.assertEqual(decision.reason, "both_detections_match_pose_right_wrist", frame)
            self.assertEqual(decision.overwrite("left"), "rejected", frame)
            self.assertEqual(decision.overwrite("right"), "accepted", frame)


class RecordedCaptureReplayTest(unittest.TestCase):
    def test_recorded_events_2063_and_2772(self) -> None:
        if not CAPTURE.is_file():
            self.skipTest(f"missing {CAPTURE}")
        replay = replay_capture(CAPTURE)
        left_dup = replay["details"][2063]
        right_dup = replay["details"][2772]
        self.assertEqual(left_dup.classification, CLASS_INSUFFICIENT)
        self.assertEqual(left_dup.reason, "missing_pose_wrist")
        self.assertEqual(left_dup.overwrite("right"), "accepted")
        self.assertEqual(left_dup.overwrite("left"), "accepted")
        self.assertEqual(right_dup.classification, CLASS_INSUFFICIENT)
        self.assertEqual(right_dup.reason, "missing_pose_wrist")
        self.assertEqual(right_dup.overwrite("left"), "accepted")
        self.assertEqual(right_dup.overwrite("right"), "accepted")
        self.assertEqual(replay["details"][2064].classification, CLASS_NORMAL)
        self.assertEqual(replay["details"][2773].classification, CLASS_NORMAL)


if __name__ == "__main__":
    unittest.main()
