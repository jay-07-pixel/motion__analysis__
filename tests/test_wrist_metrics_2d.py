"""2D wrist bend from synthetic Pose + Hands pixel keypoints. No camera.

Person faces the camera (1280x720), arm held out to the side with the palm
down, as in the game's Bend Wrist demonstration. The person's RIGHT arm is on
the image LEFT (smaller u), the LEFT arm on the image right.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import unittest

import numpy as np

import pose_game_bridge
from pose_game_bridge import POSE_BRIDGE_HOST, PoseBridgeServer, build_pose_message
from src.analysis.pose_metrics_2d import game_metrics_2d
from src.analysis.wrist_metrics_2d import WRIST_METRICS, wrist_bend_deg, wrist_metrics_2d
from src.pose.hands import HAND_BONE_INDICES, HAND_STEMS, hand_joint_name, serialize_hands_2d
from src.pose.keypoints import Keypoint2D
from tests.test_pose_metrics_2d import GATE, RIGHT_ARM_UP, body

ARMS_OUT = {
    "right_elbow": (480, 200),
    "right_wrist": (380, 200),
    "left_elbow": (800, 200),
    "left_wrist": (900, 200),
}
HAND_LENGTH_PX = 60.0


def hand(side: str, bend_deg: float, score: float = 0.95, wrist_shift=(0.0, 0.0), drop=()) -> list[Keypoint2D]:
    """21 Hands keypoints. The hand points along the forearm, turned down by bend_deg."""
    wrist_u, wrist_v = ARMS_OUT[f"{side}_wrist"]
    wrist_u += wrist_shift[0]
    wrist_v += wrist_shift[1]
    outward = -1.0 if side == "right" else 1.0
    du = outward * math.cos(math.radians(bend_deg))
    dv = math.sin(math.radians(bend_deg))
    points = []
    for index, stem in enumerate(HAND_STEMS):
        if stem in drop:
            continue
        reach = 0.0 if stem == "wrist" else HAND_LENGTH_PX * (1.0 + 0.1 * (index % 4))
        if stem == "middle_mcp":
            reach = HAND_LENGTH_PX
        points.append(
            Keypoint2D(
                name=hand_joint_name(side, stem),
                u_px=wrist_u + du * reach,
                v_px=wrist_v + dv * reach,
                confidence=score,
            )
        )
    return points


def pose(overrides=None, low=(), missing=()) -> dict[str, Keypoint2D]:
    return {kp.name: kp for kp in body({**ARMS_OUT, **(overrides or {})}, low, missing)}


def wrists(hands, **pose_kwargs) -> dict:
    return wrist_metrics_2d(pose(**pose_kwargs), hands, GATE)


class WristBendTest(unittest.TestCase):
    def test_right_wrist_straight(self) -> None:
        self.assertAlmostEqual(wrists(hand("right", 0))["right_wrist_angle_deg"], 180.0, places=6)

    def test_right_wrist_bent(self) -> None:
        for bend in (20, 30, 52, 70):
            with self.subTest(bend=bend):
                self.assertAlmostEqual(wrists(hand("right", bend))["right_wrist_angle_deg"], 180.0 - bend, places=6)

    def test_left_wrist_straight(self) -> None:
        self.assertAlmostEqual(wrists(hand("left", 0))["left_wrist_angle_deg"], 180.0, places=6)

    def test_left_wrist_bent(self) -> None:
        self.assertAlmostEqual(wrists(hand("left", 52))["left_wrist_angle_deg"], 128.0, places=6)

    def test_each_hand_drives_only_its_own_metric(self) -> None:
        m = wrists(hand("right", 40))
        self.assertAlmostEqual(m["right_wrist_angle_deg"], 140.0, places=6)
        self.assertIsNone(m["left_wrist_angle_deg"])
        both = wrists(hand("right", 40) + hand("left", 10))
        self.assertAlmostEqual(both["right_wrist_angle_deg"], 140.0, places=6)
        self.assertAlmostEqual(both["left_wrist_angle_deg"], 170.0, places=6)

    def test_bend_up_and_down_give_the_same_unsigned_value(self) -> None:
        down = wrists(hand("right", 35))["right_wrist_angle_deg"]
        up = wrists(hand("right", -35))["right_wrist_angle_deg"]
        self.assertAlmostEqual(down, up, places=6)

    def test_finger_curl_does_not_change_the_wrist(self) -> None:
        curled = [
            Keypoint2D(name=kp.name, u_px=kp.u_px + 25.0, v_px=kp.v_px + 30.0, confidence=kp.confidence)
            if kp.name.endswith(("_tip", "_pip", "_dip", "_index", "_thumb", "_pinky")) else kp
            for kp in hand("right", 30)
        ]
        self.assertAlmostEqual(wrists(curled)["right_wrist_angle_deg"], 150.0, places=6)

    def test_vertex_is_the_hands_wrist(self) -> None:
        shifted = hand("right", 0, wrist_shift=(0.0, 12.0))
        expected = math.degrees(math.acos(
            ((480 - 380) * (-HAND_LENGTH_PX) + (200 - 212) * 0.0)
            / (math.hypot(100, 12) * HAND_LENGTH_PX)
        ))
        self.assertAlmostEqual(wrists(shifted)["right_wrist_angle_deg"], expected, places=6)


class WristNullTest(unittest.TestCase):
    def test_no_hands_is_null_not_zero(self) -> None:
        for hands in (None, []):
            m = wrists(hands)
            self.assertIsNone(m["right_wrist_angle_deg"])
            self.assertIsNone(m["left_wrist_angle_deg"])

    def test_missing_hand_landmark_is_null(self) -> None:
        self.assertIsNone(wrists(hand("right", 30, drop=("middle_mcp",)))["right_wrist_angle_deg"])
        self.assertIsNone(wrists(hand("right", 30, drop=("wrist",)))["right_wrist_angle_deg"])

    def test_low_hand_confidence_is_null(self) -> None:
        self.assertIsNone(wrists(hand("right", 30, score=0.3))["right_wrist_angle_deg"])
        self.assertAlmostEqual(wrists(hand("right", 30, score=GATE))["right_wrist_angle_deg"], 150.0, places=6)

    def test_low_or_missing_pose_elbow_is_null(self) -> None:
        self.assertIsNone(wrists(hand("right", 30), low=("right_elbow",))["right_wrist_angle_deg"])
        self.assertIsNone(wrists(hand("right", 30), missing=("right_elbow",))["right_wrist_angle_deg"])

    def test_pose_wrist_must_be_tracked_for_the_hand_to_belong_to_the_arm(self) -> None:
        self.assertIsNone(wrists(hand("right", 30), low=("right_wrist",))["right_wrist_angle_deg"])

    def test_degenerate_hand_is_null(self) -> None:
        flat = [Keypoint2D(name=kp.name, u_px=380.0, v_px=200.0, confidence=0.95) for kp in hand("right", 0)]
        self.assertIsNone(wrist_bend_deg(pose(), {kp.name: kp for kp in flat}, "right", GATE))


class BridgeMessageTest(unittest.TestCase):
    def test_message_has_both_wrist_metrics(self) -> None:
        message = build_pose_message(1, 1280, 720, body(ARMS_OUT), GATE, hand_keypoints=hand("right", 52) + hand("left", 0))
        self.assertAlmostEqual(message["metrics"]["right_wrist_angle_deg"], 128.0, places=6)
        self.assertAlmostEqual(message["metrics"]["left_wrist_angle_deg"], 180.0, places=6)

    def test_without_hands_the_wrist_fields_are_json_null(self) -> None:
        encoded = json.dumps(build_pose_message(1, 1280, 720, body(ARMS_OUT), GATE))
        decoded = json.loads(encoded)["metrics"]
        for name, _side in WRIST_METRICS:
            self.assertIn(f'"{name}": null', encoded)
            self.assertIsNone(decoded[name])

    def test_existing_metrics_are_unchanged_by_hands(self) -> None:
        for overrides in (ARMS_OUT, RIGHT_ARM_UP, None):
            keypoints = body(overrides)
            existing = game_metrics_2d({kp.name: kp for kp in keypoints}, GATE)
            hands = hand("right", 40, wrist_shift=(8.0, -6.0)) + hand("left", 25, wrist_shift=(-5.0, 9.0))
            with_hands = build_pose_message(1, 1280, 720, keypoints, GATE, hand_keypoints=hands)["metrics"]
            for name, value in existing.items():
                self.assertEqual(with_hands[name], value, name)
            self.assertEqual(set(with_hands), set(existing) | {name for name, _ in WRIST_METRICS})

    def test_landmarks_stay_the_33_pose_points(self) -> None:
        message = build_pose_message(1, 1280, 720, body(ARMS_OUT), GATE, hand_keypoints=hand("right", 30))
        self.assertEqual(len(message["landmarks"]), 33)
        right_wrist = next(lm for lm in message["landmarks"] if lm["name"] == "right_wrist")
        self.assertEqual((right_wrist["u_px"], right_wrist["v_px"]), (380.0, 200.0))

    def test_websocket_carries_wrist_metrics(self) -> None:
        async def scenario() -> dict:
            import websockets

            bridge = PoseBridgeServer()
            async with websockets.serve(bridge.handler, POSE_BRIDGE_HOST, 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://{POSE_BRIDGE_HOST}:{port}") as client:
                    await asyncio.sleep(0.05)
                    message = build_pose_message(1, 1280, 720, body(ARMS_OUT), GATE, hand_keypoints=hand("left", 45))
                    await bridge.publish(json.dumps(message))
                    return json.loads(await asyncio.wait_for(client.recv(), 2))

        received = asyncio.run(scenario())["metrics"]
        self.assertAlmostEqual(received["left_wrist_angle_deg"], 135.0, places=6)
        self.assertIsNone(received["right_wrist_angle_deg"])


class HandSkeletonMessageTest(unittest.TestCase):
    def test_mediapipe_hand_connections_match_hands_py(self) -> None:
        self.assertEqual(
            HAND_BONE_INDICES,
            (
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17),
            ),
        )

    def test_detected_right_hand_has_21_landmarks(self) -> None:
        message = build_pose_message(1, 1280, 720, body(ARMS_OUT), GATE, hand_keypoints=hand("right", 40))
        right = message["hands"]["right"]
        self.assertEqual(len(right), 21)
        self.assertEqual([p["index"] for p in right], list(range(21)))
        self.assertEqual([p["name"] for p in right], list(HAND_STEMS))
        self.assertEqual(right[0]["name"], "wrist")
        self.assertEqual(right[9]["name"], "middle_mcp")
        self.assertIsNone(message["hands"]["left"])

    def test_detected_left_hand_has_21_landmarks(self) -> None:
        message = build_pose_message(1, 1280, 720, body(ARMS_OUT), GATE, hand_keypoints=hand("left", 10))
        left = message["hands"]["left"]
        self.assertEqual(len(left), 21)
        self.assertEqual(left[0]["name"], "wrist")
        self.assertIsNone(message["hands"]["right"])

    def test_missing_hand_is_json_null(self) -> None:
        encoded = json.dumps(build_pose_message(1, 1280, 720, body(ARMS_OUT), GATE))
        self.assertIn('"left": null', encoded)
        self.assertIn('"right": null', encoded)
        hands = json.loads(encoded)["hands"]
        self.assertIsNone(hands["left"])
        self.assertIsNone(hands["right"])

    def test_partial_hand_is_null_not_invented(self) -> None:
        payload = serialize_hands_2d(hand("right", 30, drop=("pinky_tip",)))
        self.assertIsNone(payload["right"])
        self.assertIsNone(payload["left"])

    def test_wrist_metrics_use_the_same_hand_keypoints(self) -> None:
        keypoints = body(ARMS_OUT)
        hands = hand("right", 52)
        message = build_pose_message(1, 1280, 720, keypoints, GATE, hand_keypoints=hands)
        self.assertAlmostEqual(message["metrics"]["right_wrist_angle_deg"], 128.0, places=6)
        self.assertEqual(message["hands"]["right"][9]["name"], "middle_mcp")
        mcp = next(kp for kp in hands if kp.name == hand_joint_name("right", "middle_mcp"))
        self.assertEqual(message["hands"]["right"][9]["u_px"], mcp.u_px)
        self.assertEqual(message["hands"]["right"][9]["v_px"], mcp.v_px)

    def test_pose_landmarks_stay_33_when_hands_are_present(self) -> None:
        message = build_pose_message(
            1, 1280, 720, body(ARMS_OUT), GATE, hand_keypoints=hand("right", 30) + hand("left", 0)
        )
        self.assertEqual(len(message["landmarks"]), 33)
        self.assertEqual(len(message["hands"]["left"]), 21)
        self.assertEqual(len(message["hands"]["right"]), 21)
        pose_wrist = next(lm for lm in message["landmarks"] if lm["name"] == "right_wrist")
        self.assertEqual((pose_wrist["u_px"], pose_wrist["v_px"]), (380.0, 200.0))

    def test_websocket_carries_body_and_hands(self) -> None:
        async def scenario() -> dict:
            import websockets

            bridge = PoseBridgeServer()
            async with websockets.serve(bridge.handler, POSE_BRIDGE_HOST, 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://{POSE_BRIDGE_HOST}:{port}") as client:
                    await asyncio.sleep(0.05)
                    message = build_pose_message(
                        1, 1280, 720, body(ARMS_OUT), GATE, hand_keypoints=hand("right", 20)
                    )
                    await bridge.publish(json.dumps(message))
                    return json.loads(await asyncio.wait_for(client.recv(), 2))

        received = asyncio.run(scenario())
        self.assertEqual(len(received["landmarks"]), 33)
        self.assertEqual(len(received["hands"]["right"]), 21)
        self.assertIsNone(received["hands"]["left"])
        self.assertIn("right_wrist_angle_deg", received["metrics"])


class BridgePumpTest(unittest.TestCase):
    def test_hands_run_on_the_pose_frame_with_the_pose_keypoints(self) -> None:
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)]
        keep_alive = list(frames)
        stop = threading.Event()
        pose_keypoints = body(ARMS_OUT)

        class Camera:
            def get_frame(self):
                if not frames:
                    stop.set()
                    return None
                return frames.pop(0)

        pose_frames: list[int] = []
        hand_calls: list[tuple[int, int]] = []
        payloads: list[str] = []

        class Extractor:
            def extract(self, image):
                pose_frames.append(id(image))
                return pose_keypoints

        class Hands:
            def extract(self, image, pose_keypoints=None):
                hand_calls.append((id(image), id(pose_keypoints)))
                return hand("right", 52)

        class Server:
            async def publish(self, payload):
                payloads.append(payload)

        original = pose_game_bridge.asyncio.run_coroutine_threadsafe

        def run_now(coro, _loop):
            try:
                coro.send(None)
            except StopIteration:
                pass

        pose_game_bridge.asyncio.run_coroutine_threadsafe = run_now
        try:
            pose_game_bridge._pump_camera(Camera(), Extractor(), Server(), object(), stop, GATE, lambda: None, None, Hands())
        finally:
            pose_game_bridge.asyncio.run_coroutine_threadsafe = original
        self.assertEqual([frame_id for frame_id, _ in hand_calls], pose_frames)
        self.assertTrue(all(kp_id == id(pose_keypoints) for _, kp_id in hand_calls))
        self.assertEqual(len(payloads), 2)
        first = json.loads(payloads[0])
        self.assertAlmostEqual(first["metrics"]["right_wrist_angle_deg"], 128.0, places=6)
        self.assertEqual(len(first["hands"]["right"]), 21)
        self.assertIsNone(first["hands"]["left"])
        del keep_alive

    def test_bridge_reuses_the_existing_hands_pipeline(self) -> None:
        import ast
        from pathlib import Path

        source = (Path(pose_game_bridge.__file__)).read_text(encoding="utf-8")
        calls = {
            getattr(node.func, "id", None)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
        }
        self.assertIn("HandsExtractor2D", calls)
        self.assertNotIn("mp.solutions.hands", source)
        names = [
            getattr(node.func, "id", None)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
        ]
        self.assertEqual(names.count("PoseExtractor2D"), 1)
        self.assertEqual(names.count("HandsExtractor2D"), 1)
        self.assertEqual(names.count("LiveRealSenseRGB"), 1)
        self.assertEqual(source.count("hands.extract("), 1)


if __name__ == "__main__":
    unittest.main()
