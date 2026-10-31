"""Contract tests for the game pose bridge. They do not open the camera."""

from __future__ import annotations

import asyncio
import json
import unittest

from src.pose.keypoints import Keypoint2D
from src.pose.skeleton import LANDMARK_NAMES
from pose_game_bridge import POSE_BRIDGE_HOST, PoseBridgeServer, build_pose_message
from validate_2d_pose import IMPORTANT


def _body(confidence: float) -> list[Keypoint2D]:
    return [
        Keypoint2D(name=name, u_px=100.0 + index, v_px=200.0 + index, confidence=confidence)
        for index, name in enumerate(LANDMARK_NAMES)
    ]


class PoseMessageContractTest(unittest.TestCase):
    def test_sends_all_33_pixel_landmarks(self) -> None:
        message = build_pose_message(4, 1280, 720, _body(0.9), 0.5, timestamp_ms=10)
        self.assertEqual(message["timestamp_ms"], 10)
        self.assertEqual(message["frame"], 4)
        self.assertEqual(message["width"], 1280)
        self.assertEqual(message["height"], 720)
        self.assertEqual(message["tracking"], "GOOD")
        self.assertEqual(len(message["landmarks"]), 33)
        nose = message["landmarks"][0]
        self.assertEqual(nose["index"], 0)
        self.assertEqual(nose["name"], "nose")
        self.assertEqual(nose["u_px"], 100.0)
        self.assertEqual(nose["v_px"], 200.0)
        self.assertNotIn("z", nose)
        self.assertNotIn("x_m", nose)
        encoded = json.dumps(message)
        for banned in ("world_landmark", "depth", "dof", "marker"):
            self.assertNotIn(banned, encoded)

    def test_tracking_uses_validate_2d_pose_gate(self) -> None:
        good = build_pose_message(1, 1280, 720, _body(0.5), 0.5)
        self.assertEqual(good["tracking"], "GOOD")

        weak = _body(0.9)
        heel = next(kp for kp in weak if kp.name == IMPORTANT[-1])
        weak[LANDMARK_NAMES.index(heel.name)] = Keypoint2D(
            name=heel.name, u_px=heel.u_px, v_px=heel.v_px, confidence=0.49
        )
        poor = build_pose_message(2, 1280, 720, weak, 0.5)
        self.assertEqual(poor["tracking"], "POOR")
        self.assertEqual(len(poor["landmarks"]), 33)

    def test_no_person_is_poor_and_sends_no_invented_points(self) -> None:
        message = build_pose_message(3, 1280, 720, [], 0.5)
        self.assertEqual(message["tracking"], "POOR")
        self.assertEqual(message["landmarks"], [])


class PoseSocketTest(unittest.TestCase):
    def test_client_can_reconnect_after_disconnect(self) -> None:
        async def scenario() -> None:
            import websockets

            bridge = PoseBridgeServer()
            stop = asyncio.Event()

            async with websockets.serve(bridge.handler, POSE_BRIDGE_HOST, 0) as server:
                port = server.sockets[0].getsockname()[1]
                url = f"ws://{POSE_BRIDGE_HOST}:{port}"
                async with websockets.connect(url) as first:
                    await asyncio.sleep(0.05)
                    await bridge.publish('{"frame":1}')
                    self.assertEqual(await asyncio.wait_for(first.recv(), 2), '{"frame":1}')
                async with websockets.connect(url) as second:
                    await asyncio.sleep(0.05)
                    await bridge.publish('{"frame":2}')
                    self.assertEqual(await asyncio.wait_for(second.recv(), 2), '{"frame":2}')
            stop.set()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
