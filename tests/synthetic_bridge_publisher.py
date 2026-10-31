"""Publish a synthetic raised-right-arm pose on the game bridge ports. No camera.

For the PatientBalanceAI end-to-end check (npm run test:bridge-e2e). Metrics are
calculated here by build_pose_message, exactly as in pose_game_bridge.py.
The preview on :8766 is a drawn test image with a dot at every keypoint, so the
game's skeleton overlay can be checked against it.

    python tests/synthetic_bridge_publisher.py [seconds]
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pose_game_bridge import POSE_BRIDGE_HOST, POSE_BRIDGE_PORT, PoseBridgeServer, build_pose_message
from src.io.preview_stream import PreviewStream
from src.pose.hands import HAND_STEMS, hand_joint_name
from src.pose.keypoints import Keypoint2D
from tests.test_pose_metrics_2d import GATE, RIGHT_ARM_UP, body

# 2D wrist bend the synthetic hands are drawn with. Python reports 180 - bend.
RIGHT_HAND_BEND_DEG = 40.0
LEFT_HAND_BEND_DEG = 0.0


def hand_on_forearm(keypoints, side: str, bend_deg: float, length_px: float = 55.0) -> list[Keypoint2D]:
    """21 Hands points: every hand joint on one line from the Pose wrist, turned bend_deg off the forearm."""
    by_name = {kp.name: kp for kp in keypoints}
    elbow, wrist = by_name[f"{side}_elbow"], by_name[f"{side}_wrist"]
    du, dv = wrist.u_px - elbow.u_px, wrist.v_px - elbow.v_px
    norm = math.hypot(du, dv)
    du, dv = du / norm, dv / norm
    turn = math.radians(bend_deg)
    hu = du * math.cos(turn) - dv * math.sin(turn)
    hv = du * math.sin(turn) + dv * math.cos(turn)
    return [
        Keypoint2D(
            name=hand_joint_name(side, stem),
            u_px=wrist.u_px + hu * (0.0 if stem == "wrist" else length_px),
            v_px=wrist.v_px + hv * (0.0 if stem == "wrist" else length_px),
            confidence=0.95,
        )
        for stem in HAND_STEMS
    ]


def test_image(keypoints, hand_keypoints=()) -> np.ndarray:
    image = np.full((720, 1280, 3), 60, dtype=np.uint8)
    cv2.putText(image, "SYNTHETIC TEST IMAGE (camera view, not mirrored)", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    for kp in keypoints:
        cv2.circle(image, (int(kp.u_px), int(kp.v_px)), 9, (0, 0, 255), -1, cv2.LINE_AA)
    for kp in hand_keypoints:
        cv2.circle(image, (int(kp.u_px), int(kp.v_px)), 6, (255, 0, 255), -1, cv2.LINE_AA)
    by_name = {kp.name: kp for kp in keypoints}
    wrist = by_name["right_wrist"]
    cv2.putText(image, "person's RIGHT wrist", (int(wrist.u_px) - 360, int(wrist.v_px) + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return image


async def main(seconds: float) -> None:
    import websockets

    bridge = PoseBridgeServer()
    keypoints = body(RIGHT_ARM_UP)
    hands = hand_on_forearm(keypoints, "right", RIGHT_HAND_BEND_DEG) + hand_on_forearm(keypoints, "left", LEFT_HAND_BEND_DEG)
    image = test_image(keypoints, hands)
    preview = PreviewStream()
    preview.start()
    try:
        async with websockets.serve(bridge.handler, POSE_BRIDGE_HOST, POSE_BRIDGE_PORT):
            print(f"Synthetic bridge on ws://{POSE_BRIDGE_HOST}:{POSE_BRIDGE_PORT}", flush=True)
            print(f"Synthetic preview on {preview.url}", flush=True)
            first = build_pose_message(0, 1280, 720, keypoints, GATE, hand_keypoints=hands)
            print("PYTHON_METRICS " + json.dumps(first["metrics"]), flush=True)
            frames = int(seconds * 30)
            for frame in range(1, frames + 1):
                message = build_pose_message(frame, 1280, 720, keypoints, GATE, hand_keypoints=hands)
                await bridge.publish(json.dumps(message))
                preview.submit(image)
                await asyncio.sleep(1 / 30)
    finally:
        preview.stop()


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 12.0))
