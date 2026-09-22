"""MediaPipe Hands: 21 finger joints per hand, in camera pixels.

Pose only has wrist + thumb tip + index tip + pinky tip. That is not a
hand skeleton. Hands adds knuckles and every fingertip so Both hands can
draw a proper stick figure.

z from this model is NOT RealSense metres. We keep 2D (u_px, v_px) only.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from src.pose.keypoints import Keypoint2D

# Official MediaPipe Hands order (index 0 .. 20).
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_STEMS: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

# Pose already uses these four names. Map Hands tips/wrist onto them so
# wrist angles and overlays keep working.
_POSE_ALIAS = {
    "wrist": "wrist",
    "thumb_tip": "thumb",
    "index_tip": "index",
    "pinky_tip": "pinky",
}

# Finger chains + palm. Stems are turned into left_/right_ names below.
_HAND_CHAINS: tuple[tuple[str, ...], ...] = (
    ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp", "index_pip", "index_dip", "index_tip"),
    ("wrist", "middle_mcp", "middle_pip", "middle_dip", "middle_tip"),
    ("wrist", "ring_mcp", "ring_pip", "ring_dip", "ring_tip"),
    ("wrist", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"),
    ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"),
)


def hand_joint_name(side: str, stem: str) -> str:
    """Build a joint name. Tips reuse Pose names (left_index, left_thumb, ...)."""
    aliased = _POSE_ALIAS.get(stem, stem)
    return f"{side}_{aliased}"


def _bones_for_side(side: str) -> tuple[tuple[str, str], ...]:
    """One hand's stick-figure links in camera pixels."""
    bones: list[tuple[str, str]] = []
    for chain in _HAND_CHAINS:
        for start, end in zip(chain, chain[1:]):
            bones.append((hand_joint_name(side, start), hand_joint_name(side, end)))
    return tuple(bones)


HAND_SKELETON_BONES: tuple[tuple[str, str], ...] = _bones_for_side("left") + _bones_for_side(
    "right"
)

# Pose draws a straight wrist→tip line. Skip those when the full finger exists.
POSE_SIMPLE_HAND_BONES: frozenset[tuple[str, str]] = frozenset(
    {
        ("left_wrist", "left_index"),
        ("right_wrist", "right_index"),
        ("left_wrist", "left_thumb"),
        ("right_wrist", "right_thumb"),
        ("left_wrist", "left_pinky"),
        ("right_wrist", "right_pinky"),
    }
)


def merge_pose_and_hands(
    pose_keypoints: list[Keypoint2D],
    hand_keypoints: list[Keypoint2D],
) -> list[Keypoint2D]:
    """Pose body first, then Hands. Hands overwrite shared names (wrist, tips)."""
    if not hand_keypoints:
        return pose_keypoints
    if not pose_keypoints:
        return hand_keypoints

    by_name = {kp.name: kp for kp in pose_keypoints}
    pose_names = [kp.name for kp in pose_keypoints]
    extras: list[Keypoint2D] = []
    for kp in hand_keypoints:
        by_name[kp.name] = kp
        if kp.name not in pose_names:
            extras.append(kp)
    merged = [by_name[name] for name in pose_names]
    merged.extend(extras)
    return merged


class HandsExtractor2D:
    """Runs MediaPipe Hands and returns 2D finger keypoints in camera pixels."""

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Create the Hands model. Settings come from config.yaml.

        Args:
            max_num_hands: 1 or 2. We want both hands for this project.
            model_complexity: 0 = faster, 1 = default (clearer fingers).
            min_detection_confidence: 0..1 first look at a hand.
            min_tracking_confidence: 0..1 keep the same hand next frame.
        """
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=int(max_num_hands),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def extract(self, bgr_frame: np.ndarray) -> list[Keypoint2D]:
        """Return up to 21 joints per seen hand, or [] if none.

        Args:
            bgr_frame: Colour image (OpenCV BGR). Same frame as Pose.

        Returns:
            left_* and/or right_* finger keypoints in camera pixels.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        height, width = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        handedness = list(results.multi_handedness or [])
        keypoints: list[Keypoint2D] = []
        for index, hand in enumerate(results.multi_hand_landmarks):
            side = _hand_side(handedness, index)
            if side is None:
                continue
            score = _hand_score(handedness, index)
            for landmark_index, landmark in enumerate(hand.landmark):
                stem = HAND_STEMS[landmark_index]
                visibility = getattr(landmark, "visibility", 0.0) or score
                keypoints.append(
                    Keypoint2D(
                        name=hand_joint_name(side, stem),
                        u_px=float(landmark.x) * width,
                        v_px=float(landmark.y) * height,
                        confidence=float(visibility),
                    )
                )
        return keypoints

    def close(self) -> None:
        """Free the MediaPipe Hands graph."""
        self._hands.close()


def _hand_side(handedness, index: int) -> str | None:
    """Left/Right from MediaPipe, person-facing the camera."""
    if index >= len(handedness):
        return None
    label = str(handedness[index].classification[0].label).strip().lower()
    if label == "left":
        return "left"
    if label == "right":
        return "right"
    return None


def _hand_score(handedness, index: int) -> float:
    """Use handedness score as confidence (Hands often has no visibility)."""
    if index >= len(handedness):
        return 1.0
    return float(handedness[index].classification[0].score)
