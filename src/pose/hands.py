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
        swap_handedness: bool = True,
        match_to_pose_wrists: bool = True,
    ) -> None:
        """Create the Hands model. Settings come from config.yaml.

        Args:
            max_num_hands: 1 or 2. We want both hands for this project.
            model_complexity: 0 = faster, 1 = default (clearer fingers).
            min_detection_confidence: 0..1 first look at a hand.
            min_tracking_confidence: 0..1 keep the same hand next frame.
            swap_handedness: Hands labels assume a selfie (mirrored) image.
                RealSense is not mirrored, so we flip Left/Right if Pose
                wrists are missing.
            match_to_pose_wrists: Prefer the Pose wrist on the same arm
                instead of the Hands Left/Right label.
        """
        self._swap_handedness = bool(swap_handedness)
        self._match_to_pose_wrists = bool(match_to_pose_wrists)
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=int(max_num_hands),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def extract(
        self,
        bgr_frame: np.ndarray,
        pose_keypoints: list[Keypoint2D] | None = None,
    ) -> list[Keypoint2D]:
        """Return up to 21 joints per seen hand, or [] if none.

        Args:
            bgr_frame: Colour image (OpenCV BGR). Same frame as Pose.
            pose_keypoints: Body joints from Pose. Used to put each hand
                on the matching arm (left_wrist / right_wrist).

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
        raw_hands: list[tuple[str, float, list[tuple[float, float]]]] = []
        for index, hand in enumerate(results.multi_hand_landmarks):
            label = _hand_side(handedness, index)
            if label is None:
                continue
            score = _hand_score(handedness, index)
            pixels = [
                (float(landmark.x) * width, float(landmark.y) * height)
                for landmark in hand.landmark
            ]
            raw_hands.append((label, score, pixels))

        sides = _assign_hand_sides(
            raw_hands,
            pose_keypoints or [],
            swap_handedness=self._swap_handedness,
            match_to_pose_wrists=self._match_to_pose_wrists,
        )

        keypoints: list[Keypoint2D] = []
        used_sides: set[str] = set()
        for (label, score, pixels), side in zip(raw_hands, sides):
            if side is None or side in used_sides:
                continue
            used_sides.add(side)
            for landmark_index, (u_px, v_px) in enumerate(pixels):
                stem = HAND_STEMS[landmark_index]
                keypoints.append(
                    Keypoint2D(
                        name=hand_joint_name(side, stem),
                        u_px=u_px,
                        v_px=v_px,
                        confidence=float(score),
                    )
                )
        return keypoints

    def close(self) -> None:
        """Free the MediaPipe Hands graph."""
        self._hands.close()


def _assign_hand_sides(
    raw_hands: list[tuple[str, float, list[tuple[float, float]]]],
    pose_keypoints: list[Keypoint2D],
    swap_handedness: bool,
    match_to_pose_wrists: bool,
) -> list[str | None]:
    """Pick left/right for each Hands detection so arms do not cross.

    MediaPipe Hands Left/Right is built for a mirrored selfie. Pose uses
    the person's true left/right. Matching to Pose wrists keeps the
    elbow→wrist bone on the same arm.
    """
    n = len(raw_hands)
    fallback = []
    for label, _score, _pixels in raw_hands:
        side = label
        if swap_handedness:
            side = "right" if label == "left" else "left"
        fallback.append(side)

    if not match_to_pose_wrists or n == 0:
        return fallback

    by_name = {kp.name: kp for kp in pose_keypoints}
    pose_uv = {}
    for side in ("left", "right"):
        wrist = by_name.get(f"{side}_wrist")
        if wrist is not None:
            pose_uv[side] = (wrist.u_px, wrist.v_px)
    if not pose_uv:
        return fallback

    hand_uv = [pixels[0] for (_label, _score, pixels) in raw_hands]

    if n == 1:
        only = hand_uv[0]
        best_side = min(pose_uv, key=lambda side: _dist2(only, pose_uv[side]))
        return [best_side]

    if n >= 2 and "left" in pose_uv and "right" in pose_uv:
        # Two pairings: keep order vs swap. Pick the shorter total distance.
        d_keep = _dist2(hand_uv[0], pose_uv["left"]) + _dist2(hand_uv[1], pose_uv["right"])
        d_swap = _dist2(hand_uv[0], pose_uv["right"]) + _dist2(hand_uv[1], pose_uv["left"])
        if d_keep <= d_swap:
            return ["left", "right"] + [None] * (n - 2)
        return ["right", "left"] + [None] * (n - 2)

    sides: list[str | None] = []
    taken: set[str] = set()
    for uv in hand_uv:
        available = [side for side in pose_uv if side not in taken]
        if not available:
            sides.append(None)
            continue
        best_side = min(available, key=lambda side: _dist2(uv, pose_uv[side]))
        taken.add(best_side)
        sides.append(best_side)
    return sides


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Squared pixel distance (no sqrt needed for comparisons)."""
    du = a[0] - b[0]
    dv = a[1] - b[1]
    return du * du + dv * dv


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
