"""Live RGB preview guide: feet should stay above this native-image row.

Visual only. Does not change keypoints, depth, diagnostics, or saved video.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.pose.keypoints import Keypoint2D

FOOT_LANDMARK_NAMES = (
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
LINE_LABEL = "FOOT SAFETY LINE"
WARNING_TEXT = "FEET TOO LOW — ADJUST CAMERA OR MOVE BACK"
_LINE_BGR = (0, 0, 255)
_DEFAULT_V_PX = 640


def feet_too_low(
    keypoints: list[Keypoint2D],
    min_visibility: float,
    guide_v_px: float,
    image_width: int,
    image_height: int,
) -> bool:
    """True if a confident foot landmark is below the guide or outside the image.

    Missing or low-confidence landmarks are ignored (unknown, not safe).
    """
    for kp in keypoints:
        if kp.name not in FOOT_LANDMARK_NAMES:
            continue
        if kp.confidence < float(min_visibility):
            continue
        if not math.isfinite(kp.u_px) or not math.isfinite(kp.v_px):
            continue
        outside = (
            kp.u_px < 0.0
            or kp.v_px < 0.0
            or kp.u_px >= float(image_width)
            or kp.v_px >= float(image_height)
        )
        below = kp.v_px > float(guide_v_px)
        if outside or below:
            return True
    return False


def draw_foot_safety_guide(
    canvas: np.ndarray,
    keypoints: list[Keypoint2D],
    min_visibility: float,
    guide_v_px: float = _DEFAULT_V_PX,
) -> np.ndarray:
    """Dashed red line + label on the native frame. Same array is returned."""
    if canvas is None or canvas.size == 0:
        return canvas
    height, width = canvas.shape[:2]
    y = int(round(float(guide_v_px)))
    if 0 <= y < height:
        _dashed_hline(canvas, y, width)
        label_y = y - 8 if y >= 22 else y + 18
        cv2.putText(
            canvas,
            LINE_LABEL,
            (16, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            _LINE_BGR,
            2,
            cv2.LINE_AA,
        )
    if feet_too_low(keypoints, min_visibility, guide_v_px, width, height):
        text = WARNING_TEXT.replace("\u2014", "-")
        cv2.putText(
            canvas,
            text,
            (16, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            _LINE_BGR,
            2,
            cv2.LINE_AA,
        )
    return canvas


def _dashed_hline(canvas: np.ndarray, y: int, width: int, dash: int = 16, gap: int = 10) -> None:
    x = 0
    while x < width:
        x2 = min(width - 1, x + dash - 1)
        cv2.line(canvas, (x, y), (x2, y), _LINE_BGR, 2, cv2.LINE_8)
        x = x2 + 1 + gap
