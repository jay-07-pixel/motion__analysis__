"""Draw 2D angle labels, a joint trail, and highlighted joint coordinates."""

from __future__ import annotations

import cv2
import numpy as np

from src.analysis.angles_2d import Angle2D
from src.pose.keypoints import Keypoint2D


def draw_angles_2d(
    canvas: np.ndarray,
    angles: list[Angle2D],
    text_color: tuple[int, int, int],
) -> None:
    """Write 'left_elbow 142 deg' next to the vertex. Draws on canvas in place."""
    for angle in angles:
        x = int(round(angle.vertex_u_px))
        y = int(round(angle.vertex_v_px)) - 12
        label = f"{angle.name} {angle.degrees:.0f} deg"
        cv2.putText(
            canvas,
            label,
            (x, max(24, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            2,
            cv2.LINE_AA,
        )


def draw_trail_2d(
    canvas: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw the recent path of one joint. Needs at least two points."""
    if len(points) < 2:
        return
    cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def _outlined_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    """Readable overlay text: dark stroke, then coloured fill."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, origin, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, font, scale, color, 2, cv2.LINE_AA)


def draw_joint_coords_2d(
    canvas: np.ndarray,
    keypoints: list[Keypoint2D],
    specs: list[dict],
    min_confidence: float,
) -> dict[str, tuple[float, float] | None]:
    """Highlight configured joints (both wrists) with camera-pixel (u, v).

    Args:
        canvas: BGR overlay (drawn in place).
        keypoints: This frame's 2D joints.
        specs: YAML list of {name, label, color_bgr}.
        min_confidence: Skip a joint below this visibility.

    Returns:
        name -> (u_px, v_px) if drawn, else None. Used by the GUI cards.
    """
    by_name = {kp.name: kp for kp in keypoints}
    height, width = canvas.shape[:2]
    shown: dict[str, tuple[float, float] | None] = {}

    for spec in specs:
        name = str(spec["name"])
        kp = by_name.get(name)
        if kp is None or kp.confidence < min_confidence:
            shown[name] = None
            continue

        color = (int(spec["color_bgr"][0]), int(spec["color_bgr"][1]), int(spec["color_bgr"][2]))
        label = str(spec.get("label", name.replace("_", " ")))
        u = int(round(kp.u_px))
        v = int(round(kp.v_px))
        shown[name] = (kp.u_px, kp.v_px)

        # Bigger than the other skeleton dots so the wrists read first.
        cv2.circle(canvas, (u, v), 16, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 6, color, -1, cv2.LINE_AA)

        # Left joints: label to the left. Right joints: label to the right.
        go_left = "left" in name
        box_w, box_h = 168, 44
        tx = u - box_w - 18 if go_left else u + 20
        ty = v - 52
        tx = max(8, min(width - box_w - 8, tx))
        ty = max(8, min(height - box_h - 8, ty))

        cv2.rectangle(canvas, (tx, ty), (tx + box_w, ty + box_h), (10, 14, 20), -1)
        cv2.rectangle(canvas, (tx, ty), (tx + box_w, ty + box_h), color, 2)
        anchor = (tx + box_w, ty + box_h // 2) if go_left else (tx, ty + box_h // 2)
        cv2.line(canvas, (u, v), anchor, color, 2, cv2.LINE_AA)

        _outlined_text(canvas, label, (tx + 8, ty + 18), color, 0.52)
        _outlined_text(canvas, f"u={u}   v={v}", (tx + 8, ty + 36), (240, 244, 248), 0.52)

    return shown
