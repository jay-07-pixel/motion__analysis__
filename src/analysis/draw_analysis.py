"""Draw 2D angle labels and a joint trail on a BGR image."""

from __future__ import annotations

import cv2
import numpy as np

from src.analysis.angles_2d import Angle2D


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
