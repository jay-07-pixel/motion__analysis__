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
    space: str = "2d",
    display_unit: str = "cm",
) -> dict[str, tuple[float, float] | None]:
    """Highlight configured joints with (u, v) in 2D or (X, Y, Z) in 3D.

    Args:
        canvas: BGR overlay (drawn in place).
        keypoints: This frame's joints.
        specs: YAML list of {name, label, color_bgr}.
        min_confidence: Skip a joint below this visibility.
        space: '2d' = pixel u,v ; '3d' = camera metres/cm (origin = optical).
        display_unit: 'cm' or 'm' for the 3D overlay text.

    Returns:
        name -> (u_px, v_px) if drawn, else None. Used by the GUI cards.
    """
    by_name = {kp.name: kp for kp in keypoints}
    height, width = canvas.shape[:2]
    shown: dict[str, tuple[float, float] | None] = {}
    is_3d = str(space).lower() == "3d"

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

        cv2.circle(canvas, (u, v), 16, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 6, color, -1, cv2.LINE_AA)

        go_left = "left" in name
        box_w = 188 if is_3d else 168
        box_h = 90 if is_3d else 44
        tx = u - box_w - 18 if go_left else u + 20
        ty = v - 52
        tx = max(8, min(width - box_w - 8, tx))
        ty = max(8, min(height - box_h - 8, ty))

        cv2.rectangle(canvas, (tx, ty), (tx + box_w, ty + box_h), (10, 14, 20), -1)
        cv2.rectangle(canvas, (tx, ty), (tx + box_w, ty + box_h), color, 2)
        anchor = (tx + box_w, ty + box_h // 2) if go_left else (tx, ty + box_h // 2)
        cv2.line(canvas, (u, v), anchor, color, 2, cv2.LINE_AA)

        _outlined_text(canvas, label, (tx + 8, ty + 18), color, 0.52)
        if is_3d:
            lines = _xyz_overlay_lines(kp, display_unit)
            _outlined_text(canvas, lines[0], (tx + 8, ty + 38), (240, 244, 248), 0.48)
            _outlined_text(canvas, lines[1], (tx + 8, ty + 56), (240, 244, 248), 0.48)
            if len(lines) > 2:
                _outlined_text(canvas, lines[2], (tx + 8, ty + 72), (240, 244, 248), 0.48)
        else:
            _outlined_text(canvas, f"u={u}   v={v}", (tx + 8, ty + 36), (240, 244, 248), 0.52)

    return shown


def highlight_readout(
    keypoints: list[Keypoint2D],
    specs: list[dict],
    min_confidence: float,
    space: str = "2d",
    display_unit: str = "cm",
) -> list[dict]:
    """Same numbers as the video boxes, for a panel above the live view.

    Returns one dict per highlight joint: name, label, lines, color_bgr.
    """
    by_name = {kp.name: kp for kp in keypoints}
    is_3d = str(space).lower() == "3d"
    rows: list[dict] = []
    for spec in specs:
        name = str(spec["name"])
        label = str(spec.get("label", name.replace("_", " ")))
        color = (
            int(spec["color_bgr"][0]),
            int(spec["color_bgr"][1]),
            int(spec["color_bgr"][2]),
        )
        kp = by_name.get(name)
        if kp is None or kp.confidence < min_confidence:
            lines = ["not visible"]
        elif is_3d:
            lines = _xyz_overlay_lines(kp, display_unit)
        else:
            lines = [f"u={int(round(kp.u_px))}   v={int(round(kp.v_px))}"]
        rows.append({"name": name, "label": label, "lines": lines, "color_bgr": color})
    return rows


def _xyz_overlay_lines(kp: Keypoint2D, display_unit: str) -> list[str]:
    """Format camera-frame X/Y/Z for the overlay box."""
    unit = str(display_unit).strip().lower()
    if kp.x_m is None or kp.y_m is None or kp.z_m is None:
        return ["X= —  Y= —", "no depth"]
    if unit == "m":
        return [
            f"X={kp.x_m:+.3f} m",
            f"Y={kp.y_m:+.3f} m",
            f"Z={kp.z_m:+.3f} m",
        ]
    return [
        f"X={kp.x_m * 100:+.1f} cm",
        f"Y={kp.y_m * 100:+.1f} cm",
        f"Z={kp.z_m * 100:+.1f} cm",
    ]
