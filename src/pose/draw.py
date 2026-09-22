"""Draw 2D keypoints and skeleton bones on an RGB frame.

This is only visualisation so we can check the joints sit on the body.
Numbers (CSV) are Step 4. Drawing uses camera pixels (u_px, v_px).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.pose.keypoints import Keypoint2D
from src.pose.skeleton import SKELETON_BONES


def draw_keypoints_2d(
    frame: np.ndarray,
    keypoints: list[Keypoint2D],
    min_visibility: float,
    point_radius: int,
    line_thickness: int,
    point_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    allowed_names: set[str] | None = None,
    extra_bones: list[tuple[str, str]] | None = None,
) -> np.ndarray:
    """Draw bones then joints on a copy of the frame.

    Args:
        frame: BGR image (will not be modified; we draw on a copy).
        keypoints: Output of PoseExtractor2D.extract.
        min_visibility: Joints below this confidence are skipped.
        point_radius: Circle size in pixels.
        line_thickness: Bone width in pixels.
        point_color: BGR colour for joints.
        line_color: BGR colour for bones.
        allowed_names: If set, only these joints and bones are drawn.
        extra_bones: Extra name pairs (e.g. face or hand links) from YAML.

    Returns:
        Annotated BGR image.
    """
    canvas = frame.copy()
    by_name = {kp.name: kp for kp in keypoints}
    bones = list(SKELETON_BONES)
    if extra_bones:
        bones.extend(extra_bones)

    for start_name, end_name in bones:
        if allowed_names is not None and (
            start_name not in allowed_names or end_name not in allowed_names
        ):
            continue
        start = by_name.get(start_name)
        end = by_name.get(end_name)
        if start is None or end is None:
            continue
        if start.confidence < min_visibility or end.confidence < min_visibility:
            continue
        cv2.line(
            canvas,
            (int(round(start.u_px)), int(round(start.v_px))),
            (int(round(end.u_px)), int(round(end.v_px))),
            line_color,
            line_thickness,
            cv2.LINE_AA,
        )

    for kp in keypoints:
        if allowed_names is not None and kp.name not in allowed_names:
            continue
        if kp.confidence < min_visibility:
            continue
        cv2.circle(
            canvas,
            (int(round(kp.u_px)), int(round(kp.v_px))),
            point_radius,
            point_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return canvas
