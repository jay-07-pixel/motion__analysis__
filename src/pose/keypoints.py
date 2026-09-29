"""Keypoint type: 2D colour pixels, optional 3D camera metres.

2D (always present):
    coord_frame = camera_2d_pixel
    origin      = top-left of the RGB image
    u_px, v_px  = pixels

3D (only when RealSense depth is aligned to colour):
    coord_frame = camera_3d_metre
    origin      = colour-camera optical centre
    x_m, y_m, z_m = metres (X right, Y down, Z forward)

We do NOT use MediaPipe 'world landmarks'. Those are a model guess,
not RealSense metres.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Keypoint2D:
    """One body joint on the RGB image, with optional camera-3D metres."""

    name: str
    u_px: float
    v_px: float
    confidence: float
    coord_frame: str = "camera_2d_pixel"
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
