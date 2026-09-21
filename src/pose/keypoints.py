"""2D keypoint type: one joint in camera pixel coordinates.

coord_frame is always 'camera_2d_pixel' in this phase:
    origin = top-left of the RGB image
    u_px   = column (rightward)
    v_px   = row (downward)
    units  = pixels

We do NOT use MediaPipe 'world landmarks' here. Those are a model guess,
not RealSense metres. Real 3D comes later from depth at (u_px, v_px).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Keypoint2D:
    """One body joint on the RGB image."""

    name: str
    u_px: float
    v_px: float
    confidence: float
    coord_frame: str = "camera_2d_pixel"
