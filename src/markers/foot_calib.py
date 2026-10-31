"""Session foot calibration: four distinct toes from live RGB-D clicks.

Does not copy foot_index, invent a small-toe offset, or run Pose2Sim's
add_smalltoe_data. Each toe is a click with valid RealSense XYZ, then
tracked with a rigid foot frame from the current ankle, heel, and foot_index.
"""

from __future__ import annotations

import numpy as np

from src.markers.catalog import TOE_INPUTS
from src.pose.keypoints import Keypoint2D

_FOOT_ROOTS = {
    "LBigToe": ("left_ankle", "left_heel", "left_foot_index"),
    "LSmallToe": ("left_ankle", "left_heel", "left_foot_index"),
    "RBigToe": ("right_ankle", "right_heel", "right_foot_index"),
    "RSmallToe": ("right_ankle", "right_heel", "right_foot_index"),
}


def _xyz(kp: Keypoint2D | None) -> np.ndarray | None:
    if kp is None or kp.x_m is None or kp.y_m is None or kp.z_m is None:
        return None
    if not (np.isfinite(kp.x_m) and np.isfinite(kp.y_m) and np.isfinite(kp.z_m)):
        return None
    return np.array([kp.x_m, kp.y_m, kp.z_m], dtype=np.float64)


def foot_rotation(ankle: np.ndarray, heel: np.ndarray, index: np.ndarray) -> np.ndarray | None:
    """Orthonormal axes at the ankle: X toward heel, Y toward foot_index, Z their cross."""
    x = heel - ankle
    y_raw = index - ankle
    z = np.cross(x, y_raw)
    y = np.cross(z, x)
    nx, ny, nz = np.linalg.norm(x), np.linalg.norm(y), np.linalg.norm(z)
    if nx < 1e-4 or ny < 1e-4 or nz < 1e-4:
        return None
    return np.column_stack([x / nx, y / ny, z / nz])


class FootCalibrator:
    """Collect four live toe clicks, then reconstruct them each frame."""

    def __init__(self) -> None:
        self.local: dict[str, np.ndarray] = {}
        self.next_index = 0
        self.message = "Click LEFT big toe on the live image (step closer if needed)"

    def remaining(self) -> tuple[str, ...]:
        return tuple(n for n in TOE_INPUTS if n not in self.local)

    def ready(self) -> bool:
        return len(self.local) == len(TOE_INPUTS)

    def prompt(self) -> str:
        left = self.remaining()
        if not left:
            return "Toe calibration complete (4/4). Lower LSTM can run."
        labels = {
            "LBigToe": "LEFT big toe",
            "LSmallToe": "LEFT small toe / 5th metatarsal",
            "RBigToe": "RIGHT big toe",
            "RSmallToe": "RIGHT small toe / 5th metatarsal",
        }
        return f"Click {labels[left[0]]}  ({4 - len(left) + 1}/4)"

    def add_click(
        self,
        u_px: float,
        v_px: float,
        xyz: tuple[float, float, float],
        keypoints: list[Keypoint2D],
    ) -> str:
        left = self.remaining()
        if not left:
            return self.prompt()
        name = left[0]
        by_name = {kp.name: kp for kp in keypoints}
        ankle_n, heel_n, index_n = _FOOT_ROOTS[name]
        ankle = _xyz(by_name.get(ankle_n))
        heel = _xyz(by_name.get(heel_n))
        index = _xyz(by_name.get(index_n))
        if ankle is None or heel is None or index is None:
            return (
                f"Need valid depth on {ankle_n}, {heel_n}, {index_n} in this frame. "
                "Stand so that foot is in view, then click again."
            )
        rot = foot_rotation(ankle, heel, index)
        if rot is None:
            return "Foot frame degenerate this frame. Reposition the foot and click again."
        point = np.array(xyz, dtype=np.float64)
        if not np.all(np.isfinite(point)):
            return "Click has no valid RealSense XYZ. Aim at the toe with depth."
        self.local[name] = rot.T @ (point - ankle)
        return self.prompt()

    def reconstruct(self, keypoints: list[Keypoint2D], min_visibility: float) -> list[Keypoint2D]:
        if not self.local:
            return []
        by_name = {kp.name: kp for kp in keypoints}
        out: list[Keypoint2D] = []
        for name, local in self.local.items():
            ankle_n, heel_n, index_n = _FOOT_ROOTS[name]
            ankle_kp = by_name.get(ankle_n)
            heel_kp = by_name.get(heel_n)
            index_kp = by_name.get(index_n)
            if ankle_kp is None or heel_kp is None or index_kp is None:
                continue
            if min(ankle_kp.confidence, heel_kp.confidence, index_kp.confidence) < min_visibility:
                continue
            ankle = _xyz(ankle_kp)
            heel = _xyz(heel_kp)
            index = _xyz(index_kp)
            if ankle is None or heel is None or index is None:
                continue
            rot = foot_rotation(ankle, heel, index)
            if rot is None:
                continue
            cam = ankle + rot @ local
            out.append(
                Keypoint2D(
                    name=name,
                    u_px=float(index_kp.u_px),
                    v_px=float(index_kp.v_px),
                    confidence=min(ankle_kp.confidence, heel_kp.confidence, index_kp.confidence),
                    coord_frame="camera_3d_metre",
                    x_m=float(cam[0]),
                    y_m=float(cam[1]),
                    z_m=float(cam[2]),
                )
            )
        return out
