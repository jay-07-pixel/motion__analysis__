"""Official Pose2Sim v0.3 lower LSTM I/O. Requires four distinct toes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.markers.catalog import POSE2SIM_LOWER_IN, POSE2SIM_LOWER_OUT, TOE_INPUTS
from src.pose.keypoints import Keypoint2D
from src.pose2sim_live.lstm_common import camera_to_yup

FEATURE_MARKERS = POSE2SIM_LOWER_IN
RESPONSE_MARKERS = POSE2SIM_LOWER_OUT
N_FEATURE_XYZ = 45
N_INPUT = 47
N_OUTPUT = 105
MODEL_DIR = Path(
    r"C:\Users\jayjo\anaconda3\envs\pose2sim-offline\Lib\site-packages"
    r"\Pose2Sim\MarkerAugmenter\LSTM\v0.3_lower"
)

MP_TO_OPENPOSE = {
    "left_shoulder": "LShoulder",
    "right_shoulder": "RShoulder",
    "left_hip": "LHip",
    "right_hip": "RHip",
    "left_knee": "LKnee",
    "right_knee": "RKnee",
    "left_ankle": "LAnkle",
    "right_ankle": "RAnkle",
    "left_heel": "LHeel",
    "right_heel": "RHeel",
}


def _finite_xyz(kp: Keypoint2D | None) -> np.ndarray | None:
    if kp is None or kp.x_m is None or kp.y_m is None or kp.z_m is None:
        return None
    if not (np.isfinite(kp.x_m) and np.isfinite(kp.y_m) and np.isfinite(kp.z_m)):
        return None
    return np.array([kp.x_m, kp.y_m, kp.z_m], dtype=np.float64)


def missing_lower_inputs(keypoints: list[Keypoint2D], min_visibility: float) -> tuple[str, ...]:
    """Names the lower LSTM still needs. Never treats foot_index as a toe."""
    by_name = {kp.name: kp for kp in keypoints}
    missing: list[str] = []
    for src, dst in MP_TO_OPENPOSE.items():
        kp = by_name.get(src)
        if kp is None or kp.confidence < min_visibility or _finite_xyz(kp) is None:
            missing.append(dst)
    for toe in TOE_INPUTS:
        kp = by_name.get(toe)
        if kp is None or kp.confidence < min_visibility or _finite_xyz(kp) is None:
            missing.append(toe)
    return tuple(dict.fromkeys(missing))


def extract_lower_frame(
    keypoints: list[Keypoint2D],
    min_visibility: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """(feature_xyz 45, hip_yup 3) or None. Drops the frame if any input is missing."""
    by_name = {kp.name: kp for kp in keypoints}
    mapped: dict[str, np.ndarray] = {}
    for src, dst in MP_TO_OPENPOSE.items():
        kp = by_name.get(src)
        if kp is None or kp.confidence < min_visibility:
            return None
        cam = _finite_xyz(kp)
        if cam is None:
            return None
        mapped[dst] = camera_to_yup(float(cam[0]), float(cam[1]), float(cam[2]))
    for toe in TOE_INPUTS:
        kp = by_name.get(toe)
        if kp is None or kp.confidence < min_visibility:
            return None
        cam = _finite_xyz(kp)
        if cam is None:
            return None
        mapped[toe] = camera_to_yup(float(cam[0]), float(cam[1]), float(cam[2]))
    mapped["Neck"] = 0.5 * (mapped["LShoulder"] + mapped["RShoulder"])
    hip = 0.5 * (mapped["LHip"] + mapped["RHip"])
    feat = np.concatenate([mapped[name] for name in FEATURE_MARKERS], axis=0)
    return feat, hip
