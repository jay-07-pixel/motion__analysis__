"""Definitive live marker inventory: observed, estimated, unavailable.

Names and sources match installed models. MediaPipe names are model landmarks,
not ISB palpable points. Pose2Sim LSTM names are the official study outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.pose.hands import HAND_STEMS
from src.pose.skeleton import LANDMARK_NAMES


@dataclass(frozen=True)
class MarkerSpec:
    name: str
    region: str
    source: str
    kind: str  # observed | estimated | geometric | calibrated | unavailable
    notes: str


POSE2SIM_UPPER_IN = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RElbow",
    "LElbow",
    "RWrist",
    "LWrist",
)
POSE2SIM_UPPER_OUT = (
    "r_lelbow_study",
    "r_melbow_study",
    "r_lwrist_study",
    "r_mwrist_study",
    "L_lelbow_study",
    "L_melbow_study",
    "L_lwrist_study",
    "L_mwrist_study",
)
POSE2SIM_LOWER_IN = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RHip",
    "LHip",
    "RKnee",
    "LKnee",
    "RAnkle",
    "LAnkle",
    "RHeel",
    "LHeel",
    "RSmallToe",
    "LSmallToe",
    "RBigToe",
    "LBigToe",
)
POSE2SIM_LOWER_OUT = (
    "r.ASIS_study",
    "L.ASIS_study",
    "r.PSIS_study",
    "L.PSIS_study",
    "r_knee_study",
    "r_mknee_study",
    "r_ankle_study",
    "r_mankle_study",
    "r_toe_study",
    "r_5meta_study",
    "r_calc_study",
    "L_knee_study",
    "L_mknee_study",
    "L_ankle_study",
    "L_mankle_study",
    "L_toe_study",
    "L_calc_study",
    "L_5meta_study",
    "r_shoulder_study",
    "L_shoulder_study",
    "C7_study",
    "r_thigh1_study",
    "r_thigh2_study",
    "r_thigh3_study",
    "L_thigh1_study",
    "L_thigh2_study",
    "L_thigh3_study",
    "r_sh1_study",
    "r_sh2_study",
    "r_sh3_study",
    "L_sh1_study",
    "L_sh2_study",
    "L_sh3_study",
    "RHJC_study",
    "LHJC_study",
)
TOE_INPUTS = ("LBigToe", "LSmallToe", "RBigToe", "RSmallToe")
GEOMETRIC_PROXIES = ("Neck", "Hip")
GEOMETRIC_CENTRES = (
    "r_elbow_centre",
    "L_elbow_centre",
    "r_wrist_centre",
    "L_wrist_centre",
    "r_knee_centre",
    "L_knee_centre",
    "r_ankle_centre",
    "L_ankle_centre",
)
UNAVAILABLE = (
    "IJ",
    "PX",
    "T8",
    "acromion_L",
    "acromion_R",
    "GHJC_L",
    "GHJC_R",
)


def _pose_region(name: str) -> str:
    if name in {"nose", "mouth_left", "mouth_right"} or "eye" in name or "ear" in name:
        return "head"
    if "shoulder" in name or "hip" in name:
        return "trunk"
    if "elbow" in name or "wrist" in name:
        return "upper_limb"
    if "knee" in name or "ankle" in name:
        return "lower_limb"
    if "heel" in name or "foot" in name:
        return "foot"
    if name in {"left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb"}:
        return "hand"
    return "other"


def _pose_note(name: str) -> str:
    if name.endswith("foot_index"):
        return "One toe-ray per foot. Not BigToe and not SmallToe."
    if name.endswith("heel"):
        return "Model heel. Not a protocol calcaneus marker."
    if name.endswith("hip"):
        return "Model hip. Not ASIS, PSIS, or hip joint centre."
    if name.endswith("knee"):
        return "Single model knee. Not femoral epicondyles."
    if name.endswith("ankle"):
        return "Single model ankle. Not malleoli."
    if name.endswith("elbow"):
        return "Single model elbow. Not epicondyles."
    if name.endswith("wrist"):
        return "Single model wrist. Not radial/ulnar styloids."
    if name.endswith("shoulder"):
        return "Model shoulder. Not GHJC or acromion."
    return "MediaPipe model landmark, not an ISB palpable point."


_MP_HANDS: tuple[MarkerSpec, ...] = tuple(
    MarkerSpec(
        name=f"{side}_{stem}",
        region="hand",
        source="mediapipe_hands + realsense_depth",
        kind="observed",
        notes="Hands graph landmark. Not an ISB hand marker.",
    )
    for side in ("left", "right")
    for stem in HAND_STEMS
)

_MP_POSE: tuple[MarkerSpec, ...] = tuple(
    MarkerSpec(
        name=name,
        region=_pose_region(name),
        source="mediapipe_pose + realsense_depth",
        kind="observed",
        notes=_pose_note(name),
    )
    for name in LANDMARK_NAMES
)


SPECS: tuple[MarkerSpec, ...] = (
    *_MP_POSE,
    *_MP_HANDS,
    MarkerSpec("Neck", "trunk", "mid_shoulders", "geometric", "Shoulder midpoint proxy. Not C7."),
    MarkerSpec("Hip", "pelvis", "mid_hips", "geometric", "Hip midpoint proxy. Not ASIS/HJC."),
    *[
        MarkerSpec(n, "upper_limb", "pose2sim_v0.3_upper_lstm", "estimated", "Official upper LSTM study marker.")
        for n in POSE2SIM_UPPER_OUT
    ],
    *[
        MarkerSpec(n, "full_body", "pose2sim_v0.3_lower_lstm", "estimated", "Official lower LSTM study marker.")
        for n in POSE2SIM_LOWER_OUT
    ],
    *[
        MarkerSpec(
            n,
            "foot",
            "session_click + rigid_foot",
            "calibrated",
            "Live RGB-D click in a foot-local frame from ankle/heel/foot_index. Not MediaPipe foot_index.",
        )
        for n in TOE_INPUTS
    ],
    *[
        MarkerSpec(n, "joint_centre", "midpoint_of_lstm_pair", "geometric", "Midpoint of medial/lateral LSTM pair.")
        for n in GEOMETRIC_CENTRES
    ],
    *[
        MarkerSpec(n, "thorax", "none", "unavailable", "Not an output of installed Pose2Sim v0.3 models.")
        for n in UNAVAILABLE
    ],
)


def specs_by_kind(kind: str) -> tuple[MarkerSpec, ...]:
    return tuple(s for s in SPECS if s.kind == kind)


def assert_no_foot_index_as_toe() -> None:
    """Guard: MediaPipe foot_index is never a Pose2Sim toe input."""
    for spec in SPECS:
        if spec.name in TOE_INPUTS:
            assert "foot_index" not in spec.source
            assert spec.kind != "observed"
