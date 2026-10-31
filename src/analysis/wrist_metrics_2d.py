"""2D wrist bend for the PatientBalanceAI game, from MediaPipe Hands. Image plane only.

right_wrist_angle_deg / left_wrist_angle_deg is the angle at the wrist between
the forearm and the hand in the colour image:

    Pose elbow -> Hands wrist (vertex) -> Hands middle_mcp

180 = hand in line with the forearm in the image. Smaller = more visible bend.
It is unsigned: the same bend up or down gives the same value.

This is a 2D wrist bend, NOT 3D wrist flexion/extension or deviation. A bend
toward or away from the camera hardly changes it. It uses no depth, no world
landmarks and no 3D hand frame.

middle_mcp, not a fingertip: the main app's 2D wrist angle (config.yaml
analysis.angles) ends at the index tip, which moves when the fingers curl.
wrist -> middle_mcp is the hand axis the 3D hand frame uses
(anatomical_frames.build_hand_frame), and it does not change with finger curl.

Hand keypoints come from src.pose.hands.HandsExtractor2D. It puts each hand on
the Pose arm whose wrist is nearest; its confidence is the MediaPipe handedness
score. The Pose elbow and Pose wrist must pass the gate too, so the hand was
matched to this arm by position. Any point missing or below min_confidence
gives None, never 0.
"""

from __future__ import annotations

from src.analysis.angles_2d import angle_at_vertex_deg
from src.analysis.pose_metrics_2d import valid
from src.pose.hands import hand_joint_name
from src.pose.keypoints import Keypoint2D

WRIST_METRICS = (
    ("right_wrist_angle_deg", "right"),
    ("left_wrist_angle_deg", "left"),
)


def wrist_bend_deg(
    pose_by_name: dict[str, Keypoint2D],
    hand_by_name: dict[str, Keypoint2D],
    side: str,
    min_confidence: float,
) -> float | None:
    """Image-plane forearm-to-hand angle at one wrist, or None if not measurable."""
    elbow_name = f"{side}_elbow"
    wrist_name = hand_joint_name(side, "wrist")
    mcp_name = hand_joint_name(side, "middle_mcp")
    if not valid(pose_by_name, elbow_name, min_confidence):
        return None
    if not valid(pose_by_name, f"{side}_wrist", min_confidence):
        return None
    if not (valid(hand_by_name, wrist_name, min_confidence) and valid(hand_by_name, mcp_name, min_confidence)):
        return None
    elbow = pose_by_name[elbow_name]
    wrist = hand_by_name[wrist_name]
    mcp = hand_by_name[mcp_name]
    return angle_at_vertex_deg((elbow.u_px, elbow.v_px), (wrist.u_px, wrist.v_px), (mcp.u_px, mcp.v_px))


def wrist_metrics_2d(
    pose_by_name: dict[str, Keypoint2D],
    hand_keypoints: list[Keypoint2D] | None,
    min_confidence: float,
) -> dict[str, float | None]:
    """Both wrist metrics. Hands keypoints stay separate: their wrist shares the Pose name."""
    hand_by_name = {kp.name: kp for kp in hand_keypoints or []}
    return {
        name: wrist_bend_deg(pose_by_name, hand_by_name, side, min_confidence)
        for name, side in WRIST_METRICS
    }
