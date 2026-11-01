"""Check assigned hand sides before Hands overwrites Pose.

This does not choose left or right. `_assign_hand_sides` already did that.
The check only accepts or withholds the overwrite of each assigned side.

A duplicate is two Hands detections in one tight 21-point cluster that
clearly sits on one Pose wrist. The matching side is kept and the other
overwrite is withheld. Two hands that are apart, and that each sit on
their own Pose wrist, stay accepted even when both are nearer one elbow.

Raw MediaPipe detections are copied into the decision and are not modified.
Rejected sides keep the Pose wrist and Pose finger tips already on that
frame. Previous-frame coordinates are never used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from src.pose.keypoints import Keypoint2D

CLASS_NORMAL = "normal"
CLASS_DUPLICATE = "duplicate_hand_suspected"
CLASS_CROSSING = "ambiguous_crossing"
CLASS_UNCERTAIN = "uncertain"
CLASS_INSUFFICIENT = "insufficient_evidence"

# Two Hands wrists inside this distance can be one repeated detection.
CLUSTER_WRIST_PX = 40.0
# Mean distance between the 21 corresponding landmarks. A real second hand
# is farther apart than this even when the wrists are only moderately close.
CLUSTER_MEAN_LANDMARK_PX = 40.0
# The other Pose wrist must be at least this much farther than the matched one.
CLEAR_POSE_MARGIN_PX = 40.0

OVERWRITE_ACCEPTED = "accepted"
OVERWRITE_REJECTED = "rejected"
OVERWRITE_NONE = "none"

WRIST_IDENTITY_FIELDS = (
    "frame",
    "t_frame_sec",
    "classification",
    "reason",
    "mp_label_0",
    "mp_score_0",
    "raw_wrist_u_0",
    "raw_wrist_v_0",
    "assigned_side_0",
    "mp_label_1",
    "mp_score_1",
    "raw_wrist_u_1",
    "raw_wrist_v_1",
    "assigned_side_1",
    "pose_left_wrist_u",
    "pose_left_wrist_v",
    "pose_left_wrist_confidence",
    "pose_right_wrist_u",
    "pose_right_wrist_v",
    "pose_right_wrist_confidence",
    "pose_left_elbow_u",
    "pose_left_elbow_v",
    "pose_left_elbow_confidence",
    "pose_right_elbow_u",
    "pose_right_elbow_v",
    "pose_right_elbow_confidence",
    "det0_dist_left_elbow_px",
    "det0_dist_right_elbow_px",
    "det1_dist_left_elbow_px",
    "det1_dist_right_elbow_px",
    "left_overwrite",
    "right_overwrite",
    "raw_hands_json",
)


@dataclass(frozen=True)
class RawHandDetection:
    """One MediaPipe Hands detection, copied off the inference result."""

    handedness: str
    score: float
    landmarks_px: tuple[tuple[float, float], ...]

    @property
    def wrist_px(self) -> tuple[float, float] | None:
        if not self.landmarks_px:
            return None
        u_px, v_px = self.landmarks_px[0]
        if not math.isfinite(u_px) or not math.isfinite(v_px):
            return None
        return (float(u_px), float(v_px))


@dataclass(frozen=True)
class PoseJointSnapshot:
    """Pose joint used as evidence. Coordinates stay even when confidence is low."""

    u_px: float | None
    v_px: float | None
    confidence: float | None

    @property
    def xy(self) -> tuple[float, float] | None:
        if self.u_px is None or self.v_px is None:
            return None
        if not math.isfinite(self.u_px) or not math.isfinite(self.v_px):
            return None
        return (float(self.u_px), float(self.v_px))


@dataclass(frozen=True)
class WristIdentityDecision:
    """Raw detections plus the overwrite decision for this frame."""

    classification: str
    reason: str
    raw_hands: tuple[RawHandDetection, ...]
    assigned_sides: tuple[str | None, ...]
    accepted_sides: tuple[str, ...]
    rejected_sides: tuple[str, ...]
    pose_left_wrist: PoseJointSnapshot
    pose_right_wrist: PoseJointSnapshot
    pose_left_elbow: PoseJointSnapshot
    pose_right_elbow: PoseJointSnapshot
    elbow_distances_px: tuple[tuple[float | None, float | None], ...]

    def overwrite(self, side: str) -> str:
        if side in self.rejected_sides:
            return OVERWRITE_REJECTED
        if side in self.accepted_sides:
            return OVERWRITE_ACCEPTED
        return OVERWRITE_NONE

    def to_csv_row(self, frame: int, t_frame_sec: float) -> dict[str, str]:
        """Flat diagnostic row. Raw landmarks stay in raw_hands_json."""
        row = {field: "" for field in WRIST_IDENTITY_FIELDS}
        row["frame"] = str(int(frame))
        row["t_frame_sec"] = f"{float(t_frame_sec):.6f}"
        row["classification"] = self.classification
        row["reason"] = self.reason
        for index in range(2):
            if index >= len(self.raw_hands):
                continue
            raw = self.raw_hands[index]
            row[f"mp_label_{index}"] = raw.handedness
            row[f"mp_score_{index}"] = _fmt(raw.score, 4)
            if raw.wrist_px is not None:
                row[f"raw_wrist_u_{index}"] = _fmt(raw.wrist_px[0], 3)
                row[f"raw_wrist_v_{index}"] = _fmt(raw.wrist_px[1], 3)
            side = self.assigned_sides[index] if index < len(self.assigned_sides) else None
            row[f"assigned_side_{index}"] = side or ""
            distances = self.elbow_distances_px[index] if index < len(self.elbow_distances_px) else (None, None)
            row[f"det{index}_dist_left_elbow_px"] = _fmt(distances[0], 3)
            row[f"det{index}_dist_right_elbow_px"] = _fmt(distances[1], 3)
        _write_joint(row, "pose_left_wrist", self.pose_left_wrist)
        _write_joint(row, "pose_right_wrist", self.pose_right_wrist)
        _write_joint(row, "pose_left_elbow", self.pose_left_elbow)
        _write_joint(row, "pose_right_elbow", self.pose_right_elbow)
        row["left_overwrite"] = self.overwrite("left")
        row["right_overwrite"] = self.overwrite("right")
        row["raw_hands_json"] = json.dumps(
            [
                {
                    "handedness": raw.handedness,
                    "score": raw.score,
                    "landmarks_px": [[u_px, v_px] for u_px, v_px in raw.landmarks_px],
                }
                for raw in self.raw_hands
            ],
            separators=(",", ":"),
        )
        return row


def validate_wrist_identity(
    raw_hands: list[tuple[str, float, list[tuple[float, float]]]],
    assigned_sides: list[str | None],
    pose_keypoints: list[Keypoint2D],
    *,
    min_pose_confidence: float = 0.5,
) -> WristIdentityDecision:
    """Accept or withhold each assigned Hands overwrite.

    `raw_hands` is not modified. `assigned_sides` comes from `_assign_hand_sides`
    and is not rewritten. A tight cluster keeps the side of the Pose wrist it
    matches and withholds the other overwrite. A crossing is recorded and
    left as assigned.
    """
    raw = _copy_raw(raw_hands)
    sides = tuple(assigned_sides[index] if index < len(assigned_sides) else None for index in range(len(raw)))
    by_name = {kp.name: kp for kp in pose_keypoints}
    pose_left_wrist = _snapshot(by_name.get("left_wrist"))
    pose_right_wrist = _snapshot(by_name.get("right_wrist"))
    pose_left_elbow = _snapshot(by_name.get("left_elbow"))
    pose_right_elbow = _snapshot(by_name.get("right_elbow"))
    left_xy = pose_left_elbow.xy
    right_xy = pose_right_elbow.xy
    distances = tuple(_elbow_distances(item.wrist_px, left_xy, right_xy) for item in raw)
    floor = float(min_pose_confidence)
    elbows_usable = _usable(pose_left_elbow, floor) and _usable(pose_right_elbow, floor)

    measured = [
        (index, raw[index], sides[index], raw[index].wrist_px)
        for index in range(len(raw))
        if sides[index] in ("left", "right") and raw[index].wrist_px is not None
    ]
    by_side: dict[str, int] = {}
    for index, _item, side, _wrist in measured:
        if side not in by_side:
            by_side[side] = index

    def finish(
        classification: str,
        reason: str,
        accepted: tuple[str, ...],
        rejected: tuple[str, ...] = (),
    ) -> WristIdentityDecision:
        return WristIdentityDecision(
            classification=classification,
            reason=reason,
            raw_hands=raw,
            assigned_sides=sides,
            accepted_sides=accepted,
            rejected_sides=rejected,
            pose_left_wrist=pose_left_wrist,
            pose_right_wrist=pose_right_wrist,
            pose_left_elbow=pose_left_elbow,
            pose_right_elbow=pose_right_elbow,
            elbow_distances_px=distances,
        )

    assigned_present = tuple(side for side in ("left", "right") if side in by_side)
    if not raw:
        return finish(CLASS_INSUFFICIENT, "no_hand_detections", ())
    if not measured:
        return finish(CLASS_INSUFFICIENT, "missing_wrist_detection", ())

    if len(assigned_present) < 2:
        if not elbows_usable:
            if pose_left_elbow.xy is None or pose_right_elbow.xy is None:
                reason = "missing_pose_elbow"
            else:
                reason = "low_confidence_pose_elbow"
            return finish(CLASS_INSUFFICIENT, reason, assigned_present)
        side = assigned_present[0]
        nearer = _nearer_elbow(raw[by_side[side]].wrist_px, left_xy, right_xy)
        if nearer is None:
            return finish(CLASS_INSUFFICIENT, "elbow_distance_tie", assigned_present)
        if nearer == side:
            return finish(CLASS_NORMAL, "single_wrist_nearer_own_elbow", assigned_present)
        return finish(CLASS_INSUFFICIENT, "single_wrist_nearer_opposite_elbow", assigned_present)

    pair = tuple(raw[by_side[side]] for side in ("left", "right"))
    wrists_usable = _usable(pose_left_wrist, floor) and _usable(pose_right_wrist, floor)
    clustered = _same_hand_cluster(pair[0], pair[1])
    pose_left_xy = pose_left_wrist.xy
    pose_right_xy = pose_right_wrist.xy
    matched_side = None
    if clustered and wrists_usable:
        matched_side = _clear_pose_side(
            (pair[0].wrist_px, pair[1].wrist_px),
            pose_left_xy,
            pose_right_xy,
        )

    if matched_side is not None:
        rejected = tuple(side for side in ("left", "right") if side != matched_side)
        return finish(
            CLASS_DUPLICATE,
            f"both_detections_match_pose_{matched_side}_wrist",
            (matched_side,),
            rejected,
        )

    if clustered and not wrists_usable:
        if pose_left_wrist.xy is None or pose_right_wrist.xy is None:
            reason = "missing_pose_wrist"
        elif not elbows_usable and (
            pose_left_elbow.xy is not None and pose_right_elbow.xy is not None
        ):
            reason = "low_confidence_pose_elbow"
        elif pose_left_wrist.confidence is not None and (
            pose_left_wrist.confidence < floor or (pose_right_wrist.confidence or 0.0) < floor
        ):
            reason = "low_confidence_pose_wrist"
        elif not elbows_usable:
            reason = "missing_pose_elbow" if pose_left_elbow.xy is None or pose_right_elbow.xy is None else "low_confidence_pose_elbow"
        else:
            reason = "missing_pose_wrist"
        return finish(CLASS_INSUFFICIENT, reason, ("left", "right"))

    if clustered:
        return finish(CLASS_UNCERTAIN, "cluster_without_clear_pose_wrist", ("left", "right"))

    nearer = {
        side: _nearer_elbow(raw[by_side[side]].wrist_px, left_xy, right_xy) if elbows_usable else None
        for side in ("left", "right")
    }
    if elbows_usable and nearer["left"] == "right" and nearer["right"] == "left":
        return finish(CLASS_CROSSING, "each_wrist_nearer_opposite_elbow", ("left", "right"))

    if wrists_usable and _each_matches_own_pose_wrist(
        pair,
        pose_left_xy,
        pose_right_xy,
    ):
        return finish(CLASS_NORMAL, "each_detection_matches_own_pose_wrist", ("left", "right"))

    if elbows_usable and nearer["left"] == "left" and nearer["right"] == "right":
        return finish(CLASS_NORMAL, "each_wrist_nearer_own_elbow", ("left", "right"))
    if elbows_usable and nearer["left"] is not None and nearer["left"] == nearer["right"]:
        return finish(CLASS_UNCERTAIN, "distinct_hands_nearer_same_elbow", ("left", "right"))
    if elbows_usable and (nearer["left"] is None or nearer["right"] is None):
        return finish(CLASS_INSUFFICIENT, "elbow_distance_tie", ("left", "right"))
    if pose_left_elbow.xy is None or pose_right_elbow.xy is None:
        return finish(CLASS_INSUFFICIENT, "missing_pose_elbow", ("left", "right"))
    return finish(CLASS_INSUFFICIENT, "low_confidence_pose_elbow", ("left", "right"))


def _copy_raw(
    raw_hands: list[tuple[str, float, list[tuple[float, float]]]],
) -> tuple[RawHandDetection, ...]:
    copied: list[RawHandDetection] = []
    for label, score, pixels in raw_hands:
        landmarks = tuple((float(u_px), float(v_px)) for u_px, v_px in pixels)
        copied.append(RawHandDetection(str(label), float(score), landmarks))
    return tuple(copied)


def _snapshot(kp: Keypoint2D | None) -> PoseJointSnapshot:
    if kp is None:
        return PoseJointSnapshot(None, None, None)
    u_px = float(kp.u_px) if math.isfinite(kp.u_px) else None
    v_px = float(kp.v_px) if math.isfinite(kp.v_px) else None
    confidence = float(kp.confidence) if math.isfinite(kp.confidence) else None
    return PoseJointSnapshot(u_px, v_px, confidence)


def _usable(joint: PoseJointSnapshot, min_confidence: float) -> bool:
    return joint.xy is not None and joint.confidence is not None and joint.confidence >= min_confidence


def _elbow_distances(
    wrist: tuple[float, float] | None,
    left_elbow: tuple[float, float] | None,
    right_elbow: tuple[float, float] | None,
) -> tuple[float | None, float | None]:
    if wrist is None or left_elbow is None or right_elbow is None:
        return (None, None)
    return (_pixel_distance(wrist, left_elbow), _pixel_distance(wrist, right_elbow))


def _nearer_elbow(
    wrist: tuple[float, float] | None,
    left_elbow: tuple[float, float] | None,
    right_elbow: tuple[float, float] | None,
) -> str | None:
    distances = _elbow_distances(wrist, left_elbow, right_elbow)
    left_px, right_px = distances
    if left_px is None or right_px is None:
        return None
    if left_px < right_px:
        return "left"
    if right_px < left_px:
        return "right"
    return None


def _pixel_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _same_hand_cluster(first: RawHandDetection, second: RawHandDetection) -> bool:
    """True when both detections are one repeated hand, not two separated hands."""
    if first.wrist_px is None or second.wrist_px is None:
        return False
    if _pixel_distance(first.wrist_px, second.wrist_px) > CLUSTER_WRIST_PX:
        return False
    mean = _mean_landmark_distance(first, second)
    if mean is None:
        return True
    return mean <= CLUSTER_MEAN_LANDMARK_PX


def _mean_landmark_distance(first: RawHandDetection, second: RawHandDetection) -> float | None:
    if len(first.landmarks_px) < 21 or len(second.landmarks_px) < 21:
        return None
    total = 0.0
    for index in range(21):
        total += _pixel_distance(first.landmarks_px[index], second.landmarks_px[index])
    return total / 21.0


def _clear_pose_side(
    wrists: tuple[tuple[float, float], tuple[float, float]],
    left_wrist: tuple[float, float] | None,
    right_wrist: tuple[float, float] | None,
) -> str | None:
    """The one Pose wrist both detections sit on, when the other wrist is clearly farther."""
    if left_wrist is None or right_wrist is None:
        return None
    votes: list[str] = []
    for wrist in wrists:
        left_px = _pixel_distance(wrist, left_wrist)
        right_px = _pixel_distance(wrist, right_wrist)
        if left_px + CLEAR_POSE_MARGIN_PX <= right_px:
            votes.append("left")
        elif right_px + CLEAR_POSE_MARGIN_PX <= left_px:
            votes.append("right")
        else:
            return None
    if len(set(votes)) == 1:
        return votes[0]
    return None


def _each_matches_own_pose_wrist(
    pair: tuple[RawHandDetection, RawHandDetection],
    left_wrist: tuple[float, float] | None,
    right_wrist: tuple[float, float] | None,
) -> bool:
    """True when the left-assigned detection is nearer the left Pose wrist, and the same for right."""
    if left_wrist is None or right_wrist is None:
        return False
    left_hand, right_hand = pair
    if left_hand.wrist_px is None or right_hand.wrist_px is None:
        return False
    left_owns = _pixel_distance(left_hand.wrist_px, left_wrist) < _pixel_distance(left_hand.wrist_px, right_wrist)
    right_owns = _pixel_distance(right_hand.wrist_px, right_wrist) < _pixel_distance(right_hand.wrist_px, left_wrist)
    return left_owns and right_owns


def _fmt(value: float | None, digits: int) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _write_joint(row: dict[str, str], prefix: str, joint: PoseJointSnapshot) -> None:
    row[f"{prefix}_u"] = _fmt(joint.u_px, 3)
    row[f"{prefix}_v"] = _fmt(joint.v_px, 3)
    row[f"{prefix}_confidence"] = _fmt(joint.confidence, 4)
