"""Upper-body IK tasks for the 12 mapped markers.

Forty model coordinates exist. Most cannot be determined by these markers.
Locked coordinates are held at the value stored on the model. No replacement
default is written here. Shoulder and elbow OpenPose markers are not treated
as glenohumeral centres, and observed wrists are not added.

Official study-marker weights are an initial reference from the full-body IK
file. They are not a validated weight for this reduced marker set.
"""

from __future__ import annotations

from src.opensim_upper.marker_map import marker_mapping
from src.opensim_upper.trc import ordered_marker_rows, read_trc

_PELVIS = (
    "No pelvis, ASIS, PSIS, or hip marker is available. "
    "The coordinate stays at the model's stored default."
)
_LEG = "No lower-limb marker is available. The coordinate stays at the model's stored default."
_LUMBAR = (
    "No C7 or pelvis marker is available, so lumbar motion cannot be separated from the torso. "
    "The coordinate stays at the model's stored default."
)
_NECK = "No head or neck marker is available. The coordinate stays at the model's stored default."
_WRIST = (
    "Changing this coordinate on the generic model moves none of the 12 markers. "
    "The wrist study markers are fixed on the radius and there is no hand marker. "
    "The coordinate stays at the model's stored default."
)
_ARM_ROT = (
    "A 0.8 rad change of this coordinate moves the generic model's 12 markers by about 0.039 m. "
    "With the model's own range used as a clamp, inverse kinematics recovered prescribed values "
    "to within 0.005 rad from markers sampled on that model. "
    "The shoulder marker is a torso station, not a glenohumeral centre. "
    "The check used model-sampled markers, not a participant recording."
)
_PRO_SUP = (
    "A 0.8 rad change of this coordinate moves the generic model's 12 markers by about 0.041 m. "
    "The wrist study markers are on the radius, so they move with forearm rotation. "
    "With the model's own range used as a clamp, inverse kinematics recovered prescribed values "
    "to within 0.012 rad from markers sampled on that model. "
    "The check used model-sampled markers, not a participant recording."
)
_SHOULDER = (
    "The elbow cluster can change this coordinate while the torso is held fixed. "
    "The shoulder marker is a torso station, not a glenohumeral joint centre. "
    "The model file leaves this coordinate unclamped, and the IK solver then leaves "
    "the stated range. The solve keeps the model's own range as a clamp. "
    "The model is unscaled, so the result is a model fit rather than a measured shoulder angle."
)
_ELBOW = (
    "The radius wrist cluster relative to the humerus elbow cluster changes elbow flexion. "
    "Those points are estimated study markers on an unscaled model, not a clinical measurement."
)

LOCKED = {
    "pelvis_tilt": _PELVIS,
    "pelvis_list": _PELVIS,
    "pelvis_rotation": _PELVIS,
    "pelvis_tx": _PELVIS,
    "pelvis_ty": _PELVIS,
    "pelvis_tz": _PELVIS,
    "hip_flexion_r": _LEG,
    "hip_adduction_r": _LEG,
    "hip_rotation_r": _LEG,
    "knee_angle_r": _LEG,
    "ankle_angle_r": _LEG,
    "subtalar_angle_r": _LEG,
    "mtp_angle_r": _LEG,
    "hip_flexion_l": _LEG,
    "hip_adduction_l": _LEG,
    "hip_rotation_l": _LEG,
    "knee_angle_l": _LEG,
    "ankle_angle_l": _LEG,
    "subtalar_angle_l": _LEG,
    "mtp_angle_l": _LEG,
    "L5_S1_Flex_Ext": _LUMBAR,
    "L5_S1_Lat_Bending": _LUMBAR,
    "L5_S1_axial_rotation": _LUMBAR,
    "neck_flexion": _NECK,
    "neck_bending": _NECK,
    "neck_rotation": _NECK,
    "wrist_flex_r": _WRIST,
    "wrist_dev_r": _WRIST,
    "wrist_flex_l": _WRIST,
    "wrist_dev_l": _WRIST,
}

SOLVED = {
    "arm_flex_r": _SHOULDER,
    "arm_add_r": _SHOULDER,
    "arm_flex_l": _SHOULDER,
    "arm_add_l": _SHOULDER,
    "elbow_flex_r": _ELBOW,
    "elbow_flex_l": _ELBOW,
    "arm_rot_r": _ARM_ROT,
    "arm_rot_l": _ARM_ROT,
    "pro_sup_r": _PRO_SUP,
    "pro_sup_l": _PRO_SUP,
}

_THORAX = (
    "The thorax is the root of the upper-body model. This coordinate places "
    "that root in the colour-camera frame: translations are metres along "
    "camera X right, Y down, and Z forward, and rotations are about those "
    "axes. It is not a lumbar joint angle."
)

# The camera-frame model has these instead of the pelvis, lumbar, and leg coordinates.
CAMERA_SOLVED = {
    "thorax_rx": _THORAX,
    "thorax_ry": _THORAX,
    "thorax_rz": _THORAX,
    "thorax_tx": _THORAX,
    "thorax_ty": _THORAX,
    "thorax_tz": _THORAX,
}
CAMERA_MODEL_LOCKED = (
    "neck_flexion",
    "neck_bending",
    "neck_rotation",
    "wrist_flex_r",
    "wrist_dev_r",
    "wrist_flex_l",
    "wrist_dev_l",
)

FORBIDDEN_MARKERS = (
    "C7_study",
    "r_shoulder_study",
    "L_shoulder_study",
    "r.ASIS_study",
    "L.ASIS_study",
    "r.PSIS_study",
    "L.PSIS_study",
    "RHJC_study",
    "LHJC_study",
    "RWrist",
    "LWrist",
    "right_wrist",
    "left_wrist",
)


def coordinate_decisions(names: list[str] | None = None) -> dict[str, dict]:
    """Classify every coordinate. An unknown name is an error, not a silent default."""
    requested = list(LOCKED) + list(SOLVED) if names is None else list(names)
    decisions = {}
    for name in requested:
        if name in SOLVED or name in CAMERA_SOLVED:
            if name in CAMERA_SOLVED:
                identifiable = "root_in_camera_frame"
                reason = CAMERA_SOLVED[name]
            elif name.startswith(("arm_flex", "arm_add")):
                identifiable = "partial"
                reason = SOLVED[name]
            elif name.startswith("elbow"):
                identifiable = "cluster_direction"
                reason = SOLVED[name]
            else:
                identifiable = "model_cluster_recovered"
                reason = SOLVED[name]
            decisions[name] = {
                "solved": True,
                "locked": False,
                "identifiable": identifiable,
                "reason": reason,
                "value_source": "ik_from_markers",
            }
            continue
        if name in LOCKED:
            decisions[name] = {
                "solved": False,
                "locked": True,
                "identifiable": "not_observable",
                "reason": LOCKED[name],
                "value_source": "model_default",
            }
            continue
        raise ValueError(f"No observability decision for coordinate {name}.")
    if names is None and set(decisions) != set(LOCKED) | set(SOLVED):
        raise ValueError("Coordinate classification is incomplete.")
    return decisions


def marker_weights(overrides: dict[str, float] | None = None, mapping: dict | None = None) -> dict[str, dict]:
    """Weights for the 12 mapped markers. Overrides must name a mapped marker."""
    table = mapping if mapping is not None else marker_mapping()
    weights = {}
    for row in table["mapped"]:
        target = row["target"]
        if target in FORBIDDEN_MARKERS:
            raise ValueError(f"{target} is not an upper-body IK marker.")
        official = row.get("marker_weight")
        if official is None:
            weights[target] = {
                "weight": 5.0,
                "source": (
                    "no official IK weight; 5 matches the study-marker reference "
                    "and is not a validated weight"
                ),
                "official_reference": False,
            }
        else:
            weights[target] = {
                "weight": float(official),
                "source": (
                    "weight from IK_Setup_Pose2Sim_withoutHands_LSTM.xml for this "
                    "marker name only; not validated for the 12-marker model"
                ),
                "official_reference": True,
            }
    if overrides:
        unknown = set(overrides).difference(weights)
        if unknown:
            raise ValueError(f"Weight override names a marker that is not in the 12-marker map: {sorted(unknown)}")
        for name, value in overrides.items():
            weight = float(value)
            if weight < 0 or weight != weight:
                raise ValueError(f"Weight for {name} must be a finite number >= 0.")
            weights[name] = {
                "weight": weight,
                "source": "caller override",
                "official_reference": False,
            }
    return weights


def ik_tasks(
    overrides: dict[str, float] | None = None,
    absent_markers: list[str] | None = None,
    mapping: dict | None = None,
) -> dict:
    """Task list. A marker that is missing on every frame is not applied."""
    rows = ordered_marker_rows(mapping)
    weights = marker_weights(overrides, {"mapped": rows, "unmapped_observed": [], "not_added": []})
    absent = set(absent_markers or [])
    unknown_absent = absent.difference(row["target"] for row in rows)
    if unknown_absent:
        raise ValueError(f"Unknown marker in absent list: {sorted(unknown_absent)}")
    tasks = []
    for row in rows:
        spec = weights[row["target"]]
        tasks.append(
            {
                "name": row["target"],
                "apply": row["target"] not in absent and spec["weight"] > 0,
                "weight": spec["weight"],
                "weight_source": spec["source"],
                "official_reference": spec["official_reference"],
                "source_landmark": row["source"],
                "body": row["body"],
                "note": row["note"],
                "glenohumeral_centre": False,
            }
        )
    names = [task["name"] for task in tasks]
    if any(name in FORBIDDEN_MARKERS for name in names):
        raise ValueError("The IK task list contains a forbidden marker.")
    return {
        "tasks": tasks,
        "coordinates": coordinate_decisions(),
        "full_body_ik_configuration_used": False,
        "clinical_accuracy": False,
    }


def validate_trc_for_ik(path, mapping: dict | None = None) -> dict:
    """Require the 12-marker TRC layout before IK starts."""
    parsed = read_trc(path)
    expected = [row["target"] for row in ordered_marker_rows(mapping)]
    if parsed["markers"] != expected:
        raise ValueError(
            "TRC markers do not match the 12-marker map. "
            f"Found {parsed['markers']}."
        )
    if parsed["num_frames"] != len(parsed["frames"]) or parsed["num_frames"] < 1:
        raise ValueError("TRC frame count does not match its data rows.")
    times = [frame["time_sec"] for frame in parsed["frames"]]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("TRC timestamps are not strictly increasing.")
    if any(name in FORBIDDEN_MARKERS for name in parsed["markers"]):
        raise ValueError("TRC contains a marker that this IK setup does not use.")
    return parsed
