"""Joint coordinates used to place markers on the generic upper-body model.

These are the inputs of the model-sampled IK check. They are not a
participant recording. Times are 0, 0.1 and 0.2 seconds.
"""

from __future__ import annotations

FRAME_TIMES_SEC = (0.0, 0.1, 0.2)

ARM_POSES_RAD = (
    {
        "arm_flex_r": 0.40,
        "arm_add_r": -1.00,
        "elbow_flex_r": 1.10,
        "arm_flex_l": -0.30,
        "arm_add_l": -0.70,
        "elbow_flex_l": 0.80,
    },
    {
        "arm_flex_r": 0.20,
        "arm_add_r": -1.20,
        "elbow_flex_r": 0.60,
        "arm_flex_l": 0.15,
        "arm_add_l": -1.10,
        "elbow_flex_l": 1.20,
    },
    {
        "arm_flex_r": -0.25,
        "arm_add_r": -0.80,
        "elbow_flex_r": 1.40,
        "arm_flex_l": 0.35,
        "arm_add_l": -0.90,
        "elbow_flex_l": 0.50,
    },
)

AXIAL_POSES_RAD = (
    {"arm_rot_r": 0.50, "arm_rot_l": -0.30, "pro_sup_r": 0.80, "pro_sup_l": 1.00},
    {"arm_rot_r": -0.40, "arm_rot_l": 0.45, "pro_sup_r": 1.20, "pro_sup_l": 0.50},
    {"arm_rot_r": 0.20, "arm_rot_l": 0.10, "pro_sup_r": 0.40, "pro_sup_l": 1.40},
)


def with_axial_free_poses() -> list[dict[str, float]]:
    """Poses in which shoulder, elbow and forearm coordinates were all prescribed."""
    return [{**arm, **axial} for arm, axial in zip(ARM_POSES_RAD, AXIAL_POSES_RAD)]
