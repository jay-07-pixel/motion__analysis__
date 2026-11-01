"""Camera metres versus the Rajagopal/Pose2Sim ground frame.

Camera (RealSense colour, right-handed):
    X right, Y down, Z forward from the optical centre. Metres.

Model (OpenSim ground in Model_Pose2Sim_simple.osim):
    Gravity is stored as (0, -9.80665, 0), so model +Y is up. Metres.
    The file is a Rajagopal full-body model. In that convention +X is anterior
    and +Z is to the model's right. This file does not contain a measurement
    of which camera axis is anterior.

A Y-axis sign flip keeps camera X as "right" and camera Z as "forward".
That is a reflection, not a rotation, and it does not move the origin
from the optical centre to a measured model point. Upper-body markers
do not supply up, heading, or the model origin.

``camera_to_model_transform()`` with no calibration record still refuses.
A transform is returned only from a ``FrameCalibration`` that contains a
measured accelerometer up, a heading, and one point whose model coordinates
were entered by the operator. Heading is the front-facing colour-camera -Z
protocol, or the earlier two-point rod measurement. Neither path uses a
Y-axis reflection.
"""

from __future__ import annotations

MISSING_REGISTRATION = (
    "Heading is unknown: nothing in the upper-body markers measures which "
    "camera direction is the model's anterior axis (+X). Do not assume the "
    "person is facing the camera.",
    "Origin is unknown: the model origin is the pelvis/ground, and this "
    "pipeline has no pelvis, ASIS, PSIS, or hip joint centre.",
    "Camera tilt relative to gravity is unknown: image Y-down is the sensor "
    "axis, not a measured vertical, unless the camera is independently leveled.",
)


class RegistrationInsufficient(Exception):
    """Raised when a full camera-to-model rigid transform is requested."""

    def __init__(self, missing: tuple[str, ...] | None = None) -> None:
        self.missing = MISSING_REGISTRATION if missing is None else tuple(missing)
        super().__init__(" ".join(self.missing))


def registration_report() -> dict:
    """What is known, and why no rotation or translation is returned."""
    return {
        "complete": False,
        "units": "metres",
        "camera_axes": {"x": "right", "y": "down", "z": "forward"},
        "camera_origin": "colour optical centre",
        "camera_handedness": "right",
        "model_y": "up, from the gravity vector stored in the osim file",
        "model_horizontal_axes": "not observed",
        "model_origin": "not observed",
        "model_handedness": "right",
        "rotation": None,
        "translation_m": None,
        "y_flip_applied": False,
        "missing": list(MISSING_REGISTRATION),
    }


def camera_to_model_transform(calibration=None) -> dict:
    """Return a rigid transform only when up, facing, and origin were measured.

    With no calibration record this still refuses. A partial record also
    refuses. Image Y is not gravity. Facing the camera is used only after
    ``confirm_facing_camera``. A rod measurement does not set that flag.
    """
    if calibration is None:
        raise RegistrationInsufficient()
    solved = calibration.solve()
    if not solved["complete"]:
        raise RegistrationInsufficient(tuple(solved["missing"]))
    return solved
