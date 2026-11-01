"""Camera-to-model rigid frame from measured up, heading, and one point.

Rotation (3 degrees of freedom):
    Up comes from the RealSense accelerometer in the colour camera frame.
    Anterior is either the front-facing protocol or a previously measured rod.
    The front-facing protocol uses the horizontal part of colour-camera -Z
    (toward the camera). The shoulder line is not used. Image Y is not gravity.

Translation (3 degrees of freedom):
    One camera point plus the three model-frame coordinates of that same
    physical point. Those coordinates are typed by the operator. This module
    does not set them to the pelvis, the ground, or zero.

Model axes, once rotation is known: +X anterior, +Y up, +Z to the model's
right. The matrix is built as a proper rotation (determinant +1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.opensim_upper.registration import RegistrationInsufficient

MIN_ANTERIOR_HORIZONTAL_M = 0.05
HEADING_OPTICAL = "camera_optical_axis_negative_z"
HEADING_ROD = "facing_rod"
_OPTICAL_ANTERIOR = np.array([0.0, 0.0, -1.0])
_BARE_ORIGIN_WORDS = frozenset(
    {"pelvis", "ground", "floor", "hip", "asis", "psis", "origin", "hjc"}
)

INSTRUCTIONS = (
    "1. Measure camera up while the live session is stopped. The accelerometer "
    "is rotated into the colour camera. Image Y is not gravity.\n"
    "2. Confirm the patient faces the camera. Anterior is the horizontal part "
    "of colour-camera -Z. The shoulder line is not used.\n"
    "3. Start a 3D session and click one physical reference point.\n"
    "4. Type that point's measured OpenSim ground coordinates in metres: "
    "X anterior, Y up, Z to the model's right.\n"
    "5. Compute. 6. Save.\n\n"
    "The patient must face the camera. Rotation away from the lens is a heading "
    "error. Leave the camera still after calibration. If it is moved or rotated, "
    "repeat the calibration. Do not type coordinates you have not measured. "
    "The ground origin is not the default pelvis position, and it is not filled in."
)


def _as_vec(values, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must be three finite metres.")
    return vector


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError(f"{label} has zero length.")
    return vector / length


class FrameCalibration:
    """Holds only references the operator or the accelerometer supplied."""

    def __init__(self) -> None:
        self.up_camera: np.ndarray | None = None
        self.up_magnitude_m_s2: float | None = None
        self.up_n_samples: int | None = None
        self.up_source: str | None = None
        self.heading_mode: str | None = None
        self.anterior_from_m: np.ndarray | None = None
        self.anterior_to_m: np.ndarray | None = None
        self.origin_camera_m: np.ndarray | None = None
        self.origin_model_m: np.ndarray | None = None
        self.origin_description: str = ""

    def set_measured_up(self, assessment: dict) -> None:
        if assessment.get("image_y_used_as_gravity"):
            raise ValueError("Image Y cannot be stored as gravity.")
        up = _unit(_as_vec(assessment["up_camera"], "Up"), "Up")
        magnitude = float(assessment["magnitude_m_s2"])
        if not np.isfinite(magnitude):
            raise ValueError("Up magnitude is not finite.")
        self.up_camera = up
        self.up_magnitude_m_s2 = magnitude
        self.up_n_samples = int(assessment["n_samples"])
        self.up_source = str(assessment.get("source") or "realsense_accel_rotated_to_color")

    def confirm_facing_camera(self) -> None:
        """Use colour-camera -Z as anterior. Clears any rod clicks."""
        self.heading_mode = HEADING_OPTICAL
        self.anterior_from_m = None
        self.anterior_to_m = None

    def set_anterior_point(self, role: str, xyz_m) -> None:
        """Rod heading kept for registrations that were measured that way."""
        if role not in ("anterior_from", "anterior_to"):
            raise ValueError("Anterior role must be anterior_from or anterior_to.")
        point = _as_vec(xyz_m, role)
        self.heading_mode = HEADING_ROD
        if role == "anterior_from":
            self.anterior_from_m = point
        else:
            self.anterior_to_m = point

    def set_origin_camera(self, xyz_m) -> None:
        self.origin_camera_m = _as_vec(xyz_m, "Origin camera point")

    def set_origin_model(self, xyz_m, description: str) -> None:
        text = " ".join(str(description).split())
        if text.lower() in _BARE_ORIGIN_WORDS or len(text) < 12:
            raise ValueError(
                "Describe the physical point you measured. "
                "A name such as pelvis or ground is not a measurement."
            )
        self.origin_model_m = _as_vec(xyz_m, "Origin model coordinates")
        self.origin_description = text

    def missing(self) -> list[str]:
        gaps: list[str] = []
        if self.up_camera is None:
            gaps.append(
                "Camera up is not measured. Read the still accelerometer. "
                "Do not use image Y as gravity."
            )
        if self.heading_mode == HEADING_OPTICAL:
            if self.up_camera is not None and _horizontal_anterior(self) is None:
                gaps.append(
                    "The camera is looking too close to vertical. The horizontal "
                    f"part of colour-camera -Z is shorter than {MIN_ANTERIOR_HORIZONTAL_M:.2f} m. "
                    "Aim the camera at the patient and measure up again."
                )
        elif self.heading_mode != HEADING_ROD or self.anterior_from_m is None or self.anterior_to_m is None:
            gaps.append(
                "Facing direction is not set. Confirm that the patient faces the camera. "
                "The shoulder line is not used."
            )
        elif self.up_camera is not None and _horizontal_anterior(self) is None:
            gaps.append(
                "The two facing points are parallel to up, or closer than "
                f"{MIN_ANTERIOR_HORIZONTAL_M:.2f} m in the horizontal plane. "
                "Place them along the facing direction and click again."
            )
        if self.origin_camera_m is None or self.origin_model_m is None:
            gaps.append(
                "Origin is not measured. Click one physical point and type its "
                "model coordinates in metres (X anterior, Y up, Z model-right). "
                "Pelvis and ground coordinates are not invented."
            )
        return gaps

    def solve(self) -> dict:
        """Return rotation and translation only for the references that exist.

        Rotation is included when up and a horizontal facing direction exist.
        Translation is included only when that rotation also has a measured
        point correspondence. ``complete`` is true only when both exist and
        the numeric checks pass. Landmarks are not transformed here.
        """
        gaps = self.missing()
        rotation = None
        translation = None
        checks = None
        axes = _rotation_if_possible(self)
        if axes is not None:
            rotation = axes["rotation"]
            gram = rotation.T @ rotation
            if not np.allclose(gram, np.eye(3), atol=1e-8) or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-8:
                rotation = None
                gaps.append("The facing direction did not produce a proper rotation. It was not stored.")
            else:
                gaps = [
                    item
                    for item in gaps
                    if item.startswith("Origin")
                ]
        if rotation is not None and self.origin_camera_m is not None and self.origin_model_m is not None:
            translation = self.origin_model_m - rotation @ self.origin_camera_m
            checks = _validate(rotation, translation, self.origin_camera_m, self.origin_model_m, axes["x_camera"])
            if not checks["accepted"]:
                rotation = None
                translation = None
                gaps.append(checks["reason"])
        complete = bool(checks and checks["accepted"] and translation is not None and rotation is not None)
        return {
            "complete": complete,
            "units": "metres",
            "rotation": None if rotation is None else rotation.tolist(),
            "translation_m": None if translation is None else [float(value) for value in translation],
            "checks": checks,
            "missing": [] if complete else gaps,
            "y_flip_applied": False,
            "assumed_facing_camera": self.heading_mode == HEADING_OPTICAL,
            "heading_source": self.heading_mode,
            "assumed_image_y_is_gravity": False,
            "origin_is_model_pelvis": False,
            "origin_is_ground": False,
            "applied_to_landmarks": False,
            "references": self._reference_payload(),
            "instructions": INSTRUCTIONS,
        }

    def save(self, path: Path) -> dict:
        payload = self.solve()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _reference_payload(self) -> dict:
        def row(vector: np.ndarray | None):
            if vector is None:
                return None
            return [float(value) for value in vector]

        return {
            "up_camera": row(self.up_camera),
            "up_magnitude_m_s2": self.up_magnitude_m_s2,
            "up_n_samples": self.up_n_samples,
            "up_source": self.up_source,
            "heading_source": self.heading_mode,
            "anterior_from_m": row(self.anterior_from_m),
            "anterior_to_m": row(self.anterior_to_m),
            "origin_camera_m": row(self.origin_camera_m),
            "origin_model_m": row(self.origin_model_m),
            "origin_description": self.origin_description,
        }


def apply_camera_to_model(point_m, solved: dict) -> np.ndarray:
    """Apply a completed rigid transform. Refuses a partial result."""
    if not solved.get("complete"):
        raise RegistrationInsufficient(tuple(solved.get("missing") or ()))
    rotation = np.asarray(solved["rotation"], dtype=float)
    translation = np.asarray(solved["translation_m"], dtype=float)
    return rotation @ _as_vec(point_m, "Point") + translation


def _horizontal_anterior(calibration: FrameCalibration) -> np.ndarray | None:
    if calibration.up_camera is None:
        return None
    if calibration.heading_mode == HEADING_OPTICAL:
        anterior = _OPTICAL_ANTERIOR.copy()
    elif calibration.anterior_from_m is None or calibration.anterior_to_m is None:
        return None
    else:
        anterior = calibration.anterior_to_m - calibration.anterior_from_m
    y_axis = _unit(calibration.up_camera, "Up")
    horizontal = anterior - y_axis * float(np.dot(anterior, y_axis))
    if float(np.linalg.norm(horizontal)) < MIN_ANTERIOR_HORIZONTAL_M:
        return None
    return horizontal


def _rotation_if_possible(calibration: FrameCalibration) -> dict | None:
    horizontal = _horizontal_anterior(calibration)
    if horizontal is None or calibration.up_camera is None:
        return None
    y_axis = _unit(calibration.up_camera, "Up")
    x_axis = _unit(horizontal, "Anterior")
    z_axis = np.cross(x_axis, y_axis)
    z_axis = _unit(z_axis, "Model right")
    rotation = np.vstack([x_axis, y_axis, z_axis])
    return {"rotation": rotation, "x_camera": x_axis}


def _validate(rotation, translation, origin_camera, origin_model, x_camera) -> dict:
    gram = rotation.T @ rotation
    orthogonal = bool(np.allclose(gram, np.eye(3), atol=1e-8))
    determinant = float(np.linalg.det(rotation))
    proper = orthogonal and abs(determinant - 1.0) < 1e-8
    mapped = rotation @ origin_camera + translation
    alignment = float(np.linalg.norm(mapped - origin_model))
    probe = origin_camera + 0.25 * x_camera
    probe_model = rotation @ probe + translation
    anterior_error = float(np.linalg.norm(probe_model - (origin_model + np.array([0.25, 0.0, 0.0]))))
    samples = np.array(
        [
            origin_camera,
            probe,
            origin_camera + np.array([0.1, -0.2, 0.3]),
            origin_camera + np.array([-0.4, 0.2, 0.5]),
        ],
        dtype=float,
    )
    inverse_error = 0.0
    for point in samples:
        model = rotation @ point + translation
        back = rotation.T @ (model - translation)
        inverse_error = max(inverse_error, float(np.linalg.norm(back - point)))
    length_error = abs(float(np.linalg.norm(rotation @ np.array([1.0, 0.0, 0.0]))) - 1.0)
    accepted = proper and alignment < 1e-8 and anterior_error < 1e-8 and inverse_error < 1e-8
    reason = ""
    if not accepted:
        reason = (
            "The solved matrix failed a rotation check "
            f"(det={determinant:.6f}, alignment={alignment:.3e} m). It was not stored."
        )
    return {
        "accepted": accepted,
        "reason": reason,
        "orthogonal": orthogonal,
        "determinant": determinant,
        "proper_rotation": proper,
        "translation_units": "metres",
        "translation_finite": bool(np.isfinite(translation).all()),
        "inverse_max_error_m": inverse_error,
        "reference_alignment_error_m": alignment,
        "anterior_axis_error_m": anterior_error,
        "length_preservation_error": length_error,
    }
