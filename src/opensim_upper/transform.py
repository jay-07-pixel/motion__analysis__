"""Apply a saved camera-to-OpenSim registration.

Direction, matching ``FrameCalibration``:

    p_model = R @ p_camera + t
    p_camera = R.T @ (p_model - t)

R and t come from the registration file. A missing or rejected file is not
replaced with the identity, and a Y-only reflection is not applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.opensim_upper.registration import RegistrationInsufficient

_Y_REFLECTION = np.diag([1.0, -1.0, 1.0])


def validate_registration(payload: dict) -> dict:
    """Accept only a complete proper rotation with a metre translation."""
    if not isinstance(payload, dict):
        raise RegistrationInsufficient(("Registration file is not an object.",))
    if payload.get("y_flip_applied") is True:
        raise RegistrationInsufficient(("The registration applied a Y-axis reflection.",))
    if payload.get("units") != "metres":
        raise RegistrationInsufficient(("Registration units are not metres.",))
    if payload.get("complete") is not True:
        missing = payload.get("missing") or ("Registration is incomplete.",)
        raise RegistrationInsufficient(tuple(str(item) for item in missing))

    rotation = np.asarray(payload.get("rotation"), dtype=float)
    translation = np.asarray(payload.get("translation_m"), dtype=float).reshape(-1)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise RegistrationInsufficient(("Rotation is missing or is not a finite 3x3 matrix.",))
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise RegistrationInsufficient(("Translation is missing or is not three finite metres.",))
    if np.allclose(rotation, _Y_REFLECTION):
        raise RegistrationInsufficient(("A Y-only reflection is not a rotation.",))

    gram = rotation.T @ rotation
    determinant = float(np.linalg.det(rotation))
    if not np.allclose(gram, np.eye(3), atol=1e-6) or abs(determinant - 1.0) > 1e-6:
        raise RegistrationInsufficient(
            (f"Rotation is not proper: determinant {determinant:.6f}.",)
        )
    checks = payload.get("checks") or {}
    if checks.get("accepted") is not True:
        raise RegistrationInsufficient(("Registration checks were not accepted.",))
    if checks.get("translation_units") != "metres":
        raise RegistrationInsufficient(("Translation units are not metres.",))
    if checks.get("proper_rotation") is not True:
        raise RegistrationInsufficient(("Registration checks do not mark a proper rotation.",))
    if payload.get("assumed_facing_camera") is True and payload.get("heading_source") != "camera_optical_axis_negative_z":
        raise RegistrationInsufficient(
            ("A front-facing registration must record heading_source camera_optical_axis_negative_z.",)
        )
    if payload.get("heading_source") == "facing_rod" and payload.get("assumed_facing_camera") is True:
        raise RegistrationInsufficient(("A rod registration cannot claim the patient faces the camera.",))
    return payload


def load_registration(path: Path) -> dict:
    """Load ``registration.json``. Raises if it is missing, partial, or invalid."""
    path = Path(path)
    if not path.is_file():
        raise RegistrationInsufficient((f"Registration file was not found: {path}",))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RegistrationInsufficient((f"Registration file is not valid JSON: {error}",)) from error
    return validate_registration(payload)


def camera_to_model_points(points, registration: dict) -> np.ndarray:
    """Map camera metres to model metres. Refuses an incomplete registration."""
    validate_registration(registration)
    rotation = np.asarray(registration["rotation"], dtype=float)
    translation = np.asarray(registration["translation_m"], dtype=float)
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(cloud).all():
        raise ValueError("Landmark coordinates must be finite metres.")
    return (rotation @ cloud.T).T + translation


def model_to_camera_points(points, registration: dict) -> np.ndarray:
    """Inverse of ``camera_to_model_points``."""
    validate_registration(registration)
    rotation = np.asarray(registration["rotation"], dtype=float)
    translation = np.asarray(registration["translation_m"], dtype=float)
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(cloud).all():
        raise ValueError("Landmark coordinates must be finite metres.")
    return (rotation.T @ (cloud - translation).T).T


def transform_static_pose(summary: dict, registration: dict) -> dict:
    """Transform every static landmark with the same validated matrix.

    Observed and predicted points use one path. Wrist points are transformed
    here; marker mapping decides later whether they have a model marker.
    """
    validate_registration(registration)
    if summary.get("units") != "metres":
        raise ValueError("Static pose units must be metres.")
    if summary.get("coord_frame") != "camera_3d_metre":
        raise ValueError("Static pose must still be in camera metres before registration.")
    landmarks = {}
    for name, row in summary["landmarks"].items():
        camera = np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
        model = camera_to_model_points(camera, registration)[0]
        landmarks[name] = {
            "x_m": float(model[0]),
            "y_m": float(model[1]),
            "z_m": float(model[2]),
            "camera_x_m": float(camera[0]),
            "camera_y_m": float(camera[1]),
            "camera_z_m": float(camera[2]),
            "source": row["source"],
            "coord_frame": "opensim_ground_metre",
            "estimated_not_palpated": row["source"] == "estimated_lstm_study",
        }
    return {
        "coord_frame": "opensim_ground_metre",
        "units": "metres",
        "axes": {"x": "anterior", "y": "up", "z": "model right"},
        "participant_height_m": summary.get("participant_height_m"),
        "participant_mass_kg": summary.get("participant_mass_kg"),
        "n_valid_frames": summary.get("n_valid_frames"),
        "landmarks": landmarks,
        "model_registration": "applied",
        "transform": "p_model = R @ p_camera + t",
        "inverse_kinematics": False,
        "trc_written": False,
    }


def find_completed_registration(directory: Path) -> Path | None:
    """Return the first registration.json in ``directory`` that validates."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("**/registration.json")):
        try:
            load_registration(path)
        except (RegistrationInsufficient, OSError, ValueError):
            continue
        return path
    return None
