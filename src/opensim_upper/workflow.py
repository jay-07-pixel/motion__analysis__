"""Build one upper-body calibration folder from already captured frames.

Does not open the camera, does not run the LSTM, and does not write a
scaled model or a TRC file.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.opensim_upper.marker_map import map_static_landmarks, marker_mapping
from src.opensim_upper.model_copy import write_upper_model
from src.opensim_upper.registration import RegistrationInsufficient, registration_report
from src.opensim_upper.residuals import same_body_residuals
from src.opensim_upper.scaling import subject_lengths
from src.opensim_upper.static_pose import StaticPoseCapture
from src.opensim_upper.transform import find_completed_registration, load_registration, transform_static_pose
from src.opensim_upper.visualize import save_model_frame_plot, save_reference_plot


def write_calibration(
    capture: StaticPoseCapture,
    folder: Path,
    height_m: float,
    mass_kg: float,
) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pose = capture.save(folder / "static_pose.json", height_m, mass_kg)
    registration = registration_report()
    lengths = subject_lengths(pose)
    plot = save_reference_plot(pose, folder / "static_pose_camera.png")
    payload = {
        "static_pose": "static_pose.json",
        "plot": plot.name,
        "registration": registration,
        "scaling": lengths,
        "inverse_kinematics": False,
        "trc_written": False,
    }
    (folder / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def registration_for_capture(diag_dir: Path) -> Path | None:
    """Return this capture's own registration. Other captures are not searched."""
    path = Path(diag_dir) / "registration.json"
    if not path.is_file():
        return None
    try:
        load_registration(path)
    except (RegistrationInsufficient, OSError, ValueError):
        return None
    return path


def write_capture_registration(diag_dir: Path, payload: dict) -> Path:
    """Store one completed registration beside that capture's landmarks."""
    checked = load_registration_payload(payload)
    path = Path(diag_dir) / "registration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checked, indent=2), encoding="utf-8")
    return path


def load_registration_payload(payload: dict) -> dict:
    """Validate a registration dict without reading another capture's file."""
    from src.opensim_upper.transform import validate_registration

    return validate_registration(payload)


def device_registration_status(project_root: Path) -> dict:
    """Whether a completed device calibration has been saved for this project."""
    path = find_completed_registration(Path(project_root) / "output" / "opensim_frame")
    if path is None:
        return {
            "end_to_end_device_calibration": False,
            "registration_json": None,
            "pending_reason": (
                "No complete registration.json was found under output/opensim_frame. "
                "That archive is not applied to a capture automatically. "
                "Save the calibration into the capture that will be analysed."
            ),
        }
    return {
        "end_to_end_device_calibration": True,
        "registration_json": str(path),
        "pending_reason": None,
    }


def apply_registered_markers(
    summary: dict,
    registration_path: Path,
    folder: Path,
    model_dest: Path,
    project_root: Path | None = None,
) -> dict:
    """Transform, map, and score one static pose. Does not write a TRC or run IK.

    ``registration_path`` must already be a validated calibration file.
    A missing device capture is reported and is not replaced with the identity.
    """
    registration = load_registration(registration_path)
    transformed = transform_static_pose(summary, registration)
    mapping = marker_mapping()
    mapped = map_static_landmarks(transformed, mapping)
    residuals = same_body_residuals(mapped["assignments"])
    model_note = write_upper_model(model_dest)
    plot = save_model_frame_plot(transformed, Path(folder) / "registered_landmarks.png")
    status = device_registration_status(project_root or Path.cwd())
    payload = {
        "registration_file": str(registration_path),
        "device_calibration": status,
        "transformed_landmarks": transformed,
        "mapping": mapping,
        "assignments": mapped,
        "residuals": residuals,
        "model_copy": model_note,
        "plot": plot.name,
        "inverse_kinematics": False,
        "trc_written": False,
        "placement_adjusted": False,
    }
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "registered_markers.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
