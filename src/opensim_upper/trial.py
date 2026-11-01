"""Write one upper-body TRC trial. Does not run inverse kinematics.

A trial directory contains ``markers.trc``, ``trial_metadata.json``,
``data_quality.json``, and ``processing.log``. Synthetic trials are marked
``participant_capture: false`` and are not described as a calibrated recording.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.opensim_upper.registration import RegistrationInsufficient
from src.opensim_upper.residuals import same_body_residuals
from src.opensim_upper.transform import camera_to_model_points, load_registration
from src.opensim_upper.trc import ordered_marker_rows, write_trc
from src.opensim_upper.trajectory import linear_fill, load_diagnostics, nominal_rate_hz
from src.opensim_upper.workflow import device_registration_status


def trial_directory(project_root: Path, trial_id: str) -> Path:
    _check_trial_id(trial_id)
    return Path(project_root) / "output" / "opensim_upper" / "trials" / trial_id


def write_camera_trial_from_diagnostics(
    diagnostics_dir: Path,
    trial_dir: Path,
    trial_id: str,
    project_root: Path | None = None,
    interpolate: bool = False,
) -> dict:
    """Copy camera-metre markers into a TRC. No registration is applied."""
    from src.opensim_upper.trajectory import assemble_camera_frames

    landmarks, frames = load_diagnostics(diagnostics_dir)
    camera_frames, quality = assemble_camera_frames(landmarks, frames)
    return write_trial(
        camera_frames,
        None,
        trial_dir,
        trial_id=trial_id,
        synthetic=False,
        source_description=str(Path(diagnostics_dir)),
        project_root=project_root,
        static_pose=None,
        interpolate=interpolate,
        quality=quality,
        camera_ground=True,
    )


def write_trial_from_diagnostics(
    diagnostics_dir: Path,
    registration_path: Path,
    trial_dir: Path,
    trial_id: str,
    project_root: Path | None = None,
    static_pose: dict | None = None,
    interpolate: bool = False,
) -> dict:
    """Convert a diagnostics folder. Raises before writing if registration is invalid."""
    from src.opensim_upper.trajectory import assemble_camera_frames

    landmarks, frames = load_diagnostics(diagnostics_dir)
    camera_frames, quality = assemble_camera_frames(landmarks, frames)
    return write_trial(
        camera_frames,
        registration_path,
        trial_dir,
        trial_id=trial_id,
        synthetic=False,
        source_description=str(Path(diagnostics_dir)),
        project_root=project_root,
        static_pose=static_pose,
        interpolate=interpolate,
        quality=quality,
    )


def write_synthetic_trial(
    registration_path: Path,
    trial_dir: Path,
    trial_id: str,
    n_frames: int = 4,
    project_root: Path | None = None,
) -> dict:
    """Build a labelled synthetic trial. This is not a participant capture."""
    if n_frames < 2:
        raise ValueError("A synthetic trial needs at least two frames so the rate is measured.")
    rows = ordered_marker_rows()
    camera_frames = []
    for index in range(n_frames):
        time_sec = 0.1 * index
        points = {}
        for row_index, row in enumerate(rows):
            base = 0.2 * row_index
            points[row["source"]] = (
                0.15 + 0.01 * index + base,
                -0.4 - 0.002 * index,
                1.4 + 0.005 * row_index,
            )
        camera_frames.append(
            {"source_frame": index, "t_sec": time_sec, "points": points, "dropped_before": 0}
        )
    # One interior gap so missing-data handling is visible and not filled.
    camera_frames[1]["points"][rows[0]["source"]] = None
    quality = {
        "n_input_frames": n_frames,
        "n_kept_frames": n_frames,
        "ignored_unmapped_rows": 0,
        "invalid_coordinate_rows": 0,
        "duplicate_landmark_rows": 0,
        "duplicate_timestamps_removed": [],
        "non_increasing_timestamps_removed": [],
        "dropped_frames_recorded": 0,
        "dropped_frames_interpolated": False,
        "recorded_fps_median": None,
        "interpolation": "none",
        "synthetic": True,
    }
    return write_trial(
        camera_frames,
        registration_path,
        trial_dir,
        trial_id=trial_id,
        synthetic=True,
        source_description="synthetic_test_data",
        project_root=project_root,
        static_pose=None,
        interpolate=False,
        quality=quality,
        sampled_from_opensim_model=False,
        point_cloud=(
            "Each marker is placed on a straight line in the camera frame, "
            "0.2 m apart, then transformed by the registration. "
            "The points are not positions of the OpenSim model markers."
        ),
    )


def write_trial(
    camera_frames: list[dict],
    registration_path: Path | None,
    trial_dir: Path,
    trial_id: str,
    synthetic: bool,
    source_description: str,
    project_root: Path | None = None,
    static_pose: dict | None = None,
    interpolate: bool = False,
    quality: dict | None = None,
    sampled_from_opensim_model: bool | None = None,
    point_cloud: str | None = None,
    camera_ground: bool = False,
) -> dict:
    """Write the trial directory in the model frame requested by the caller."""
    _check_trial_id(trial_id)
    if camera_ground:
        if registration_path is not None or static_pose is not None:
            raise ValueError("A camera-ground trial does not use a registration or a static-pose transform.")
        registration = None
    else:
        if registration_path is None:
            raise RegistrationInsufficient(("Registration file was not provided.",))
        registration = load_registration(registration_path)
    rows = ordered_marker_rows()
    sources = [row["source"] for row in rows]
    missing_before = [
        {"source_frame": int(frame["source_frame"]), "source": source}
        for frame in camera_frames
        for source in sources
        if frame["points"].get(source) is None
    ]
    working = camera_frames
    filled = []
    if interpolate:
        working, filled = linear_fill(camera_frames, sources)
    if not working:
        raise ValueError("No frames left after timestamp checks.")
    rate_hz, variable_rate, rate_measured = nominal_rate_hz(
        working,
        None if quality is None else quality.get("recorded_fps_median"),
    )
    samples = {row["target"]: [] for row in rows}
    missing_original = []
    for frame in working:
        for row in rows:
            point = frame["points"].get(row["source"])
            if point is None:
                samples[row["target"]].append(None)
                missing_original.append({"source_frame": frame["source_frame"], "marker": row["target"]})
                continue
            if camera_ground:
                copied = (float(point[0]), float(point[1]), float(point[2]))
            else:
                model = camera_to_model_points(point, registration)[0]
                copied = (float(model[0]), float(model[1]), float(model[2]))
            samples[row["target"]].append(copied)
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    trc_path = trial_dir / "markers.trc"
    write_trc(
        trc_path,
        [float(frame["t_sec"]) for frame in working],
        samples,
        rate_hz,
        rows,
    )
    if camera_ground:
        device = {
            "used": False,
            "reason": "Model ground is the colour camera. Camera metres are copied unchanged.",
        }
    else:
        device = device_registration_status(project_root or Path.cwd())
    spacing = _spacing_report(samples, rows)
    static_report = None
    if static_pose is not None:
        static_report = _static_alignment(static_pose, registration, samples, rows)
    quality_out = dict(quality or {})
    quality_out.update(
        {
            "interpolation": "linear_between_nearest_finite_samples" if interpolate else "none",
            "interpolated_samples": filled,
            "missing_before_interpolation": missing_before,
            "missing_samples": missing_original,
            "n_missing_samples": len(missing_original),
            "missing_markers_not_fabricated": True,
            "variable_frame_rate": variable_rate,
            "data_rate_hz": rate_hz,
            "data_rate_measured_from_timestamps": rate_measured,
            "time_column_is_authoritative": True,
            "units": "metres",
            "trc_units_token": "m",
            "source_frames": [int(frame["source_frame"]) for frame in working],
            "timestamps_sec": [float(frame["t_sec"]) for frame in working],
            "spacing_vs_marker_xml": spacing,
            "static_pose_alignment": static_report,
            "compared_to_generic_default_pose": False,
        }
    )
    metadata = {
        "trial_id": trial_id,
        "synthetic": bool(synthetic),
        "participant_capture": False if synthetic else True,
        "source": source_description,
        "registration_file": None if camera_ground else str(Path(registration_path).resolve()),
        "device_calibration": device,
        "physical_validation": "not_performed",
        "markers": [row["target"] for row in rows],
        "marker_sources": {row["target"]: row["source"] for row in rows},
        "n_frames": len(working),
        "n_markers": len(rows),
        "units": "metres",
        "coord_frame": "camera_3d_metre" if camera_ground else "opensim_ground",
        "transform": (
            "camera metres copied unchanged; OpenSim ground is the colour camera"
            if camera_ground
            else "p_model = R @ p_camera + t"
        ),
        "inverse_kinematics": False,
        "trc_file": "markers.trc",
        "model_copy": (
            "models/opensim_upper/Model_upper_camera.osim"
            if camera_ground
            else "models/opensim_upper/Model_Pose2Sim_upper.osim"
        ),
    }
    if sampled_from_opensim_model is not None:
        metadata["sampled_from_opensim_model"] = bool(sampled_from_opensim_model)
    if point_cloud is not None:
        metadata["point_cloud"] = point_cloud
    if synthetic:
        metadata["physical_validation"] = "not_performed_synthetic_trial"
    elif static_report is not None:
        metadata["physical_validation"] = "static_pose_residuals_computed"
    log = _log_text(metadata, quality_out)
    (trial_dir / "trial_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (trial_dir / "data_quality.json").write_text(json.dumps(quality_out, indent=2), encoding="utf-8")
    (trial_dir / "processing.log").write_text(log, encoding="utf-8")
    return {"metadata": metadata, "quality": quality_out, "trc": str(trc_path)}


def _spacing_report(samples: dict, rows: list[dict]) -> dict | None:
    complete = None
    for index in range(len(next(iter(samples.values())))):
        if all(samples[row["target"]][index] is not None for row in rows):
            complete = index
            break
    if complete is None:
        return None
    assignments = []
    for row in rows:
        assignments.append({**row, "mapped": True, "model_m": list(samples[row["target"]][complete])})
    report = same_body_residuals(assignments)
    report["frame_index_in_trc"] = complete
    report["compared_to_generic_default_pose"] = False
    report["participant_measurement"] = False
    return report


def _static_alignment(static_pose: dict, registration: dict, samples: dict, rows: list[dict]) -> dict:
    from src.opensim_upper.transform import transform_static_pose

    transformed = transform_static_pose(static_pose, registration)
    distances = []
    for row in rows:
        landmark = transformed["landmarks"][row["source"]]
        static_point = np.array([landmark["x_m"], landmark["y_m"], landmark["z_m"]], dtype=float)
        column = [point for point in samples[row["target"]] if point is not None]
        if not column:
            distances.append({"marker": row["target"], "distance_m": None, "reason": "no finite TRC samples"})
            continue
        median = np.median(np.asarray(column, dtype=float), axis=0)
        distances.append(
            {
                "marker": row["target"],
                "distance_m": float(np.linalg.norm(median - static_point)),
                "comparison": "median_of_finite_trc_samples_vs_transformed_static_pose",
            }
        )
    return {
        "compared_to_generic_default_pose": False,
        "markers": distances,
    }


def _log_text(metadata: dict, quality: dict) -> str:
    lines = [
        f"trial_id={metadata['trial_id']}",
        f"synthetic={metadata['synthetic']}",
        f"participant_capture={metadata['participant_capture']}",
        f"source={metadata['source']}",
        f"transform={metadata['transform']}",
        f"markers={','.join(metadata['markers'])}",
        f"n_frames={metadata['n_frames']}",
        f"interpolation={quality.get('interpolation')}",
        f"n_missing_samples={quality.get('n_missing_samples')}",
        f"duplicate_timestamps_removed={quality.get('duplicate_timestamps_removed')}",
        f"non_increasing_timestamps_removed={quality.get('non_increasing_timestamps_removed')}",
        f"dropped_frames_recorded={quality.get('dropped_frames_recorded')}",
        f"variable_frame_rate={quality.get('variable_frame_rate')}",
        f"data_rate_hz={quality.get('data_rate_hz')}",
        "inverse_kinematics=false",
        "missing markers were written as nan and were not fabricated",
        f"physical_validation={metadata['physical_validation']}",
    ]
    return "\n".join(lines) + "\n"


def _check_trial_id(trial_id: str) -> None:
    if not trial_id or any(character in trial_id for character in "\\/:"):
        raise ValueError("trial_id must be a single directory name.")


def require_registration(path: Path) -> dict:
    """Load a registration or raise. Does not invent an identity transform."""
    if path is None:
        raise RegistrationInsufficient(("Registration file was not provided.",))
    return load_registration(path)
