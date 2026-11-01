"""Run upper-body IK on one finished capture in the colour-camera frame.

The landmarks are already in camera metres. The camera-frame model uses those
numbers directly. The older ground-frame registration is not read.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.opensim_upper.angles import CAMERA_GROUND, analysis_directory, write_angle_analysis
from src.opensim_upper.camera_model import CAMERA_MODEL
from src.opensim_upper.elbow_gate import write_gated_ik_trc
from src.opensim_upper.ik import ik_directory, run_upper_body_ik
from src.opensim_upper.scaling import write_scaled_capture_model
from src.opensim_upper.trial import trial_directory, write_camera_trial_from_diagnostics
from src.opensim_upper.visualize import save_ik_fit_plot


ANGLE_LABELS = (
    ("arm_flex_r", "Right shoulder flexion"),
    ("arm_add_r", "Right shoulder adduction"),
    ("arm_rot_r", "Right humeral rotation"),
    ("elbow_flex_r", "Right elbow flexion"),
    ("pro_sup_r", "Right forearm pronation/supination"),
    ("arm_flex_l", "Left shoulder flexion"),
    ("arm_add_l", "Left shoulder adduction"),
    ("arm_rot_l", "Left humeral rotation"),
    ("elbow_flex_l", "Left elbow flexion"),
    ("pro_sup_l", "Left forearm pronation/supination"),
)


def describe_results(result: dict) -> str:
    """Text for the results window. A missing angle is unavailable, not zero."""
    lines = [
        "Clinical accuracy is not claimed.",
        "The OpenSim ground is the colour camera: +X right, +Y down, +Z forward.",
        "OpenSim joint angles are the model coordinates. They are not the live calculator.",
        "",
    ]
    if result.get("scale_statement"):
        lines.append(str(result["scale_statement"]))
        lines.append("")
    lines.extend(
        [
            str(result.get("message") or ""),
            "",
        ]
    )
    analysis_dir = result.get("analysis_dir")
    extracted = None
    if analysis_dir:
        path = Path(analysis_dir) / "joint_angles.json"
        if path.is_file():
            extracted = json.loads(path.read_text(encoding="utf-8"))
    last = {}
    root = None
    if extracted and extracted.get("frames"):
        last_frame = extracted["frames"][-1] or {}
        last = last_frame.get("solved") or {}
        root = last_frame.get("root_camera")
    if root:
        lines.append(
            "Trunk origin in the camera: "
            f"X {float(root['x_m']):.3f} m right, "
            f"Y {float(root['y_m']):.3f} m down, "
            f"Z {float(root['z_m']):.3f} m forward."
        )
        lines.append("")
    for name, label in ANGLE_LABELS:
        sample = last.get(name)
        if not sample or sample.get("reported_as_measurement") is False or "degrees" not in sample:
            lines.append(f"{label}: unavailable")
            continue
        lines.append(f"{label}: {float(sample['degrees']):.2f}°")
    lines.append("")
    lines.append("2D skeleton: shown on the camera preview during capture.")
    skeleton = Path(analysis_dir) / "plots" / "skeleton_fit.png" if analysis_dir else None
    if skeleton is not None and skeleton.is_file():
        lines.append(f"3D fit image: {skeleton}")
    else:
        lines.append("3D fit image: unavailable")
    timeline = Path(analysis_dir) / "plots" / "timeline.png" if analysis_dir else None
    if timeline is not None and timeline.is_file():
        lines.append(f"Joint-angle timeline: {timeline}")
    else:
        lines.append("Joint-angle timeline: unavailable")
    lines.append("")
    if result.get("landmarks_csv"):
        lines.append(f"Recorded camera-metre landmarks: {result['landmarks_csv']}")
    if analysis_dir:
        folder = Path(analysis_dir)
        for name in ("joint_angles.csv", "joint_angles.json", "analysis_report.md"):
            path = folder / name
            lines.append(f"{name}: {path if path.is_file() else 'unavailable'}")
        report = folder / "analysis_report.md"
        if report.is_file():
            lines.extend(["", report.read_text(encoding="utf-8")])
    for reason in result.get("reliability_reasons") or []:
        lines.append(f"- {reason}")
    return "\n".join(lines)


def analyze_capture(diag_dir: Path, project_root: Path) -> dict:
    """Write a camera-frame TRC and run IK on the thorax-root model."""
    diag_dir = Path(diag_dir)
    project_root = Path(project_root)
    landmarks = diag_dir / "landmarks.csv"
    base = {
        "ok": False,
        "opensim_ran": False,
        "clinical_accuracy": False,
        "participant_capture": True,
        "landmarks_csv": str(landmarks) if landmarks.is_file() else None,
        "depth_xyz": "camera_3d_metre",
        "model": str(CAMERA_MODEL),
        "registration_applied": False,
        "scale_statement": None,
        "analysis_dir": None,
        "plots": [],
        "reliability_reasons": [],
    }
    if not landmarks.is_file():
        return {**base, "message": "This capture has no landmarks.csv, so 3D analysis cannot start."}
    if not CAMERA_MODEL.is_file():
        return {**base, "message": f"The camera-frame model is missing: {CAMERA_MODEL}"}
    trial_id = diag_dir.name
    trial_dir = trial_directory(project_root, trial_id)
    try:
        written = write_camera_trial_from_diagnostics(
            diag_dir,
            trial_dir,
            trial_id,
            project_root=project_root,
        )
        scaled = write_scaled_capture_model(
            CAMERA_MODEL,
            trial_dir / "Model_upper_scaled.osim",
            trial_dir / "markers.trc",
        )
        gated = write_gated_ik_trc(
            trial_dir / "markers.trc",
            trial_dir / "markers_ik.trc",
            Path(scaled["model_path"]),
        )
        ik_dir = ik_directory(project_root, trial_id)
        report = run_upper_body_ik(
            Path(gated["trc"]),
            ik_dir,
            trial_metadata=trial_dir / "trial_metadata.json",
            model_path=Path(scaled["model_path"]),
        )
    except Exception as error:
        return {
            **base,
            "message": str(error),
            "trial_dir": str(trial_dir),
        }
    angle_dir = analysis_directory(project_root, trial_id)
    plots: list[str] = []
    statement = scaled["statement"]
    if gated["statement"]:
        statement = f"{statement} {gated['statement']}"
    if report.get("execution_success"):
        summary = write_angle_analysis(
            ik_dir / "ik_results.mot",
            ik_dir / "ik_report.json",
            angle_dir,
            ground_frame=CAMERA_GROUND,
            scale_statement=statement,
        )
        plots = list(summary.get("plots") or [])
        locations = sorted(ik_dir.glob("*model_marker_locations.sto"))
        if locations:
            fit_image = angle_dir / "plots" / "skeleton_fit.png"
            save_ik_fit_plot(trial_dir / "markers.trc", locations[0], fit_image)
            plots.append(fit_image.name)
            report_md = angle_dir / "analysis_report.md"
            if report_md.is_file() and "skeleton_fit.png" not in report_md.read_text(encoding="utf-8"):
                report_md.write_text(
                    report_md.read_text(encoding="utf-8").rstrip() + "\n- `plots/skeleton_fit.png`\n",
                    encoding="utf-8",
                )
    finished = bool(report.get("execution_success"))
    return {
        **base,
        "ok": finished,
        "opensim_ran": True,
        "message": (
            "Inverse kinematics finished on the sized copy in the camera frame. "
            "The generic model file was not loaded. This is a model fit, not a clinical measurement."
            if finished
            else report.get("error") or "Inverse kinematics did not finish."
        ),
        "model": scaled["model_path"],
        "scale_statement": statement,
        "trial_dir": str(trial_dir),
        "ik_dir": str(ik_dir),
        "analysis_dir": str(angle_dir) if finished else None,
        "trc": written["trc"],
        "plots": plots,
        "reliability_reasons": list(report.get("reliability_reasons") or []),
    }
