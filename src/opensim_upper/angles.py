"""Upper-body joint coordinates from an OpenSim IK motion file.

The ten arm coordinates are the model's own joint angles. They are not the
live calculator's shoulder plane, shoulder elevation, or signed elbow flexion,
and this module does not convert one set into the other.

Locked coordinates stay out of the measured table. A model default is not
reported as a measurement.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src.opensim_upper.mot import read_mot

MEASURED = (
    "arm_flex_r",
    "arm_add_r",
    "arm_rot_r",
    "elbow_flex_r",
    "pro_sup_r",
    "arm_flex_l",
    "arm_add_l",
    "arm_rot_l",
    "elbow_flex_l",
    "pro_sup_l",
)

SIDES = {
    "arm_flex_r": "right",
    "arm_add_r": "right",
    "arm_rot_r": "right",
    "elbow_flex_r": "right",
    "pro_sup_r": "right",
    "arm_flex_l": "left",
    "arm_add_l": "left",
    "arm_rot_l": "left",
    "elbow_flex_l": "left",
    "pro_sup_l": "left",
}

GROUPS = {
    "arm_flex_r": "shoulder",
    "arm_add_r": "shoulder",
    "arm_rot_r": "shoulder",
    "elbow_flex_r": "elbow",
    "pro_sup_r": "forearm",
    "arm_flex_l": "shoulder",
    "arm_add_l": "shoulder",
    "arm_rot_l": "shoulder",
    "elbow_flex_l": "elbow",
    "pro_sup_l": "forearm",
}

CAMERA_GROUND = "+X right, +Y down, +Z forward, origin at the colour camera, metres."

# Axes are the TransformAxis vectors in Model_Pose2Sim_upper.osim.
# Positive follows the right-hand rule about that axis. Left adduction and
# rotation axes point the opposite way from the right arm.
CONVENTIONS = {
    "ground_frame": "+X anterior, +Y up, +Z to the model's right, metres.",
    "mot_rotational_unit": "degrees when inDegrees=yes; radians when inDegrees=no.",
    "mot_translational_unit": "metres. inDegrees does not convert translations.",
    "reported_raw_unit": "radians",
    "reported_display_unit": "degrees",
    "shoulder_right": {
        "joint": "acromial_r",
        "order": [
            {"coordinate": "arm_flex_r", "axis": [0, 0, 1]},
            {"coordinate": "arm_add_r", "axis": [1, 0, 0]},
            {"coordinate": "arm_rot_r", "axis": [0, 1, 0]},
        ],
    },
    "shoulder_left": {
        "joint": "acromial_l",
        "order": [
            {"coordinate": "arm_flex_l", "axis": [0, 0, 1]},
            {"coordinate": "arm_add_l", "axis": [-1, 0, 0]},
            {"coordinate": "arm_rot_l", "axis": [0, -1, 0]},
        ],
    },
    "elbow": (
        "elbow_flex is the PinJoint coordinate. Positive rotation is about the "
        "Z axis of the humerus offset frame, whose orientation is not ground Z. "
        "The model range starts at 0."
    ),
    "forearm": (
        "pro_sup is the radioulnar PinJoint coordinate. Positive rotation is about "
        "the Z axis of the ulna offset frame. The coordinate name is pro_sup. "
        "This report does not rename its sign to pronation or supination."
    ),
    "live_calculator": (
        "src/analysis/dof_angles_3d.py reports shoulder plane of elevation, "
        "shoulder elevation, and signed elbow flexion from camera landmarks. "
        "It does not report humeral long-axis rotation or pro_sup. "
        "Those live names are not these OpenSim coordinates."
    ),
}

# One plain name per coordinate. The sign is not renamed to pronation or supination.
PLAIN_NAME = {
    "arm_flex_r": "Right shoulder bend",
    "arm_add_r": "Right shoulder side angle",
    "arm_rot_r": "Right shoulder twist",
    "elbow_flex_r": "Right elbow bend",
    "pro_sup_r": "Right forearm turn",
    "arm_flex_l": "Left shoulder bend",
    "arm_add_l": "Left shoulder side angle",
    "arm_rot_l": "Left shoulder twist",
    "elbow_flex_l": "Left elbow bend",
    "pro_sup_l": "Left forearm turn",
}
PLAIN_NOTE = {
    "elbow_flex_r": "0° is a straight elbow on this skeleton",
    "elbow_flex_l": "0° is a straight elbow on this skeleton",
}

PLOTS = (
    ("shoulder_right.png", ("arm_flex_r", "arm_add_r", "arm_rot_r"), "Right shoulder"),
    ("shoulder_left.png", ("arm_flex_l", "arm_add_l", "arm_rot_l"), "Left shoulder"),
    ("elbow_right.png", ("elbow_flex_r",), "Right elbow"),
    ("elbow_left.png", ("elbow_flex_l",), "Left elbow"),
    ("forearm.png", ("pro_sup_r", "pro_sup_l"), "Forearm turn"),
    ("timeline.png", ("elbow_flex_l", "elbow_flex_r", "arm_flex_l", "arm_flex_r"), "Read these four first"),
)


def analysis_directory(project_root: Path, trial_id: str) -> Path:
    if not trial_id or any(character in trial_id for character in "\\/:"):
        raise ValueError("trial_id must be a single directory name.")
    return Path(project_root) / "output" / "opensim_upper" / "analysis" / trial_id


def radians_from_mot(value: float, in_degrees: bool, rotational: bool) -> float:
    """Return radians for a rotational coordinate. Translations are unchanged."""
    if not rotational:
        return float(value)
    if in_degrees:
        return math.radians(float(value))
    return float(value)


def _metrics(errors: list[float]) -> dict:
    if not errors:
        raise ValueError("Validation needs at least one error.")
    square = sum(error * error for error in errors) / len(errors)
    return {
        "n": len(errors),
        "mean_absolute_error_rad": sum(abs(error) for error in errors) / len(errors),
        "maximum_absolute_error_rad": max(abs(error) for error in errors),
        "rmse_rad": math.sqrt(square),
        "mean_absolute_error_deg": math.degrees(sum(abs(error) for error in errors) / len(errors)),
        "maximum_absolute_error_deg": math.degrees(max(abs(error) for error in errors)),
        "rmse_deg": math.degrees(math.sqrt(square)),
    }


def extract_angles(motion: dict, ik_report: dict) -> dict:
    """Split solved arm coordinates from locked and unsupported coordinates."""
    classified = ik_report.get("coordinates")
    if not isinstance(classified, dict):
        raise ValueError("IK report has no coordinate classification.")
    in_degrees = bool(motion["in_degrees"])
    frames = []
    for row in motion["rows"]:
        solved = {}
        for name in MEASURED:
            if name not in classified:
                raise ValueError(f"IK report does not classify {name}.")
            info = classified[name]
            if info.get("locked") or not info.get("solved"):
                continue
            if name not in row:
                raise ValueError(f"MOT file has no column {name}.")
            radians = radians_from_mot(row[name], in_degrees, info.get("motion_type") == "rotational")
            solved[name] = {
                "side": SIDES[name],
                "group": GROUPS[name],
                "radians": radians,
                "degrees": math.degrees(radians),
                "reported_as_measurement": True,
            }
        root = _root_camera(row, classified, in_degrees)
        frame = {"time_sec": float(row["time"]), "solved": solved}
        if root is not None:
            frame["root_camera"] = root
        frames.append(frame)
    locked = []
    for name, info in classified.items():
        if not info.get("locked"):
            continue
        status = "unsupported" if info.get("identifiable") == "not_observable" else "locked"
        locked.append(
            {
                "coordinate": name,
                "status": status,
                "reason": info.get("reason"),
                "model_default_rad": info.get("default_value"),
                "reported_as_measurement": False,
            }
        )
    locked.sort(key=lambda item: item["coordinate"])
    return {
        "clinical_accuracy": False,
        "participant_capture": False,
        "mot_in_degrees": in_degrees,
        "mot_angular_unit": "degrees" if in_degrees else "radians",
        "raw_unit": "radians",
        "display_unit": "degrees",
        "conventions": CONVENTIONS,
        "timestamps_sec": [frame["time_sec"] for frame in frames],
        "frames": frames,
        "locked_not_measured": locked,
        "measured_coordinates": [name for name in MEASURED if any(name in frame["solved"] for frame in frames)],
    }


def _root_camera(row: dict, classified: dict, in_degrees: bool) -> dict | None:
    """Thorax root pose in camera metres and degrees, when the model has that root."""
    names = ("thorax_tx", "thorax_ty", "thorax_tz", "thorax_rx", "thorax_ry", "thorax_rz")
    if any(name not in row or name not in classified or not classified[name].get("solved") for name in names):
        return None
    rotations = {}
    for name in names[3:]:
        radians = radians_from_mot(row[name], in_degrees, True)
        rotations[name] = {"radians": radians, "degrees": math.degrees(radians)}
    return {
        "x_m": float(row["thorax_tx"]),
        "y_m": float(row["thorax_ty"]),
        "z_m": float(row["thorax_tz"]),
        "rotations": rotations,
    }


def validate_against_known(extracted: dict, known_rad: list[dict[str, float]], ik_report: dict) -> dict:
    """Compare solved coordinates with the coordinates that placed the model markers."""
    if ik_report.get("sampled_from_opensim_model") is not True:
        raise ValueError("Known-coordinate validation requires markers sampled from the OpenSim model.")
    if ik_report.get("geometry", {}).get("consistent_with_model_marker_spacing") is False:
        raise ValueError("The TRC marker spacing does not match the model, so it is not a validation set.")
    frames = extracted["frames"]
    if len(known_rad) != len(frames):
        raise ValueError("Known poses and MOT frames do not have the same length.")
    per_coordinate = {}
    pooled = []
    for name in extracted["measured_coordinates"]:
        errors = []
        for frame, pose in zip(frames, known_rad):
            if name not in pose:
                raise ValueError(f"Known poses have no value for measured coordinate {name}.")
            errors.append(frame["solved"][name]["radians"] - float(pose[name]))
        per_coordinate[name] = _metrics(errors)
        pooled.extend(errors)
    return {
        "label": "model_generated_numerical_validation",
        "clinical_accuracy": False,
        "participant_capture": False,
        "participant_accuracy": False,
        "compared_with": "joint coordinates used to place markers on the generic model",
        "per_coordinate": per_coordinate,
        "pooled": _metrics(pooled),
    }


def _write_csv(path: Path, extracted: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_sec", "side", "group", "coordinate", "radians", "degrees"])
        for frame in extracted["frames"]:
            for name in MEASURED:
                row = frame["solved"].get(name)
                if row is None:
                    continue
                writer.writerow(
                    [
                        f"{frame['time_sec']:.8f}",
                        row["side"],
                        row["group"],
                        name,
                        f"{row['radians']:.8f}",
                        f"{row['degrees']:.8f}",
                    ]
                )


def _degrees(extracted: dict, name: str) -> list[float]:
    return [frame["solved"][name]["degrees"] for frame in extracted["frames"]]


def _draw_angle(ax, times: list[float], series: list[float], name: str) -> None:
    note = PLAIN_NOTE.get(name)
    title = PLAIN_NAME[name]
    ax.plot(times, series, color="#1f4e79", linewidth=1.8)
    ax.axhline(0, color="#c5cdd6", linewidth=0.8)
    ax.set_title(f"{title} — {note}" if note else title, loc="left", fontsize=12, color="#1a2332")
    ax.set_ylabel("degrees")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)


def _write_plots(directory: Path, extracted: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    times = extracted["timestamps_sec"]
    written = []
    for filename, names, title in PLOTS:
        present = [name for name in names if name in extracted["measured_coordinates"]]
        if not present:
            continue
        columns = 2 if filename == "timeline.png" else 1
        rows = (len(present) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(10, 2.5 * rows), sharex=True, squeeze=False)
        flat = list(axes.ravel())
        for ax, name in zip(flat, present):
            _draw_angle(ax, times, _degrees(extracted, name), name)
        for ax in flat[len(present) :]:
            ax.set_visible(False)
        flat[min(len(present), len(flat)) - 1].set_xlabel("time (s)")
        if columns == 2 and len(present) > 1:
            flat[len(present) - 2].set_xlabel("time (s)")
        fig.suptitle(f"{title}\nSkeleton angles. Not the live dials.", fontsize=13, color="#0b1f3a")
        fig.tight_layout()
        path = directory / filename
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(filename)
    return written


def _write_markdown(path: Path, summary: dict) -> None:
    validation = summary.get("validation")
    conventions = summary["extraction"]["conventions"]
    if summary.get("participant_capture"):
        sizing = summary.get("scale_statement")
        intro = [
            "These are OpenSim coordinates from a recorded capture.",
            "Marker positions were copied in camera metres. The model ground is the colour camera.",
            sizing
            if sizing
            else "This is a model fit on an unscaled skeleton. It is not a clinical measurement.",
        ]
    else:
        intro = [
            "These are OpenSim model coordinates recovered from markers that were placed on the generic model.",
            "This is a numerical check of that recovery. It is not a participant recording and it is not a clinical measurement.",
        ]
    lines = [
        "# Upper-body joint angles",
        "",
        *intro,
        "",
        "## Source",
        "",
        f"- Motion file: `{summary['mot_file']}`",
        f"- IK report: `{summary['ik_report_file']}`",
        f"- Rotational values in the MOT file: {summary['extraction']['mot_angular_unit']}.",
        "- The CSV and JSON keep the radian values. The table below and the plots use degrees.",
        "",
        "## Conventions",
        "",
        f"- Ground: {conventions['ground_frame']}",
        "- Right shoulder, in order: arm_flex_r about (0, 0, 1), arm_add_r about (1, 0, 0), arm_rot_r about (0, 1, 0).",
        "- Left shoulder, in order: arm_flex_l about (0, 0, 1), arm_add_l about (-1, 0, 0), arm_rot_l about (0, -1, 0).",
        f"- Elbow: {CONVENTIONS['elbow']}",
        f"- Forearm: {CONVENTIONS['forearm']}",
        f"- Live calculator: {CONVENTIONS['live_calculator']}",
        "",
        "## Solved coordinates (degrees)",
        "",
        "| time (s) | coordinate | side | degrees |",
        "| --- | --- | --- | ---: |",
    ]
    for frame in summary["extraction"]["frames"]:
        for name in MEASURED:
            row = frame["solved"].get(name)
            if row is None:
                continue
            lines.append(f"| {frame['time_sec']:.3f} | {name} | {row['side']} | {row['degrees']:.3f} |")
    lines.extend(["", "## Comparison with the coordinates that placed the markers", ""])
    if validation is None:
        lines.append("No known input coordinates were supplied.")
    else:
        lines.append(validation["label"] + ". Not participant accuracy.")
        lines.extend(["", "| coordinate | MAE (deg) | max (deg) | RMSE (deg) |", "| --- | ---: | ---: | ---: |"])
        for name, metrics in validation["per_coordinate"].items():
            lines.append(
                f"| {name} | {metrics['mean_absolute_error_deg']:.4f} | "
                f"{metrics['maximum_absolute_error_deg']:.4f} | {metrics['rmse_deg']:.4f} |"
            )
        pooled = validation["pooled"]
        lines.extend(
            [
                "",
                (
                    f"Pooled mean absolute error {pooled['mean_absolute_error_deg']:.4f} deg, "
                    f"maximum {pooled['maximum_absolute_error_deg']:.4f} deg, "
                    f"RMSE {pooled['rmse_deg']:.4f} deg."
                ),
            ]
        )
    lines.extend(["", "## Not measured", ""])
    lines.append("Locked and unsupported coordinates are omitted from the table above.")
    for row in summary["extraction"]["locked_not_measured"]:
        lines.append(f"- `{row['coordinate']}` ({row['status']}): {row['reason']}")
    lines.extend(["", "## Plots", ""])
    for name in summary["plots"]:
        lines.append(f"- `plots/{name}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_angle_analysis(
    mot_path: Path,
    ik_report_path: Path,
    output_dir: Path,
    known_rad: list[dict[str, float]] | None = None,
    ground_frame: str | None = None,
    scale_statement: str | None = None,
) -> dict:
    """Write the angle table, validation, plots, and markdown report."""
    motion = read_mot(mot_path)
    ik_report = json.loads(Path(ik_report_path).read_text(encoding="utf-8"))
    extracted = extract_angles(motion, ik_report)
    if ground_frame:
        extracted["conventions"] = {**extracted["conventions"], "ground_frame": ground_frame}
    validation = None
    if known_rad is not None:
        validation = validate_against_known(extracted, known_rad, ik_report)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "joint_angles.csv", extracted)
    (output_dir / "joint_angles.json").write_text(json.dumps(extracted, indent=2), encoding="utf-8")
    if validation is not None:
        (output_dir / "angle_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    plots = _write_plots(output_dir / "plots", extracted)
    summary = {
        "mot_file": str(Path(mot_path)),
        "ik_report_file": str(Path(ik_report_path)),
        "clinical_accuracy": False,
        "participant_capture": bool(ik_report.get("participant_capture")),
        "scale_statement": scale_statement,
        "extraction": extracted,
        "validation": validation,
        "plots": plots,
    }
    _write_markdown(output_dir / "analysis_report.md", summary)
    return summary
