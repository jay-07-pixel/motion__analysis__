"""Run upper-body IK with the isolated OpenSim 4.4 interpreter.

    C:\\Users\\jayjo\\anaconda3\\envs\\opensim-upper\\python.exe tools/opensim_ik_run.py <plan.json>

The plan lists the model, the TRC, the output directory, and optional weight
overrides. Locked coordinates are changed only in memory. The model file and
the geometry directory are not written. Inverse kinematics is the only
OpenSim tool used.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.opensim_upper.ik_fit import marker_spacing, model_marker_rows, per_marker_residuals  # noqa: E402
from src.opensim_upper.scaling import spacing_mapping  # noqa: E402
from src.opensim_upper.ik_tasks import coordinate_decisions, ik_tasks, validate_trc_for_ik  # noqa: E402
from src.opensim_upper.mot import read_mot as _read_mot  # noqa: E402


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def _all_nan(column, n_rows: int) -> bool:
    for index in range(n_rows):
        sample = column[index]
        if not any(math.isnan(sample[axis]) for axis in range(3)):
            return False
    return True


def _marker_errors(folder: Path) -> dict | None:
    files = sorted(folder.glob("*marker*error*.sto")) + sorted(folder.glob("*Marker*Error*.sto"))
    if not files:
        files = [path for path in folder.glob("*.sto") if path.name != "ik_model_marker_locations.sto"]
    if not files:
        return None
    table = _read_mot(files[0])
    summary = {}
    for name in table["columns"]:
        if name == "time":
            continue
        values = [abs(row[name]) for row in table["rows"] if name in row and _finite(row[name])]
        if not values:
            continue
        summary[name] = {
            "mean_abs_m": sum(values) / len(values),
            "max_abs_m": max(values),
        }
    return {"file": files[0].name, "columns": summary}


def run(plan_path: Path) -> int:
    import opensim

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    log_lines = []
    model_path = Path(plan["model"])
    trc_path = Path(plan["trc"])
    before = model_path.read_bytes()
    report: dict = {
        "execution_success": False,
        "biomechanical_reliability": False,
        "clinical_accuracy": False,
        "full_body_ik_configuration_used": False,
        "model_file_modified": False,
        "synthetic": plan.get("synthetic"),
        "participant_capture": plan.get("participant_capture"),
        "physical_validation": plan.get("physical_validation", "not_performed"),
    }
    try:
        parsed = validate_trc_for_ik(trc_path)
        absent = []
        table = opensim.TimeSeriesTableVec3(str(trc_path))
        labels = list(table.getColumnLabels())
        for index, label in enumerate(labels):
            if _all_nan(table.getDependentColumnAtIndex(index), table.getNumRows()):
                absent.append(label)
        setup = ik_tasks(plan.get("weight_overrides"), absent)
        model = opensim.Model(str(model_path))
        model_names = [model.getCoordinateSet().get(i).getName() for i in range(model.getCoordinateSet().getSize())]
        decisions = coordinate_decisions(model_names)
        unlock = set(plan.get("unlock_coordinates") or [])
        extra_lock = set(plan.get("lock_coordinates") or [])
        unknown_unlock = unlock.difference(decisions)
        unknown_lock = extra_lock.difference(decisions)
        if unknown_unlock or unknown_lock:
            raise ValueError(
                f"Coordinate override names an unknown coordinate: {sorted(unknown_unlock | unknown_lock)}"
            )
        if unlock & extra_lock:
            raise ValueError(f"A coordinate cannot be both locked and unlocked: {sorted(unlock & extra_lock)}")
        held = {}
        for name, decision in decisions.items():
            coordinate = model.updCoordinateSet().get(name)
            default = float(coordinate.getDefaultValue())
            locked = name in extra_lock or (decision["locked"] and name not in unlock)
            held[name] = {
                **decision,
                "locked": locked,
                "solved": not locked and (decision["solved"] or name in unlock),
                "default_value": default,
                "range": [float(coordinate.getRangeMin()), float(coordinate.getRangeMax())],
                "motion_type": "rotational" if int(coordinate.getMotionType()) == 1 else "translational",
                "unlocked_for_observability_check": name in unlock,
                "locked_for_observability_check": name in extra_lock,
                "clamped_to_model_range": False,
            }
            if locked:
                coordinate.set_locked(True)
            elif held[name]["motion_type"] == "rotational":
                # The model file leaves the shoulder angles unclamped. The IK
                # solver then walks those angles far outside the stated range
                # even when the markers were taken from the model. The clamp
                # uses that coordinate's own range and is not written to the file.
                coordinate.set_clamped(True)
                held[name]["clamped_to_model_range"] = True
        loaded_name = model.getName()
        state = model.initSystem()
        for name, info in held.items():
            if info["locked"]:
                model.updCoordinateSet().get(name).setValue(state, info["default_value"])
        tool = opensim.InverseKinematicsTool()
        tool.setName("upper_body_12_marker")
        tool.setModel(model)
        tool.set_model_file(str(model_path))
        tool.setMarkerDataFileName(str(trc_path))
        tool.setStartTime(parsed["frames"][0]["time_sec"])
        tool.setEndTime(parsed["frames"][-1]["time_sec"])
        mot_path = output / "ik_results.mot"
        tool.setOutputMotionFileName(str(mot_path))
        tool.setResultsDir(str(output))
        tool.set_results_directory(str(output))
        tool.set_accuracy(1.0e-5)
        tool.set_report_errors(True)
        tool.set_report_marker_locations(True)
        for task_row in setup["tasks"]:
            task = opensim.IKMarkerTask()
            task.setName(task_row["name"])
            task.setWeight(float(task_row["weight"]))
            task.setApply(bool(task_row["apply"]))
            tool.upd_IKTaskSet().adoptAndAppend(task)
        setup_path = output / "ik_setup.xml"
        tool.printToXML(str(setup_path))
        log_lines.append(
            f"IK from {parsed['frames'][0]['time_sec']} to {parsed['frames'][-1]['time_sec']} s, "
            f"{parsed['num_frames']} TRC frames, {len(setup['tasks'])} marker tasks."
        )
        tool.run()
        if model_path.read_bytes() != before:
            raise RuntimeError("The model file changed during IK.")
        motion = _read_mot(mot_path)
        # OpenSim writes rotational IK coordinates in degrees. The model ranges
        # and the report use radians. Translational coordinates stay in metres.
        if motion["in_degrees"]:
            for name in motion["columns"]:
                if name == "time" or held[name]["motion_type"] != "rotational":
                    continue
                for row in motion["rows"]:
                    row[name] = math.radians(row[name])
        times = [row["time"] for row in motion["rows"]]
        trc_times = [frame["time_sec"] for frame in parsed["frames"]]
        time_ok = len(times) == len(trc_times) and all(abs(a - b) <= 1e-4 for a, b in zip(times, trc_times))
        coordinate_report = {}
        if loaded_name == "UpperBody_CameraFrame_Scaled":
            model_reason = (
                "Inverse kinematics ran on the sized copy of the camera-frame model. "
                "The generic model file was not loaded."
            )
        else:
            model_reason = "The model is the generic unscaled Pose2Sim model."
        reliability_reasons = [
            model_reason,
            "No participant calibration has been accepted as a biomechanical validation.",
        ]
        if plan.get("synthetic") is True:
            reliability_reasons.append("The TRC is labelled synthetic test data, not a participant recording.")
        limit_hits = False
        for name in motion["columns"]:
            if name == "time":
                continue
            decision = held[name]
            values = [row[name] for row in motion["rows"]]
            low, high = decision["range"]
            at_limit = sum(1 for value in values if abs(value - low) <= 1e-3 or abs(value - high) <= 1e-3)
            fraction = at_limit / len(values) if values else 0.0
            unchanged = all(abs(value - decision["default_value"]) <= 1e-5 for value in values)
            coordinate_report[name] = {
                "locked": decision["locked"],
                "solved": decision["solved"],
                "identifiable": decision["identifiable"],
                "reason": decision["reason"],
                "value_source": decision["value_source"],
                "default_value": decision["default_value"],
                "motion_type": decision["motion_type"],
                "range": decision["range"],
                "min_result": min(values),
                "max_result": max(values),
                "fraction_at_limit": fraction,
                "unchanged_from_model_default": unchanged,
                "clamped_to_model_range": decision.get("clamped_to_model_range", False),
                "unit": "metres" if decision["motion_type"] == "translational" else "radians",
            }
            outside = any(value < low - 1e-3 or value > high + 1e-3 for value in values)
            coordinate_report[name]["outside_model_range"] = outside
            if decision["locked"] and not unchanged:
                reliability_reasons.append(f"{name} was locked but the MOT value moved.")
            if decision["solved"] and fraction > 0.5:
                limit_hits = True
            if decision["solved"] and outside:
                reliability_reasons.append(
                    f"{name} left the model range ({low:.3f} to {high:.3f} {coordinate_report[name]['unit']})."
                )
        errors = _marker_errors(output)
        if errors is None:
            reliability_reasons.append("OpenSim did not write a marker-error table.")
        else:
            rms_values = errors["columns"].get("marker_error_RMS")
            if rms_values is None:
                reliability_reasons.append("OpenSim marker-error table has no marker_error_RMS column.")
            elif rms_values["mean_abs_m"] > 0.05:
                reliability_reasons.append(
                    f"marker_error_RMS mean is {rms_values['mean_abs_m']:.3f} m."
                )
        geometry = marker_spacing(parsed, spacing_mapping(model_path))
        location_files = sorted(output.glob("*model_marker_locations.sto"))
        if not location_files:
            raise ValueError("OpenSim did not write model marker locations.")
        fit = per_marker_residuals(parsed["frames"], model_marker_rows(_read_mot(location_files[0])))
        if not geometry["consistent_with_model_marker_spacing"]:
            pair = geometry["largest_pair"]
            reliability_reasons.append(
                "TRC same-parent spacing does not match the model: "
                f"{geometry['n_mismatch']} of {geometry['n_pairs']} pairs. "
                f"Largest pair is {pair['marker_a']} to {pair['marker_b']}: "
                f"model {pair['model_distance_m']:.3f} m, TRC {pair['experimental_distance_m']:.3f} m. "
                "The registration does not change these distances."
            )
            if fit["ranked_by_rms"]:
                top = fit["ranked_by_rms"][0]
                reliability_reasons.append(
                    f"{top} has the largest IK residual, RMS {fit['markers'][top]['rms_m']:.3f} m."
                )
            for name, row in coordinate_report.items():
                if not (row["solved"] and row["fraction_at_limit"] > 0.5):
                    continue
                low, high = row["range"]
                at_min = abs(row["min_result"] - low) <= 1e-3 and abs(row["max_result"] - low) <= 1e-3
                at_max = abs(row["min_result"] - high) <= 1e-3 and abs(row["max_result"] - high) <= 1e-3
                edge = "minimum" if at_min else "maximum" if at_max else "boundary"
                reliability_reasons.append(
                    f"{name} remains at the model range {edge} "
                    f"({low:.5f} to {high:.5f} rad). The range was not widened. "
                    "The input cloud is not a reachable pose of this marker set."
                )
        else:
            for name, row in coordinate_report.items():
                if row["solved"] and row["fraction_at_limit"] > 0.5:
                    reliability_reasons.append(f"{name} is at a joint limit for more than half of the frames.")
        report.update(
            {
                "execution_success": time_ok and all(
                    coordinate_report[name]["unchanged_from_model_default"]
                    for name, decision in held.items()
                    if decision["locked"]
                ),
                "biomechanical_reliability": False,
                "reliability_reasons": reliability_reasons,
                "solved_coordinate_hit_limits": limit_hits,
                "n_frames": len(motion["rows"]),
                "trc_n_frames": parsed["num_frames"],
                "timestamps_match_trc": time_ok,
                "coordinate_names": [name for name in motion["columns"] if name != "time"],
                "n_coordinates": len(motion["columns"]) - 1,
                "mot_file_angular_unit": "degrees" if motion["in_degrees"] else "radians",
                "reported_angular_unit": "radians",
                "coordinates": coordinate_report,
                "marker_tasks": setup["tasks"],
                "unlocked_for_observability_check": sorted(unlock),
                "locked_for_observability_check": sorted(extra_lock),
                "sampled_from_opensim_model": plan.get("sampled_from_opensim_model"),
                "geometry": geometry,
                "marker_fit": fit,
                "markers_absent_on_every_frame": absent,
                "marker_errors": errors,
                "time_range_sec": [trc_times[0], trc_times[-1]],
            }
        )
        log_lines.append(f"execution_success={report['execution_success']}")
        log_lines.append("biomechanical_reliability=false")
        log_lines.extend(reliability_reasons)
    except Exception as error:
        log_lines.append(traceback.format_exc())
        report["error"] = str(error)
        report["execution_success"] = False
        report["biomechanical_reliability"] = False
    if model_path.exists() and model_path.read_bytes() != before:
        report["model_file_modified"] = True
        report["execution_success"] = False
    (output / "ik_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "ik_logs.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0 if report.get("execution_success") else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: opensim_ik_run.py plan.json")
        raise SystemExit(2)
    raise SystemExit(run(Path(sys.argv[1])))
