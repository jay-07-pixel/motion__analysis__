"""Check which upper-body coordinates the 12 markers can recover.

Markers are sampled from the model itself at prescribed coordinates, then
solved with the same IK runner. This is a model-fit check. It is not a
participant recording and it is not a clinical accuracy claim.

    C:\\Users\\jayjo\\anaconda3\\envs\\opensim-upper\\python.exe tools/opensim_ik_observe.py <output_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

from src.opensim_upper.model_poses import ARM_POSES_RAD as _ARM  # noqa: E402
from src.opensim_upper.model_poses import AXIAL_POSES_RAD as _AXIAL  # noqa: E402
from src.opensim_upper.trc import ordered_marker_rows, write_trc  # noqa: E402

_spec = importlib.util.spec_from_file_location("opensim_ik_run", ROOT / "tools" / "opensim_ik_run.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)

MODEL = ROOT / "models" / "opensim_upper" / "Model_Pose2Sim_upper.osim"
SHOULDER_ELBOW = ("arm_flex_r", "arm_add_r", "elbow_flex_r", "arm_flex_l", "arm_add_l", "elbow_flex_l")
AXIAL = ("arm_rot_r", "arm_rot_l", "pro_sup_r", "pro_sup_l")
WRIST = ("wrist_flex_r", "wrist_dev_r", "wrist_flex_l", "wrist_dev_l")


def _location(model, state, name: str) -> tuple[float, float, float]:
    point = model.getMarkerSet().get(name).getLocationInGround(state)
    return (float(point[0]), float(point[1]), float(point[2]))


def _pose_markers(model, prescribed: list[dict]) -> dict[str, list[tuple[float, float, float]]]:
    names = [row["target"] for row in ordered_marker_rows()]
    state = model.initSystem()
    samples = {name: [] for name in names}
    for pose in prescribed:
        for name, value in pose.items():
            model.updCoordinateSet().get(name).setValue(state, float(value))
        model.realizePosition(state)
        for name in names:
            samples[name].append(_location(model, state, name))
        for name in pose:
            coordinate = model.updCoordinateSet().get(name)
            coordinate.setValue(state, float(coordinate.getDefaultValue()))
    return samples


def _write_pose_trc(model, folder: Path, prescribed: list[dict]) -> Path:
    path = folder / "markers.trc"
    times = [index * 0.1 for index in range(len(prescribed))]
    write_trc(path, times, _pose_markers(model, prescribed), 10.0, ordered_marker_rows())
    return path


def _max_shift(before: dict, after: dict) -> float:
    worst = 0.0
    for name, start in before.items():
        end = after[name]
        shift = sum((end[axis] - start[axis]) ** 2 for axis in range(3)) ** 0.5
        worst = max(worst, shift)
    return worst


def _single_shift(model, coordinate: str, delta: float) -> float:
    names = [row["target"] for row in ordered_marker_rows()]
    state = model.initSystem()
    model.realizePosition(state)
    before = {name: _location(model, state, name) for name in names}
    current = float(model.updCoordinateSet().get(coordinate).getValue(state))
    model.updCoordinateSet().get(coordinate).setValue(state, current + delta)
    model.realizePosition(state)
    after = {name: _location(model, state, name) for name in names}
    return _max_shift(before, after)


def _run_ik(model_path: Path, trc: Path, folder: Path, lock: list[str]) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    plan = folder / "ik_plan.json"
    plan.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "trc": str(trc),
                "output_dir": str(folder),
                "weight_overrides": {},
                "synthetic": True,
                "participant_capture": False,
                "physical_validation": "not_performed_model_generated_markers",
                "sampled_from_opensim_model": True,
                "lock_coordinates": lock,
            }
        ),
        encoding="utf-8",
    )
    _runner.run(plan)
    return json.loads((folder / "ik_report.json").read_text(encoding="utf-8"))


def _frame_errors(folder: Path, prescribed: list[dict], names: tuple[str, ...]) -> dict:
    import math

    motion = _runner._read_mot(folder / "ik_results.mot")
    if motion["in_degrees"]:
        for row in motion["rows"]:
            for name in names:
                row[name] = math.radians(row[name])
    per_coordinate = {}
    for name in names:
        errors = [abs(motion["rows"][index][name] - pose[name]) for index, pose in enumerate(prescribed)]
        per_coordinate[name] = {"max_abs_rad": max(errors), "errors_rad": errors}
    return per_coordinate


def main(folder: Path) -> dict:
    import opensim

    folder.mkdir(parents=True, exist_ok=True)
    model = opensim.Model(str(MODEL))
    model.initSystem()
    wrist_pose = {name: 0.4 for name in WRIST}
    wrist_shift = _max_shift(
        {name: samples[0] for name, samples in _pose_markers(model, [{}]).items()},
        {name: samples[0] for name, samples in _pose_markers(model, [wrist_pose]).items()},
    )
    signal = {
        name: _single_shift(model, name, 0.8 if not name.startswith("pro_sup") else -0.8)
        for name in AXIAL
    }
    arm_prescribed = _ARM
    axial_prescribed = [{**arm, **axial} for arm, axial in zip(_ARM, _AXIAL)]
    arm_trc = _write_pose_trc(model, folder / "arm_only", arm_prescribed)
    axial_trc = _write_pose_trc(model, folder / "with_axial", axial_prescribed)
    locked = _run_ik(MODEL, arm_trc, folder / "arm_only", [])
    axial_locked = _run_ik(MODEL, axial_trc, folder / "with_axial_locked", list(AXIAL))
    axial_free = _run_ik(MODEL, axial_trc, folder / "with_axial_free", [])
    summary = {
        "synthetic": True,
        "participant_capture": False,
        "clinical_accuracy": False,
        "markers_sampled_from_model": True,
        "physical_validation": "not_performed_model_generated_markers",
        "wrist_coordinate_max_marker_shift_m": wrist_shift,
        "axial_coordinate_max_marker_shift_m": signal,
        "arm_only_recovery_rad": _frame_errors(folder / "arm_only", arm_prescribed, SHOULDER_ELBOW),
        "axial_locked_shoulder_elbow_rad": _frame_errors(
            folder / "with_axial_locked", axial_prescribed, SHOULDER_ELBOW
        ),
        "axial_free_recovery_rad": _frame_errors(
            folder / "with_axial_free", axial_prescribed, SHOULDER_ELBOW + AXIAL
        ),
        "documented_tolerances": {
            "coordinate_abs_error_rad": {
                "arm_flex_arm_add_elbow_flex": 0.01,
                "arm_rot": 0.02,
                "pro_sup": 0.02,
            },
            "per_marker_rms_m": 0.001,
            "meaning": (
                "Numerical agreement with the joint coordinates that placed these "
                "markers on the generic model. Not a participant or clinical tolerance."
            ),
        },
        "arm_only_rms_m": locked["marker_errors"]["columns"]["marker_error_RMS"]["mean_abs_m"],
        "axial_locked_rms_m": axial_locked["marker_errors"]["columns"]["marker_error_RMS"]["mean_abs_m"],
        "axial_free_rms_m": axial_free["marker_errors"]["columns"]["marker_error_RMS"]["mean_abs_m"],
        "arm_only_marker_fit_rms_m": locked["marker_fit"]["unweighted_rms_m"],
        "axial_free_marker_fit_rms_m": axial_free["marker_fit"]["unweighted_rms_m"],
        "arm_only_largest_marker_rms_m": locked["marker_fit"]["markers"][locked["marker_fit"]["ranked_by_rms"][0]]["rms_m"],
        "axial_free_largest_marker_rms_m": axial_free["marker_fit"]["markers"][axial_free["marker_fit"]["ranked_by_rms"][0]]["rms_m"],
        "arm_only_spacing_consistent": locked["geometry"]["consistent_with_model_marker_spacing"],
        "axial_free_spacing_consistent": axial_free["geometry"]["consistent_with_model_marker_spacing"],
        "arm_only_execution_success": locked["execution_success"],
        "axial_free_execution_success": axial_free["execution_success"],
    }
    (folder / "observability.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: opensim_ik_observe.py output_dir")
        raise SystemExit(2)
    print(json.dumps(main(Path(sys.argv[1])), indent=2))
