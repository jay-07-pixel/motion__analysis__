"""Offline Marker Enhancer arm LSTM. Not used by the live session.

This script does not compute shoulder rotation, pronation, or wrist angles.
It predicts eight arm markers and writes a validation report.

Height and weight are required arguments. They are not read from config
and they are not filled in from the published test subject.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fixed lab mapping for a person facing this camera. Do not rebuild it
# from the shoulder line.
R_CAM_TO_ME = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)

# MediaPipe name -> Marker Enhancer input name, in model order after Neck.
MEASURED_MARKERS = (
    ("right_shoulder", "RShoulder"),
    ("left_shoulder", "LShoulder"),
    ("right_elbow", "RElbow"),
    ("left_elbow", "LElbow"),
    ("right_wrist", "RWrist"),
    ("left_wrist", "LWrist"),
    ("right_hip", "RHip"),
    ("left_hip", "LHip"),
)

# Order verified in utilities.getMarkersPoseDetector_upperExtremity.
INPUT_MARKER_ORDER = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RElbow",
    "LElbow",
    "RWrist",
    "LWrist",
)

# Order verified in utilities.getMarkersAugmenter_upperExtremity.
OUTPUT_MARKERS = (
    "RElbow_augmenter",
    "RMElbow_augmenter",
    "RWrist_augmenter",
    "RMWrist_augmenter",
    "LElbow_augmenter",
    "LMElbow_augmenter",
    "LWrist_augmenter",
    "LMWrist_augmenter",
)

# Measured joint used only for a distance check. Medial markers have none.
MEASURED_COMPARISON = {
    "RElbow_augmenter": "RElbow",
    "LElbow_augmenter": "LElbow",
    "RWrist_augmenter": "RWrist",
    "LWrist_augmenter": "LWrist",
}

PAIRS = (
    ("right_elbow_pair", "RElbow_augmenter", "RMElbow_augmenter"),
    ("left_elbow_pair", "LElbow_augmenter", "LMElbow_augmenter"),
    ("right_wrist_pair", "RWrist_augmenter", "RMWrist_augmenter"),
    ("left_wrist_pair", "LWrist_augmenter", "LMWrist_augmenter"),
)

# Exploratory only. These are not anatomical acceptance limits.
HEURISTIC_PAIR_MIN_M = 0.02
HEURISTIC_PAIR_MAX_M = 0.12
HEURISTIC_STEP_MAX_M = 0.15
HEURISTIC_ABS_COORD_MAX_M = 5.0

MIN_CONFIDENCE = 0.5
DEFAULT_CSV = PROJECT_ROOT / "data" / "output" / "20261005-150538" / "keypoints_2d.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "experiments" / "marker_enhancer_20261005"
NECK_LABEL = "shoulder_midpoint_neck_proxy"


def _require_anthropometrics(height_m: float | None, weight_kg: float | None) -> tuple[float, float]:
    """Stop before TensorFlow is imported when stature or mass was not given."""
    missing = []
    if height_m is None:
        missing.append("--height-m")
    if weight_kg is None:
        missing.append("--weight-kg")
    if missing:
        raise SystemExit(
            "Inference stopped before the model was loaded. "
            "Required anthropometric input(s) were not supplied: "
            + ", ".join(missing)
            + ". Pass this subject's height in metres and weight in kilograms. "
            "Repository averages and the published test subject are not used."
        )
    if height_m <= 0.0 or not math.isfinite(height_m):
        raise SystemExit("--height-m must be a positive finite number of metres.")
    if weight_kg <= 0.0 or not math.isfinite(weight_kg):
        raise SystemExit("--weight-kg must be a positive finite number of kilograms.")
    return float(height_m), float(weight_kg)


def _import_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit(
            "Model inference blocked because the required runtime is not installed "
            f"in this interpreter ({sys.executable}). {exc}"
        ) from exc
    return tf


def load_arm_model(model_dir: Path, tf):
    """Load the shipped arm LSTM. Fail if the graph is not the expected shape."""
    model_json = model_dir / "model.json"
    weights = model_dir / "weights.h5"
    if not model_json.is_file() or not weights.is_file():
        raise SystemExit(f"Arm model files are missing in {model_dir}.")
    model = tf.keras.models.model_from_json(model_json.read_text(encoding="utf-8"))
    model.load_weights(weights)
    in_shape = tuple(model.input_shape)
    out_shape = tuple(model.output_shape)
    if in_shape[-1] != 23 or out_shape[-1] != 24:
        raise SystemExit(
            f"Unexpected model shapes input={in_shape} output={out_shape}. "
            "Expected the last dimensions to be 23 and 24."
        )
    return model


def runtime_report(model_dir: Path) -> dict:
    """Load the arm LSTM and print versions. Does not read the recording."""
    tf = _import_tensorflow()
    import keras

    model = load_arm_model(model_dir, tf)
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "compatible_with_batch_464_23": (
            model.input_shape[0] is None
            and model.input_shape[1] is None
            and model.input_shape[2] == 23
        ),
    }
    print(json.dumps(info, indent=2))
    if not info["compatible_with_batch_464_23"]:
        raise SystemExit("Model input shape is not compatible with (1, 464, 23).")
    return info


def _to_me(points: np.ndarray) -> np.ndarray:
    """Camera metres to the fixed Marker Enhancer lab frame. points is (N, 3)."""
    return points @ R_CAM_TO_ME.T


def _to_camera(points: np.ndarray) -> np.ndarray:
    return points @ R_CAM_TO_ME


def load_recording(csv_path: Path) -> tuple[list[dict], dict]:
    """Read camera-metre rows. A frame is kept only when all eight joints exist."""
    by_frame: dict[int, dict] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["coord_frame"] != "camera_3d_metre":
                continue
            if not row["x_m"] or not row["y_m"] or not row["z_m"]:
                continue
            if float(row["confidence"]) < MIN_CONFIDENCE:
                continue
            point = np.array(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                dtype=float,
            )
            if not np.all(np.isfinite(point)):
                continue
            frame = int(row["frame"])
            slot = by_frame.setdefault(frame, {"time_sec": float(row["time_sec"]), "joints": {}})
            slot["joints"][row["joint"]] = point

    needed = [name for name, _ in MEASURED_MARKERS]
    frames = []
    for frame in sorted(by_frame):
        joints = by_frame[frame]["joints"]
        if all(name in joints for name in needed):
            frames.append(
                {
                    "frame": frame,
                    "time_sec": by_frame[frame]["time_sec"],
                    "joints": joints,
                }
            )
    if not frames:
        raise SystemExit(f"No frame in {csv_path} has all eight required camera points.")
    times = np.array([item["time_sec"] for item in frames], dtype=float)
    deltas = np.diff(times) if times.size > 1 else np.array([])
    timing = {
        "frames": len(frames),
        "duration_s": float(times[-1] - times[0]) if times.size > 1 else 0.0,
        "median_dt_s": float(np.median(deltas)) if deltas.size else math.nan,
        "median_fps": float(1.0 / np.median(deltas)) if deltas.size and np.median(deltas) > 0 else math.nan,
        "min_dt_s": float(np.min(deltas)) if deltas.size else math.nan,
        "max_dt_s": float(np.max(deltas)) if deltas.size else math.nan,
        "training_hz": 60.0,
        "note": "Input temporal distribution differs from training distribution.",
    }
    return frames, timing


def build_features(frames: list[dict], height_m: float, weight_kg: float, mean: np.ndarray, std: np.ndarray):
    """Return model input (1, T, 23) and the ME-frame midHip and measured markers."""
    if mean.shape != (23,) or std.shape != (23,):
        raise SystemExit(f"mean/std must both have shape (23,). Got {mean.shape} and {std.shape}.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std == 0.0):
        raise SystemExit("mean.npy or std.npy is not a usable 23-feature scale.")

    features = np.zeros((len(frames), 23), dtype=float)
    measured_me = {name: np.zeros((len(frames), 3), dtype=float) for name in INPUT_MARKER_ORDER if name != "Neck"}
    measured_me["Neck"] = np.zeros((len(frames), 3), dtype=float)
    midhip_me = np.zeros((len(frames), 3), dtype=float)

    for index, item in enumerate(frames):
        joints = item["joints"]
        named = {me_name: _to_me(joints[mp_name].reshape(1, 3))[0] for mp_name, me_name in MEASURED_MARKERS}
        mid = _to_me(((joints["right_hip"] + joints["left_hip"]) / 2.0).reshape(1, 3))[0]
        neck = (named["RShoulder"] + named["LShoulder"]) / 2.0
        named["Neck"] = neck
        midhip_me[index] = mid
        for name in measured_me:
            measured_me[name][index] = named[name]
        centered = []
        for name in INPUT_MARKER_ORDER:
            centered.append((named[name] - mid) / height_m)
        marker_part = np.concatenate(centered)
        raw = np.concatenate([marker_part, np.array([height_m, weight_kg], dtype=float)])
        features[index] = (raw - mean) / std

    return features.reshape(1, len(frames), 23), measured_me, midhip_me


def _series_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {key: math.nan for key in ("median", "mean", "std", "min", "max", "p05", "p95")}
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _step_stats(points: np.ndarray) -> dict:
    if points.shape[0] < 2:
        return {"median_m": math.nan, "p95_m": math.nan, "max_m": math.nan}
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return {
        "median_m": float(np.median(steps)),
        "p95_m": float(np.percentile(steps, 95)),
        "max_m": float(np.max(steps)),
        "heuristic_jump_count": int(np.sum(steps > HEURISTIC_STEP_MAX_M)),
    }


def validate(predicted_camera: dict[str, np.ndarray], measured_me: dict[str, np.ndarray], predicted_me: dict[str, np.ndarray]) -> dict:
    """Numerical checks. Pair-length limits are labeled heuristic."""
    non_finite = {}
    extreme = {}
    for name, points in predicted_camera.items():
        bad = ~np.isfinite(points)
        non_finite[name] = int(np.sum(bad))
        extreme[name] = int(np.sum(np.any(np.abs(points) > HEURISTIC_ABS_COORD_MAX_M, axis=1)))

    comparisons = {}
    for predicted_name, measured_name in MEASURED_COMPARISON.items():
        distance = np.linalg.norm(predicted_me[predicted_name] - measured_me[measured_name], axis=1)
        comparisons[predicted_name] = {"against": measured_name, **_series_stats(distance)}

    pairs = {}
    heuristic_pair_flags = []
    for label, left, right in PAIRS:
        length = np.linalg.norm(predicted_camera[left] - predicted_camera[right], axis=1)
        stats = _series_stats(length)
        outside = int(np.sum((length < HEURISTIC_PAIR_MIN_M) | (length > HEURISTIC_PAIR_MAX_M)))
        pairs[label] = stats
        pairs[label]["heuristic_outside_count"] = outside
        pairs[label]["heuristic_band_m"] = [HEURISTIC_PAIR_MIN_M, HEURISTIC_PAIR_MAX_M]
        if outside:
            heuristic_pair_flags.append(label)

    steps = {name: _step_stats(points) for name, points in predicted_camera.items()}
    asymmetry = {}
    for side in ("elbow", "wrist"):
        right = pairs[f"right_{side}_pair"]["median"]
        left = pairs[f"left_{side}_pair"]["median"]
        asymmetry[side] = {
            "right_median_m": right,
            "left_median_m": left,
            "left_over_right": float(left / right) if right else math.nan,
            "note": "Descriptive ratio of median pair lengths. No acceptance cutoff is applied.",
        }
    return {
        "non_finite_count": non_finite,
        "heuristic_extreme_count": extreme,
        "heuristic_abs_coord_max_m": HEURISTIC_ABS_COORD_MAX_M,
        "measured_vs_predicted_m": comparisons,
        "pair_lengths_m": pairs,
        "left_right_pair_asymmetry": asymmetry,
        "steps": steps,
        "heuristic_step_max_m": HEURISTIC_STEP_MAX_M,
        "heuristic_pair_flags": heuristic_pair_flags,
        "threshold_note": (
            "Pair-length and step limits are exploratory quality thresholds, "
            "not validated anatomical thresholds."
        ),
    }


def write_predictions(path: Path, frames: list[dict], predicted_camera: dict[str, np.ndarray], predicted_me: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "time_sec",
                "marker",
                "x_m",
                "y_m",
                "z_m",
                "x_me_m",
                "y_me_m",
                "z_me_m",
                "coord_frame",
                "neck_definition",
            ],
        )
        writer.writeheader()
        for index, item in enumerate(frames):
            for name in OUTPUT_MARKERS:
                camera = predicted_camera[name][index]
                me = predicted_me[name][index]
                writer.writerow(
                    {
                        "frame": item["frame"],
                        "time_sec": f"{item['time_sec']:.4f}",
                        "marker": name,
                        "x_m": f"{camera[0]:.6f}",
                        "y_m": f"{camera[1]:.6f}",
                        "z_m": f"{camera[2]:.6f}",
                        "x_me_m": f"{me[0]:.6f}",
                        "y_me_m": f"{me[1]:.6f}",
                        "z_me_m": f"{me[2]:.6f}",
                        "coord_frame": "camera_3d_metre",
                        "neck_definition": NECK_LABEL,
                    }
                )


def format_text(report: dict) -> str:
    lines = [
        "Marker Enhancer offline arm inference",
        f"classification: {report['classification']}",
        f"height_m: {report['height_m']}",
        f"weight_kg: {report['weight_kg']}",
        f"neck: {NECK_LABEL}",
        f"frames: {report['timing']['frames']}",
        (
            f"duration_s: {report['timing']['duration_s']:.3f}  "
            f"median_fps: {report['timing']['median_fps']:.2f}  "
            "training_hz: 60"
        ),
        "Input temporal distribution differs from training distribution.",
        "",
        "Measured joint vs same-name predicted lateral marker (metres)",
    ]
    for name, item in report["validation"]["measured_vs_predicted_m"].items():
        lines.append(
            f"  {name} vs {item['against']}: median {item['median']:.4f}  "
            f"mean {item['mean']:.4f}  std {item['std']:.4f}  "
            f"min {item['min']:.4f}  max {item['max']:.4f}  "
            f"p05 {item['p05']:.4f}  p95 {item['p95']:.4f}"
        )
    lines.append("")
    lines.append("Predicted pair lengths (metres). Band is heuristic only.")
    for name, item in report["validation"]["pair_lengths_m"].items():
        lines.append(
            f"  {name}: median {item['median']:.4f}  mean {item['mean']:.4f}  "
            f"std {item['std']:.4f}  min {item['min']:.4f}  max {item['max']:.4f}  "
            f"p05 {item['p05']:.4f}  p95 {item['p95']:.4f}  "
            f"outside_heuristic {item['heuristic_outside_count']}"
        )
    lines.append("")
    lines.append("Left/right median pair-length ratio. Descriptive only.")
    for name, item in report["validation"]["left_right_pair_asymmetry"].items():
        lines.append(
            f"  {name}: left {item['left_median_m']:.4f}  right {item['right_median_m']:.4f}  "
            f"left/right {item['left_over_right']:.3f}"
        )
    lines.append("")
    lines.append("Frame-to-frame steps (metres)")
    for name, item in report["validation"]["steps"].items():
        lines.append(
            f"  {name}: median {item['median_m']:.4f}  p95 {item['p95_m']:.4f}  "
            f"max {item['max_m']:.4f}  heuristic_jumps {item['heuristic_jump_count']}"
        )
    lines.append("")
    lines.append(report["validation"]["threshold_note"])
    reasons = report["validation"].get("review_reasons", [])
    if reasons:
        lines.append("Heuristic review flags (not anatomical limits):")
        lines.extend(f"  {reason}" for reason in reasons)
    return "\n".join(lines)


def run_inference(args: argparse.Namespace) -> None:
    height_m, weight_kg = _require_anthropometrics(args.height_m, args.weight_kg)
    frames, timing = load_recording(args.csv)
    mean = np.load(args.model_dir / "mean.npy")
    std = np.load(args.model_dir / "std.npy")
    features, measured_me, midhip_me = build_features(frames, height_m, weight_kg, mean, std)

    tf = _import_tensorflow()
    import keras

    model = load_arm_model(args.model_dir, tf)
    raw = model.predict(features, verbose=0)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape != (len(frames), 24):
        raise SystemExit(f"Unexpected prediction shape {raw.shape}.")
    if not np.all(np.isfinite(raw)):
        raise SystemExit("The model returned a non-finite value.")

    predicted_me = {}
    predicted_camera = {}
    for marker_index, name in enumerate(OUTPUT_MARKERS):
        local = raw[:, marker_index * 3 : marker_index * 3 + 3] * height_m + midhip_me
        predicted_me[name] = local
        predicted_camera[name] = _to_camera(local)

    validation = validate(predicted_camera, measured_me, predicted_me)
    # Heuristic only: a transverse pair used as a frame axis should not
    # change length by a large fraction of its own median during one take.
    # This is not an anatomical width limit.
    relative_range_limit = 0.40
    review_reasons = []
    for label, stats in validation["pair_lengths_m"].items():
        median = stats["median"]
        relative_range = (stats["max"] - stats["min"]) / median if median else math.nan
        stats["heuristic_relative_range"] = float(relative_range)
        if relative_range > relative_range_limit:
            review_reasons.append(
                f"{label} length range is {relative_range:.2f} of its median "
                f"(heuristic limit {relative_range_limit:.2f})"
            )
    validation["heuristic_relative_range_limit"] = relative_range_limit
    validation["review_reasons"] = review_reasons
    non_finite = sum(validation["non_finite_count"].values())
    extreme = sum(validation["heuristic_extreme_count"].values())
    jumps = sum(item["heuristic_jump_count"] for item in validation["steps"].values())
    if non_finite or extreme:
        classification = "INFERENCE RAN BUT OUTPUT VALIDATION FAILED"
    elif validation["heuristic_pair_flags"] or jumps or review_reasons:
        classification = "SUCCESSFUL INFERENCE + PLAUSIBILITY VALIDATION NEEDS REVIEW"
    else:
        classification = "SUCCESSFUL INFERENCE + PLAUSIBILITY VALIDATION PASSED"

    runtime = {
        "python": sys.version,
        "executable": sys.executable,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "prediction_shape": list(raw.shape),
    }
    report = {
        "classification": classification,
        "height_m": height_m,
        "weight_kg": weight_kg,
        "neck_definition": NECK_LABEL,
        "timing": timing,
        "transform_cam_to_me": R_CAM_TO_ME.tolist(),
        "output_markers": list(OUTPUT_MARKERS),
        "runtime": runtime,
        "validation": validation,
        "temporal_note": "Input temporal distribution differs from training distribution.",
        "biomechanical_note": (
            "These are predicted lateral and medial elbow and wrist markers. "
            "They are not true epicondyles or styloids, and no new degree of freedom was calculated."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out_dir / "inference_predictions.csv", frames, predicted_camera, predicted_me)
    (args.out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "validation_report.txt").write_text(format_text(report), encoding="utf-8")
    (args.out_dir / "runtime_info.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(format_text(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Marker Enhancer arm inference")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUT / "reference_models" / "lstm" / "arm")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--height-m", type=float, default=None)
    parser.add_argument("--weight-kg", type=float, default=None)
    parser.add_argument(
        "--runtime-check",
        action="store_true",
        help="Load the arm model and print shapes. Does not read the recording.",
    )
    args = parser.parse_args()
    if args.runtime_check:
        info = runtime_report(args.model_dir)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "runtime_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return
    run_inference(args)


if __name__ == "__main__":
    main()
