"""Coordinate-contract check for Marker Enhancer. No model is loaded.

The Marker Enhancer repository subtracts midHip, divides by stature, and
later multiplies by stature and adds midHip back. Its published test file
is metres with Y up. It does not name the two horizontal axes.

This script only checks whether our measured camera points can be written
in a right-handed Y-up frame and recovered. It does not run the network.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.dof_angles_3d import compute_initial_dofs
from src.pose.keypoints import Keypoint2D
from src.utils.config_loader import load_config

# Explicit stand-in. No stature is stored on the recordings.
SYNTHETIC_HEIGHT_M = 1.70

# Previous validations used this cutoff. Rows below it are not used.
MIN_CONFIDENCE = 0.5

# Camera axis order is X, Y, Z. These words describe our camera, not theirs.
_CAMERA_AXIS = ("right", "down", "forward")

# MediaPipe name -> OpenPose-style name. Only joints that are the same point.
LANDMARK_MAP: dict[str, str] = {
    "right_shoulder": "RShoulder",
    "left_shoulder": "LShoulder",
    "right_elbow": "RElbow",
    "left_elbow": "LElbow",
    "right_wrist": "RWrist",
    "left_wrist": "LWrist",
    "right_hip": "RHip",
    "left_hip": "LHip",
    "right_knee": "RKnee",
    "left_knee": "LKnee",
    "right_ankle": "RAnkle",
    "left_ankle": "LAnkle",
    "right_heel": "RHeel",
    "left_heel": "LHeel",
}

# Present in some of our files, but not the OpenPose toe landmarks.
UNMAPPED_BECAUSE_DIFFERENT = ("right_foot_index", "left_foot_index", "nose")

# Arm-model inputs. midHip is the reference point, not one of these channels.
ARM_MODEL_INPUTS = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RElbow",
    "LElbow",
    "RWrist",
    "LWrist",
)

# Body-model inputs. midHip is again only the reference.
BODY_MODEL_INPUTS = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RHip",
    "LHip",
    "RKnee",
    "LKnee",
    "RAnkle",
    "LAnkle",
    "RHeel",
    "LHeel",
    "RSmallToe",
    "LSmallToe",
    "RBigToe",
    "LBigToe",
)

SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("shoulder_width", "left_shoulder", "right_shoulder"),
    ("hip_width", "left_hip", "right_hip"),
    ("right_upper_arm", "right_shoulder", "right_elbow"),
    ("left_upper_arm", "left_shoulder", "left_elbow"),
    ("right_forearm", "right_elbow", "right_wrist"),
    ("left_forearm", "left_elbow", "left_wrist"),
    ("right_thigh", "right_hip", "right_knee"),
    ("left_thigh", "left_hip", "left_knee"),
    ("right_shank", "right_knee", "right_ankle"),
    ("left_shank", "left_knee", "left_ankle"),
)

# Production angles that do not need the missing transverse markers.
ANGLE_NAMES = (
    "right_shoulder_elevation",
    "left_shoulder_elevation",
    "right_elbow_flexion",
    "left_elbow_flexion",
)

PREFERRED_RECORDING = (
    PROJECT_ROOT / "data" / "output" / "20261005-150538" / "keypoints_2d.csv"
)


@dataclass(frozen=True)
class AxisMap:
    """One right-handed map from camera metres into a Y-up frame."""

    name: str
    matrix: np.ndarray
    meaning: str


def enumerate_y_up_maps() -> tuple[AxisMap, ...]:
    """Every axis-aligned right-handed map that sends camera-down to -Y.

    Marker Enhancer's test positions have Y up. Our camera Y is down, so
    their Y has to be the opposite of our Y. The two horizontal camera
    axes can then be assigned to their X and Z in four ways that keep a
    right-handed triad. No other axis-aligned right-handed choice does.
    """
    maps: list[AxisMap] = []
    # Camera X is index 0, camera Z is index 2. Camera Y is fixed.
    assignments = ((0, 2), (2, 0))
    signs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    for x_source, z_source in assignments:
        for x_sign, z_sign in signs:
            matrix = np.zeros((3, 3), dtype=float)
            matrix[1, 1] = -1.0
            matrix[0, x_source] = x_sign
            matrix[2, z_source] = z_sign
            if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-9:
                continue
            maps.append(
                AxisMap(
                    name=f"map_{len(maps) + 1}",
                    matrix=matrix,
                    meaning=_describe(matrix),
                )
            )
    return tuple(maps)


def _describe(matrix: np.ndarray) -> str:
    words = []
    labels = ("X", "Y", "Z")
    for row, label in enumerate(labels):
        column = int(np.flatnonzero(np.abs(matrix[row]) > 0.5)[0])
        sign = float(matrix[row, column])
        direction = _CAMERA_AXIS[column]
        if sign < 0.0:
            direction = _OPPOSITE[direction]
        words.append(f"{label} = {direction}")
    return ", ".join(words)


_OPPOSITE = {
    "right": "left",
    "left": "right",
    "down": "up",
    "up": "down",
    "forward": "backward",
    "backward": "forward",
}


def apply_map(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map (N, 3) camera points. Rows of ``matrix`` are the new axes."""
    return points @ matrix.T


def invert_map(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Inverse of a rotation: multiply by the transpose."""
    return points @ matrix


def round_trip_errors(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Distance, in metres, after map then inverse map."""
    recovered = invert_map(apply_map(points, matrix), matrix)
    return np.linalg.norm(points - recovered, axis=1)


def error_summary(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {"max": math.nan, "mean": math.nan, "median": math.nan, "p95": math.nan}
    return {
        "max": float(np.max(errors)),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
    }


def segment_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def height_round_trip(
    points: np.ndarray,
    mid_hip: np.ndarray,
    height_m: float,
) -> np.ndarray:
    """midHip off, divide by height, multiply by height, midHip back."""
    if height_m == 0.0:
        raise ValueError("height must be non-zero")
    shifted = points - mid_hip
    restored = (shifted / height_m) * height_m + mid_hip
    return np.linalg.norm(points - restored, axis=1)


def load_recording(csv_path: Path) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, float]]:
    """Read camera-metre rows at or above the confidence cutoff.

    Returns:
        Points by frame and joint, and one timestamp per frame.
    """
    frames: dict[int, dict[str, np.ndarray]] = {}
    times: dict[int, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["coord_frame"] != "camera_3d_metre":
                continue
            if not row["x_m"] or not row["y_m"] or not row["z_m"]:
                continue
            confidence = float(row["confidence"])
            if confidence < MIN_CONFIDENCE:
                continue
            point = np.array(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                dtype=float,
            )
            if not np.all(np.isfinite(point)):
                continue
            frame = int(row["frame"])
            frames.setdefault(frame, {})[row["joint"]] = point
            times[frame] = float(row["time_sec"])
    return frames, times


def _keypoints_from_frame(joints: dict[str, np.ndarray]) -> list[Keypoint2D]:
    keypoints: list[Keypoint2D] = []
    for name, point in joints.items():
        keypoints.append(
            Keypoint2D(
                name=name,
                u_px=0.0,
                v_px=0.0,
                confidence=1.0,
                coord_frame="camera_3d_metre",
                x_m=float(point[0]),
                y_m=float(point[1]),
                z_m=float(point[2]),
            )
        )
    return keypoints


def _dof_settings() -> tuple[float, float]:
    dof = load_config()["analysis"]["dof"]
    return float(dof["plane_singular_deg"]), float(dof["elbow_sign_deadband_deg"])


def _production_angles(
    joints: dict[str, np.ndarray],
    plane_singular_deg: float,
    elbow_sign_deadband_deg: float,
) -> dict[str, float | None]:
    angles = compute_initial_dofs(
        _keypoints_from_frame(joints),
        MIN_CONFIDENCE,
        plane_singular_deg,
        elbow_sign_deadband_deg,
    )
    out: dict[str, float | None] = {}
    for name in ANGLE_NAMES:
        value = angles[name]
        out[name] = None if value.degrees is None else float(value.degrees)
    return out


def _timing(times: dict[int, float]) -> dict[str, float]:
    ordered = [times[frame] for frame in sorted(times)]
    if len(ordered) < 2:
        return {"frames": float(len(ordered)), "duration": 0.0, "median_dt": math.nan,
                "median_fps": math.nan, "min_dt": math.nan, "max_dt": math.nan}
    deltas = np.diff(np.asarray(ordered, dtype=float))
    median_dt = float(np.median(deltas))
    return {
        "frames": float(len(ordered)),
        "duration": float(ordered[-1] - ordered[0]),
        "median_dt": median_dt,
        "median_fps": (1.0 / median_dt) if median_dt > 0.0 else math.nan,
        "min_dt": float(np.min(deltas)),
        "max_dt": float(np.max(deltas)),
    }


def _stack_points(frames: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    rows = [point for joints in frames.values() for point in joints.values()]
    if not rows:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(rows)


def evaluate(csv_path: Path, height_m: float = SYNTHETIC_HEIGHT_M) -> dict:
    """Run the contract checks. Does not write the recording or call a model."""
    frames, times = load_recording(csv_path)
    maps = enumerate_y_up_maps()
    points = _stack_points(frames)
    present = {name for joints in frames.values() for name in joints}

    round_trips = []
    for axis_map in maps:
        summary = error_summary(round_trip_errors(points, axis_map.matrix))
        round_trips.append({"name": axis_map.name, "meaning": axis_map.meaning, **summary})

    lengths = []
    for label, start_name, end_name in SEGMENTS:
        original_lengths = []
        transformed_lengths = []
        abs_errors = []
        for joints in frames.values():
            if start_name not in joints or end_name not in joints:
                continue
            start = joints[start_name]
            end = joints[end_name]
            original_length = segment_length(start, end)
            original_lengths.append(original_length)
            for axis_map in maps:
                mapped = apply_map(np.vstack((start, end)), axis_map.matrix)
                transformed_length = segment_length(mapped[0], mapped[1])
                transformed_lengths.append(transformed_length)
                abs_errors.append(abs(transformed_length - original_length))
        if not original_lengths:
            lengths.append(
                {
                    "metric": label,
                    "frames": 0,
                    "original_m": math.nan,
                    "transformed_m": math.nan,
                    "abs_error_m": math.nan,
                    "rel_error_percent": math.nan,
                }
            )
            continue
        original = np.asarray(original_lengths)
        transformed = np.asarray(transformed_lengths)
        abs_error = np.asarray(abs_errors)
        # Repeat each original length once per candidate map.
        original_repeated = np.repeat(original, len(maps))
        relative = np.divide(
            abs_error,
            original_repeated,
            out=np.zeros_like(abs_error),
            where=original_repeated > 0.0,
        )
        lengths.append(
            {
                "metric": label,
                "frames": int(original.size),
                "original_m": float(np.median(original)),
                "transformed_m": float(np.median(transformed)),
                "abs_error_m": float(np.max(abs_error)),
                "rel_error_percent": float(np.max(relative) * 100.0),
            }
        )

    angle_rows = _angle_rows_with_recovered_median(frames, maps[0].matrix)

    height_errors = []
    for joints in frames.values():
        if "left_hip" not in joints or "right_hip" not in joints:
            continue
        mid = (joints["left_hip"] + joints["right_hip"]) / 2.0
        stacked = np.vstack(list(joints.values()))
        height_errors.append(height_round_trip(stacked, mid, height_m))
    height_summary = error_summary(
        np.concatenate(height_errors) if height_errors else np.zeros((0,))
    )

    mapped_names = {LANDMARK_MAP[name] for name in present if name in LANDMARK_MAP}
    missing_arm = [name for name in ARM_MODEL_INPUTS if name not in mapped_names]
    missing_body = [name for name in BODY_MODEL_INPUTS if name not in mapped_names]

    max_round = max(item["max"] for item in round_trips) if round_trips else math.inf
    max_length = max(
        (item["abs_error_m"] for item in lengths if item["frames"] > 0),
        default=math.inf,
    )
    max_angle = max(
        (item["abs_diff_deg"] for item in angle_rows if item["frames"] > 0),
        default=math.inf,
    )
    math_ok = (
        max_round < 1e-9
        and max_length < 1e-9
        and max_angle < 1e-6
        and height_summary["max"] < 1e-9
    )
    # A zero round trip only shows that each map is invertible.
    classification = (
        "B. COORDINATE CONTRACT MATHEMATICALLY VALID BUT AXIS ORIENTATION UNKNOWN"
        if math_ok
        else "C. COORDINATE CONTRACT FAILED"
    )

    return {
        "recording": str(csv_path),
        "classification": classification,
        "maps": [{"name": item.name, "meaning": item.meaning} for item in maps],
        "round_trips": round_trips,
        "lengths": lengths,
        "angles": angle_rows,
        "height": height_summary,
        "height_m": height_m,
        "height_is_synthetic": True,
        "timing": _timing(times),
        "present_mediapipe": sorted(present),
        "mapped": sorted(mapped_names),
        "missing_arm": missing_arm,
        "missing_body": missing_body,
        "frames_with_midhip": sum(
            1
            for joints in frames.values()
            if "left_hip" in joints and "right_hip" in joints
        ),
        "frame_count": len(frames),
    }


def _angle_rows_with_recovered_median(
    frames: dict[int, dict[str, np.ndarray]],
    matrix: np.ndarray,
) -> list[dict]:
    plane_singular_deg, elbow_deadband_deg = _dof_settings()
    original_values: dict[str, list[float]] = {name: [] for name in ANGLE_NAMES}
    recovered_values: dict[str, list[float]] = {name: [] for name in ANGLE_NAMES}
    for joints in frames.values():
        before = _production_angles(joints, plane_singular_deg, elbow_deadband_deg)
        recovered_joints = {
            name: invert_map(apply_map(point.reshape(1, 3), matrix), matrix)[0]
            for name, point in joints.items()
        }
        after = _production_angles(recovered_joints, plane_singular_deg, elbow_deadband_deg)
        for name in ANGLE_NAMES:
            if before[name] is None or after[name] is None:
                continue
            original_values[name].append(before[name])
            recovered_values[name].append(after[name])
    rows = []
    for name in ANGLE_NAMES:
        original = np.asarray(original_values[name], dtype=float)
        recovered = np.asarray(recovered_values[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "frames": int(original.size),
                "original_deg": float(np.median(original)) if original.size else math.nan,
                "round_trip_deg": float(np.median(recovered)) if recovered.size else math.nan,
                "abs_diff_deg": (
                    float(np.max(np.abs(recovered - original))) if original.size else math.nan
                ),
            }
        )
    return rows


def format_report(result: dict) -> str:
    lines: list[str] = []
    lines.append("A. Recording")
    lines.append(result["recording"])
    lines.append(
        f"usable 3D frames: {result['frame_count']}; "
        f"frames with both hips: {result['frames_with_midhip']}"
    )
    lines.append("")
    lines.append("B. Landmark mapping")
    for source, target in LANDMARK_MAP.items():
        lines.append(f"  {source} -> {target}")
    lines.append("  midHip is computed as (left_hip + right_hip) / 2")
    lines.append("  nose is not Neck")
    lines.append("  foot_index is not RBigToe, LBigToe, RSmallToe, or LSmallToe")
    lines.append("")
    lines.append("C. Candidate coordinate mappings")
    lines.append("  Each map is right-handed and uses Y = up. None is selected.")
    for item in result["maps"]:
        lines.append(f"  {item['name']}: {item['meaning']}")
    lines.append("")
    lines.append("D. Round-trip position error (metres)")
    lines.append("  A near-zero error shows the map is invertible. It does not show it is the training frame.")
    for item in result["round_trips"]:
        lines.append(
            f"  {item['name']}: max {item['max']:.3e}  mean {item['mean']:.3e}  "
            f"median {item['median']:.3e}  p95 {item['p95']:.3e}"
        )
    lines.append("")
    lines.append("E. Segment lengths (median metres; max absolute error)")
    for item in result["lengths"]:
        if item["frames"] == 0:
            lines.append(f"  {item['metric']}: no usable pair in this recording")
            continue
        lines.append(
            f"  {item['metric']}: original {item['original_m']:.6f}  "
            f"transformed {item['transformed_m']:.6f}  "
            f"abs {item['abs_error_m']:.3e} m  rel {item['rel_error_percent']:.3e} %"
        )
    lines.append("")
    lines.append("F. Existing angle preservation (degrees, after the inverse map)")
    for item in result["angles"]:
        if item["frames"] == 0:
            lines.append(f"  {item['metric']}: unavailable")
            continue
        lines.append(
            f"  {item['metric']}: original {item['original_deg']:.6f}  "
            f"round-trip {item['round_trip_deg']:.6f}  "
            f"abs diff {item['abs_diff_deg']:.3e}  frames {item['frames']}"
        )
    lines.append("")
    lines.append("G. Height normalization round-trip")
    lines.append(
        f"  synthetic height {result['height_m']:.2f} m "
        "(not a measured subject height)"
    )
    height = result["height"]
    lines.append(
        f"  max {height['max']:.3e} m  mean {height['mean']:.3e} m  "
        f"median {height['median']:.3e} m"
    )
    lines.append("")
    lines.append(
        "Weight is required by Marker Enhancer but is irrelevant to this "
        "geometric round-trip experiment because the model itself is not being executed."
    )
    lines.append("")
    lines.append("H. Recording timing versus 60 Hz training")
    timing = result["timing"]
    lines.append(
        f"  frames {int(timing['frames'])}  duration {timing['duration']:.3f} s  "
        f"median dt {timing['median_dt']:.4f} s  median fps {timing['median_fps']:.2f}  "
        f"min dt {timing['min_dt']:.4f} s  max dt {timing['max_dt']:.4f} s"
    )
    lines.append("  Marker Enhancer training default is 60 Hz. This file was not resampled.")
    lines.append("")
    lines.append("I. Missing Marker Enhancer inputs")
    lines.append("  arm model: " + ", ".join(result["missing_arm"]))
    lines.append("  body model: " + ", ".join(result["missing_body"]))
    lines.append("")
    lines.append("J. Still unknown")
    lines.append("  Which of the four horizontal conventions the training TRC uses.")
    lines.append("  Subject stature and mass.")
    lines.append("  A neck landmark, and OpenPose big-toe and small-toe landmarks.")
    lines.append("")
    lines.append(result["classification"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Marker Enhancer coordinate-contract check")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=PREFERRED_RECORDING,
        help="keypoints_2d.csv from an existing 3D recording",
    )
    args = parser.parse_args()
    result = evaluate(args.csv)
    print(format_report(result))


if __name__ == "__main__":
    main()
