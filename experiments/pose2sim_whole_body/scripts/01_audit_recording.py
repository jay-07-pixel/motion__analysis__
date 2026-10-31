"""Audit one existing keypoints CSV. Does not edit the recording or open a camera.

Default input:
    data/output/20261005-150538/keypoints_2d.csv

Writes JSON + text under experiments/pose2sim_whole_body/output/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "output" / "20261005-150538" / "keypoints_2d.csv"
DEFAULT_OUT = EXPERIMENT_ROOT / "output"

# Official MediaPipe Pose order. Presence in a recording is not assumed.
MEDIAPIPE_POSE_33: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

# Same physical point family as HALPE_26 / OpenSim-style labels. Not ISB anatomy.
# left_foot_index is not LBigToe or LSmallToe.
HALPE_NAME_MAP: dict[str, str] = {
    "nose": "Nose",
    "left_eye": "LEye",
    "right_eye": "REye",
    "left_ear": "LEar",
    "right_ear": "REar",
    "left_shoulder": "LShoulder",
    "right_shoulder": "RShoulder",
    "left_elbow": "LElbow",
    "right_elbow": "RElbow",
    "left_wrist": "LWrist",
    "right_wrist": "RWrist",
    "left_hip": "LHip",
    "right_hip": "RHip",
    "left_knee": "LKnee",
    "right_knee": "RKnee",
    "left_ankle": "LAnkle",
    "right_ankle": "RAnkle",
    "left_heel": "LHeel",
    "right_heel": "RHeel",
}

NOT_HALPE_EQUIVALENT: dict[str, str] = {
    "left_foot_index": "MediaPipe foot_index is not HALPE LBigToe or LSmallToe",
    "right_foot_index": "MediaPipe foot_index is not HALPE RBigToe or RSmallToe",
    "left_eye_inner": "No HALPE_26 inner-eye landmark",
    "left_eye_outer": "No HALPE_26 outer-eye landmark",
    "right_eye_inner": "No HALPE_26 inner-eye landmark",
    "right_eye_outer": "No HALPE_26 outer-eye landmark",
    "mouth_left": "No HALPE_26 mouth landmark",
    "mouth_right": "No HALPE_26 mouth landmark",
}

BODY_REGIONS: dict[str, tuple[str, ...]] = {
    "head_face": (
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
    ),
    "torso": (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ),
    "arms": (
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
    ),
    "hands_extra": (),  # filled from detected names that are not Pose-33
    "legs": ("left_knee", "right_knee"),
    "feet": (
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ),
}

SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("shoulder_width", "left_shoulder", "right_shoulder"),
    ("hip_width", "left_hip", "right_hip"),
    ("left_upper_arm", "left_shoulder", "left_elbow"),
    ("right_upper_arm", "right_shoulder", "right_elbow"),
    ("left_forearm", "left_elbow", "left_wrist"),
    ("right_forearm", "right_elbow", "right_wrist"),
    ("left_thigh", "left_hip", "left_knee"),
    ("right_thigh", "right_hip", "right_knee"),
    ("left_shank", "left_knee", "left_ankle"),
    ("right_shank", "right_knee", "right_ankle"),
)


def _finite_xyz(row: dict[str, str]) -> tuple[float, float, float] | None:
    raw = (row.get("x_m", ""), row.get("y_m", ""), row.get("z_m", ""))
    if not all(part.strip() for part in raw):
        return None
    try:
        point = (float(raw[0]), float(raw[1]), float(raw[2]))
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in point):
        return None
    return point


def _parse_confidence(row: dict[str, str]) -> float | None:
    text = (row.get("confidence") or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _stdev(values: list[float]) -> float | None:
    return float(statistics.pstdev(values)) if len(values) >= 2 else None


def _detected_frame_rate(frame_times: dict[int, float]) -> dict[str, float | int | None]:
    frames = sorted(frame_times)
    if len(frames) < 2:
        return {
            "unique_frames": len(frames),
            "time_first_sec": frame_times[frames[0]] if frames else None,
            "time_last_sec": frame_times[frames[-1]] if frames else None,
            "span_sec": None,
            "median_dt_sec": None,
            "implied_fps": None,
            "note": "Need at least two frames to estimate rate from timestamps.",
        }
    times = [frame_times[frame] for frame in frames]
    dts = [times[i] - times[i - 1] for i in range(1, len(times)) if times[i] > times[i - 1]]
    median_dt = _median(dts)
    span = times[-1] - times[0]
    implied = (len(frames) - 1) / span if span > 0 else None
    return {
        "unique_frames": len(frames),
        "frame_min": frames[0],
        "frame_max": frames[-1],
        "time_first_sec": times[0],
        "time_last_sec": times[-1],
        "span_sec": span,
        "median_dt_sec": median_dt,
        "implied_fps_from_median_dt": (1.0 / median_dt) if median_dt else None,
        "implied_fps_from_span": implied,
        "note": (
            "Rate is from CSV time_sec, not from a live camera query. "
            "First time_sec may be > 0 because the clock starts before frame 0 is written."
        ),
    }


def audit(csv_path: Path) -> dict:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Recording not found: {csv_path}")

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    joints_by_frame: dict[int, dict[str, dict]] = defaultdict(dict)
    frame_times: dict[int, float] = {}
    coord_frames = Counter()
    sources = Counter()
    joint_counter = Counter()

    for row in rows:
        frame = int(row["frame"])
        frame_times[frame] = float(row["time_sec"])
        name = row["joint"]
        joint_counter[name] += 1
        coord_frames[row.get("coord_frame", "")] += 1
        sources[row.get("source", "")] += 1
        joints_by_frame[frame][name] = row

    detected_names = sorted(joint_counter)
    pose_present = [name for name in MEDIAPIPE_POSE_33 if name in joint_counter]
    pose_absent = [name for name in MEDIAPIPE_POSE_33 if name not in joint_counter]
    extra_names = [name for name in detected_names if name not in MEDIAPIPE_POSE_33]

    per_joint = {}
    n_frames = len(joints_by_frame)
    for name in detected_names:
        n_rows = 0
        n_xyz = 0
        n_invalid_depth = 0
        n_2d_only = 0
        confs: list[float] = []
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for frame, by_name in joints_by_frame.items():
            row = by_name.get(name)
            if row is None:
                continue
            n_rows += 1
            conf = _parse_confidence(row)
            if conf is not None:
                confs.append(conf)
            point = _finite_xyz(row)
            frame_label = row.get("coord_frame", "")
            if point is None:
                n_invalid_depth += 1
                if frame_label == "camera_2d_pixel":
                    n_2d_only += 1
                continue
            n_xyz += 1
            xs.append(point[0])
            ys.append(point[1])
            zs.append(point[2])
        per_joint[name] = {
            "rows": n_rows,
            "frames_absent": n_frames - n_rows,
            "valid_camera_xyz": n_xyz,
            "missing_or_invalid_xyz": n_frames - n_xyz,
            "coord_frame_camera_2d_pixel": n_2d_only,
            "coverage_xyz_fraction_of_frames": (n_xyz / n_frames) if n_frames else None,
            "confidence_min": min(confs) if confs else None,
            "confidence_median": _median(confs),
            "confidence_max": max(confs) if confs else None,
            "x_m_min": min(xs) if xs else None,
            "x_m_max": max(xs) if xs else None,
            "y_m_min": min(ys) if ys else None,
            "y_m_max": max(ys) if ys else None,
            "z_m_min": min(zs) if zs else None,
            "z_m_max": max(zs) if zs else None,
            "halpe26_name": HALPE_NAME_MAP.get(name),
            "anatomical_claim": False,
            "mapping_note": NOT_HALPE_EQUIVALENT.get(name)
            or (
                "Same point family as HALPE_26; still a MediaPipe model landmark, not a verified bony landmark."
                if name in HALPE_NAME_MAP
                else "No HALPE_26 equivalent; keep the MediaPipe name."
            ),
        }

    region_report = {}
    for region, names in BODY_REGIONS.items():
        if region == "hands_extra":
            names = tuple(extra_names)
        present = [name for name in names if name in joint_counter]
        absent = [name for name in names if name not in joint_counter]
        xyz_rows = sum(per_joint[name]["valid_camera_xyz"] for name in present)
        total_possible = n_frames * len(names) if names else 0
        region_report[region] = {
            "expected_names": list(names),
            "present_in_csv": present,
            "absent_from_csv": absent,
            "valid_xyz_rows": xyz_rows,
            "valid_xyz_fraction_if_expected": (xyz_rows / total_possible) if total_possible else None,
        }

    segment_report = []
    for label, start_name, end_name in SEGMENTS:
        lengths: list[float] = []
        used = 0
        skipped_missing_joint = 0
        skipped_invalid_xyz = 0
        for by_name in joints_by_frame.values():
            start_row = by_name.get(start_name)
            end_row = by_name.get(end_name)
            if start_row is None or end_row is None:
                skipped_missing_joint += 1
                continue
            start = _finite_xyz(start_row)
            end = _finite_xyz(end_row)
            if start is None or end is None:
                skipped_invalid_xyz += 1
                continue
            length = math.dist(start, end)
            lengths.append(length)
            used += 1
        segment_report.append(
            {
                "segment": label,
                "start": start_name,
                "end": end_name,
                "frames_with_both_xyz": used,
                "frames_missing_joint_row": skipped_missing_joint,
                "frames_invalid_xyz": skipped_invalid_xyz,
                "length_median_m": _median(lengths),
                "length_min_m": min(lengths) if lengths else None,
                "length_max_m": max(lengths) if lengths else None,
                "length_stdev_m": _stdev(lengths),
            }
        )

    rows_per_frame = [len(by_name) for by_name in joints_by_frame.values()]
    return {
        "csv_path": str(csv_path.resolve()),
        "csv_modified": False,
        "n_rows": len(rows),
        "columns": columns,
        "source_values": dict(sources),
        "coord_frame_counts": dict(coord_frames),
        "coordinate_system": {
            "pixels": "u_px, v_px; origin top-left of the RGB image; coord_frame camera_2d_pixel",
            "metres": (
                "x_m, y_m, z_m from RealSense deprojection when coord_frame is "
                "camera_3d_metre. Origin colour optical centre. X right, Y down, Z forward. "
                "Not MediaPipe world landmarks."
            ),
            "invalid_depth": (
                "Empty x_m/y_m/z_m and/or coord_frame camera_2d_pixel. "
                "Do not fill these with interpolation in the TRC export."
            ),
        },
        "detected_frame_rate": _detected_frame_rate(frame_times),
        "rows_per_frame_min": min(rows_per_frame) if rows_per_frame else 0,
        "rows_per_frame_max": max(rows_per_frame) if rows_per_frame else 0,
        "rows_per_frame_median": _median([float(n) for n in rows_per_frame]),
        "detected_landmark_names": detected_names,
        "mediapipe_pose_33_present": pose_present,
        "mediapipe_pose_33_absent": pose_absent,
        "extra_landmarks_not_in_pose_33": extra_names,
        "whole_body_note": (
            "This file is not a whole-body landmark set. Knees, ankles, heels and "
            "foot_index are absent. That matches an upper-body-plus-hands CSV filter, "
            "not proof that MediaPipe failed to see the legs."
        ),
        "regions": region_report,
        "per_joint": per_joint,
        "segments_camera_metres": segment_report,
        "halpe26_unmapped_in_this_file": [
            name for name in detected_names if name not in HALPE_NAME_MAP
        ],
        "augmentation": "Not applied. Stanford 47-marker augmenter stays off.",
    }


def _write_text(report: dict, path: Path) -> None:
    rate = report["detected_frame_rate"]
    lines = [
        f"CSV: {report['csv_path']}",
        f"Rows: {report['n_rows']}",
        f"Columns: {', '.join(report['columns'])}",
        (
            f"Frames: {rate.get('unique_frames')} "
            f"(index {rate.get('frame_min')} .. {rate.get('frame_max')})"
        ),
        (
            f"Time_sec: {rate.get('time_first_sec')} .. {rate.get('time_last_sec')} "
            f"(span {rate.get('span_sec')} s)"
        ),
        f"Detected frame rate (median dt): {rate.get('implied_fps_from_median_dt')} fps",
        f"Detected frame rate (span): {rate.get('implied_fps_from_span')} fps",
        f"Sources: {report['source_values']}",
        f"coord_frame counts: {report['coord_frame_counts']}",
        "",
        report["whole_body_note"],
        "",
        "MediaPipe Pose 33 present:",
        "  " + ", ".join(report["mediapipe_pose_33_present"] or ["(none)"]),
        "MediaPipe Pose 33 absent:",
        "  " + ", ".join(report["mediapipe_pose_33_absent"] or ["(none)"]),
        "Extra landmarks (Hands etc.):",
        "  " + ", ".join(report["extra_landmarks_not_in_pose_33"] or ["(none)"]),
        "",
        "Regions:",
    ]
    for region, block in report["regions"].items():
        lines.append(
            f"  {region}: present={block['present_in_csv'] or '[]'} "
            f"absent={block['absent_from_csv'] or '[]'}"
        )
    lines.append("")
    lines.append("Per-joint valid camera XYZ / frames absent / conf median:")
    for name, block in report["per_joint"].items():
        lines.append(
            f"  {name}: xyz={block['valid_camera_xyz']} "
            f"absent_rows={block['frames_absent']} "
            f"invalid_xyz={block['missing_or_invalid_xyz']} "
            f"conf_median={block['confidence_median']} "
            f"halpe={block['halpe26_name'] or 'none'}"
        )
    lines.append("")
    lines.append("Segment lengths in camera metres (only when both ends have XYZ):")
    for block in report["segments_camera_metres"]:
        lines.append(
            f"  {block['segment']}: n={block['frames_with_both_xyz']} "
            f"median_m={block['length_median_m']} "
            f"min_m={block['length_min_m']} max_m={block['length_max_m']} "
            f"stdev_m={block['length_stdev_m']} "
            f"missing_joint={block['frames_missing_joint_row']} "
            f"invalid_xyz={block['frames_invalid_xyz']}"
        )
    lines.append("")
    lines.append(report["augmentation"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Existing keypoints CSV")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="New experiment output folder")
    args = parser.parse_args(argv)

    csv_path = args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv
    report = audit(csv_path)
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "01_audit_recording.json"
    txt_path = out_dir / "01_audit_recording.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_text(report, txt_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Frames: {report['detected_frame_rate'].get('unique_frames')}")
    print(f"Landmarks: {len(report['detected_landmark_names'])}")
    print(report["whole_body_note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
