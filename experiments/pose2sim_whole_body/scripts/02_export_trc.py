"""Export an existing keypoints CSV to OpenSim-style TRC. Does not edit the CSV.

Two files are written:

  *_camera_3d_metre.trc
      x,y,z as recorded (RealSense colour optical origin: X right, Y down, Z forward).

  *_opensim_yup.trc
      X_osim = X_cam, Y_osim = -Y_cam, Z_osim = Z_cam (metres, Y up).
      This is a documented axis flip only. It is not OpenSim inverse kinematics.

Missing or invalid depth stays empty. No interpolation. No marker augmentation.
Marker names stay MediaPipe names; they are model landmarks, not verified anatomy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "output" / "20261005-150538" / "keypoints_2d.csv"
DEFAULT_OUT = EXPERIMENT_ROOT / "output"

# Import HALPE notes from the audit script without adding a third public tool.
# Same Pose-33 order as 01_audit_recording.py. Keep names in that order in the TRC.
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
}


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


def camera_to_opensim_yup(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Axis-aligned map: keep X and Z, flip camera Y (down) to OpenSim Y (up)."""
    x_cam, y_cam, z_cam = point
    return (x_cam, -y_cam, z_cam)


def _load(csv_path: Path) -> tuple[list[int], dict[int, float], dict[int, dict[str, tuple[float, float, float] | None]], list[str]]:
    joints_by_frame: dict[int, dict[str, tuple[float, float, float] | None]] = defaultdict(dict)
    frame_times: dict[int, float] = {}
    names: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            frame_times[frame] = float(row["time_sec"])
            name = row["joint"]
            names.add(name)
            joints_by_frame[frame][name] = _finite_xyz(row)
    frames = sorted(joints_by_frame)
    ordered_names = _stable_marker_order(names)
    return frames, frame_times, joints_by_frame, ordered_names


def _stable_marker_order(names: set[str]) -> list[str]:
    pose = [name for name in MEDIAPIPE_POSE_33 if name in names]
    extra = sorted(name for name in names if name not in MEDIAPIPE_POSE_33)
    return pose + extra


def _data_rate(frames: list[int], frame_times: dict[int, float]) -> float:
    if len(frames) < 2:
        return 1.0
    dts = [
        frame_times[frames[i]] - frame_times[frames[i - 1]]
        for i in range(1, len(frames))
        if frame_times[frames[i]] > frame_times[frames[i - 1]]
    ]
    if not dts:
        return 1.0
    median_dt = float(statistics.median(dts))
    return 1.0 / median_dt if median_dt > 0 else 1.0


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def write_trc(
    path: Path,
    frames: list[int],
    frame_times: dict[int, float],
    joints_by_frame: dict[int, dict[str, tuple[float, float, float] | None]],
    marker_names: list[str],
    transform,
    units: str,
) -> dict[str, int]:
    rate = _data_rate(frames, frame_times)
    n_frames = len(frames)
    n_markers = len(marker_names)
    orig_start = frames[0] if frames else 0
    header2 = (
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
        "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    )
    header3 = (
        f"{rate:.6f}\t{rate:.6f}\t{n_frames}\t{n_markers}\t{units}\t"
        f"{rate:.6f}\t{orig_start}\t{n_frames}"
    )
    name_cells = ["Frame#", "Time"]
    for name in marker_names:
        name_cells.extend([name, "", ""])
    xyz_cells = ["", ""]
    for index in range(1, n_markers + 1):
        xyz_cells.extend([f"X{index}", f"Y{index}", f"Z{index}"])

    n_written_xyz = 0
    n_empty_xyz = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
        handle.write(header2 + "\n")
        handle.write(header3 + "\n")
        handle.write("\t".join(name_cells) + "\n")
        handle.write("\t".join(xyz_cells) + "\n")
        for seq, frame in enumerate(frames, start=1):
            cells = [str(seq), f"{frame_times[frame]:.6f}"]
            by_name = joints_by_frame[frame]
            for name in marker_names:
                point = by_name.get(name)
                if point is None:
                    cells.extend(["", "", ""])
                    n_empty_xyz += 1
                    continue
                x, y, z = transform(point)
                cells.extend([_fmt(x), _fmt(y), _fmt(z)])
                n_written_xyz += 1
            handle.write("\t".join(cells) + "\n")
    return {
        "n_frames": n_frames,
        "n_markers": n_markers,
        "n_written_xyz": n_written_xyz,
        "n_empty_xyz": n_empty_xyz,
    }


def _identity(point: tuple[float, float, float]) -> tuple[float, float, float]:
    return point


def export(csv_path: Path, out_dir: Path) -> dict:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Recording not found: {csv_path}")
    frames, frame_times, joints_by_frame, marker_names = _load(csv_path)
    stem = csv_path.parent.name
    camera_path = out_dir / f"{stem}_camera_3d_metre.trc"
    opensim_path = out_dir / f"{stem}_opensim_yup.trc"
    camera_stats = write_trc(
        camera_path,
        frames,
        frame_times,
        joints_by_frame,
        marker_names,
        _identity,
        units="m",
    )
    opensim_stats = write_trc(
        opensim_path,
        frames,
        frame_times,
        joints_by_frame,
        marker_names,
        camera_to_opensim_yup,
        units="m",
    )
    meta = {
        "source_csv": str(csv_path.resolve()),
        "source_csv_modified": False,
        "marker_names": marker_names,
        "marker_naming_rule": (
            "MediaPipe CSV joint names. Not ISB anatomical landmark names. "
            "HALPE_26 aliases are listed only as a correspondence table."
        ),
        "halpe26_correspondence": {
            name: HALPE_NAME_MAP[name]
            for name in marker_names
            if name in HALPE_NAME_MAP
        },
        "not_halpe_equivalent": {
            name: NOT_HALPE_EQUIVALENT[name]
            for name in marker_names
            if name in NOT_HALPE_EQUIVALENT
        },
        "camera_trc": {"path": str(camera_path.resolve()), **camera_stats},
        "opensim_yup_trc": {"path": str(opensim_path.resolve()), **opensim_stats},
        "camera_axes": {
            "origin": "colour-camera optical centre",
            "X": "right",
            "Y": "down",
            "Z": "forward",
            "units": "metres",
        },
        "opensim_yup_transform": {
            "X_osim": "X_cam",
            "Y_osim": "-Y_cam",
            "Z_osim": "Z_cam",
            "units": "metres",
            "determinant": 1.0,
            "note": "Right-handed. Does not run OpenSim IK or change millimetres.",
        },
        "missing_depth": "Empty TRC cells. No interpolation. No zeros.",
        "augmentation": False,
        "triangulation": False,
        "inverse_kinematics": False,
    }
    meta_path = out_dir / f"{stem}_trc_export.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["meta_path"] = str(meta_path.resolve())
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    csv_path = args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    meta = export(csv_path, out_dir)
    print(f"Wrote {meta['camera_trc']['path']}")
    print(f"Wrote {meta['opensim_yup_trc']['path']}")
    print(f"Wrote {meta['meta_path']}")
    print(
        f"Markers={meta['camera_trc']['n_markers']} "
        f"frames={meta['camera_trc']['n_frames']} "
        f"empty_xyz_cells={meta['camera_trc']['n_empty_xyz']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
