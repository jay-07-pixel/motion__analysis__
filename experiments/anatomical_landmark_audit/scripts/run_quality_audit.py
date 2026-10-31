"""Observed MediaPipe + RealSense 3D quality audit. Does not edit the recording.

Reads data/output/20261011-144812/keypoints_2d.csv and writes only under
experiments/anatomical_landmark_audit/output/20261011-144812_quality/.

No interpolation. No augmenter. Landmark names stay MediaPipe model names.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = PROJECT_ROOT / "data" / "output" / "20261011-144812" / "keypoints_2d.csv"
OUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "anatomical_landmark_audit"
    / "output"
    / "20261011-144812_quality"
)

POSE_33 = (
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
POSE_SET = set(POSE_33)

# Hands-only names (not Pose-33). Wrist/tips can be Pose or Hands overwrite.
HAND_EXTRA = (
    "left_thumb_cmc",
    "left_thumb_mcp",
    "left_thumb_ip",
    "left_index_mcp",
    "left_index_pip",
    "left_index_dip",
    "left_middle_mcp",
    "left_middle_pip",
    "left_middle_dip",
    "left_middle_tip",
    "left_ring_mcp",
    "left_ring_pip",
    "left_ring_dip",
    "left_ring_tip",
    "left_pinky_mcp",
    "left_pinky_pip",
    "left_pinky_dip",
    "right_thumb_cmc",
    "right_thumb_mcp",
    "right_thumb_ip",
    "right_index_mcp",
    "right_index_pip",
    "right_index_dip",
    "right_middle_mcp",
    "right_middle_pip",
    "right_middle_dip",
    "right_middle_tip",
    "right_ring_mcp",
    "right_ring_pip",
    "right_ring_dip",
    "right_ring_tip",
    "right_pinky_mcp",
    "right_pinky_pip",
    "right_pinky_dip",
)

SEGMENTS = (
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

CORE_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Length vs robust session median. Documented in quiet_window.json.
LENGTH_LOW = 0.50
LENGTH_HIGH = 1.50
# Consecutive-frame jump of mid-hip (metres). 7 fps × 0.05 m/frame ≈ 0.35 m/s.
HIP_JUMP_M = 0.050
# Any core joint step above this is a jump (depth spike or tracking hop).
JOINT_JUMP_M = 0.080
MIN_QUIET_SEC = 4.0
HIP_STEP_QUIET_M = 0.020
MIN_CORE_SEGMENTS_OK = 6  # of 10 listed segments with valid length and not implausible


def _xyz(row: dict[str, str]) -> tuple[float, float, float] | None:
    raw = (row.get("x_m", ""), row.get("y_m", ""), row.get("z_m", ""))
    if not all(part.strip() for part in raw):
        return None
    try:
        point = (float(raw[0]), float(raw[1]), float(raw[2]))
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in point):
        return None
    return point


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a, b)


def _mid(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def load_csv(path: Path):
    by_frame: dict[int, dict[str, dict]] = defaultdict(dict)
    times: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            times[frame] = float(row["time_sec"])
            by_frame[frame][row["joint"]] = row
    frames = sorted(by_frame)
    return frames, times, by_frame


def segment_length(by_name: dict, start: str, end: str) -> tuple[float | None, str]:
    """Return length and why it is missing: missing_row, invalid_depth, or ok."""
    a = by_name.get(start)
    b = by_name.get(end)
    if a is None or b is None:
        return None, "missing_row"
    pa, pb = _xyz(a), _xyz(b)
    if pa is None or pb is None:
        return None, "invalid_depth"
    return _dist(pa, pb), "ok"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarise(values: list[float], n_missing: int) -> dict:
    if not values:
        return {
            "valid_frame_count": 0,
            "missing_frame_count": n_missing,
            "median_m": None,
            "stdev_m": None,
            "p05_m": None,
            "p95_m": None,
            "min_m": None,
            "max_m": None,
            "stdev_over_median": None,
        }
    med = float(statistics.median(values))
    stdev = float(statistics.pstdev(values)) if len(values) >= 2 else 0.0
    return {
        "valid_frame_count": len(values),
        "missing_frame_count": n_missing,
        "median_m": med,
        "stdev_m": stdev,
        "p05_m": percentile(values, 5),
        "p95_m": percentile(values, 95),
        "min_m": min(values),
        "max_m": max(values),
        "stdev_over_median": (stdev / med) if med > 1e-9 else None,
    }


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)
    frames, times, by_frame = load_csv(CSV_PATH)
    n = len(frames)

    # Robust session medians for implausible-length flags (all valid lengths).
    all_lengths: dict[str, list[float]] = {name: [] for name, _, _ in SEGMENTS}
    length_by_frame: dict[str, list[float | None]] = {name: [] for name, _, _ in SEGMENTS}
    length_reason: dict[str, list[str]] = {name: [] for name, _, _ in SEGMENTS}
    for frame in frames:
        by_name = by_frame[frame]
        for name, start, end in SEGMENTS:
            length, reason = segment_length(by_name, start, end)
            length_by_frame[name].append(length)
            length_reason[name].append(reason)
            if length is not None:
                all_lengths[name].append(length)
    median_length = {
        name: (float(statistics.median(vals)) if vals else None) for name, vals in all_lengths.items()
    }

    hip_step = [None] * n
    max_core_step = [None] * n
    midhip = [None] * n
    for i, frame in enumerate(frames):
        by_name = by_frame[frame]
        lh = _xyz(by_name["left_hip"]) if "left_hip" in by_name else None
        rh = _xyz(by_name["right_hip"]) if "right_hip" in by_name else None
        if lh and rh:
            midhip[i] = _mid(lh, rh)
        if i == 0:
            continue
        prev = frames[i - 1]
        if frames[i] != prev + 1:
            continue
        if midhip[i] is not None and midhip[i - 1] is not None:
            hip_step[i] = _dist(midhip[i], midhip[i - 1])
        steps = []
        for joint in CORE_JOINTS:
            pa = _xyz(by_frame[prev][joint]) if joint in by_frame[prev] else None
            pb = _xyz(by_frame[frame][joint]) if joint in by_frame[frame] else None
            if pa and pb:
                steps.append(_dist(pa, pb))
        if steps:
            max_core_step[i] = max(steps)

    implausible = []
    jump_hip = []
    jump_joint = []
    accepted = []
    pose_xyz_count = []
    hands_xyz_count = []
    pose_row_count = []
    hands_row_count = []
    flags_text = []

    for i, frame in enumerate(frames):
        by_name = by_frame[frame]
        bad_len = []
        ok_seg = 0
        for name, _, _ in SEGMENTS:
            length = length_by_frame[name][i]
            med = median_length[name]
            if length is None or med is None:
                continue
            if length < LENGTH_LOW * med or length > LENGTH_HIGH * med:
                bad_len.append(name)
            else:
                ok_seg += 1
        implausible.append(bool(bad_len))
        jh = hip_step[i] is not None and hip_step[i] > HIP_JUMP_M
        jj = max_core_step[i] is not None and max_core_step[i] > JOINT_JUMP_M
        jump_hip.append(jh)
        jump_joint.append(jj)
        pose_rows = [nm for nm in by_name if nm in POSE_SET]
        hand_rows = [nm for nm in by_name if nm in HAND_EXTRA]
        pose_xyz = sum(1 for nm in pose_rows if _xyz(by_name[nm]) is not None)
        hand_xyz = sum(1 for nm in hand_rows if _xyz(by_name[nm]) is not None)
        pose_row_count.append(len(pose_rows))
        hands_row_count.append(len(hand_rows))
        pose_xyz_count.append(pose_xyz)
        hands_xyz_count.append(hand_xyz)
        keep = (not bad_len) and (not jh) and (not jj) and ok_seg >= MIN_CORE_SEGMENTS_OK
        accepted.append(keep)
        bits = []
        if bad_len:
            bits.append("implausible_length:" + ",".join(bad_len))
        if jh:
            bits.append("hip_jump")
        if jj:
            bits.append("core_joint_jump")
        if ok_seg < MIN_CORE_SEGMENTS_OK:
            bits.append(f"few_valid_segments:{ok_seg}")
        # Distinguish missing detection vs invalid depth on core joints.
        missing = [j for j in CORE_JOINTS if j not in by_name]
        depth = [
            j
            for j in CORE_JOINTS
            if j in by_name and _xyz(by_name[j]) is None
        ]
        if missing:
            bits.append("missing_row:" + ",".join(missing))
        if depth:
            bits.append("invalid_depth:" + ",".join(depth))
        flags_text.append(";".join(bits))

    # Quiet standing: consecutive kept frames, low hip motion, duration >= MIN_QUIET_SEC.
    quiet_mask = []
    for i, frame in enumerate(frames):
        step_ok = hip_step[i] is None or hip_step[i] <= HIP_STEP_QUIET_M
        if i == 0:
            step_ok = True
        quiet_mask.append(bool(accepted[i] and step_ok))

    runs: list[tuple[int, int]] = []
    start = None
    for i in range(n):
        if quiet_mask[i] and start is None:
            start = i
        if (not quiet_mask[i] or i == n - 1) and start is not None:
            end = i if quiet_mask[i] and i == n - 1 else i - 1
            if end >= start:
                runs.append((start, end))
            start = None

    candidates = []
    for a, b in runs:
        t0 = times[frames[a]]
        t1 = times[frames[b]]
        dur = t1 - t0
        n_frames = b - a + 1
        steps = [hip_step[i] for i in range(a, b + 1) if hip_step[i] is not None]
        med_step = float(statistics.median(steps)) if steps else None
        candidates.append(
            {
                "index_start": a,
                "index_end": b,
                "frame_start": frames[a],
                "frame_end": frames[b],
                "time_start_sec": t0,
                "time_end_sec": t1,
                "duration_sec": dur,
                "n_frames": n_frames,
                "median_hip_step_m": med_step,
            }
        )
    long_enough = [c for c in candidates if c["duration_sec"] >= MIN_QUIET_SEC]
    selected = None
    if long_enough:
        selected = max(long_enough, key=lambda c: (c["duration_sec"], c["n_frames"]))
    # If none meet duration, keep the longest run as a reported candidate only.
    best_any = max(candidates, key=lambda c: (c["duration_sec"], c["n_frames"])) if candidates else None

    def metrics_for_indices(name: str, indices: list[int] | None, subset: str) -> dict:
        vals = []
        missing = 0
        if indices is None:
            series = length_by_frame[name]
            reasons = length_reason[name]
            for length, reason in zip(series, reasons):
                if length is None:
                    missing += 1
                else:
                    vals.append(length)
        else:
            for i in indices:
                length = length_by_frame[name][i]
                if length is None:
                    missing += 1
                else:
                    vals.append(length)
        row = summarise(vals, missing)
        row["segment"] = name
        row["subset"] = subset
        row["start_joint"] = next(s for n, s, e in SEGMENTS if n == name)
        row["end_joint"] = next(e for n, s, e in SEGMENTS if n == name)
        row["session_median_m"] = median_length[name]
        return row

    all_idx = list(range(n))
    kept_idx = [i for i, keep in enumerate(accepted) if keep]
    quiet_idx = list(range(selected["index_start"], selected["index_end"] + 1)) if selected else None

    metric_rows = []
    for name, _, _ in SEGMENTS:
        metric_rows.append(metrics_for_indices(name, all_idx, "all_person_frames"))
        metric_rows.append(metrics_for_indices(name, kept_idx, "quality_filtered"))
        if quiet_idx is not None:
            metric_rows.append(metrics_for_indices(name, quiet_idx, "quiet_window"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "quality_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "subset",
            "segment",
            "start_joint",
            "end_joint",
            "valid_frame_count",
            "missing_frame_count",
            "median_m",
            "stdev_m",
            "p05_m",
            "p95_m",
            "min_m",
            "max_m",
            "stdev_over_median",
            "session_median_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({k: row.get(k) for k in fields})

    with (OUT_DIR / "frame_quality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "time_sec",
                "accepted",
                "in_quiet_window",
                "implausible_length",
                "hip_jump",
                "core_joint_jump",
                "hip_step_m",
                "max_core_joint_step_m",
                "pose_33_rows",
                "pose_33_valid_xyz",
                "hands_extra_rows",
                "hands_extra_valid_xyz",
                "flags",
            ]
            + [f"{name}_m" for name, _, _ in SEGMENTS],
        )
        writer.writeheader()
        q0 = selected["index_start"] if selected else -1
        q1 = selected["index_end"] if selected else -1
        for i, frame in enumerate(frames):
            row = {
                "frame": frame,
                "time_sec": f"{times[frame]:.4f}",
                "accepted": int(accepted[i]),
                "in_quiet_window": int(q0 <= i <= q1) if selected else 0,
                "implausible_length": int(implausible[i]),
                "hip_jump": int(jump_hip[i]),
                "core_joint_jump": int(jump_joint[i]),
                "hip_step_m": "" if hip_step[i] is None else f"{hip_step[i]:.6f}",
                "max_core_joint_step_m": "" if max_core_step[i] is None else f"{max_core_step[i]:.6f}",
                "pose_33_rows": pose_row_count[i],
                "pose_33_valid_xyz": pose_xyz_count[i],
                "hands_extra_rows": hands_row_count[i],
                "hands_extra_valid_xyz": hands_xyz_count[i],
                "flags": flags_text[i],
            }
            for name, _, _ in SEGMENTS:
                val = length_by_frame[name][i]
                row[f"{name}_m"] = "" if val is None else f"{val:.6f}"
            writer.writerow(row)

    def rel_stdev(subset: str, segment: str) -> float | None:
        for row in metric_rows:
            if row["subset"] == subset and row["segment"] == segment:
                return row["stdev_over_median"]
        return None

    go_segments = []
    no_go_segments = []
    if selected:
        for name, _, _ in SEGMENTS:
            ratio = rel_stdev("quiet_window", name)
            # Decision.md: thigh/shank stdev small vs median. 5% relative stdev.
            if ratio is not None and ratio <= 0.05:
                go_segments.append(name)
            else:
                no_go_segments.append(name)

    quiet_payload = {
        "source_csv": str(CSV_PATH.resolve()),
        "source_modified": False,
        "n_person_frames": n,
        "frame_min": frames[0],
        "frame_max": frames[-1],
        "time_first_sec": times[frames[0]],
        "time_last_sec": times[frames[-1]],
        "criteria": {
            "implausible_length": (
                f"segment length outside [{LENGTH_LOW}, {LENGTH_HIGH}] × "
                "session median of that segment (all valid frames)"
            ),
            "hip_jump_m": HIP_JUMP_M,
            "core_joint_jump_m": JOINT_JUMP_M,
            "quality_filter": (
                "accepted if no implausible length, no hip jump, no core-joint jump, "
                f"and at least {MIN_CORE_SEGMENTS_OK} of 10 segments valid and in-band"
            ),
            "quiet_standing": (
                f"accepted frames whose mid-hip step is <= {HIP_STEP_QUIET_M} m "
                "(or first frame / after a gap with no step), grouped into consecutive runs"
            ),
            "min_quiet_sec": MIN_QUIET_SEC,
            "go_relative_stdev": 0.05,
            "motion_signal": "mid-hip XYZ step between consecutive frame indices (not interpolated across gaps)",
            "no_interpolation": True,
            "landmark_names": "MediaPipe model names; not ISB anatomical landmarks",
        },
        "session_median_lengths_m": median_length,
        "n_accepted_frames": int(sum(accepted)),
        "n_implausible_length": int(sum(implausible)),
        "n_hip_jump": int(sum(jump_hip)),
        "n_core_joint_jump": int(sum(jump_joint)),
        "quiet_runs_all": candidates,
        "quiet_runs_meeting_duration": long_enough,
        "selected_quiet_window": selected,
        "longest_run_if_none_selected": None if selected else best_any,
        "pose_33_frames_with_all_xyz": int(sum(1 for c in pose_xyz_count if c == 33)),
        "hands_extra_frames_with_any_xyz": int(sum(1 for c in hands_xyz_count if c > 0)),
        "hands_extra_frames_with_all_34_xyz": int(sum(1 for c in hands_xyz_count if c == 34)),
        "go_segments_quiet_stdev_over_median_le_0p05": go_segments,
        "no_go_segments_quiet": no_go_segments,
    }
    (OUT_DIR / "quiet_window.json").write_text(json.dumps(quiet_payload, indent=2), encoding="utf-8")

    t = np.array([times[f] for f in frames], dtype=float)
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    for ax, (name, _, _) in zip(axes, SEGMENTS):
        y = [np.nan if v is None else v for v in length_by_frame[name]]
        ax.plot(t, y, color="#1f4e79", lw=0.8, label="observed length")
        rejected_t = [t[i] for i in range(n) if not accepted[i] and length_by_frame[name][i] is not None]
        rejected_y = [length_by_frame[name][i] for i in range(n) if not accepted[i] and length_by_frame[name][i] is not None]
        ax.scatter(rejected_t, rejected_y, s=8, c="#c0392b", zorder=3, label="rejected frame")
        if selected:
            ax.axvspan(
                selected["time_start_sec"],
                selected["time_end_sec"],
                color="#2ecc71",
                alpha=0.18,
                label="quiet window",
            )
        if median_length[name]:
            ax.axhline(median_length[name], color="#7f8c8d", ls="--", lw=0.7)
            ax.axhline(LENGTH_LOW * median_length[name], color="#e67e22", ls=":", lw=0.6)
            ax.axhline(LENGTH_HIGH * median_length[name], color="#e67e22", ls=":", lw=0.6)
        ax.set_ylabel("m")
        ax.set_title(name.replace("_", " "))
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time_sec")
    axes[-2].set_xlabel("time_sec")
    fig.suptitle(
        "Segment lengths (MediaPipe model landmarks + RealSense metres). "
        "Red = quality-rejected. Green band = selected quiet interval."
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "segment_lengths.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, [np.nan if v is None else v for v in hip_step], color="#1f4e79", lw=0.8)
    axes[0].axhline(HIP_JUMP_M, color="#c0392b", ls="--", lw=0.8, label=f"hip jump {HIP_JUMP_M} m")
    axes[0].axhline(HIP_STEP_QUIET_M, color="#27ae60", ls="--", lw=0.8, label=f"quiet hip {HIP_STEP_QUIET_M} m")
    axes[0].set_ylabel("mid-hip step (m)")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, [np.nan if v is None else v for v in max_core_step], color="#6c3483", lw=0.8)
    axes[1].axhline(JOINT_JUMP_M, color="#c0392b", ls="--", lw=0.8)
    axes[1].set_ylabel("max core-joint step (m)")
    axes[2].plot(t, pose_xyz_count, label="Pose-33 valid XYZ", color="#1f4e79")
    axes[2].plot(t, hands_xyz_count, label="Hands-extra valid XYZ", color="#d35400")
    axes[2].set_ylabel("count")
    axes[2].legend(fontsize=8)
    if selected:
        for ax in axes:
            ax.axvspan(selected["time_start_sec"], selected["time_end_sec"], color="#2ecc71", alpha=0.18)
    for i, frame_i in enumerate(range(n)):
        if not accepted[i]:
            axes[0].axvline(t[i], color="#e74c3c", alpha=0.08, lw=0.4)
    axes[2].set_xlabel("time_sec")
    fig.suptitle("Frame quality: motion, jumps, Pose vs Hands coverage. Green = quiet window.")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frame_quality.png", dpi=140)
    plt.close(fig)

    _write_markdown(
        OUT_DIR / "quality_audit.md",
        quiet_payload,
        metric_rows,
        n,
        selected,
        long_enough,
        best_any,
        go_segments,
        no_go_segments,
        accepted,
        pose_xyz_count,
        hands_xyz_count,
        pose_row_count,
        hands_row_count,
    )
    print(f"Wrote {OUT_DIR}")
    print("selected", selected)
    print("go", go_segments)
    print("no-go", no_go_segments)


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(
    path: Path,
    payload: dict,
    metric_rows: list[dict],
    n: int,
    selected: dict | None,
    long_enough: list,
    best_any: dict | None,
    go_segments: list,
    no_go_segments: list,
    accepted: list[bool],
    pose_xyz_count: list[int],
    hands_xyz_count: list[int],
    pose_row_count: list[int],
    hands_row_count: list[int],
) -> None:
    def table(subset: str) -> list[str]:
        lines = [
            "| Segment | n valid | n missing | median m | stdev m | p05 | p95 | stdev/median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in metric_rows:
            if row["subset"] != subset:
                continue
            lines.append(
                "| {segment} | {valid_frame_count} | {missing_frame_count} | {median_m} | {stdev_m} | {p05_m} | {p95_m} | {stdev_over_median} |".format(
                    segment=row["segment"],
                    valid_frame_count=row["valid_frame_count"],
                    missing_frame_count=row["missing_frame_count"],
                    median_m=_fmt(row["median_m"]),
                    stdev_m=_fmt(row["stdev_m"]),
                    p05_m=_fmt(row["p05_m"]),
                    p95_m=_fmt(row["p95_m"]),
                    stdev_over_median=_fmt(row["stdev_over_median"]),
                )
            )
        return lines

    lines = [
        "# 3D landmark quality audit — 20261011-144812",
        "",
        "Observed MediaPipe Pose/Hands keypoints with RealSense camera metres. "
        "**Not** ISB anatomical landmarks. No interpolation. No Marker Enhancer. No Pose2Sim augmenter.",
        "",
        f"Source (read-only): `{payload['source_csv']}`",
        f"Person-tracked CSV frames: **{n}** (index {payload['frame_min']}–{payload['frame_max']}, "
        f"{payload['time_first_sec']:.2f}–{payload['time_last_sec']:.2f} s).",
        "",
        "## Selection criteria (reproducible)",
        "",
        "Quiet standing is **not guessed**. It is the longest consecutive run of frames that pass all of:",
        "",
        f"1. Segment length in `[{LENGTH_LOW}, {LENGTH_HIGH}] ×` that segment’s **session median** (all valid lengths).",
        f"2. Mid-hip step between consecutive frame IDs ≤ **{HIP_JUMP_M} m** (else `hip_jump`).",
        f"3. Max step of any of {', '.join(CORE_JOINTS)} ≤ **{JOINT_JUMP_M} m** (else `core_joint_jump`).",
        f"4. At least **{MIN_CORE_SEGMENTS_OK}/10** segments have a valid in-band length.",
        f"5. Mid-hip step ≤ **{HIP_STEP_QUIET_M} m** (quiet motion gate).",
        f"6. Run duration ≥ **{MIN_QUIET_SEC} s**.",
        "",
        "A missing CSV row is `missing_row` (not detected / not written). Empty XYZ with a row present is `invalid_depth`. Gaps in frame index are not interpolated.",
        "",
        f"Quality-filtered frames (gates 1–4, not requiring quiet hip): **{int(sum(accepted))} / {n}**.",
        f"Implausible-length frames: {payload['n_implausible_length']}. Hip jumps: {payload['n_hip_jump']}. Core-joint jumps: {payload['n_core_joint_jump']}.",
        "",
        "## Pose vs Hands coverage",
        "",
        f"- Pose-33 rows present, median per frame: {statistics.median(pose_row_count)} (max 33).",
        f"- Pose-33 valid XYZ, median: {statistics.median(pose_xyz_count)}; frames with all 33 XYZ: {payload['pose_33_frames_with_all_xyz']}.",
        f"- Hands-extra rows (34 names, not Pose-33), median: {statistics.median(hands_row_count)}.",
        f"- Hands-extra valid XYZ, median: {statistics.median(hands_xyz_count)}; any Hands XYZ: {payload['hands_extra_frames_with_any_xyz']}; all 34: {payload['hands_extra_frames_with_all_34_xyz']}.",
        "",
        "Wrist/index/pinky/thumb names can come from Pose or Hands overwrite. Hands extras are the unambiguous Hands-only set.",
        "",
        "## Quiet interval",
        "",
    ]
    if selected:
        s = selected
        lines += [
            f"**Selected:** frames {s['frame_start']}–{s['frame_end']} "
            f"({s['time_start_sec']:.3f}–{s['time_end_sec']:.3f} s), "
            f"{s['n_frames']} frames, {s['duration_sec']:.2f} s, "
            f"median mid-hip step {s['median_hip_step_m']:.4f} m.",
            "",
            f"Other runs ≥ {MIN_QUIET_SEC} s: {len(long_enough) - 1} besides the selected one. "
            f"All consecutive quiet runs: {len(payload['quiet_runs_all'])}.",
        ]
    else:
        lines += [
            f"**No run met the {MIN_QUIET_SEC} s quiet-standing gate.**",
            "",
            f"Longest run: {best_any}" if best_any else "No quiet frames at all.",
        ]
    lines += ["", "## Segment metrics — all person-frames", ""]
    lines += table("all_person_frames")
    lines += ["", "## Segment metrics — quality-filtered (rejected frames excluded, not filled)", ""]
    lines += table("quality_filtered")
    if selected:
        lines += ["", "## Segment metrics — selected quiet window", ""]
        lines += table("quiet_window")
    lines += [
        "",
        "## Plots",
        "",
        "- `segment_lengths.png` — ten segment lengths vs time; red = rejected; green band = quiet window.",
        "- `frame_quality.png` — mid-hip step, core-joint step, Pose vs Hands XYZ counts.",
        "",
        "## Decision",
        "",
        "Go rule from `decision.md`: on the quiet subset, thigh/shank **stdev / median ≤ 0.05**. "
        "The same rule is reported for the other segments for completeness.",
        "",
        f"- **Stable on quiet window:** {', '.join(go_segments) if go_segments else '(none)'}",
        f"- **Not stable enough:** {', '.join(no_go_segments) if no_go_segments else '(none)'}",
        "",
        "These lengths are distances between **MediaPipe model points**, not bone lengths between palpable landmarks.",
        "",
        "### Next test",
        "",
        "- If thighs/shanks pass: keep using observed lower-limb model points on quiet/filtered frames; do **not** enable augmenter.",
        "- Forearms/wrists: expect worse stability (depth + Hands). Do not treat them as ISB forearm bones.",
        "- Hands extras remain intermittent; practical hand frames only when MCP XYZ exist.",
        "- Do not modify `dof_angles_3d.py` from this audit.",
        "",
        "## Limitations",
        "",
        "- Session median used for the length band is computed on **all** valid frames, including motion and outliers; it is robust (median) but not a standing-only anthropometry.",
        "- ~7 fps timestamps; a 5 cm/frame hip jump is a strong spike, not gait.",
        "- Camera metres include depth noise; filtering drops frames rather than smoothing XYZ.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
