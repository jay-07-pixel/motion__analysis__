"""Lower-limb observed-landmark stability on the existing quiet window.

Reads the 20261011-144812 CSV and the previous quality-audit quiet_window.json.
Writes only under output/20261011-144812_lower_limb/. Does not edit sources.
Does not build anatomical frames or DOFs. Names stay MediaPipe model landmarks.
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
QUIET_JSON = (
    PROJECT_ROOT
    / "experiments"
    / "anatomical_landmark_audit"
    / "output"
    / "20261011-144812_quality"
    / "quiet_window.json"
)
FRAME_QUALITY = (
    PROJECT_ROOT
    / "experiments"
    / "anatomical_landmark_audit"
    / "output"
    / "20261011-144812_quality"
    / "frame_quality.csv"
)
OUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "anatomical_landmark_audit"
    / "output"
    / "20261011-144812_lower_limb"
)

LOWER_JOINTS = (
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

SEGMENTS = (
    ("hip_width", "left_hip", "right_hip"),
    ("left_thigh", "left_hip", "left_knee"),
    ("right_thigh", "right_hip", "right_knee"),
    ("left_shank", "left_knee", "left_ankle"),
    ("right_shank", "right_knee", "right_ankle"),
    ("left_ankle_heel", "left_ankle", "left_heel"),
    ("right_ankle_heel", "right_ankle", "right_heel"),
    ("left_heel_foot_index", "left_heel", "left_foot_index"),
    ("right_heel_foot_index", "right_heel", "right_foot_index"),
)

GO_REL_STDEV = 0.05
# Quiet standing: millimetre–centimetre jitter. 2 cm/frame is a hop at ~7 fps.
JOINT_STEP_WARN_M = 0.020
SYMMETRY_WARN = 0.05  # |L-R| / mean of L and R medians


def _xyz(row: dict[str, str] | None) -> tuple[float, float, float] | None:
    if not row:
        return None
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


def _conf(row: dict[str, str] | None) -> float | None:
    if not row:
        return None
    text = (row.get("confidence") or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def load_csv(path: Path):
    by_frame: dict[int, dict[str, dict]] = defaultdict(dict)
    times: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            times[frame] = float(row["time_sec"])
            by_frame[frame][row["joint"]] = row
    return times, by_frame


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarise(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "median": None,
            "stdev": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev_over_median": None,
        }
    med = float(statistics.median(values))
    stdev = float(statistics.pstdev(values)) if len(values) >= 2 else 0.0
    return {
        "n": len(values),
        "median": med,
        "stdev": stdev,
        "p05": percentile(values, 5),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
        "stdev_over_median": (stdev / med) if med > 1e-9 else None,
    }


def main() -> None:
    for path in (CSV_PATH, QUIET_JSON, FRAME_QUALITY):
        if not path.is_file():
            raise FileNotFoundError(path)

    quiet = json.loads(QUIET_JSON.read_text(encoding="utf-8"))
    selected = quiet["selected_quiet_window"]
    frame_start = int(selected["frame_start"])
    frame_end = int(selected["frame_end"])
    if (frame_start, frame_end) != (357, 504):
        raise RuntimeError(
            f"quiet_window.json selected frames {frame_start}–{frame_end}, expected 357–504"
        )

    times, by_frame = load_csv(CSV_PATH)
    frames = [f for f in sorted(by_frame) if frame_start <= f <= frame_end]
    if not frames:
        raise RuntimeError("No CSV frames inside the quiet window.")

    fq_quiet = []
    with FRAME_QUALITY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["in_quiet_window"] == "1":
                fq_quiet.append(int(row["frame"]))
    if fq_quiet != frames:
        raise RuntimeError(
            f"frame_quality.csv quiet frames {fq_quiet[0]}–{fq_quiet[-1]} "
            f"n={len(fq_quiet)} do not match CSV window n={len(frames)}"
        )

    # Per-joint series
    joint_rows = []
    joint_steps: dict[str, list[float]] = {j: [] for j in LOWER_JOINTS}
    joint_xyz: dict[str, list[tuple]] = {j: [] for j in LOWER_JOINTS}
    joint_valid = {j: 0 for j in LOWER_JOINTS}
    joint_missing_row = {j: 0 for j in LOWER_JOINTS}
    joint_invalid_depth = {j: 0 for j in LOWER_JOINTS}
    prev_xyz: dict[str, tuple | None] = {j: None for j in LOWER_JOINTS}
    prev_frame: int | None = None

    for frame in frames:
        by_name = by_frame[frame]
        for joint in LOWER_JOINTS:
            row = by_name.get(joint)
            point = _xyz(row)
            conf = _conf(row)
            if row is None:
                joint_missing_row[joint] += 1
                kind = "missing_row"
            elif point is None:
                joint_invalid_depth[joint] += 1
                kind = "invalid_depth"
            else:
                joint_valid[joint] += 1
                joint_xyz[joint].append(point)
                kind = "ok"
                if prev_frame is not None and frame == prev_frame + 1 and prev_xyz[joint] is not None:
                    joint_steps[joint].append(math.dist(point, prev_xyz[joint]))
            prev_xyz[joint] = point
            joint_rows.append(
                {
                    "frame": frame,
                    "time_sec": times[frame],
                    "joint": joint,
                    "status": kind,
                    "confidence": "" if conf is None else f"{conf:.4f}",
                    "x_m": "" if point is None else f"{point[0]:.6f}",
                    "y_m": "" if point is None else f"{point[1]:.6f}",
                    "z_m": "" if point is None else f"{point[2]:.6f}",
                    "step_m": "",
                }
            )
        prev_frame = frame

    # Fill step on the ok consecutive rows (second pass on joint_rows by joint order is messy;
    # recompute from series). Already stored in joint_steps.

    seg_values: dict[str, list[float]] = {n: [] for n, _, _ in SEGMENTS}
    seg_missing = {n: 0 for n, _, _ in SEGMENTS}
    seg_by_frame = {n: [] for n, _, _ in SEGMENTS}
    for frame in frames:
        by_name = by_frame[frame]
        for name, a, b in SEGMENTS:
            pa = _xyz(by_name.get(a))
            pb = _xyz(by_name.get(b))
            if pa is None or pb is None:
                seg_missing[name] += 1
                seg_by_frame[name].append(None)
            else:
                length = math.dist(pa, pb)
                seg_values[name].append(length)
                seg_by_frame[name].append(length)

    n = len(frames)
    metric_rows = []
    for joint in LOWER_JOINTS:
        steps = joint_steps[joint]
        pts = np.array(joint_xyz[joint], dtype=float) if joint_xyz[joint] else np.zeros((0, 3))
        confs = []
        for frame in frames:
            row = by_frame[frame].get(joint)
            c = _conf(row)
            if c is not None and _xyz(row) is not None:
                confs.append(c)
        span = None
        if len(pts) >= 2:
            span = float(np.max(np.ptp(pts, axis=0)))
        s = summarise(steps)
        metric_rows.append(
            {
                "kind": "joint_step",
                "name": joint,
                "valid_xyz": joint_valid[joint],
                "missing_row": joint_missing_row[joint],
                "invalid_depth": joint_invalid_depth[joint],
                "n_window": n,
                "coverage": joint_valid[joint] / n,
                "confidence_median": float(statistics.median(confs)) if confs else None,
                "step_median_m": s["median"],
                "step_stdev_m": s["stdev"],
                "step_p95_m": s["p95"],
                "step_max_m": s["max"],
                "n_steps_over_warn": int(sum(1 for v in steps if v > JOINT_STEP_WARN_M)),
                "xyz_span_max_axis_m": span,
                "median_m": None,
                "stdev_m": None,
                "stdev_over_median": None,
                "p05_m": None,
                "p95_m": None,
            }
        )
    for name, a, b in SEGMENTS:
        s = summarise(seg_values[name])
        go = s["stdev_over_median"] is not None and s["stdev_over_median"] <= GO_REL_STDEV
        metric_rows.append(
            {
                "kind": "segment",
                "name": name,
                "start": a,
                "end": b,
                "valid_xyz": s["n"],
                "missing_or_invalid": seg_missing[name],
                "n_window": n,
                "coverage": s["n"] / n,
                "median_m": s["median"],
                "stdev_m": s["stdev"],
                "p05_m": s["p05"],
                "p95_m": s["p95"],
                "min_m": s["min"],
                "max_m": s["max"],
                "stdev_over_median": s["stdev_over_median"],
                "go_rel_stdev_le_0p05": go,
            }
        )

    symmetry = {}
    for label, left, right in (
        ("thigh", "left_thigh", "right_thigh"),
        ("shank", "left_shank", "right_shank"),
        ("ankle_heel", "left_ankle_heel", "right_ankle_heel"),
        ("heel_foot_index", "left_heel_foot_index", "right_heel_foot_index"),
    ):
        lm = next(r["median_m"] for r in metric_rows if r["name"] == left)
        rm = next(r["median_m"] for r in metric_rows if r["name"] == right)
        if lm and rm:
            mean = 0.5 * (lm + rm)
            rel = abs(lm - rm) / mean
        else:
            rel = None
        symmetry[label] = {
            "left_median_m": lm,
            "right_median_m": rm,
            "abs_diff_m": None if lm is None or rm is None else abs(lm - rm),
            "rel_diff": rel,
            "warn_if_rel_gt": SYMMETRY_WARN,
        }

    go_segments = [
        r["name"]
        for r in metric_rows
        if r["kind"] == "segment" and r.get("go_rel_stdev_le_0p05")
    ]
    no_go_segments = [
        r["name"]
        for r in metric_rows
        if r["kind"] == "segment" and not r.get("go_rel_stdev_le_0p05")
    ]
    stable_joints = [
        r["name"]
        for r in metric_rows
        if r["kind"] == "joint_step"
        and r["coverage"] >= 0.99
        and r["step_median_m"] is not None
        and r["step_median_m"] <= 0.005
        and (r["n_steps_over_warn"] or 0) == 0
    ]
    unstable_joints = [j for j in LOWER_JOINTS if j not in stable_joints]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "lower_limb_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({k for row in metric_rows for k in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    with (OUT_DIR / "lower_limb_frame_joints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joint_rows[0].keys()))
        writer.writeheader()
        writer.writerows(joint_rows)

    summary = {
        "source_csv": str(CSV_PATH.resolve()),
        "source_modified": False,
        "quiet_window": selected,
        "verified_against_frame_quality_csv": True,
        "n_frames_in_window": n,
        "frame_index_min": frames[0],
        "frame_index_max": frames[-1],
        "time_start_sec": times[frames[0]],
        "time_end_sec": times[frames[-1]],
        "criteria": {
            "segment_go": f"stdev/median <= {GO_REL_STDEV} on quiet-window valid lengths",
            "joint_stable": (
                "coverage >= 0.99, median consecutive XYZ step <= 5 mm, "
                f"no steps > {JOINT_STEP_WARN_M} m"
            ),
            "no_interpolation": True,
            "names": "MediaPipe model landmarks, not ISB bones",
        },
        "symmetry": symmetry,
        "go_segments": go_segments,
        "no_go_segments": no_go_segments,
        "stable_joints": stable_joints,
        "not_meeting_joint_stable_rule": unstable_joints,
        "previous_audit_stable_segments": [
            "hip_width",
            "left_thigh",
            "right_thigh",
            "left_shank",
            "right_shank",
        ],
    }
    (OUT_DIR / "lower_limb_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    t = np.array([times[f] for f in frames], dtype=float)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    axes = axes.ravel()
    for ax, (name, _, _) in zip(axes, SEGMENTS):
        y = [np.nan if v is None else v for v in seg_by_frame[name]]
        ax.plot(t, y, color="#1f4e79", lw=1.0)
        row = next(r for r in metric_rows if r["name"] == name)
        if row["median_m"]:
            ax.axhline(row["median_m"], color="#7f8c8d", ls="--", lw=0.7)
        ax.set_title(name.replace("_", " "))
        ax.set_ylabel("m")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Quiet window 357–504: lower-limb segment lengths (MediaPipe + RealSense metres)"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lower_limb_segment_lengths.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(13, 10), sharex=True)
    axes = axes.ravel()
    for ax, joint in zip(axes, LOWER_JOINTS):
        steps_plot = []
        t_step = []
        prev = None
        prev_f = None
        for frame in frames:
            point = _xyz(by_frame[frame].get(joint))
            if (
                prev is not None
                and prev_f is not None
                and frame == prev_f + 1
                and point is not None
            ):
                t_step.append(times[frame])
                steps_plot.append(math.dist(point, prev))
            prev = point
            prev_f = frame
        ax.plot(t_step, steps_plot, color="#6c3483", lw=0.8)
        ax.axhline(JOINT_STEP_WARN_M, color="#c0392b", ls="--", lw=0.7)
        ax.set_title(joint)
        ax.set_ylabel("step m")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time_sec")
    axes[-2].set_xlabel("time_sec")
    fig.suptitle("Quiet window: consecutive-frame XYZ step of lower-limb model landmarks")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lower_limb_joint_steps.png", dpi=140)
    plt.close(fig)

    _write_md(OUT_DIR / "lower_limb_stability.md", summary, metric_rows, n, times, frames)
    print(f"Wrote {OUT_DIR}")
    print("go_segments", go_segments)
    print("no_go_segments", no_go_segments)
    print("stable_joints", stable_joints)


def _fmt(v, d=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def _write_md(path, summary, metric_rows, n, times, frames):
    q = summary["quiet_window"]
    lines = [
        "# Lower-limb 3D landmark stability — quiet window",
        "",
        "Observed MediaPipe Pose points + RealSense camera metres. "
        "**Not** ISB hip, knee, ankle or foot landmarks. No interpolation, augmenter, or DOF changes.",
        "",
        f"Source (read-only): `{summary['source_csv']}`",
        "",
        "## Window (from previous quality audit, re-checked)",
        "",
        f"- Frames **{q['frame_start']}–{q['frame_end']}** ({q['time_start_sec']:.3f}–{q['time_end_sec']:.3f} s)",
        f"- **{n}** CSV frames in that closed range; matches `frame_quality.csv` `in_quiet_window=1`",
        f"- Previous median mid-hip step **{q['median_hip_step_m']:.4f} m**",
        "",
        "Previous quality audit (stdev/median ≤ 0.05): hip width, both thighs, both shanks. "
        "This run **recomputes** those lengths on the same window and adds per-joint XYZ steps, "
        "heels, foot_index, ankle–heel and heel–foot_index distances, and left/right symmetry.",
        "",
        "## Joint coverage and jitter (quiet window)",
        "",
        "| Joint | valid XYZ | missing row | invalid depth | coverage | conf median | step median m | step p95 m | step max m | steps > 20 mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        if row["kind"] != "joint_step":
            continue
        lines.append(
            "| {name} | {valid_xyz} | {missing_row} | {invalid_depth} | {coverage} | {confidence_median} | {step_median_m} | {step_p95_m} | {step_max_m} | {n_steps_over_warn} |".format(
                name=row["name"],
                valid_xyz=row["valid_xyz"],
                missing_row=row["missing_row"],
                invalid_depth=row["invalid_depth"],
                coverage=_fmt(row["coverage"], 3),
                confidence_median=_fmt(row["confidence_median"], 3),
                step_median_m=_fmt(row["step_median_m"], 4),
                step_p95_m=_fmt(row["step_p95_m"], 4),
                step_max_m=_fmt(row["step_max_m"], 4),
                n_steps_over_warn=row["n_steps_over_warn"],
            )
        )
    lines += [
        "",
        "Joint “stable” rule: coverage ≥ 0.99, median step ≤ 5 mm, no step > 20 mm.",
        "",
        f"- **Meets joint-stable rule:** {', '.join(summary['stable_joints']) or '(none)'}",
        f"- **Does not:** {', '.join(summary['not_meeting_joint_stable_rule']) or '(none)'}",
        "",
        "## Segment lengths (quiet window)",
        "",
        "Go: stdev/median ≤ 0.05 (same as `decision.md` / quality audit).",
        "",
        "| Segment | n valid | missing | median m | stdev m | p05 | p95 | stdev/median | go |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metric_rows:
        if row["kind"] != "segment":
            continue
        lines.append(
            "| {name} | {valid_xyz} | {missing_or_invalid} | {median_m} | {stdev_m} | {p05_m} | {p95_m} | {stdev_over_median} | {go} |".format(
                name=row["name"],
                valid_xyz=row["valid_xyz"],
                missing_or_invalid=row["missing_or_invalid"],
                median_m=_fmt(row["median_m"]),
                stdev_m=_fmt(row["stdev_m"]),
                p05_m=_fmt(row["p05_m"]),
                p95_m=_fmt(row["p95_m"]),
                stdev_over_median=_fmt(row["stdev_over_median"]),
                go=_fmt(row["go_rel_stdev_le_0p05"]),
            )
        )
    lines += ["", "## Left/right median symmetry", "", "| Pair | left m | right m | |diff| m | rel |", "|---|---:|---:|---:|---:|"]
    for label, block in summary["symmetry"].items():
        lines.append(
            f"| {label} | {_fmt(block['left_median_m'])} | {_fmt(block['right_median_m'])} | "
            f"{_fmt(block['abs_diff_m'])} | {_fmt(block['rel_diff'])} |"
        )
    lines += [
        "",
        f"Relative difference > {SYMMETRY_WARN} is a warning (model-point geometry, not bone inequality).",
        "",
        "## Decision",
        "",
        f"- **Segment go:** {', '.join(summary['go_segments']) or '(none)'}",
        f"- **Segment no-go:** {', '.join(summary['no_go_segments']) or '(none)'}",
        "",
        "Reconfirm vs previous audit: hip width, both thighs, both shanks should still pass. "
        "Foot segments (ankle–heel, heel–foot_index) were not in that go-list; they are judged here on the same numeric rule.",
        "",
        "Reliable enough to **keep as observed lower-limb model landmarks** on quiet/filtered frames: "
        "hips, knees, ankles (and the thigh/shank distances they form). ",
        "Heels and foot_index: use only with the joint-stable / segment-go result; they are weaker model points "
        "(lower confidence, possible depth outliers historically).",
        "",
        "**Do not** build ISB knee/ankle frames from these points (single model knee ≠ two epicondyles; "
        "single ankle ≠ two malleoli; foot_index ≠ hallux and 5th metatarsal).",
        "",
        "**Do not** enable augmenter or edit `dof_angles_3d.py` / `anatomical_frames.py`.",
        "",
        "### Next",
        "",
        "- If thighs/shanks/hips still go: a later experiment may **report** observed lower-limb segment lengths over time (filtered), still as model distances.",
        "- Anatomical coordinate frames for the lower limb remain **blocked** until bony landmarks exist.",
        "- Upper limb remains on the previous quality-audit no-go list for standing stability.",
        "",
        "## Plots",
        "",
        "- `lower_limb_segment_lengths.png`",
        "- `lower_limb_joint_steps.png`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
