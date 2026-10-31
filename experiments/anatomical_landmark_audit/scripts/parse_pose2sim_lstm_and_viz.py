"""Parse Pose2Sim LSTM TRC (header/data mismatch) and write CSV + figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = Path(
    r"C:\Projects\motion__analysis__\experiments\anatomical_landmark_audit"
    r"\output\20261011-144812_full_body_augmentation"
)
LSTM = OUT / "trial" / "pose-3d" / "input_mapped_yup_LSTM.trc"
UPPER_OUT = [
    "r_lelbow_study",
    "r_melbow_study",
    "r_lwrist_study",
    "r_mwrist_study",
    "L_lelbow_study",
    "L_melbow_study",
    "L_lwrist_study",
    "L_mwrist_study",
]
QUIET_T0, QUIET_T1 = 43.0532, 62.9664
POSE_BONES = [
    ("LShoulder", "RShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("LShoulder", "LHip"),
    ("RShoulder", "RHip"),
    ("LHip", "RHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("LAnkle", "LHeel"),
    ("RAnkle", "RHeel"),
]
OBS = [
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
    "LHeel",
    "RHeel",
    "Neck",
]
PAIRS = [
    ("right_elbow_pair", "r_lelbow_study", "r_melbow_study"),
    ("left_elbow_pair", "L_lelbow_study", "L_melbow_study"),
    ("right_wrist_pair", "r_lwrist_study", "r_mwrist_study"),
    ("left_wrist_pair", "L_lwrist_study", "L_mwrist_study"),
]


def parse_trc(path: Path):
    text = path.read_text(encoding="utf-8").splitlines()
    header_names = [m.strip() for m in text[3].split("\t")[2::3] if m.strip()]
    rows = []
    for line in text[5:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        frame = int(float(parts[0]))
        time_sec = float(parts[1])
        nums = []
        for p in parts[2:]:
            p = p.strip()
            if p == "":
                nums.append(None)
            else:
                nums.append(float(p))
        while len(nums) % 3 != 0:
            nums.append(None)
        n_markers = len(nums) // 3
        names = list(header_names)
        if n_markers > len(names):
            extra = n_markers - len(names)
            if extra == 8:
                names.extend(UPPER_OUT)
            else:
                names.extend([f"unnamed_{i}" for i in range(extra)])
        pts = {}
        for i, name in enumerate(names[:n_markers]):
            trip = nums[i * 3 : i * 3 + 3]
            if any(v is None for v in trip) or not all(math.isfinite(v) for v in trip):
                pts[name] = None
            else:
                pts[name] = np.array(trip, dtype=float)
        rows.append({"frame": frame, "time_sec": time_sec, "pts": pts})
    return rows, names


def stats(vals):
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(len(a)),
        "median_m": float(np.median(a)),
        "mean_m": float(np.mean(a)),
        "stdev_m": float(np.std(a)),
        "min_m": float(np.min(a)),
        "max_m": float(np.max(a)),
    }


def draw(ax, obs, est, title):
    for a, b in POSE_BONES:
        if a in obs and b in obs and obs[a] is not None and obs[b] is not None:
            pa, pb = obs[a], obs[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="#1f4e79", lw=1.3)
    op = [obs[k] for k in obs if obs[k] is not None]
    if op:
        P = np.vstack(op)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="#1f4e79", s=26, label="observed (MediaPipe model)")
    ep = [est[k] for k in est if est[k] is not None]
    if ep:
        Q = np.vstack(ep)
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c="#e67e22", s=40, marker="^", label="estimated LSTM")
    ax.set_title(title)
    ax.set_xlabel("X m")
    ax.set_ylabel("Y m (up)")
    ax.set_zlabel("Z m")
    ax.legend(loc="upper left", fontsize=7)
    pts = op + ep
    if pts:
        P = np.vstack(pts)
        c = P.mean(axis=0)
        r = max(0.55, float(np.max(np.linalg.norm(P - c, axis=1))) * 1.25)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)


def main():
    rows, names = parse_trc(LSTM)
    estimated = [n for n in names if n.endswith("_study") or n in UPPER_OUT]
    print("n_frames", len(rows), "n_names", len(names), "n_estimated", len(estimated))
    print("estimated", estimated)

    csv_path = OUT / "augmented_markers_yup.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=["frame", "time_sec", "marker", "x_m", "y_m", "z_m", "coord_frame", "provenance"],
        )
        w.writeheader()
        n_finite = 0
        for row in rows:
            for name in estimated:
                p = row["pts"].get(name)
                if p is None:
                    w.writerow(
                        {
                            "frame": row["frame"],
                            "time_sec": f"{row['time_sec']:.4f}",
                            "marker": name,
                            "x_m": "",
                            "y_m": "",
                            "z_m": "",
                            "coord_frame": "opensim_yup_metre",
                            "provenance": "estimated_pose2sim_lstm_v0.3_upper",
                        }
                    )
                else:
                    n_finite += 1
                    w.writerow(
                        {
                            "frame": row["frame"],
                            "time_sec": f"{row['time_sec']:.4f}",
                            "marker": name,
                            "x_m": f"{p[0]:.6f}",
                            "y_m": f"{p[1]:.6f}",
                            "z_m": f"{p[2]:.6f}",
                            "coord_frame": "opensim_yup_metre",
                            "provenance": "estimated_pose2sim_lstm_v0.3_upper",
                        }
                    )

    pair_stats = {}
    for label, a, b in PAIRS:
        vals = []
        for row in rows:
            pa, pb = row["pts"].get(a), row["pts"].get(b)
            if pa is not None and pb is not None:
                vals.append(float(np.linalg.norm(pa - pb)))
        pair_stats[label] = stats(vals)

    jumps = {}
    for name in estimated:
        steps = []
        prev = None
        prev_f = None
        for row in rows:
            p = row["pts"].get(name)
            if prev is not None and prev_f is not None and row["frame"] == prev_f + 1 and p is not None:
                steps.append(float(np.linalg.norm(p - prev)))
            prev, prev_f = p, row["frame"]
        jumps[name] = stats(steps)

    quiet = [i for i, r in enumerate(rows) if QUIET_T0 <= r["time_sec"] <= QUIET_T1]
    vis_i = quiet[len(quiet) // 2] if quiet else len(rows) // 2
    obs = {k: rows[vis_i]["pts"].get(k) for k in OBS}
    est = {k: rows[vis_i]["pts"].get(k) for k in estimated}

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    t = rows[vis_i]["time_sec"]
    draw(ax1, obs, {}, f"Observed model landmarks  t={t:.2f}s")
    draw(ax2, obs, est, f"Observed + 8 arm estimates  t={t:.2f}s")
    fig.suptitle(
        "Pose2Sim LSTM v0.3 upper (Y-up metres). Orange = estimated, not epicondyles/styloids.\n"
        "Lower-body augmenter skipped: BigToe/SmallToe not mapped from foot_index."
    )
    fig.tight_layout()
    fig.savefig(OUT / "viz_observed_vs_estimated.png", dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    for j, (elev, azim) in enumerate(((18, -70), (12, 25), (75, -90))):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        draw(ax, obs, est, f"elev {elev} az {azim}")
        ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(OUT / "viz_estimated_viewpoints.png", dpi=140)
    plt.close(fig)

    if len(quiet) >= 4:
        idxs = quiet[:: max(1, len(quiet) // 20)]
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        def update(k):
            ax.cla()
            i = idxs[k]
            o = {n: rows[i]["pts"].get(n) for n in OBS}
            e = {n: rows[i]["pts"].get(n) for n in estimated}
            draw(ax, o, e, f"quiet t={rows[i]['time_sec']:.2f}s")

        anim = FuncAnimation(fig, update, frames=len(idxs), interval=220)
        anim.save(str(OUT / "viz_quiet_window.gif"), writer=PillowWriter(fps=5))
        plt.close(fig)

    summary = {
        "lstm_trc": str(LSTM),
        "n_frames": len(rows),
        "n_header_names": len(names),
        "estimated_markers": estimated,
        "n_estimated": len(estimated),
        "n_finite_estimated_xyz": n_finite,
        "lower_body_ran": False,
        "upper_body_ran": len(estimated) == 8,
        "pair_stats_m": pair_stats,
        "frame_to_frame_step_m": jumps,
        "viz_time_sec": t,
        "n_quiet_frames": len(quiet),
        "coord_frame": "opensim_yup_metre (X right, Y up, Z forward)",
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_frames", "n_estimated", "upper_body_ran", "lower_body_ran", "pair_stats_m")}, indent=2))


if __name__ == "__main__":
    main()
