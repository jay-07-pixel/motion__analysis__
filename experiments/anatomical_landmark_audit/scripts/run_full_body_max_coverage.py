"""Maximum honest full-body marker coverage for 20261011-144812.

Writes a NEW subfolder. Does not overwrite the previous augmenter run.
Does not invent BigToe/SmallToe. Does not run the lower LSTM.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

PROJECT = Path(r"C:\Projects\motion__analysis__")
CSV_PATH = PROJECT / "data" / "output" / "20261011-144812" / "keypoints_2d.csv"
PARENT = (
    PROJECT
    / "experiments"
    / "anatomical_landmark_audit"
    / "output"
    / "20261011-144812_full_body_augmentation"
)
EST_CSV = PARENT / "augmented_markers_yup.csv"
OUT = PARENT / "max_coverage"
QUIET_T0, QUIET_T1 = 43.0532, 62.9664

POSE_33 = [
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
]
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
LOWER_OUT = [
    "r.ASIS_study",
    "L.ASIS_study",
    "r.PSIS_study",
    "L.PSIS_study",
    "r_knee_study",
    "r_mknee_study",
    "r_ankle_study",
    "r_mankle_study",
    "r_toe_study",
    "r_5meta_study",
    "r_calc_study",
    "L_knee_study",
    "L_mknee_study",
    "L_ankle_study",
    "L_mankle_study",
    "L_toe_study",
    "L_calc_study",
    "L_5meta_study",
    "r_shoulder_study",
    "L_shoulder_study",
    "C7_study",
    "r_thigh1_study",
    "r_thigh2_study",
    "r_thigh3_study",
    "L_thigh1_study",
    "L_thigh2_study",
    "L_thigh3_study",
    "r_sh1_study",
    "r_sh2_study",
    "r_sh3_study",
    "L_sh1_study",
    "L_sh2_study",
    "L_sh3_study",
    "RHJC_study",
    "LHJC_study",
]
UPPER_IN = ["Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist", "LWrist"]
LOWER_IN = [
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
]
BONES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
]
SEGMENTS = [
    ("shoulder_width", "left_shoulder", "right_shoulder"),
    ("hip_width", "left_hip", "right_hip"),
    ("left_thigh", "left_hip", "left_knee"),
    ("right_thigh", "right_hip", "right_knee"),
    ("left_shank", "left_knee", "left_ankle"),
    ("right_shank", "right_knee", "right_ankle"),
    ("left_heel_foot_index", "left_heel", "left_foot_index"),
    ("right_heel_foot_index", "right_heel", "right_foot_index"),
]
PAIRS = [
    ("right_elbow_pair", "r_lelbow_study", "r_melbow_study"),
    ("left_elbow_pair", "L_lelbow_study", "L_melbow_study"),
    ("right_wrist_pair", "r_lwrist_study", "r_mwrist_study"),
    ("left_wrist_pair", "L_lwrist_study", "L_mwrist_study"),
]
ANATOMY_NEITHER_MODEL = [
    "glenohumeral_joint_centre",
    "acromion_protocol",
    "incisura_jugularis",
    "xiphoid",
    "T8",
    "radial_styloid",
    "ulnar_styloid",
    "individual_vertebrae",
    "true_hallux_palpated",
    "true_5th_metatarsal_palpated",
]


def cam_to_yup(x, y, z):
    return np.array([x, -y, z], dtype=float)


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
        "stdev_over_median": None if abs(float(np.median(a))) < 1e-9 else float(np.std(a) / np.median(a)),
    }


def load_observed(path: Path):
    by_frame: dict[int, dict] = {}
    times: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            times[frame] = float(row["time_sec"])
            joint = row["joint"]
            xs, ys, zs = row["x_m"].strip(), row["y_m"].strip(), row["z_m"].strip()
            rec = by_frame.setdefault(frame, {})
            if xs == "" or ys == "" or zs == "":
                rec[joint] = {
                    "xyz_yup": None,
                    "u_px": float(row["u_px"]) if row["u_px"] else None,
                    "v_px": float(row["v_px"]) if row["v_px"] else None,
                    "confidence": float(row["confidence"]) if row["confidence"] else None,
                    "invalid_depth": True,
                }
            else:
                rec[joint] = {
                    "xyz_yup": cam_to_yup(float(xs), float(ys), float(zs)),
                    "u_px": float(row["u_px"]) if row["u_px"] else None,
                    "v_px": float(row["v_px"]) if row["v_px"] else None,
                    "confidence": float(row["confidence"]) if row["confidence"] else None,
                    "invalid_depth": False,
                }
    return by_frame, times


def load_estimates(path: Path):
    by_frame: dict[int, dict] = {}
    times: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            times[frame] = float(row["time_sec"])
            xs, ys, zs = row["x_m"].strip(), row["y_m"].strip(), row["z_m"].strip()
            rec = by_frame.setdefault(frame, {})
            if xs == "" or ys == "" or zs == "":
                rec[row["marker"]] = None
            else:
                rec[row["marker"]] = np.array([float(xs), float(ys), float(zs)], dtype=float)
    return by_frame, times


def draw(ax, obs, est, title):
    for a, b in BONES:
        pa = obs.get(a)
        pb = obs.get(b)
        if pa is not None and pb is not None:
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="#1f4e79", lw=1.2)
    op = [p for p in obs.values() if p is not None]
    if op:
        P = np.vstack(op)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="#1f4e79", s=18, label="observed (MediaPipe+depth)")
    ep = [p for p in est.values() if p is not None]
    if ep:
        Q = np.vstack(ep)
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c="#e67e22", s=36, marker="^", label="estimated LSTM upper")
    ax.set_title(title)
    ax.set_xlabel("X m")
    ax.set_ylabel("Y m (up)")
    ax.set_zlabel("Z m")
    ax.legend(loc="upper left", fontsize=7)
    pts = op + ep
    if pts:
        P = np.vstack(pts)
        c = P.mean(axis=0)
        r = max(0.7, float(np.max(np.linalg.norm(P - c, axis=1))) * 1.2)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observed, obs_times = load_observed(CSV_PATH)
    estimates, est_times = load_estimates(EST_CSV)
    frames = sorted(observed)
    quiet = [f for f in frames if QUIET_T0 <= obs_times[f] <= QUIET_T1]
    vis = quiet[len(quiet) // 2] if quiet else frames[len(frames) // 2]

    # Coverage counts
    coverage_rows = []
    extra_names = sorted({n for rec in observed.values() for n in rec} - set(POSE_33))

    def coverage_for(name, category, note):
        n_xyz = n_inv = n_present = 0
        for rec in observed.values():
            if name not in rec:
                continue
            n_present += 1
            if rec[name]["invalid_depth"] or rec[name]["xyz_yup"] is None:
                n_inv += 1
            else:
                n_xyz += 1
        coverage_rows.append(
            {
                "marker": name,
                "category": category,
                "n_frames_present_2d": n_present,
                "n_finite_xyz": n_xyz,
                "n_invalid_depth": n_inv,
                "note": note,
            }
        )

    for name in POSE_33:
        note = "MediaPipe Pose model landmark + RealSense depth"
        if "foot_index" in name:
            note = "One toe-ray model point. NOT BigToe and NOT SmallToe."
        coverage_for(name, "observed", note)
    for name in extra_names:
        coverage_for(name, "observed", "MediaPipe Hands extra + RealSense depth")

    coverage_rows.append(
        {
            "marker": "Neck",
            "category": "proxy",
            "n_frames_present_2d": "",
            "n_finite_xyz": "",
            "n_invalid_depth": "",
            "note": "midpoint(left_shoulder, right_shoulder); LSTM feature, not C7",
        }
    )
    coverage_rows.append(
        {
            "marker": "Hip",
            "category": "proxy",
            "n_frames_present_2d": "",
            "n_finite_xyz": "",
            "n_invalid_depth": "",
            "note": "midpoint(left_hip, right_hip); LSTM root, not ASIS/HJC",
        }
    )

    n_est_frames = len(estimates)
    for name in UPPER_OUT:
        n_xyz = sum(1 for rec in estimates.values() if rec.get(name) is not None)
        coverage_rows.append(
            {
                "marker": name,
                "category": "estimated_upper_lstm_v0.3",
                "n_frames_present_2d": n_est_frames,
                "n_finite_xyz": n_xyz,
                "n_invalid_depth": n_est_frames - n_xyz,
                "note": "Stanford upper LSTM. Not epicondyle/styloid ground truth.",
            }
        )
    for name in LOWER_OUT:
        coverage_rows.append(
            {
                "marker": name,
                "category": "missing_lower_lstm_blocked",
                "n_frames_present_2d": 0,
                "n_finite_xyz": 0,
                "n_invalid_depth": "",
                "note": "Lower LSTM skipped: R/L BigToe and SmallToe not observed.",
            }
        )
    for name in ["RBigToe", "LBigToe", "RSmallToe", "LSmallToe"]:
        coverage_rows.append(
            {
                "marker": name,
                "category": "missing_required_input",
                "n_frames_present_2d": 0,
                "n_finite_xyz": 0,
                "n_invalid_depth": "",
                "note": "Required lower-LSTM input. Not in this recording as distinct 3D points.",
            }
        )
    for name in ANATOMY_NEITHER_MODEL:
        coverage_rows.append(
            {
                "marker": name,
                "category": "unavailable_neither_model_nor_observation",
                "n_frames_present_2d": 0,
                "n_finite_xyz": 0,
                "n_invalid_depth": "",
                "note": "Not an LSTM v0.3 output and not a MediaPipe landmark.",
            }
        )

    cov_path = OUT / "marker_coverage.csv"
    with cov_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "marker",
                "category",
                "n_frames_present_2d",
                "n_finite_xyz",
                "n_invalid_depth",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    # Combined XYZ CSV (observed Y-up + estimates), quiet + all frames would be huge;
    # write all observed pose-33 + extras + estimates where present.
    comb_path = OUT / "full_body_markers_yup.csv"
    with comb_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "time_sec",
                "marker",
                "x_m",
                "y_m",
                "z_m",
                "coord_frame",
                "provenance",
            ],
        )
        writer.writeheader()
        for frame in frames:
            t = obs_times[frame]
            rec = observed[frame]
            for name, item in rec.items():
                p = item["xyz_yup"]
                writer.writerow(
                    {
                        "frame": frame,
                        "time_sec": f"{t:.4f}",
                        "marker": name,
                        "x_m": "" if p is None else f"{p[0]:.6f}",
                        "y_m": "" if p is None else f"{p[1]:.6f}",
                        "z_m": "" if p is None else f"{p[2]:.6f}",
                        "coord_frame": "opensim_yup_metre",
                        "provenance": "observed_mediapipe_realsense" if p is not None else "invalid_depth",
                    }
                )
            ls, rs = rec.get("left_shoulder"), rec.get("right_shoulder")
            if ls and rs and ls["xyz_yup"] is not None and rs["xyz_yup"] is not None:
                p = 0.5 * (ls["xyz_yup"] + rs["xyz_yup"])
                writer.writerow(
                    {
                        "frame": frame,
                        "time_sec": f"{t:.4f}",
                        "marker": "Neck",
                        "x_m": f"{p[0]:.6f}",
                        "y_m": f"{p[1]:.6f}",
                        "z_m": f"{p[2]:.6f}",
                        "coord_frame": "opensim_yup_metre",
                        "provenance": "proxy_mid_shoulders",
                    }
                )
            lh, rh = rec.get("left_hip"), rec.get("right_hip")
            if lh and rh and lh["xyz_yup"] is not None and rh["xyz_yup"] is not None:
                p = 0.5 * (lh["xyz_yup"] + rh["xyz_yup"])
                writer.writerow(
                    {
                        "frame": frame,
                        "time_sec": f"{t:.4f}",
                        "marker": "Hip",
                        "x_m": f"{p[0]:.6f}",
                        "y_m": f"{p[1]:.6f}",
                        "z_m": f"{p[2]:.6f}",
                        "coord_frame": "opensim_yup_metre",
                        "provenance": "proxy_mid_hips",
                    }
                )
            est = estimates.get(frame, {})
            for name in UPPER_OUT:
                p = est.get(name)
                writer.writerow(
                    {
                        "frame": frame,
                        "time_sec": f"{t:.4f}",
                        "marker": name,
                        "x_m": "" if p is None else f"{p[0]:.6f}",
                        "y_m": "" if p is None else f"{p[1]:.6f}",
                        "z_m": "" if p is None else f"{p[2]:.6f}",
                        "coord_frame": "opensim_yup_metre",
                        "provenance": "estimated_pose2sim_lstm_v0.3_upper",
                    }
                )

    # Quality on quiet window
    quiet_stats = {}
    for label, a, b in SEGMENTS:
        vals = []
        for f in quiet:
            pa = observed[f].get(a, {}).get("xyz_yup") if a in observed[f] else None
            pb = observed[f].get(b, {}).get("xyz_yup") if b in observed[f] else None
            if pa is not None and pb is not None:
                vals.append(float(np.linalg.norm(pa - pb)))
        quiet_stats[label] = stats(vals)

    pair_stats = {}
    for label, a, b in PAIRS:
        vals = []
        for f in quiet:
            rec = estimates.get(f, {})
            pa, pb = rec.get(a), rec.get(b)
            if pa is not None and pb is not None:
                vals.append(float(np.linalg.norm(pa - pb)))
        pair_stats[label] = stats(vals)

    jumps = {}
    prev = {}
    prev_f = None
    for f in quiet:
        rec = observed[f]
        for name in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"):
            p = rec.get(name, {}).get("xyz_yup") if name in rec else None
            if prev_f is not None and f == prev_f + 1 and p is not None and name in prev and prev[name] is not None:
                jumps.setdefault(name, []).append(float(np.linalg.norm(p - prev[name])))
            prev[name] = p
        prev_f = f
    jump_stats = {k: stats(v) for k, v in jumps.items()}

    obs_vis = {
        n: observed[vis][n]["xyz_yup"]
        for n in POSE_33
        if n in observed[vis] and observed[vis][n]["xyz_yup"] is not None
    }
    est_vis = {n: estimates.get(vis, {}).get(n) for n in UPPER_OUT}

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    t = obs_times[vis]
    draw(ax1, obs_vis, {}, f"Observed full body  t={t:.2f}s frame={vis}")
    draw(ax2, obs_vis, est_vis, f"Observed + 8 arm estimates  t={t:.2f}s")
    fig.suptitle(
        "Maximum honest coverage (Y-up metres). Orange = LSTM estimates, not anatomy.\n"
        "Lower LSTM not run: four distinct toe 3D landmarks are missing."
    )
    fig.tight_layout()
    fig.savefig(OUT / "viz_observed_vs_estimated.png", dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    for j, (elev, azim) in enumerate(((18, -70), (12, 25), (75, -90))):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        draw(ax, obs_vis, est_vis, f"elev {elev} az {azim}")
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
            f = idxs[k]
            o = {
                n: observed[f][n]["xyz_yup"]
                for n in POSE_33
                if n in observed[f] and observed[f][n]["xyz_yup"] is not None
            }
            e = {n: estimates.get(f, {}).get(n) for n in UPPER_OUT}
            draw(ax, o, e, f"quiet t={obs_times[f]:.2f}s")

        anim = FuncAnimation(fig, update, frames=len(idxs), interval=220)
        anim.save(str(OUT / "viz_quiet_window.gif"), writer=PillowWriter(fps=5))
        plt.close(fig)

    # Rotatable HTML
    payload = {
        "t": t,
        "frame": vis,
        "obs": {"x": [], "y": [], "z": [], "text": []},
        "est": {"x": [], "y": [], "z": [], "text": []},
        "bones": [],
    }
    for name, p in obs_vis.items():
        payload["obs"]["x"].append(float(p[0]))
        payload["obs"]["y"].append(float(p[1]))
        payload["obs"]["z"].append(float(p[2]))
        payload["obs"]["text"].append(name)
    for name, p in est_vis.items():
        if p is None:
            continue
        payload["est"]["x"].append(float(p[0]))
        payload["est"]["y"].append(float(p[1]))
        payload["est"]["z"].append(float(p[2]))
        payload["est"]["text"].append(name)
    for a, b in BONES:
        if a in obs_vis and b in obs_vis:
            pa, pb = obs_vis[a], obs_vis[b]
            payload["bones"].append(
                {
                    "x": [float(pa[0]), float(pb[0]), None],
                    "y": [float(pa[1]), float(pb[1]), None],
                    "z": [float(pa[2]), float(pb[2]), None],
                }
            )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Max coverage Y-up m</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body>
<p>Drag to rotate. Blue = observed MediaPipe+depth. Orange = upper LSTM <em>estimates</em>. Lower LSTM did not run.</p>
<p>t={t:.4f}s frame={vis}. OpenSim Y-up metres.</p>
<div id="p" style="width:100%;height:720px"></div>
<script>
const d={json.dumps(payload)};
const traces=[];
for (const b of d.bones) traces.push({{type:'scatter3d',mode:'lines',x:b.x,y:b.y,z:b.z,line:{{color:'#1f4e79',width:4}},showlegend:false,hoverinfo:'skip'}});
traces.push({{type:'scatter3d',mode:'markers',x:d.obs.x,y:d.obs.y,z:d.obs.z,text:d.obs.text,name:'observed',marker:{{size:4,color:'#1f4e79'}}}});
traces.push({{type:'scatter3d',mode:'markers',x:d.est.x,y:d.est.y,z:d.est.z,text:d.est.text,name:'estimated LSTM upper',marker:{{size:6,color:'#e67e22',symbol:'diamond'}}}});
Plotly.newPlot('p', traces, {{scene:{{xaxis:{{title:'X m'}},yaxis:{{title:'Y m (up)'}},zaxis:{{title:'Z m'}},aspectmode:'data'}},margin:{{l:0,r:0,t:20,b:0}}}}, {{responsive:true}});
</script></body></html>"""
    (OUT / "viz_rotatable_3d.html").write_text(html, encoding="utf-8")

    inventory = {
        "coord_frame": "opensim_yup_metre (X right, Y up= -Y_cam, Z forward)",
        "observed_mediapipe_pose33": POSE_33,
        "observed_hands_extras_in_this_csv": extra_names,
        "upper_lstm_required_inputs": UPPER_IN,
        "upper_lstm_plus_hip_centering": UPPER_IN + ["Hip"],
        "upper_lstm_outputs": UPPER_OUT,
        "lower_lstm_required_inputs": LOWER_IN,
        "lower_lstm_outputs": LOWER_OUT,
        "input_status": {
            "directly_observed_then_renamed": {
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
            },
            "proxies": {"Neck": "mid-shoulders", "Hip": "mid-hips"},
            "not_a_defensible_map": {
                "left_foot_index": "not LBigToe or LSmallToe",
                "right_foot_index": "not RBigToe or RSmallToe",
            },
            "missing_required_lower_inputs": ["RBigToe", "LBigToe", "RSmallToe", "LSmallToe"],
        },
        "anatomical_markers_neither_model_provides": ANATOMY_NEITHER_MODEL,
        "pose2sim_blazepose_note": (
            "Installed Pose2Sim BLAZEPOSE skeleton names MediaPipe ids 31/32 as LBigToe/RBigToe "
            "and has no SmallToe. That is a skeleton alias, not four observed toe landmarks. "
            "v0.10.49 augmenter does not synthesise SmallToe; it skips v0.3_lower."
        ),
        "n_observed_person_frames": len(frames),
        "n_upper_estimate_frames": n_est_frames,
        "visualization_frame": vis,
        "visualization_time_sec": t,
    }
    (OUT / "marker_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    quality = {
        "quiet_window_sec": [QUIET_T0, QUIET_T1],
        "n_quiet_frames": len(quiet),
        "observed_segment_lengths_m": quiet_stats,
        "estimated_pair_lengths_m": pair_stats,
        "quiet_joint_step_m": jump_stats,
        "interpretation": (
            "Stable trajectories are not anatomical accuracy. Pair lengths of LSTM "
            "lateral/medial estimates remain variable."
        ),
    }
    (OUT / "quality_checks.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print("vis frame", vis, "t", t, "n_obs_frames", len(frames), "n_est", n_est_frames)


if __name__ == "__main__":
    main()
