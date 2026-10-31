"""Combined rest-of-body visualization: observed + upper LSTM only.

Writes a NEW folder. Does not overwrite prior experiment outputs.
Does not run the lower LSTM or invent toes.
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
OUT = PARENT / "rest_of_body"
QUIET_T0, QUIET_T1 = 43.0532, 62.9664

POSE_BODY = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
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
POSE_HEAD = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
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
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("mouth_left", "mouth_right"),
]


def cam_to_yup(x, y, z):
    return np.array([x, -y, z], dtype=float)


def load_observed(path: Path):
    by_frame: dict[int, dict] = {}
    times: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            times[frame] = float(row["time_sec"])
            xs, ys, zs = row["x_m"].strip(), row["y_m"].strip(), row["z_m"].strip()
            rec = by_frame.setdefault(frame, {})
            if xs == "" or ys == "" or zs == "":
                rec[row["joint"]] = None
            else:
                rec[row["joint"]] = cam_to_yup(float(xs), float(ys), float(zs))
    return by_frame, times


def load_estimates(path: Path):
    by_frame: dict[int, dict] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rec = by_frame.setdefault(int(row["frame"]), {})
            xs, ys, zs = row["x_m"].strip(), row["y_m"].strip(), row["z_m"].strip()
            rec[row["marker"]] = None if xs == "" else np.array([float(xs), float(ys), float(zs)])
    return by_frame


def proxies(rec: dict):
    out = {}
    ls, rs = rec.get("left_shoulder"), rec.get("right_shoulder")
    if ls is not None and rs is not None:
        out["Neck"] = 0.5 * (ls + rs)
    lh, rh = rec.get("left_hip"), rec.get("right_hip")
    if lh is not None and rh is not None:
        out["Hip"] = 0.5 * (lh + rh)
    return out


def scatter_group(ax, pts, **kwargs):
    finite = [p for p in pts.values() if p is not None]
    if not finite:
        return
    P = np.vstack(finite)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], **kwargs)


def draw(ax, rec, est, title):
    obs = {n: rec.get(n) for n in POSE_BODY + POSE_HEAD if rec.get(n) is not None}
    extra = {
        n: p
        for n, p in rec.items()
        if p is not None and n not in obs and n not in ("Neck", "Hip")
    }
    prox = proxies(rec)
    for a, b in BONES:
        pa, pb = rec.get(a), rec.get(b)
        if pa is not None and pb is not None:
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="#1f4e79", lw=1.15)
    scatter_group(ax, obs, c="#1f4e79", s=22, label="observed (MediaPipe+depth)")
    scatter_group(ax, extra, c="#5b8db8", s=10, label="observed extras (hands/face)")
    scatter_group(ax, prox, c="#2ecc71", s=48, marker="s", label="proxy (Neck/Hip midpoints)")
    est_pts = {n: est.get(n) for n in UPPER_OUT if est.get(n) is not None}
    scatter_group(ax, est_pts, c="#e67e22", s=40, marker="^", label="estimated (upper LSTM)")
    ax.set_title(title)
    ax.set_xlabel("X m")
    ax.set_ylabel("Y m (up)")
    ax.set_zlabel("Z m")
    ax.legend(loc="upper left", fontsize=6)
    pts = [p for p in list(obs.values()) + list(prox.values()) + list(est_pts.values()) if p is not None]
    if pts:
        P = np.vstack(pts)
        c = P.mean(axis=0)
        r = max(0.75, float(np.max(np.linalg.norm(P - c, axis=1))) * 1.2)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observed, times = load_observed(CSV_PATH)
    estimates = load_estimates(EST_CSV)
    frames = sorted(observed)
    quiet = [f for f in frames if QUIET_T0 <= times[f] <= QUIET_T1]
    vis = quiet[len(quiet) // 2] if quiet else frames[len(frames) // 2]
    t = times[vis]

    csv_path = OUT / "combined_markers_yup.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "time_sec", "marker", "x_m", "y_m", "z_m", "coord_frame", "provenance"],
        )
        writer.writeheader()

        def write_pt(frame, name, p, provenance):
            writer.writerow(
                {
                    "frame": frame,
                    "time_sec": f"{times[frame]:.4f}",
                    "marker": name,
                    "x_m": "" if p is None else f"{p[0]:.6f}",
                    "y_m": "" if p is None else f"{p[1]:.6f}",
                    "z_m": "" if p is None else f"{p[2]:.6f}",
                    "coord_frame": "opensim_yup_metre",
                    "provenance": provenance if p is not None else "invalid_depth",
                }
            )

        n_obs_xyz = n_proxy = n_est = 0
        extras = set()
        for frame in frames:
            rec = observed[frame]
            for name, p in rec.items():
                if name in POSE_BODY + POSE_HEAD:
                    prov = "observed_mediapipe_realsense"
                else:
                    extras.add(name)
                    prov = "observed_mediapipe_hands_or_face_extra"
                if p is not None:
                    n_obs_xyz += 1
                write_pt(frame, name, p, prov)
            for name, p in proxies(rec).items():
                n_proxy += 1
                write_pt(
                    frame,
                    name,
                    p,
                    "proxy_mid_shoulders" if name == "Neck" else "proxy_mid_hips",
                )
            est = estimates.get(frame, {})
            for name in UPPER_OUT:
                p = est.get(name)
                if p is not None:
                    n_est += 1
                write_pt(frame, name, p, "estimated_pose2sim_lstm_v0.3_upper")

    coverage = []
    names_obs = sorted({n for rec in observed.values() for n in rec})
    for name in names_obs:
        n_xyz = sum(1 for rec in observed.values() if rec.get(name) is not None)
        n_pres = sum(1 for rec in observed.values() if name in rec)
        kind = "observed"
        note = "MediaPipe model landmark + RealSense depth. Not ISB anatomy."
        if "foot_index" in name:
            note = "Observed toe-ray model point. Not BigToe or SmallToe."
        coverage.append(
            {
                "marker": name,
                "class": kind,
                "n_present": n_pres,
                "n_finite_xyz": n_xyz,
                "included_in_visualization": True,
                "note": note,
            }
        )
    for name in ("Neck", "Hip"):
        coverage.append(
            {
                "marker": name,
                "class": "proxy",
                "n_present": "",
                "n_finite_xyz": n_proxy // 2,
                "included_in_visualization": True,
                "note": "Midpoint proxy for LSTM features only. Not C7 / not ASIS/HJC.",
            }
        )
    for name in UPPER_OUT:
        n_xyz = sum(1 for rec in estimates.values() if rec.get(name) is not None)
        coverage.append(
            {
                "marker": name,
                "class": "estimated_upper_lstm",
                "n_present": len(estimates),
                "n_finite_xyz": n_xyz,
                "included_in_visualization": True,
                "note": "Stanford v0.3 upper LSTM. Not epicondyle or styloid.",
            }
        )
    for name in LOWER_OUT:
        coverage.append(
            {
                "marker": name,
                "class": "unavailable",
                "n_present": 0,
                "n_finite_xyz": 0,
                "included_in_visualization": False,
                "note": "Lower LSTM not run: toes postponed; no supported subset without them.",
            }
        )
    cov_path = OUT / "marker_coverage.csv"
    with cov_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["marker", "class", "n_present", "n_finite_xyz", "included_in_visualization", "note"],
        )
        writer.writeheader()
        writer.writerows(coverage)

    rec = observed[vis]
    est = estimates.get(vis, {})
    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    draw(ax1, rec, {}, f"Observed rest of body  t={t:.2f}s  frame={vis}")
    draw(ax2, rec, est, f"Observed + 8 arm estimates  t={t:.2f}s")
    fig.suptitle(
        "Rest of body from existing recording (Y-up metres). "
        "Blue=observed, green squares=proxy, orange triangles=estimated.\n"
        "No lower/torso LSTM: four toe inputs postponed. MediaPipe hips ≠ ASIS/HJC."
    )
    fig.tight_layout()
    fig.savefig(OUT / "viz_overview.png", dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    for j, (elev, azim) in enumerate(((18, -70), (12, 25), (75, -90))):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        draw(ax, rec, est, f"elev {elev} az {azim}")
        ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(OUT / "viz_viewpoints.png", dpi=140)
    plt.close(fig)

    if len(quiet) >= 4:
        idxs = quiet[:: max(1, len(quiet) // 20)]
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        def update(k):
            ax.cla()
            f = idxs[k]
            draw(ax, observed[f], estimates.get(f, {}), f"t={times[f]:.2f}s")

        anim = FuncAnimation(fig, update, frames=len(idxs), interval=220)
        anim.save(str(OUT / "viz_sequence.gif"), writer=PillowWriter(fps=5))
        plt.close(fig)

    payload = {"t": t, "frame": vis, "obs": [], "extra": [], "proxy": [], "est": [], "bones": []}
    for name in POSE_BODY + POSE_HEAD:
        p = rec.get(name)
        if p is not None:
            payload["obs"].append({"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "name": name})
    for name, p in rec.items():
        if p is None or name in POSE_BODY or name in POSE_HEAD:
            continue
        payload["extra"].append({"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "name": name})
    for name, p in proxies(rec).items():
        payload["proxy"].append({"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "name": name})
    for name in UPPER_OUT:
        p = est.get(name)
        if p is not None:
            payload["est"].append({"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "name": name})
    for a, b in BONES:
        pa, pb = rec.get(a), rec.get(b)
        if pa is not None and pb is not None:
            payload["bones"].append(
                {
                    "x": [float(pa[0]), float(pb[0]), None],
                    "y": [float(pa[1]), float(pb[1]), None],
                    "z": [float(pa[2]), float(pb[2]), None],
                }
            )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Rest of body Y-up m</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body>
<p>Drag to rotate. Blue circles = observed MediaPipe+depth. Light blue = hands/face extras.
Green squares = Neck/Hip <em>proxies</em>. Orange diamonds = upper LSTM <em>estimates</em> (not anatomy).
No pelvis/torso LSTM outputs.</p>
<p>t={t:.4f}s frame={vis}. OpenSim Y-up metres (X right, Y up, Z forward).</p>
<div id="p" style="width:100%;height:740px"></div>
<script>
const d={json.dumps(payload)};
const traces=[];
for (const b of d.bones) traces.push({{type:'scatter3d',mode:'lines',x:b.x,y:b.y,z:b.z,line:{{color:'#1f4e79',width:4}},showlegend:false,hoverinfo:'skip'}});
function g(arr,name,color,sym,size){{
  return {{type:'scatter3d',mode:'markers',x:arr.map(p=>p.x),y:arr.map(p=>p.y),z:arr.map(p=>p.z),text:arr.map(p=>p.name),name:name,marker:{{size:size,color:color,symbol:sym}}}};
}}
traces.push(g(d.obs,'observed','#1f4e79','circle',5));
traces.push(g(d.extra,'observed extras','#5b8db8','circle',3));
traces.push(g(d.proxy,'proxy Neck/Hip','#2ecc71','square',6));
traces.push(g(d.est,'estimated upper LSTM','#e67e22','diamond',6));
Plotly.newPlot('p', traces, {{scene:{{xaxis:{{title:'X m'}},yaxis:{{title:'Y m (up)'}},zaxis:{{title:'Z m'}},aspectmode:'data'}},margin:{{l:0,r:0,t:24,b:0}}}}, {{responsive:true}});
</script></body></html>"""
    (OUT / "viz_rotatable_3d.html").write_text(html, encoding="utf-8")

    model_check = {
        "installed_models": ["LSTM/v0.3_lower", "LSTM/v0.3_upper"],
        "lower_onnx_input_features": 47,
        "lower_feature_breakdown": "15 named markers * 3 XYZ + height + mass",
        "lower_required_markers": 15,
        "includes_four_toes": True,
        "supported_subset_without_toes": False,
        "official_behavior_if_toes_missing": "skip v0.3_lower",
        "other_torso_or_pelvis_models_installed": False,
        "upper_already_ran": True,
        "lower_ran_this_stage": False,
        "height_mass_used_this_stage": "not required; no new inference",
    }
    (OUT / "model_path_check.json").write_text(json.dumps(model_check, indent=2), encoding="utf-8")

    report = f"""# Rest-of-body visualization (toes postponed)

No new lower/torso/pelvis estimates. Installed augmenter has only `v0.3_upper` and `v0.3_lower`. Lower ONNX input is **47** features (15 markers × 3 + height + mass), including BigToe/SmallToe. Official code skips lower if those names are absent. There is no documented subset and no other installed model.

## Included

**Observed (MediaPipe + RealSense, Y-up metres):** shoulders, elbows, wrists, hips, knees, ankles, heels, `foot_index`, head/face Pose points, and Hands extras when XYZ is finite.

**Proxy (green squares):** `Neck` = mid-shoulders; `Hip` = mid-hips. Not C7 / not ASIS / not HJC.

**Estimated (orange, 8):** `{", ".join(UPPER_OUT)}` from the existing upper LSTM run.

## Not generated

All 35 lower LSTM outputs (ASIS, PSIS, C7, HJC, thigh/shank clusters, medial/lateral knees and ankles, toe/5meta/calc, shoulder_study).

## Files

- `viz_overview.png`
- `viz_viewpoints.png`
- `viz_sequence.gif`
- `viz_rotatable_3d.html`
- `combined_markers_yup.csv` ({n_obs_xyz} observed XYZ rows, {n_proxy} proxy, {n_est} estimated)
- `marker_coverage.csv`
- `model_path_check.json`

Convention: OpenSim Y-up metres. Estimates are not verified anatomy.
"""
    (OUT / "coverage_report.md").write_text(report, encoding="utf-8")
    print("out", OUT)
    print("frame", vis, "t", t, "obs_xyz", n_obs_xyz, "proxy", n_proxy, "est", n_est)


if __name__ == "__main__":
    main()
