"""Read-only analysis of data/diagnostics/20261012-150535. Writes only into this folder."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(r"C:\Projects\motion__analysis__\data\diagnostics\20261012-150535")
OUT = Path(__file__).resolve().parent

frames = pd.read_csv(SRC / "frames.csv")
lm = pd.read_csv(SRC / "landmarks.csv")
summary = json.loads((SRC / "summary.json").read_text(encoding="utf-8"))

for c in [
    "raw_x_m",
    "raw_y_m",
    "raw_z_m",
    "visibility",
    "t_frame_sec",
    "t_pred_input_sec",
    "t_pred_done_sec",
    "infer_latency_ms",
    "pred_lag_ms",
]:
    if c in lm.columns:
        lm[c] = pd.to_numeric(lm[c], errors="coerce")
for c in [
    "t_frame_sec",
    "process_ms",
    "pred_lag_ms",
    "infer_latency_ms",
    "fps",
    "t_pred_input_sec",
    "t_pred_done_sec",
]:
    if c in frames.columns:
        frames[c] = pd.to_numeric(frames[c], errors="coerce")

FEET = [
    "left_ankle",
    "left_heel",
    "left_foot_index",
    "right_ankle",
    "right_heel",
    "right_foot_index",
]
CTRL = ["left_knee", "right_knee", "left_hip", "right_hip"]
OBS_ARM = ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]


def land(name: str) -> pd.DataFrame:
    return lm[lm["landmark"] == name].sort_values("frame")


def series_xyz(name: str) -> pd.DataFrame:
    d = land(name)
    return d.loc[d["xyz_missing"].astype(int) == 0, ["frame", "t_frame_sec", "raw_x_m", "raw_y_m", "raw_z_m", "visibility"]].copy()


def f2f(xyz: pd.DataFrame) -> dict:
    xyz = xyz.dropna()
    if len(xyz) < 2:
        return {}
    arr = xyz.to_numpy()
    d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return {
        "n": int(len(arr)),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "n_jump_5cm": int((d > 0.05).sum()),
        "n_jump_10cm": int((d > 0.10).sum()),
        "n_jump_20cm": int((d > 0.20).sum()),
    }


foot_rows = []
for name in FEET + CTRL:
    d = land(name)
    vis = d["visibility"]
    miss = d["xyz_missing"].astype(int)
    valid = d[miss == 0]
    foot_rows.append(
        {
            "landmark": name,
            "n_rows": int(len(d)),
            "n_unique_frames": int(d["frame"].nunique()),
            "n_xyz_valid": int((miss == 0).sum()),
            "n_xyz_missing": int((miss == 1).sum()),
            "invalid_pct": float(100.0 * (miss == 1).mean()) if len(d) else None,
            "vis_min": float(vis.min()) if len(d) else None,
            "vis_median": float(vis.median()) if len(d) else None,
            "vis_max": float(vis.max()) if len(d) else None,
            "n_2d_present_xyz_missing": int(((miss == 1) & vis.notna() & (vis > 0)).sum()),
            "n_vis_ge_0.5_xyz_missing": int(((miss == 1) & (vis >= 0.5)).sum()),
            "n_vis_lt_0.5_xyz_missing": int(((miss == 1) & (vis < 0.5)).sum()),
            "valid_frames": valid["frame"].astype(int).tolist(),
            "valid_z_m": [float(z) for z in valid["raw_z_m"].tolist()],
            "valid_x_m": [float(z) for z in valid["raw_x_m"].tolist()],
            "valid_y_m": [float(z) for z in valid["raw_y_m"].tolist()],
            "valid_vis": [float(z) for z in valid["visibility"].tolist()],
        }
    )

valid_context = []
for name in FEET:
    d = land(name)
    for _, r in d[d["xyz_missing"].astype(int) == 0].iterrows():
        fr = int(r["frame"])
        ctx = {
            "foot": name,
            "frame": fr,
            "foot_z": float(r["raw_z_m"]),
            "foot_y": float(r["raw_y_m"]),
            "foot_x": float(r["raw_x_m"]),
            "vis": float(r["visibility"]),
        }
        for other in CTRL:
            o = land(other)
            hit = o[o["frame"] == fr]
            if len(hit) and int(hit.iloc[0]["xyz_missing"]) == 0:
                ctx[f"{other}_z"] = float(hit.iloc[0]["raw_z_m"])
                ctx[f"{other}_y"] = float(hit.iloc[0]["raw_y_m"])
        valid_context.append(ctx)

arm_names = OBS_ARM + [
    "L_lelbow_study",
    "L_melbow_study",
    "L_elbow_centre",
    "L_lwrist_study",
    "L_mwrist_study",
    "L_wrist_centre",
    "r_lelbow_study",
    "r_melbow_study",
    "r_elbow_centre",
    "r_lwrist_study",
    "r_mwrist_study",
    "r_wrist_centre",
]
arm_stats = {}
for name in arm_names:
    d = land(name)
    if d.empty:
        continue
    xyz = d.loc[d["xyz_missing"].astype(int) == 0, ["raw_x_m", "raw_y_m", "raw_z_m"]]
    st = {
        "class": str(d["class"].iloc[0]),
        "n_rows": int(len(d)),
        "n_valid": int((d["xyz_missing"].astype(int) == 0).sum()),
        "n_missing": int((d["xyz_missing"].astype(int) == 1).sum()),
        "std": {a: float(xyz[f"raw_{a}_m"].std(ddof=0)) if len(xyz) else None for a in "xyz"},
        "range": {a: float(xyz[f"raw_{a}_m"].max() - xyz[f"raw_{a}_m"].min()) if len(xyz) else None for a in "xyz"},
        "f2f": f2f(xyz),
        "mean_xyz": {a: float(xyz[f"raw_{a}_m"].mean()) if len(xyz) else None for a in "xyz"},
    }
    if "predicted" in st["class"] or "geometric" in st["class"]:
        lag = d["pred_lag_ms"].dropna()
        st["pred_lag_ms"] = {
            "n": int(len(lag)),
            "mean": float(lag.mean()) if len(lag) else None,
            "median": float(lag.median()) if len(lag) else None,
            "p95": float(np.percentile(lag, 95)) if len(lag) else None,
            "max": float(lag.max()) if len(lag) else None,
        }
        inf = d["infer_latency_ms"].dropna()
        st["infer_latency_ms"] = {
            "mean": float(inf.mean()) if len(inf) else None,
            "max": float(inf.max()) if len(inf) else None,
        }
    arm_stats[name] = st

pairs = [
    ("left_elbow", "L_elbow_centre"),
    ("right_elbow", "r_elbow_centre"),
    ("left_wrist", "L_wrist_centre"),
    ("right_wrist", "r_wrist_centre"),
    ("left_elbow", "L_lelbow_study"),
    ("left_elbow", "L_melbow_study"),
    ("right_elbow", "r_lelbow_study"),
    ("right_elbow", "r_melbow_study"),
    ("left_wrist", "L_lwrist_study"),
    ("left_wrist", "L_mwrist_study"),
    ("right_wrist", "r_lwrist_study"),
    ("right_wrist", "r_mwrist_study"),
]
obs_pred = {}
for a, b in pairs:
    da, db = series_xyz(a), series_xyz(b)
    m = da.merge(db, on="frame", suffixes=("_obs", "_pred"))
    if m.empty:
        obs_pred[f"{a}__{b}"] = {"n": 0}
        continue
    dist = np.sqrt(
        (m["raw_x_m_obs"] - m["raw_x_m_pred"]) ** 2
        + (m["raw_y_m_obs"] - m["raw_y_m_pred"]) ** 2
        + (m["raw_z_m_obs"] - m["raw_z_m_pred"]) ** 2
    )
    obs_pred[f"{a}__{b}"] = {
        "n": int(len(m)),
        "mean_dist_m": float(dist.mean()),
        "median_dist_m": float(dist.median()),
        "p95_dist_m": float(np.percentile(dist, 95)),
        "max_dist_m": float(dist.max()),
        "mean_dx": float((m["raw_x_m_pred"] - m["raw_x_m_obs"]).mean()),
        "mean_dy": float((m["raw_y_m_pred"] - m["raw_y_m_obs"]).mean()),
        "mean_dz": float((m["raw_z_m_pred"] - m["raw_z_m_obs"]).mean()),
        "corr_x": float(m["raw_x_m_obs"].corr(m["raw_x_m_pred"])) if len(m) > 2 else None,
        "corr_y": float(m["raw_y_m_obs"].corr(m["raw_y_m_pred"])) if len(m) > 2 else None,
        "corr_z": float(m["raw_z_m_obs"].corr(m["raw_z_m_pred"])) if len(m) > 2 else None,
    }

pred_upper = lm[lm["class"] == "predicted_upper"]
pred_span = {
    "first_obs_frame": int(lm[lm["class"] == "observed"]["frame"].min()),
    "last_obs_frame": int(lm[lm["class"] == "observed"]["frame"].max()),
    "first_pred_frame": int(pred_upper["frame"].min()) if len(pred_upper) else None,
    "last_pred_frame": int(pred_upper["frame"].max()) if len(pred_upper) else None,
    "n_obs_landmark_types": int(lm[lm["class"] == "observed"]["landmark"].nunique()),
    "n_pred_landmark_types": int(pred_upper["landmark"].nunique()) if len(pred_upper) else 0,
}

pm = frames["process_ms"].dropna()
perf = {
    "n_frames": int(len(frames)),
    "dropped_max": int(frames["dropped_before"].max()) if "dropped_before" in frames else None,
    "t_span_sec": float(frames["t_frame_sec"].max() - frames["t_frame_sec"].min()),
    "dt_median": float(frames["t_frame_sec"].diff().median()),
    "dt_mean": float(frames["t_frame_sec"].diff().mean()),
    "process_ms_mean": float(pm.mean()),
    "process_ms_median": float(pm.median()),
    "process_ms_p95": float(np.percentile(pm, 95)),
    "process_ms_max": float(pm.max()),
    "process_ms_min": float(pm.min()),
    "implied_processing_fps": float(1000.0 / pm.mean()) if pm.mean() else None,
    "wall_fps_summary": summary.get("wall_fps"),
    "pred_lag_ms_mean": float(frames["pred_lag_ms"].mean()) if frames["pred_lag_ms"].notna().any() else None,
    "pred_lag_ms_p95": float(np.percentile(frames["pred_lag_ms"].dropna(), 95)) if frames["pred_lag_ms"].notna().any() else None,
    "infer_latency_ms_mean": float(frames["infer_latency_ms"].mean()) if frames["infer_latency_ms"].notna().any() else None,
}

payload = {
    "source_recording": str(SRC),
    "foot": foot_rows,
    "valid_foot_context": valid_context,
    "arm_stats": arm_stats,
    "obs_vs_pred": obs_pred,
    "pred_span": pred_span,
    "perf": perf,
    "classes": lm.groupby("class")["landmark"].nunique().to_dict(),
    "note_no_upx": "landmarks.csv does not store u_px/v_px; 2D presence inferred from row+visibility.",
}
(OUT / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=140)
    plt.close()


fig, ax = plt.subplots(figsize=(9, 4.2))
names = [r["landmark"] for r in foot_rows]
pcts = [r["invalid_pct"] for r in foot_rows]
colors = ["#c44747" if n in FEET else "#4aa3ff" for n in names]
ax.bar(names, pcts, color=colors)
ax.set_ylabel("Invalid XYZ (%)")
ax.set_title("Invalid camera-XYZ rate (observed feet vs knee/hip controls)")
ax.set_ylim(0, 110)
plt.xticks(rotation=30, ha="right")
ax.set_xlabel("Landmark")
savefig("fig_A_invalid_xyz_feet.png")

fig, ax = plt.subplots(figsize=(8, 4))
for name in FEET:
    d = land(name)
    miss = d[d["xyz_missing"].astype(int) == 1]
    ax.plot(miss["frame"], miss["visibility"], ".", ms=2, label=name, alpha=0.6)
ax.axhline(0.5, color="gray", ls="--", lw=1, label="min_visibility 0.5")
ax.set_xlabel("Frame")
ax.set_ylabel("MediaPipe visibility")
ax.set_title("Foot landmark visibility on frames with invalid XYZ")
ax.legend(fontsize=8, ncol=2)
savefig("fig_A_foot_visibility_when_xyz_missing.png")

layout = [
    ("left_elbow", "L_elbow_centre", "left elbow Z"),
    ("right_elbow", "r_elbow_centre", "right elbow Z"),
    ("left_wrist", "L_wrist_centre", "left wrist Z"),
    ("right_wrist", "r_wrist_centre", "right wrist Z"),
]
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, (o, p, title) in zip(axes.ravel(), layout):
    a, b = series_xyz(o), series_xyz(p)
    ax.plot(a["frame"], a["raw_z_m"], lw=1.0, label="observed", color="#4aa3ff")
    ax.plot(b["frame"], b["raw_z_m"], lw=1.0, label="LSTM centre", color="#c000c0")
    ax.set_title(title)
    ax.set_ylabel("Z (m)")
    ax.legend(fontsize=8)
axes[1, 0].set_xlabel("Frame")
axes[1, 1].set_xlabel("Frame")
fig.suptitle("Observed vs predicted centre depth (camera Z, metres)")
savefig("fig_B_obs_vs_pred_z.png")

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, (o, p, title) in zip(axes.ravel(), layout):
    a, b = series_xyz(o), series_xyz(p)
    ax.plot(a["frame"], a["raw_x_m"], lw=1.0, label="observed X", color="#4aa3ff")
    ax.plot(b["frame"], b["raw_x_m"], lw=1.0, label="LSTM centre X", color="#c000c0")
    ax.set_title(title.replace("Z", "X"))
    ax.set_ylabel("X (m)")
    ax.legend(fontsize=8)
axes[1, 0].set_xlabel("Frame")
axes[1, 1].set_xlabel("Frame")
fig.suptitle("Observed vs predicted centre X (camera metres, +X right)")
savefig("fig_B_obs_vs_pred_x.png")

fig, ax = plt.subplots(figsize=(8, 4.2))
labels, means, p95s = [], [], []
for name in OBS_ARM + ["L_elbow_centre", "r_elbow_centre", "L_wrist_centre", "r_wrist_centre"]:
    st = arm_stats.get(name, {}).get("f2f") or {}
    if not st:
        continue
    labels.append(name)
    means.append(st["mean"] * 1000)
    p95s.append(st["p95"] * 1000)
x = np.arange(len(labels))
ax.bar(x - 0.18, means, 0.35, label="mean f2f (mm)")
ax.bar(x + 0.18, p95s, 0.35, label="p95 f2f (mm)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Displacement (mm)")
ax.set_title("Frame-to-frame displacement, observed vs LSTM centres")
ax.legend()
savefig("fig_B_f2f_displacement.png")

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(frames["frame"], frames["process_ms"], lw=0.8, color="#2f9e6a")
ax.axhline(frames["process_ms"].median(), color="gray", ls="--", label=f"median {frames['process_ms'].median():.1f} ms")
ax.set_xlabel("Frame")
ax.set_ylabel("process_ms")
ax.set_title("Per-frame processing time (diagnostics loop)")
ax.legend()
savefig("fig_C_process_ms.png")

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(frames["frame"], frames["pred_lag_ms"], lw=0.8, color="#c000c0")
ax.set_xlabel("Frame")
ax.set_ylabel("pred_lag_ms")
ax.set_title("Prediction lag (camera frame time minus LSTM input-frame time)")
savefig("fig_C_pred_lag_ms.png")

print(OUT)
print(json.dumps({"perf": perf, "pred_span": pred_span}, indent=2))
