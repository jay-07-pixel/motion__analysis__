"""Read-only audit of data/diagnostics/20261012-155914. Does not change production data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(r"c:\Projects\motion__analysis__\data\diagnostics\20261012-155914")
OUT = Path(__file__).resolve().parent
W, H = 1280, 720
MIN_M, MAX_M = 0.3, 6.0
MIN_VIS = 0.5
FOOT = [
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]
BODY = [
    "nose",
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
    *FOOT,
]
PAIRS = [
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]
PRED_PAIRS = [
    ("left_elbow", "L_elbow_centre", "L_lelbow_study", "L_melbow_study"),
    ("right_elbow", "r_elbow_centre", "r_lelbow_study", "r_melbow_study"),
    ("left_wrist", "L_wrist_centre", "L_lwrist_study", "L_mwrist_study"),
    ("right_wrist", "r_wrist_centre", "r_lwrist_study", "r_mwrist_study"),
]


def num(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


frames = pd.read_csv(SRC / "frames.csv")
lm = pd.read_csv(SRC / "landmarks.csv")
frames = num(
    frames,
    [
        "frame",
        "t_frame_sec",
        "process_ms",
        "n_observed",
        "n_observed_xyz",
        "n_predicted",
        "infer_latency_ms",
        "pred_lag_ms",
        "fps",
        "dropped_before",
    ],
)
lm = num(
    lm,
    [
        "frame",
        "t_frame_sec",
        "raw_x_m",
        "raw_y_m",
        "raw_z_m",
        "visibility",
        "u_px",
        "v_px",
        "depth_sample_m",
        "depth_n_valid_in_window",
        "in_image",
        "depth_valid",
        "xyz_missing",
        "t_pred_input_sec",
        "pred_lag_ms",
        "infer_latency_ms",
    ],
)

n_frames = int(frames["frame"].nunique())
t0 = float(frames["t_frame_sec"].min())
t1 = float(frames["t_frame_sec"].max())
obs = lm[lm["class"] == "observed"].copy()
pred = lm[lm["class"].isin(["predicted_upper", "geometric_centre"])].copy()

# Per-frame motion from hips if present
hip = obs[obs["landmark"].isin(["left_hip", "right_hip"])].copy()
hip_xyz = hip.dropna(subset=["raw_x_m", "raw_y_m", "raw_z_m"])
hip_mid = (
    hip_xyz.groupby("frame")[["raw_x_m", "raw_y_m", "raw_z_m", "t_frame_sec"]]
    .mean()
    .reset_index()
    .sort_values("frame")
)
hip_mid["dxyz"] = np.sqrt(
    (hip_mid["raw_x_m"].diff()) ** 2
    + (hip_mid["raw_y_m"].diff()) ** 2
    + (hip_mid["raw_z_m"].diff()) ** 2
)
# rolling 1 s ~ 8 frames
win = 8
hip_mid["roll_d"] = hip_mid["dxyz"].rolling(win, min_periods=4).median()
hip_mid["roll_std_z"] = hip_mid["raw_z_m"].rolling(win, min_periods=4).std()
hip_mid["roll_std_x"] = hip_mid["raw_x_m"].rolling(win, min_periods=4).std()

# ankle v for framing
ank_v = (
    obs[obs["landmark"].isin(["left_ankle", "right_ankle", "left_foot_index", "right_foot_index"])]
    .groupby("frame")["v_px"]
    .max()
    .rename("foot_v_max")
)
merged_f = frames.merge(hip_mid[["frame", "dxyz", "roll_d", "roll_std_z", "raw_z_m"]], on="frame", how="left")
merged_f = merged_f.merge(ank_v, on="frame", how="left")

# Heuristic flags (documented, not assumed a priori)
# setup: sparse pose or no hips or foot_v very high / missing
# walk: hip dxyz persistently large
# still: roll_d small AND roll_std_z small AND hips present
still = (
    merged_f["roll_d"].notna()
    & (merged_f["roll_d"] < 0.015)
    & (merged_f["roll_std_z"] < 0.04)
    & (merged_f["n_observed"] >= 30)
)
# consecutive still runs
still_runs = []
run_start = None
for fr, ok in zip(merged_f["frame"], still.fillna(False)):
    if ok and run_start is None:
        run_start = int(fr)
    if (not ok) and run_start is not None:
        still_runs.append((run_start, int(fr) - 1))
        run_start = None
if run_start is not None:
    still_runs.append((run_start, int(merged_f["frame"].max())))
still_runs_n = [(a, b, b - a + 1) for a, b in still_runs]
longest_still = max(still_runs_n, key=lambda x: x[2]) if still_runs_n else None

# hip z trajectory to describe approach
z_by_frame = hip_mid.set_index("frame")["raw_z_m"]


def interval_stats(a, b, label):
    sl = merged_f[(merged_f["frame"] >= a) & (merged_f["frame"] <= b)]
    return {
        "label": label,
        "frames": [int(a), int(b)],
        "t_sec": [
            float(frames.loc[frames.frame == a, "t_frame_sec"].iloc[0]) if (frames.frame == a).any() else None,
            float(frames.loc[frames.frame == b, "t_frame_sec"].iloc[0]) if (frames.frame == b).any() else None,
        ],
        "n": int(len(sl)),
        "median_hip_dxyz_m": None if sl["dxyz"].dropna().empty else float(sl["dxyz"].median()),
        "p95_hip_dxyz_m": None if sl["dxyz"].dropna().empty else float(sl["dxyz"].quantile(0.95)),
        "median_hip_z_m": None if sl["raw_z_m"].dropna().empty else float(sl["raw_z_m"].median()),
        "median_process_ms": None if sl["process_ms"].dropna().empty else float(sl["process_ms"].median()),
        "median_n_observed": None if sl["n_observed"].dropna().empty else float(sl["n_observed"].median()),
        "median_foot_v": None if sl["foot_v_max"].dropna().empty else float(sl["foot_v_max"].median()),
        "max_foot_v": None if sl["foot_v_max"].dropna().empty else float(sl["foot_v_max"].max()),
    }


# Candidate regions from data, not a preset 60-499
# First hip XYZ frame
first_hip = int(hip_mid["frame"].min()) if len(hip_mid) else None
last_hip = int(hip_mid["frame"].max()) if len(hip_mid) else None
first_pred = int(pred["frame"].min()) if len(pred) else None

# Foot validity series
def foot_counts(name: str) -> dict:
    d = obs[obs["landmark"] == name]
    n = len(d)
    vis = d["visibility"]
    in_img = d["in_image"]
    nval = d["depth_n_valid_in_window"]
    zsamp = d["depth_sample_m"]
    miss = d["xyz_missing"]
    valid = (miss == 0) & d["raw_x_m"].notna()
    oob = in_img == 0
    hole = (in_img == 1) & (nval == 0)
    # positive window but rejected by min/max (sample not in [0.3,6])
    rejected_minmax = (
        (in_img == 1)
        & (nval > 0)
        & zsamp.notna()
        & ((zsamp < MIN_M) | (zsamp > MAX_M))
    )
    low_vis = vis < MIN_VIS
    # frames in whole recording without this landmark row
    present_frames = set(d["frame"].astype(int))
    all_frames = set(frames["frame"].astype(int))
    no_row = len(all_frames - present_frames)
    # Cross-check depth_valid vs xyz_missing
    disagree = int(((d["depth_valid"] == 1) != (miss == 0)).sum()) if "depth_valid" in d else None
    return {
        "landmark": name,
        "detected_rows": int(n),
        "frames_without_row": int(no_row),
        "valid_xyz": int(valid.sum()),
        "xyz_missing_flag": int((miss == 1).sum()),
        "in_image_0": int(oob.sum()),
        "in_image_1_n_valid_0": int(hole.sum()),
        "in_image_positive_depth_rejected_minmax": int(rejected_minmax.sum()),
        "visibility_lt_0.5": int(low_vis.sum()),
        "visibility_lt_0.5_and_valid_xyz": int((low_vis & valid).sum()),
        "visibility_ge_0.5_and_xyz_missing": int((~low_vis & (miss == 1)).sum()),
        "oob_and_xyz_missing": int((oob & (miss == 1)).sum()),
        "hole_and_xyz_missing": int((hole & (miss == 1)).sum()),
        "depth_valid_vs_xyz_missing_disagree": disagree,
        "u_min": None if d["u_px"].dropna().empty else float(d["u_px"].min()),
        "u_max": None if d["u_px"].dropna().empty else float(d["u_px"].max()),
        "v_min": None if d["v_px"].dropna().empty else float(d["v_px"].min()),
        "v_median": None if d["v_px"].dropna().empty else float(d["v_px"].median()),
        "v_max": None if d["v_px"].dropna().empty else float(d["v_px"].max()),
        "n_v_ge_720": int((d["v_px"] >= H).sum()),
        "n_v_gt_640": int((d["v_px"] > 640).sum()),
        "vis_median": None if vis.dropna().empty else float(vis.median()),
        "z_median_when_valid": None if d.loc[valid, "raw_z_m"].dropna().empty else float(d.loc[valid, "raw_z_m"].median()),
        "z_min_when_valid": None if d.loc[valid, "raw_z_m"].dropna().empty else float(d.loc[valid, "raw_z_m"].min()),
        "z_max_when_valid": None if d.loc[valid, "raw_z_m"].dropna().empty else float(d.loc[valid, "raw_z_m"].max()),
    }


foot_table = [foot_counts(n) for n in FOOT]

# Body XYZ quality
def series_quality(name: str, cls: str = "observed") -> dict:
    d = lm[(lm["landmark"] == name) & (lm["class"] == cls)].sort_values("frame")
    valid = d.dropna(subset=["raw_x_m", "raw_y_m", "raw_z_m"])
    if len(valid) < 2:
        return {"name": name, "class": cls, "n_valid": int(len(valid)), "n_rows": int(len(d))}
    xyz = valid[["raw_x_m", "raw_y_m", "raw_z_m"]].to_numpy()
    dxyz = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    fr = valid["frame"].to_numpy()
    gap = np.diff(fr)
    # only consecutive-frame steps
    step = dxyz[gap == 1] if np.any(gap == 1) else dxyz
    jumps5 = int((step > 0.05).sum())
    jumps10 = int((step > 0.10).sum())
    jumps20 = int((step > 0.20).sum())
    return {
        "name": name,
        "class": cls,
        "n_rows": int(len(d)),
        "n_valid": int(len(valid)),
        "mean_xyz": [float(x) for x in xyz.mean(0)],
        "std_xyz": [float(x) for x in xyz.std(0)],
        "min_xyz": [float(x) for x in xyz.min(0)],
        "max_xyz": [float(x) for x in xyz.max(0)],
        "range_xyz": [float(x) for x in (xyz.max(0) - xyz.min(0))],
        "mean_f2f_m": float(np.mean(step)) if len(step) else None,
        "median_f2f_m": float(np.median(step)) if len(step) else None,
        "p95_f2f_m": float(np.quantile(step, 0.95)) if len(step) else None,
        "max_f2f_m": float(np.max(step)) if len(step) else None,
        "jumps_gt_5cm": jumps5,
        "jumps_gt_10cm": jumps10,
        "jumps_gt_20cm": jumps20,
    }


body_q = [series_quality(n) for n in BODY]
pred_q = [
    series_quality(n, c)
    for n, c in [
        ("L_lelbow_study", "predicted_upper"),
        ("L_melbow_study", "predicted_upper"),
        ("L_lwrist_study", "predicted_upper"),
        ("L_mwrist_study", "predicted_upper"),
        ("r_lelbow_study", "predicted_upper"),
        ("r_melbow_study", "predicted_upper"),
        ("r_lwrist_study", "predicted_upper"),
        ("r_mwrist_study", "predicted_upper"),
        ("L_elbow_centre", "geometric_centre"),
        ("r_elbow_centre", "geometric_centre"),
        ("L_wrist_centre", "geometric_centre"),
        ("r_wrist_centre", "geometric_centre"),
    ]
]


def segment_lengths(a, b):
    aa = obs[obs.landmark == a][["frame", "raw_x_m", "raw_y_m", "raw_z_m"]].dropna()
    bb = obs[obs.landmark == b][["frame", "raw_x_m", "raw_y_m", "raw_z_m"]].dropna()
    m = aa.merge(bb, on="frame", suffixes=("_a", "_b"))
    if m.empty:
        return {"a": a, "b": b, "n": 0}
    L = np.sqrt(
        (m["raw_x_m_a"] - m["raw_x_m_b"]) ** 2
        + (m["raw_y_m_a"] - m["raw_y_m_b"]) ** 2
        + (m["raw_z_m_a"] - m["raw_z_m_b"]) ** 2
    )
    return {
        "a": a,
        "b": b,
        "n": int(len(L)),
        "median_m": float(np.median(L)),
        "std_m": float(np.std(L)),
        "min_m": float(np.min(L)),
        "max_m": float(np.max(L)),
        "p05_m": float(np.quantile(L, 0.05)),
        "p95_m": float(np.quantile(L, 0.95)),
        "cv": float(np.std(L) / np.median(L)) if np.median(L) > 1e-6 else None,
    }


segs = [segment_lengths(a, b) for a, b in PAIRS]


def pred_vs_obs(obs_name, centre, lat, med):
    o = obs[obs.landmark == obs_name][["frame", "raw_x_m", "raw_y_m", "raw_z_m"]].dropna()
    c = pred[pred.landmark == centre][["frame", "raw_x_m", "raw_y_m", "raw_z_m"]].dropna()
    la = pred[pred.landmark == lat][["frame", "raw_x_m"]].dropna()
    me = pred[pred.landmark == med][["frame", "raw_x_m"]].dropna()
    m = o.merge(c, on="frame", suffixes=("_o", "_c"))
    if m.empty:
        return {"obs": obs_name, "n": 0}
    dist = np.sqrt(
        (m["raw_x_m_o"] - m["raw_x_m_c"]) ** 2
        + (m["raw_y_m_o"] - m["raw_y_m_c"]) ** 2
        + (m["raw_z_m_o"] - m["raw_z_m_c"]) ** 2
    )
    corr = []
    for ax in ("raw_x_m", "raw_y_m", "raw_z_m"):
        if m[f"{ax}_o"].std() < 1e-9 or m[f"{ax}_c"].std() < 1e-9:
            corr.append(None)
        else:
            corr.append(float(np.corrcoef(m[f"{ax}_o"], m[f"{ax}_c"])[0, 1]))
    laterality = None
    mm = la.merge(me, on="frame", suffixes=("_lat", "_med")).merge(
        o[["frame", "raw_x_m"]], on="frame"
    )
    if len(mm):
        laterality = {
            "mean_X_lateral": float(mm["raw_x_m_lat"].mean()),
            "mean_X_medial": float(mm["raw_x_m_med"].mean()),
            "mean_X_observed": float(mm["raw_x_m"].mean()),
            "lateral_minus_medial_X": float((mm["raw_x_m_lat"] - mm["raw_x_m_med"]).mean()),
        }
    return {
        "obs": obs_name,
        "centre": centre,
        "n": int(len(m)),
        "mean_dist_m": float(dist.mean()),
        "median_dist_m": float(np.median(dist)),
        "corr_xyz": corr,
        "laterality": laterality,
    }


pred_cmp = [pred_vs_obs(*p) for p in PRED_PAIRS]

# LSTM timing
pred_frames = frames[frames["n_predicted"] > 0]
lag = frames["pred_lag_ms"].dropna()
inf = frames["infer_latency_ms"].dropna()

# Performance
proc = frames["process_ms"].dropna()

# Classes present
classes = lm.groupby("class")["landmark"].nunique().to_dict()
pred_names = sorted(lm[lm["class"] == "predicted_upper"]["landmark"].unique().tolist())
geo_names = sorted(lm[lm["class"] == "geometric_centre"]["landmark"].unique().tolist())
lower_names = sorted(lm[lm["class"] == "predicted_lower"]["landmark"].unique().tolist()) if (
    lm["class"] == "predicted_lower"
).any() else []

# CSV vs summary.json cross-check for left_ankle
summary = json.loads((SRC / "summary.json").read_text(encoding="utf-8"))
la_csv = foot_table[0]
la_sum = summary["landmarks"]["left_ankle"]

payload = {
    "recording": str(SRC),
    "n_frames_csv": n_frames,
    "t_sec_range": [t0, t1],
    "wall_from_summary": {
        "frames": summary.get("frames"),
        "dropped": summary.get("dropped_frames"),
        "wall_fps": summary.get("wall_fps"),
        "mean_process_ms": summary.get("mean_process_ms"),
        "t_start_utc": summary.get("t_start_utc"),
        "t_end_utc": summary.get("t_end_utc"),
    },
    "csv_columns_landmarks": list(lm.columns),
    "classes_unique_landmarks": classes,
    "predicted_upper_names": pred_names,
    "geometric_centre_names": geo_names,
    "predicted_lower_names": lower_names,
    "first_last_hip_frame": [first_hip, last_hip],
    "first_prediction_frame": first_pred,
    "still_runs_frames": still_runs_n,
    "longest_still": longest_still,
    "intervals": [
        interval_stats(0, first_hip - 1, "before_first_hip_xyz") if first_hip and first_hip > 0 else None,
        interval_stats(first_hip, first_pred - 1, "hips_before_lstm") if first_hip is not None and first_pred else None,
        interval_stats(*longest_still[:2], "longest_low_hip_motion") if longest_still else None,
        interval_stats(int(frames["frame"].max()) - 40, int(frames["frame"].max()), "last_40_frames"),
    ],
    "foot_counts": foot_table,
    "left_ankle_csv_vs_summary": {
        "csv_valid": la_csv["valid_xyz"],
        "summary_valid": la_sum.get("n_valid_xyz"),
        "csv_rows": la_csv["detected_rows"],
        "summary_seen": la_sum.get("n_frames_seen"),
    },
    "body_quality": body_q,
    "pred_quality": pred_q,
    "segment_lengths": segs,
    "pred_vs_obs": pred_cmp,
    "lstm_timing": {
        "frames_with_predictions": int(len(pred_frames)),
        "first_pred_frame": first_pred,
        "pred_lag_ms": {
            "n": int(lag.count()),
            "median": None if lag.empty else float(lag.median()),
            "mean": None if lag.empty else float(lag.mean()),
            "p95": None if lag.empty else float(lag.quantile(0.95)),
            "max": None if lag.empty else float(lag.max()),
            "share_eq_0": None if lag.empty else float((lag == 0).mean()),
        },
        "infer_latency_ms": {
            "n": int(inf.count()),
            "median": None if inf.empty else float(inf.median()),
            "mean": None if inf.empty else float(inf.mean()),
            "p95": None if inf.empty else float(inf.quantile(0.95)),
            "max": None if inf.empty else float(inf.max()),
        },
    },
    "performance": {
        "process_ms": {
            "n": int(proc.count()),
            "mean": float(proc.mean()),
            "median": float(proc.median()),
            "p95": float(proc.quantile(0.95)),
            "max": float(proc.max()),
        },
        "implied_processing_fps": 1000.0 / float(proc.mean()) if proc.mean() else None,
        "wall_fps_summary": summary.get("wall_fps"),
        "per_stage_timers_in_csv": False,
        "infer_share_of_mean_process": (
            float(inf.mean() / proc.mean()) if len(inf) and proc.mean() else None
        ),
    },
    "thresholds_from_config_not_csv": {"min_m": MIN_M, "max_m": MAX_M, "min_visibility_used_in_audit": MIN_VIS},
}

(OUT / "analysis.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

# Plots
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(merged_f["frame"], merged_f["raw_z_m"], lw=0.8, label="mid-hip Z m")
ax.plot(merged_f["frame"], merged_f["dxyz"], lw=0.6, alpha=0.7, label="hip f2f m")
if longest_still:
    ax.axvspan(longest_still[0], longest_still[1], color="0.85", label="longest low-motion hip run")
ax.set_xlabel("frame")
ax.set_ylabel("metres")
ax.legend(fontsize=8)
ax.set_title("Hip Z and frame-to-frame displacement (observed)")
fig.tight_layout()
fig.savefig(OUT / "fig_hip_motion.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(10, 3.6))
for name in FOOT:
    d = obs[obs.landmark == name]
    ax.plot(d["frame"], d["v_px"], lw=0.7, label=name)
ax.axhline(720, color="k", ls="--", lw=1, label="v=720 image bottom")
ax.axhline(640, color="r", ls=":", lw=1, label="v=640 safety guide (config, not in CSV)")
ax.set_ylim(200, 850)
ax.legend(fontsize=7, ncol=2)
ax.set_title("Foot landmark v_px")
fig.tight_layout()
fig.savefig(OUT / "fig_foot_v.png", dpi=120)
plt.close()

fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
for ax, name in zip(axes.ravel(), FOOT):
    d = obs[obs.landmark == name].sort_values("frame")
    valid = d["xyz_missing"] == 0
    ax.scatter(d.loc[valid, "frame"], d.loc[valid, "raw_z_m"], s=4, c="C0", label="valid Z")
    miss = d["xyz_missing"] == 1
    oob = miss & (d["in_image"] == 0)
    hole = miss & (d["in_image"] == 1) & (d["depth_n_valid_in_window"] == 0)
    ax.scatter(d.loc[oob, "frame"], np.full(int(oob.sum()), 0.05), s=8, c="C3", marker="x", label="OOB")
    ax.scatter(d.loc[hole, "frame"], np.full(int(hole.sum()), 0.12), s=8, c="C1", marker="+", label="in-image hole")
    ax.set_title(name)
    ax.set_ylabel("Z m / flags")
axes[0, 1].legend(fontsize=7)
axes[-1, 0].set_xlabel("frame")
fig.suptitle("Foot depth: valid Z vs OOB vs in-image zero-window")
fig.tight_layout()
fig.savefig(OUT / "fig_foot_depth_validity.png", dpi=120)
plt.close()

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, axis, col in zip(axes, "XYZ", ["raw_x_m", "raw_y_m", "raw_z_m"]):
    for name in ["left_shoulder", "left_hip", "left_ankle", "left_wrist"]:
        d = obs[obs.landmark == name].dropna(subset=[col])
        ax.plot(d["frame"], d[col], lw=0.7, label=name)
    ax.set_ylabel(f"{axis} m")
    ax.legend(fontsize=7, ncol=4)
axes[0].set_title("Observed camera metres (left side)")
axes[-1].set_xlabel("frame")
fig.tight_layout()
fig.savefig(OUT / "fig_obs_xyz_left.png", dpi=120)
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
for ax, obs_n, pred_n, ylab in [
    (axes[0], "left_elbow", "L_elbow_centre", "X m"),
    (axes[1], "left_elbow", "L_elbow_centre", "Z m"),
]:
    col = "raw_x_m" if ylab.startswith("X") else "raw_z_m"
    o = obs[obs.landmark == obs_n].dropna(subset=[col])
    p = pred[pred.landmark == pred_n].dropna(subset=[col])
    ax.plot(o["frame"], o[col], lw=0.8, label=f"{obs_n} observed")
    ax.plot(p["frame"], p[col], lw=0.8, label=f"{pred_n} geometric")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=8)
axes[0].set_title("Observed elbow vs LSTM pair-centre (not proof of anatomy)")
axes[-1].set_xlabel("frame")
fig.tight_layout()
fig.savefig(OUT / "fig_elbow_obs_vs_pred.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(frames["frame"], frames["process_ms"], lw=0.6, label="process_ms (whole loop)")
ax.plot(frames["frame"], frames["infer_latency_ms"], lw=0.8, label="LSTM infer_latency_ms")
ax.set_xlabel("frame")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.set_title("Timing in CSV: loop vs LSTM only (no Pose/Hands split)")
fig.tight_layout()
fig.savefig(OUT / "fig_timing.png", dpi=120)
plt.close()

print("wrote", OUT)
print("still_runs", still_runs_n)
print("longest", longest_still)
print("first_pred", first_pred, "first_hip", first_hip)
