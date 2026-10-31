"""Read-only analysis of foot pixels vs image bounds and depth holes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SRC = Path(r"c:\Projects\motion__analysis__\data\diagnostics\20261012-153538")
OUT = Path(__file__).resolve().parent
W, H = 1280, 720
NAMES = [
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
    "left_knee",
    "right_knee",
]

lm = pd.read_csv(SRC / "landmarks.csv")
obs = lm[lm["class"] == "observed"].copy()
for col in ("u_px", "v_px", "depth_sample_m", "visibility", "raw_x_m", "raw_y_m", "raw_z_m"):
    obs[col] = pd.to_numeric(obs[col], errors="coerce")
obs["n_valid"] = pd.to_numeric(obs["depth_n_valid_in_window"], errors="coerce")
obs["in_image"] = pd.to_numeric(obs["in_image"], errors="coerce")
obs["xyz_missing"] = pd.to_numeric(obs["xyz_missing"], errors="coerce")
obs["depth_valid"] = pd.to_numeric(obs["depth_valid"], errors="coerce")

rows = []
for name in NAMES:
    d = obs[obs["landmark"] == name]
    n = len(d)
    oob = int((d["in_image"] == 0).sum())
    inn = int((d["in_image"] == 1).sum())
    miss = int((d["xyz_missing"] == 1).sum())
    hole = int(((d["in_image"] == 1) & (d["n_valid"] == 0)).sum())
    hole_xyz = int(((d["in_image"] == 1) & (d["xyz_missing"] == 1)).sum())
    oob_xyz = int(((d["in_image"] == 0) & (d["xyz_missing"] == 1)).sum())
    inrange_reject = int(
        ((d["in_image"] == 1) & (d["n_valid"] > 0) & (d["xyz_missing"] == 1)).sum()
    )
    v = d["v_px"]
    u = d["u_px"]
    rows.append(
        {
            "landmark": name,
            "n": n,
            "valid_xyz": n - miss,
            "xyz_missing": miss,
            "in_image": inn,
            "out_of_image": oob,
            "oob_and_xyz_missing": oob_xyz,
            "in_image_n_valid_0": hole,
            "in_image_xyz_missing": hole_xyz,
            "in_image_depth_but_rejected": inrange_reject,
            "u_min": None if u.dropna().empty else float(u.min()),
            "u_max": None if u.dropna().empty else float(u.max()),
            "v_min": None if v.dropna().empty else float(v.min()),
            "v_p50": None if v.dropna().empty else float(v.median()),
            "v_p95": None if v.dropna().empty else float(v.quantile(0.95)),
            "v_max": None if v.dropna().empty else float(v.max()),
            "n_v_ge_720": int((v >= H).sum()),
            "n_v_lt_0": int((v < 0).sum()),
            "n_u_ge_1280": int((u >= W).sum()),
            "n_u_lt_0": int((u < 0).sum()),
            "n_v_ge_700": int((v >= 700).sum()),
            "n_v_ge_710": int((v >= 710).sum()),
            "vis_median": None if d["visibility"].dropna().empty else float(d["visibility"].median()),
        }
    )

# OOB pixel locations
oob_all = obs[(obs["landmark"].isin(NAMES)) & (obs["in_image"] == 0)]
oob_summary = []
for name, g in oob_all.groupby("landmark"):
    oob_summary.append(
        {
            "landmark": name,
            "n": int(len(g)),
            "u_min": float(g["u_px"].min()),
            "u_max": float(g["u_px"].max()),
            "v_min": float(g["v_px"].min()),
            "v_max": float(g["v_px"].max()),
            "frames": [int(x) for x in g["frame"].head(20).tolist()],
        }
    )

# in-image holes
holes = obs[(obs["landmark"].isin(NAMES[:6])) & (obs["in_image"] == 1) & (obs["n_valid"] == 0)]
hole_summary = []
for name, g in holes.groupby("landmark"):
    hole_summary.append(
        {
            "landmark": name,
            "n": int(len(g)),
            "u_min": float(g["u_px"].min()),
            "u_max": float(g["u_px"].max()),
            "v_min": float(g["v_px"].min()),
            "v_max": float(g["v_px"].max()),
            "v_median": float(g["v_px"].median()),
            "depth_sample_unique": sorted(set(g["depth_sample_m"].fillna(-1).tolist()))[:8],
        }
    )

payload = {
    "recording": str(SRC),
    "image": {"width": W, "height": H},
    "by_landmark": rows,
    "oob_details": oob_summary,
    "in_image_zero_depth": hole_summary,
    "csv_columns": list(lm.columns),
}

(OUT / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
