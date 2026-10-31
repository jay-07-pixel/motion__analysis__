"""Run Pose2Sim markerAugmentation on a COPY of the 20261011 Y-up TRC.

Must be executed with the isolated pose2sim-offline interpreter.
Does not modify the source recording, original TRCs, or DOF code.
Does not map foot_index to BigToe/SmallToe.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from importlib.metadata import version as pkg_version
from Pose2Sim.markerAugmentation import (
    augment_markers_all,
    getMarkers_upperExtremity_noPelvis2,
    getOpenPoseMarkers_lowerExtremity2,
)

PROJECT = Path(r"C:\Projects\motion__analysis__")
SRC_TRC = (
    PROJECT
    / "experiments"
    / "pose2sim_whole_body"
    / "output"
    / "20261011-144812"
    / "20261011-144812_opensim_yup.trc"
)
CAM_TRC = (
    PROJECT
    / "experiments"
    / "pose2sim_whole_body"
    / "output"
    / "20261011-144812"
    / "20261011-144812_camera_3d_metre.trc"
)
OUT = (
    PROJECT
    / "experiments"
    / "anatomical_landmark_audit"
    / "output"
    / "20261011-144812_full_body_augmentation"
)
TRIAL = OUT / "trial"
POSE3D = TRIAL / "pose-3d"
HEIGHT_M = 1.80
MASS_KG = 61.0
QUIET_T0 = 43.0532
QUIET_T1 = 62.9664

# Defensible model-keypoint name map only. Not anatomy.
NAME_MAP = {
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

OBSERVED_KEEP = list(NAME_MAP.keys())  # MediaPipe names also stored
NOT_MAPPED = ("left_foot_index", "right_foot_index")

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


def _finite3(row, name: str) -> np.ndarray | None:
    cols = [c for c in row.index if c == name]
    if len(cols) < 3:
        return None
    # After Pose2Sim read_trc, three consecutive columns share the marker name.
    vals = row.loc[name]
    if isinstance(vals, pd.Series):
        xyz = vals.to_numpy(dtype=float)
    else:
        xyz = np.asarray(vals, dtype=float)
    if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
        return None
    return xyz[:3]


def read_trc(path: Path):
    from Pose2Sim.common import read_trc as p2s_read

    return p2s_read(str(path))


def write_simple_trc(path: Path, frames, times, marker_order, points_by_marker, rate: float):
    """Write OpenSim TRC. points_by_marker[name][i] is (3,) or None."""
    n_frames = len(frames)
    n_markers = len(marker_order)
    path.parent.mkdir(parents=True, exist_ok=True)
    name_cells = ["Frame#", "Time"]
    for name in marker_order:
        name_cells.extend([name, "", ""])
    xyz_cells = ["", ""]
    for i in range(1, n_markers + 1):
        xyz_cells.extend([f"X{i}", f"Y{i}", f"Z{i}"])
    orig_start = frames[0] if frames else 1
    h2 = "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    h3 = f"{rate:.6f}\t{rate:.6f}\t{n_frames}\t{n_markers}\tm\t{rate:.6f}\t{orig_start}\t{n_frames}"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
        handle.write(h2 + "\n")
        handle.write(h3 + "\n")
        handle.write("\t".join(name_cells) + "\n")
        handle.write("\t".join(xyz_cells) + "\n")
        for i, (fr, t) in enumerate(zip(frames, times)):
            cells = [str(int(fr)), f"{float(t):.6f}"]
            for name in marker_order:
                p = points_by_marker[name][i]
                if p is None:
                    cells.extend(["", "", ""])
                else:
                    cells.extend([f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])
            handle.write("\t".join(cells) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SRC_TRC.is_file():
        raise FileNotFoundError(SRC_TRC)

    shutil.copy2(SRC_TRC, OUT / "source_opensim_yup_copy.trc")
    trc_data, frames_col, time_col, markers, header = read_trc(SRC_TRC)
    n = len(trc_data)
    frames = frames_col.to_numpy()
    times = time_col.to_numpy(dtype=float)
    dts = np.diff(times)
    dts = dts[dts > 0]
    rate = float(1.0 / np.median(dts)) if len(dts) else 7.3

    # Build OpenPose-named series from MediaPipe columns (same 3-col layout).
    mapped: dict[str, list] = {dst: [] for dst in NAME_MAP.values()}
    for src, dst in NAME_MAP.items():
        if src not in markers:
            raise RuntimeError(f"Source TRC missing {src}")
    for i in range(n):
        row = trc_data.iloc[i]
        for src, dst in NAME_MAP.items():
            mapped[dst].append(_finite3(row, src))

    # Geometric proxies required by the LSTM (not measured landmarks).
    mapped["Neck"] = []
    mapped["Hip"] = []
    for i in range(n):
        ls, rs = mapped["LShoulder"][i], mapped["RShoulder"][i]
        lh, rh = mapped["LHip"][i], mapped["RHip"][i]
        mapped["Neck"].append(None if ls is None or rs is None else (ls + rs) / 2.0)
        mapped["Hip"].append(None if lh is None or rh is None else (lh + rh) / 2.0)

    upper_features, upper_out = getMarkers_upperExtremity_noPelvis2()
    lower_features, lower_out = getOpenPoseMarkers_lowerExtremity2()
    # Hip is required for centering even though it is not an upper feature.
    need_for_upper = list(dict.fromkeys(list(upper_features) + ["Hip"]))

    keep_idx = []
    dropped = []
    for i in range(n):
        missing = [name for name in need_for_upper if mapped[name][i] is None]
        if missing:
            dropped.append({"i": i, "frame": int(frames[i]), "time": float(times[i]), "missing": missing})
        else:
            keep_idx.append(i)

    if not keep_idx:
        raise RuntimeError("No frames have complete upper-body LSTM inputs.")

    # Dense subsequence for LSTM: consecutive kept rows only (no NaN fill).
    order = (
        list(NAME_MAP.values())
        + ["Neck", "Hip"]
        + [m for m in OBSERVED_KEEP if m in markers]
    )
    # Avoid duplicate names if MediaPipe name equals OpenPose (they don't).
    seen = set()
    marker_order = []
    for name in order:
        if name not in seen:
            marker_order.append(name)
            seen.add(name)

    # Also keep unmapped observed (face, hands, foot_index) for visualization, still not as toes.
    extra_obs = [m for m in markers if m not in NAME_MAP and m not in seen]
    marker_order.extend(extra_obs)

    points = {name: [] for name in marker_order}
    for i in keep_idx:
        row = trc_data.iloc[i]
        for name in marker_order:
            if name in mapped:
                points[name].append(mapped[name][i])
            else:
                points[name].append(_finite3(row, name))

    out_frames = [int(frames[i]) for i in keep_idx]
    out_times = [float(times[i]) for i in keep_idx]

    if POSE3D.exists():
        shutil.rmtree(POSE3D)
    POSE3D.mkdir(parents=True)
    input_trc = POSE3D / "input_mapped_yup.trc"
    write_simple_trc(input_trc, out_frames, out_times, marker_order, points, rate)

    mapping_doc = {
        "source_trc": str(SRC_TRC),
        "source_not_modified": True,
        "coordinate_convention": {
            "file": "OpenSim Y-up copy of RealSense metres",
            "X": "camera X (right)",
            "Y": "minus camera Y (up)",
            "Z": "camera Z (forward)",
            "units": "metres",
            "origin": "colour-camera optical centre",
        },
        "height_m": HEIGHT_M,
        "mass_kg": MASS_KG,
        "name_map_observed_model_keypoints": NAME_MAP,
        "not_mapped": {
            "left_foot_index": "Not BigToe or SmallToe",
            "right_foot_index": "Not BigToe or SmallToe",
        },
        "proxies": {
            "Neck": "midpoint(LShoulder, RShoulder); labelled estimated/proxy not a C7/neck landmark",
            "Hip": "midpoint(LHip, RHip); LSTM root; not ASIS/PSIS/HJC",
        },
        "missing_data": "Frames lacking any upper LSTM feature or Hip were dropped. No interpolation.",
        "n_source_frames": n,
        "n_kept_frames": len(keep_idx),
        "n_dropped_frames": len(dropped),
        "pose2sim_upper_features": upper_features,
        "pose2sim_upper_outputs": upper_out,
        "pose2sim_lower_features": lower_features,
        "pose2sim_lower_outputs": lower_out,
        "lower_body_note": (
            "Lower LSTM needs RBigToe, LBigToe, RSmallToe, LSmallToe. "
            "Those were NOT synthesised from foot_index. Official augmenter should skip v0.3_lower."
        ),
    }
    (OUT / "applied_mapping.json").write_text(json.dumps(mapping_doc, indent=2), encoding="utf-8")
    (OUT / "dropped_frames.json").write_text(json.dumps(dropped[:200], indent=2), encoding="utf-8")

    config = {
        "project": {
            "project_dir": str(TRIAL.resolve()),
            "participant_height": HEIGHT_M,
            "participant_mass": MASS_KG,
            "frame_range": "all",
        },
        "markerAugmentation": {"feet_on_floor": False, "make_c3d": False},
        "kinematics": {
            "large_hip_knee_angles": 90,
            "trimmed_extrema_percent": 50,
            "default_height": 1.7,
        },
    }
    (OUT / "pose2sim_config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Running augment_markers_all ...", flush=True)
    try:
        augment_markers_all(config)
        ran = True
        error = None
    except Exception as exc:
        ran = False
        error = repr(exc)
        print("AUGMENT FAILED", error, flush=True)

    lstm_files = sorted(POSE3D.glob("*LSTM*.trc"))
    result = {
        "inference_ran": ran,
        "error": error,
        "python": sys.version,
        "pose2sim_version": pkg_version("pose2sim"),
        "onnxruntime": __import__("onnxruntime").__version__,
        "model_assets": {
            "lower": str(
                Path(__import__("Pose2Sim").__file__).parent
                / "MarkerAugmenter"
                / "LSTM"
                / "v0.3_lower"
            ),
            "upper": str(
                Path(__import__("Pose2Sim").__file__).parent
                / "MarkerAugmenter"
                / "LSTM"
                / "v0.3_upper"
            ),
        },
        "lstm_trc_files": [str(p) for p in lstm_files],
        "n_input_frames": len(keep_idx),
        "height_m": HEIGHT_M,
        "mass_kg": MASS_KG,
    }
    (OUT / "inference_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not ran or not lstm_files:
        print("No LSTM TRC produced. Stop before visualization of estimates.")
        return

    out_trc = lstm_files[0]
    pred_data, pred_frames, pred_times, pred_markers, _ = read_trc(out_trc)
    estimated = [m for m in pred_markers if m.endswith("_study")]
    print("estimated markers", len(estimated), estimated)

    # CSV of estimates in Y-up metres.
    csv_path = OUT / "augmented_markers_yup.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
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
        for i in range(len(pred_data)):
            row = pred_data.iloc[i]
            for name in estimated:
                p = _finite3(row, name)
                writer.writerow(
                    {
                        "frame": int(pred_frames.iloc[i]),
                        "time_sec": f"{float(pred_times.iloc[i]):.4f}",
                        "marker": name,
                        "x_m": "" if p is None else f"{p[0]:.6f}",
                        "y_m": "" if p is None else f"{p[1]:.6f}",
                        "z_m": "" if p is None else f"{p[2]:.6f}",
                        "coord_frame": "opensim_yup_metre",
                        "provenance": "estimated_pose2sim_lstm_v0.3",
                    }
                )

    def pair_len(a, b, i):
        pa = _finite3(pred_data.iloc[i], a)
        pb = _finite3(pred_data.iloc[i], b)
        if pa is None or pb is None:
            return None
        return float(np.linalg.norm(pa - pb))

    pairs = [
        ("right_elbow_pair", "r_lelbow_study", "r_melbow_study"),
        ("left_elbow_pair", "L_lelbow_study", "L_melbow_study"),
        ("right_wrist_pair", "r_lwrist_study", "r_mwrist_study"),
        ("left_wrist_pair", "L_lwrist_study", "L_mwrist_study"),
    ]
    pair_stats = {}
    for label, a, b in pairs:
        vals = [pair_len(a, b, i) for i in range(len(pred_data))]
        vals = [v for v in vals if v is not None]
        if not vals:
            pair_stats[label] = {"n": 0}
            continue
        arr = np.asarray(vals)
        pair_stats[label] = {
            "n": int(len(arr)),
            "median_m": float(np.median(arr)),
            "mean_m": float(np.mean(arr)),
            "stdev_m": float(np.std(arr)),
            "min_m": float(np.min(arr)),
            "max_m": float(np.max(arr)),
        }

    # Quiet-window indices
    t_arr = pred_times.to_numpy(dtype=float)
    quiet = np.where((t_arr >= QUIET_T0) & (t_arr <= QUIET_T1))[0]
    vis_i = int(quiet[len(quiet) // 2]) if len(quiet) else len(pred_data) // 2

    def xyz_of(names, i):
        out = {}
        row = pred_data.iloc[i]
        for name in names:
            p = _finite3(row, name)
            if p is not None:
                out[name] = p
        return out

    obs_names = [m for m in ("LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "LHeel", "RHeel", "Neck") if m in pred_markers]
    obs = xyz_of(obs_names, vis_i)
    est = xyz_of(estimated, vis_i)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    def draw(ax, title, with_est: bool):
        for a, b in POSE_BONES:
            if a in obs and b in obs:
                pa, pb = obs[a], obs[b]
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="#1f4e79", lw=1.4)
        if obs:
            P = np.vstack(list(obs.values()))
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="#1f4e79", s=28, label="observed (model)")
        if with_est and est:
            Q = np.vstack(list(est.values()))
            ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c="#e67e22", s=36, marker="^", label="estimated LSTM")
        ax.set_title(title)
        ax.set_xlabel("X m (right)")
        ax.set_ylabel("Y m (up)")
        ax.set_zlabel("Z m (forward)")
        ax.legend(loc="upper left", fontsize=8)
        if obs:
            P = np.vstack(list(obs.values()) + (list(est.values()) if with_est and est else []))
            c = P.mean(axis=0)
            r = max(0.6, float(np.max(np.linalg.norm(P - c, axis=1))) * 1.2)
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)

    t_vis = float(pred_times.iloc[vis_i])
    draw(ax1, f"Observed only  t={t_vis:.2f}s", False)
    draw(ax2, f"Observed + estimated  t={t_vis:.2f}s", True)
    fig.suptitle(
        "Pose2Sim LSTM v0.3 estimates (Y-up metres). Orange triangles are predicted, not anatomy."
    )
    fig.tight_layout()
    fig.savefig(OUT / "viz_observed_vs_estimated.png", dpi=140)
    plt.close(fig)

    # Extra viewpoints of overlay
    fig = plt.figure(figsize=(12, 4))
    for j, elev_azim in enumerate(((20, -60), (10, 20), (80, -90))):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        draw(ax, f"elev={elev_azim[0]} az={elev_azim[1]}", True)
        ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
    fig.tight_layout()
    fig.savefig(OUT / "viz_estimated_viewpoints.png", dpi=140)
    plt.close(fig)

    # Quiet-window animation (subsample)
    if len(quiet) >= 4:
        step = max(1, len(quiet) // 24)
        idxs = quiet[::step]
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        def update(k):
            ax.cla()
            i = int(idxs[k])
            o = xyz_of(obs_names, i)
            e = xyz_of(estimated, i)
            for a, b in POSE_BONES:
                if a in o and b in o:
                    pa, pb = o[a], o[b]
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="#1f4e79", lw=1.2)
            if o:
                P = np.vstack(list(o.values()))
                ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="#1f4e79", s=20)
            if e:
                Q = np.vstack(list(e.values()))
                ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c="#e67e22", s=28, marker="^")
            ax.set_title(f"quiet window t={float(pred_times.iloc[i]):.2f}s")
            if o:
                P = np.vstack(list(o.values()) + (list(e.values()) if e else []))
                c = P.mean(axis=0)
                r = 0.9
                ax.set_xlim(c[0] - r, c[0] + r)
                ax.set_ylim(c[1] - r, c[1] + r)
                ax.set_zlim(c[2] - r, c[2] + r)
            ax.set_xlabel("X")
            ax.set_ylabel("Y up")
            ax.set_zlabel("Z")

        anim = FuncAnimation(fig, update, frames=len(idxs), interval=200)
        anim.save(OUT / "viz_quiet_window.gif", writer=PillowWriter(fps=5))
        plt.close(fig)

    summary = {
        **result,
        "n_estimated_markers": len(estimated),
        "estimated_marker_names": estimated,
        "lower_body_outputs_present": any("ASIS" in m or "HJC" in m or "knee_study" in m for m in estimated),
        "upper_body_outputs_present": any("elbow_study" in m or "wrist_study" in m for m in estimated),
        "pair_stats_m": pair_stats,
        "visualization_frame_time_sec": t_vis,
        "n_quiet_frames_in_output": int(len(quiet)),
        "files": {
            "input_trc": str(input_trc),
            "lstm_trc": str(out_trc),
            "csv": str(csv_path),
            "png_side": str(OUT / "viz_observed_vs_estimated.png"),
        },
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("inference_ran", "n_estimated_markers", "lower_body_outputs_present", "upper_body_outputs_present")}, indent=2))


if __name__ == "__main__":
    main()
