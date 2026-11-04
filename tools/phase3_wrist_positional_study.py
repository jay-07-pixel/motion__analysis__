"""Phase 3: wrist positional decomposition + controlled IK sensitivity (isolated).

Does not edit original markers.trc, gates, model geometry, or LSTM weights.
Outputs under output/opensim_upper/phase3/.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

from src.opensim_upper.ik import run_upper_body_ik
from src.opensim_upper.mot import read_mot
from src.opensim_upper.trc import read_trc, write_trc

ROOT = Path(__file__).resolve().parents[1]
TRIAL = "20261104-140115"
TRIALS = ("20261104-140115", "20261104-103341")
PHASE3 = ROOT / "output" / "opensim_upper" / "phase3"
WRISTS = (
    "r_lwrist_study",
    "r_mwrist_study",
    "L_lwrist_study",
    "L_mwrist_study",
)
SIDES = (
    ("R", "r_lelbow_study", "r_melbow_study", "r_lwrist_study", "r_mwrist_study", "right_wrist"),
    ("L", "L_lelbow_study", "L_melbow_study", "L_lwrist_study", "L_mwrist_study", "left_wrist"),
)


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def _pt(points: dict, name: str) -> np.ndarray | None:
    p = points.get(name)
    return None if p is None else np.asarray(p, dtype=float)


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _decomp_frame(rec: dict, model: dict, el, em, wl, wm) -> dict | None:
    a, b, c, d = _pt(rec, el), _pt(rec, em), _pt(rec, wl), _pt(rec, wm)
    ma, mb, mc, md = _pt(model, el), _pt(model, em), _pt(model, wl), _pt(model, wm)
    if any(x is None for x in (a, b, c, d, ma, mb, mc, md)):
        return None
    rec_mid = 0.5 * (c + d)
    mod_mid = 0.5 * (mc + md)
    e_mid = 0.5 * (a + b)
    forearm = _unit(rec_mid - e_mid)
    if forearm is None:
        forearm = _unit(mod_mid - 0.5 * (ma + mb))
    if forearm is None:
        return None
    dmid = rec_mid - mod_mid
    along = float(np.dot(dmid, forearm))
    across_vec = dmid - along * forearm
    across = float(np.linalg.norm(across_vec))
    tip_l = float(np.linalg.norm(c - mc))
    tip_m = float(np.linalg.norm(d - md))
    width_rec = float(np.linalg.norm(c - d))
    width_mod = float(np.linalg.norm(mc - md))
    # orientation residual in forearm-perp plane
    e_ml = a - b
    w_ml = c - d
    e_ml = e_ml - np.dot(e_ml, forearm) * forearm
    w_ml = w_ml - np.dot(w_ml, forearm) * forearm
    wm_ml = (mc - md) - np.dot(mc - md, forearm) * forearm
    eu, wu, wmu = _unit(e_ml), _unit(w_ml), _unit(wm_ml)
    roll_rec = None
    roll_mod = None
    if eu is not None and wu is not None:
        roll_rec = float(math.degrees(math.acos(float(np.clip(np.dot(eu, wu), -1, 1)))))
    if eu is not None and wmu is not None:
        roll_mod = float(math.degrees(math.acos(float(np.clip(np.dot(eu, wmu), -1, 1)))))
    return {
        "mid_miss_m": float(np.linalg.norm(dmid)),
        "mid_along_m": along,
        "mid_across_m": across,
        "tip_lat_m": tip_l,
        "tip_med_m": tip_m,
        "width_rec_m": width_rec,
        "width_mod_m": width_mod,
        "width_err_m": width_rec - width_mod,
        "forearm_len_rec_m": float(np.linalg.norm(rec_mid - e_mid)),
        "forearm_len_mod_m": float(np.linalg.norm(mod_mid - 0.5 * (ma + mb))),
        "roll_rec_deg": roll_rec,
        "roll_mod_deg": roll_mod,
    }


def _median(xs: list[float]) -> float | None:
    return None if not xs else float(np.median(xs))


def decompose_trial(trial: str) -> dict:
    trial_dir = ROOT / "output/opensim_upper/trials" / trial
    ik_dir = ROOT / "output/opensim_upper/ik" / trial
    frames = read_trc(trial_dir / "markers.trc")["frames"]
    ik_frames = read_trc(trial_dir / "markers_ik.trc")["frames"]
    loc = [p for p in sorted(ik_dir.glob("*model_marker_locations.sto")) if "axial" not in p.name][0]
    model_rows = read_mot(loc)["rows"]
    model_points = []
    for row in model_rows:
        points = {}
        for key in row:
            if key.endswith("_tx"):
                name = key[:-3]
                points[name] = (
                    float(row[f"{name}_tx"]),
                    float(row[f"{name}_ty"]),
                    float(row[f"{name}_tz"]),
                )
        model_points.append(points)
    report = json.loads((ik_dir / "ik_report.json").read_text(encoding="utf-8"))
    out: dict = {
        "trial": trial,
        "markers_trc_sha16": _sha16(trial_dir / "markers.trc"),
        "take_rms_mm": report["marker_errors"]["columns"]["marker_error_RMS"]["mean_abs_m"] * 1000,
        "clinical_accuracy": report.get("clinical_accuracy"),
        "sides": {},
    }
    for side, el, em, wl, wm, mp_wrist in SIDES:
        rows = []
        used = 0
        for fr, ik, mp in zip(frames, ik_frames, model_points):
            if _pt(ik["points"], wl) is None or _pt(ik["points"], wm) is None:
                continue
            used += 1
            row = _decomp_frame(fr["points"], mp, el, em, wl, wm)
            if row is not None:
                # Optional: LSTM mid vs MediaPipe observed wrist (prediction vs observed joint)
                obs = _pt(fr["points"], mp_wrist) if mp_wrist in fr["points"] else None
                # MP wrists are often not in TRC 12-marker set — check landmarks? TRC only has 12.
                rows.append(row)
        # MediaPipe wrists are unmapped — compare LSTM mid to elbow→ implied length only from TRC.
        keys = [
            "mid_miss_m",
            "mid_along_m",
            "mid_across_m",
            "tip_lat_m",
            "tip_med_m",
            "width_rec_m",
            "width_mod_m",
            "width_err_m",
            "forearm_len_rec_m",
            "forearm_len_mod_m",
            "roll_rec_deg",
            "roll_mod_deg",
        ]
        summary = {"n_ik_used_frames": used, "n_decomp": len(rows)}
        for key in keys:
            vals = [r[key] for r in rows if r.get(key) is not None]
            summary[f"median_{key}"] = _median(vals)
            if vals:
                summary[f"std_{key}"] = float(np.std(vals))
        # Classification heuristic (evidence, not a claim of sole cause)
        mid = summary.get("median_mid_miss_m") or 0
        tip = max(summary.get("median_tip_lat_m") or 0, summary.get("median_tip_med_m") or 0)
        width = abs(summary.get("median_width_err_m") or 0)
        across = summary.get("median_mid_across_m") or 0
        along = abs(summary.get("median_mid_along_m") or 0)
        primary = []
        if mid >= 0.015:
            primary.append("A_translation")
        if width >= 0.015:
            primary.append("B_width")
        # orientation: tip >> mid suggests clocking/width beyond pure translation
        if tip - mid >= 0.015:
            primary.append("C_orientation_or_pair_shape")
        if along > across and mid >= 0.01:
            primary.append("along_forearm_dominant")
        if across >= along and mid >= 0.01:
            primary.append("across_forearm_dominant")
        summary["primary_error_classes"] = primary or ["insufficient_frames"]
        out["sides"][side] = summary
    return out


def _copy_trc_blank_wrists(source: Path, dest: Path) -> None:
    parsed = read_trc(source)
    samples = {name: [] for name in parsed["markers"]}
    for frame in parsed["frames"]:
        for name in parsed["markers"]:
            if name in WRISTS:
                samples[name].append(None)
            else:
                samples[name].append(frame["points"].get(name))
    write_trc(
        dest,
        [frame["time_sec"] for frame in parsed["frames"]],
        samples,
        float(parsed["data_rate_hz"]),
        [{"target": name} for name in parsed["markers"]],
    )


def _default_weights() -> dict[str, float]:
    # Equal 5.0 policy used by UI path for all 12 markers.
    return {
        "RShoulder": 5.0,
        "LShoulder": 5.0,
        "RElbow": 5.0,
        "LElbow": 5.0,
        "r_lelbow_study": 5.0,
        "r_melbow_study": 5.0,
        "L_lelbow_study": 5.0,
        "L_melbow_study": 5.0,
        "r_lwrist_study": 5.0,
        "r_mwrist_study": 5.0,
        "L_lwrist_study": 5.0,
        "L_mwrist_study": 5.0,
    }


def run_sensitivity() -> dict:
    trial_dir = ROOT / "output/opensim_upper/trials" / TRIAL
    model = trial_dir / "Model_upper_scaled.osim"
    meta = trial_dir / "trial_metadata.json"
    src_ik = trial_dir / "markers_ik.trc"
    # Prove original TRC untouched before/after.
    original = trial_dir / "markers.trc"
    before = original.read_bytes()
    work = PHASE3 / TRIAL
    work.mkdir(parents=True, exist_ok=True)
    (work / "README.txt").write_text(
        "Phase 3 isolated IK experiments. Original trials/ markers.trc not edited.\n",
        encoding="utf-8",
    )
    trc_with = work / "markers_ik_with_wrists.trc"
    trc_without = work / "markers_ik_without_wrists.trc"
    shutil.copy2(src_ik, trc_with)
    _copy_trc_blank_wrists(src_ik, trc_without)

    experiments = [
        ("A_wrists_excluded", trc_without, _default_weights()),
        ("B_wrists_included_w5", trc_with, _default_weights()),
        ("C_wrists_w0p5", trc_with, {**_default_weights(), **{n: 0.5 for n in WRISTS}}),
        ("C_wrists_w1", trc_with, {**_default_weights(), **{n: 1.0 for n in WRISTS}}),
        ("C_wrists_w10", trc_with, {**_default_weights(), **{n: 10.0 for n in WRISTS}}),
        ("C_wrists_w20", trc_with, {**_default_weights(), **{n: 20.0 for n in WRISTS}}),
    ]
    results = []
    for name, trc, weights in experiments:
        out_dir = work / name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        report = run_upper_body_ik(
            trc,
            out_dir,
            trial_metadata=meta,
            model_path=model,
            weight_overrides=weights,
            axial_refinement=False,
        )
        fit = report.get("fit") or {}
        markers = (fit.get("markers") or {})
        # Also from marker_errors aggregate
        rms = ((report.get("marker_errors") or {}).get("columns") or {}).get("marker_error_RMS", {})
        row = {
            "name": name,
            "take_rms_mm": None if not rms else rms.get("mean_abs_m", 0) * 1000,
            "execution_success": report.get("execution_success"),
            "wrist_rms_mm": {},
            "elbow_rms_mm": {},
            "shoulder_rms_mm": {},
            "unweighted_rms_m": fit.get("unweighted_rms_m"),
        }
        for marker, stats in markers.items():
            if stats.get("rms_m") is None:
                continue
            mm = stats["rms_m"] * 1000
            if "wrist" in marker:
                row["wrist_rms_mm"][marker] = mm
            elif "elbow" in marker.lower() or marker in ("RElbow", "LElbow"):
                row["elbow_rms_mm"][marker] = mm
            elif "Shoulder" in marker:
                row["shoulder_rms_mm"][marker] = mm
        # Joint coord ranges from MOT
        mot = out_dir / "ik_results.mot"
        if mot.is_file():
            motion = read_mot(mot)
            coords = {}
            for key in ("pro_sup_r", "pro_sup_l", "elbow_flex_r", "elbow_flex_l", "arm_rot_r", "arm_rot_l"):
                if key not in motion["rows"][0]:
                    continue
                # MOT angular values are degrees in this pipeline's OpenSim export.
                vals = [float(r[key]) for r in motion["rows"]]
                coords[key] = {
                    "median_deg": float(np.median(vals)),
                    "std_deg": float(np.std(vals)),
                    "min_deg": float(min(vals)),
                    "max_deg": float(max(vals)),
                }
            row["coordinates_deg"] = coords
        results.append(row)
        (out_dir / "phase3_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    after = original.read_bytes()
    if after != before:
        raise RuntimeError("Original markers.trc changed during Phase 3 experiments.")
    return {
        "trial": TRIAL,
        "original_trc_sha16": _sha16(original),
        "experiments": results,
        "clinical_accuracy": False,
        "biomechanical_reliability": False,
    }


def lstm_vs_observed_geometry() -> dict:
    """LSTM wrist mid vs elbow-implied forearm; width; L/R asymmetry (TRC only)."""
    out = {}
    for trial in TRIALS:
        frames = read_trc(ROOT / f"output/opensim_upper/trials/{trial}/markers.trc")["frames"]
        side_stats = {}
        for side, el, em, wl, wm, _ in SIDES:
            widths, forearm, tip_sep_from_mid = [], [], []
            for fr in frames:
                a, b, c, d = _pt(fr["points"], el), _pt(fr["points"], em), _pt(fr["points"], wl), _pt(fr["points"], wm)
                if any(x is None for x in (a, b, c, d)):
                    continue
                mid_e = 0.5 * (a + b)
                mid_w = 0.5 * (c + d)
                widths.append(float(np.linalg.norm(c - d)))
                forearm.append(float(np.linalg.norm(mid_w - mid_e)))
                tip_sep_from_mid.append(0.5 * float(np.linalg.norm(c - d)))
            side_stats[side] = {
                "median_width_m": _median(widths),
                "std_width_m": None if not widths else float(np.std(widths)),
                "median_forearm_m": _median(forearm),
                "std_forearm_m": None if not forearm else float(np.std(forearm)),
                "n": len(widths),
            }
        # Model wrist pair distance from scaled osim locals (same body — valid)
        from src.opensim_upper.scaling import spacing_mapping

        model = ROOT / f"output/opensim_upper/trials/{trial}/Model_upper_scaled.osim"
        locs = {
            row["target"]: np.asarray(row["location_m"], float)
            for row in spacing_mapping(model)["mapped"]
        }
        model_w = float(np.linalg.norm(locs["r_lwrist_study"] - locs["r_mwrist_study"]))
        side_stats["model_wrist_width_m"] = model_w
        side_stats["R_width_err_m"] = None if side_stats["R"]["median_width_m"] is None else side_stats["R"]["median_width_m"] - model_w
        side_stats["L_width_err_m"] = None if side_stats["L"]["median_width_m"] is None else side_stats["L"]["median_width_m"] - model_w
        out[trial] = side_stats
    return out


def model_constraint_notes() -> dict:
    return {
        "wrist_markers_parent": "both lateral and medial study wrists on radius_* (not ulna)",
        "locked_dofs": ["wrist_flex_r/l", "wrist_dev_r/l", "neck_*"],
        "solved_forearm_axial": "pro_sup_r/l",
        "solved_humerus_axial": "arm_rot_r/l",
        "pro_sup_range_rad": [0.0, 2.09],
        "implication": (
            "IK can move wrist stations only by posing thorax/shoulder/elbow/pro_sup. "
            "Locked wrist_flex/dev cannot absorb tip error in the hand plane. "
            "Because both tips are fixed on radius, width is almost rigid on the model; "
            "a wrong predicted width cannot be fixed by DOFs."
        ),
        "clinical_accuracy": False,
    }


def main() -> None:
    PHASE3.mkdir(parents=True, exist_ok=True)
    decomp = {trial: decompose_trial(trial) for trial in TRIALS}
    geom = lstm_vs_observed_geometry()
    constraints = model_constraint_notes()
    sensitivity = run_sensitivity()
    payload = {
        "phase": 3,
        "decomposition": decomp,
        "lstm_geometry": geom,
        "opensim_constraints": constraints,
        "sensitivity_140115": sensitivity,
        "clinical_accuracy": False,
        "biomechanical_reliability": False,
    }
    path = PHASE3 / "phase3_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(path), "experiments": [e["name"] for e in sensitivity["experiments"]]}, indent=2))


if __name__ == "__main__":
    main()
