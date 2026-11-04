"""Phase 4 read-only geometry study. Writes isolated JSON; does not touch TRC/gates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.opensim_upper.mot import read_mot
from src.opensim_upper.scaling import forearm_reference_distance, measure_body_scales, spacing_mapping
from src.opensim_upper.trc import read_trc

ROOT = Path(__file__).resolve().parents[1]
GENERIC = ROOT / "models" / "opensim_upper" / "Model_upper_camera.osim"
OUT = ROOT / "output" / "opensim_upper" / "phase4"
TRIALS = ("20261104-140115", "20261104-103341")
SIDES = {
    "R": ("RElbow", "r_lwrist_study", "r_mwrist_study"),
    "L": ("LElbow", "L_lwrist_study", "L_mwrist_study"),
}


def _model_rows(table):
    keys = table["rows"][0].keys()
    names = [k[:-3] for k in keys if k.endswith("_tx")]
    rows = []
    for row in table["rows"]:
        pts = {
            n: np.array(
                [float(row[f"{n}_tx"]), float(row[f"{n}_ty"]), float(row[f"{n}_tz"])],
                dtype=float,
            )
            for n in names
        }
        rows.append(pts)
    return names, rows


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "median_m": None, "mean_m": None, "std_m": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "median_m": float(np.median(arr)),
        "mean_m": float(np.mean(arr)),
        "std_m": float(np.std(arr)),
    }


def predicted_geometry(trc_path: Path) -> dict:
    frames = read_trc(trc_path)["frames"]
    out = {}
    for side, (elbow_n, lat_n, med_n) in SIDES.items():
        forearm = []
        width = []
        elbow_to_lat = []
        elbow_to_med = []
        lat_bias_along = []
        med_bias_along = []
        for fr in frames:
            p = fr["points"]
            elbow, lat, med = p.get(elbow_n), p.get(lat_n), p.get(med_n)
            if elbow is None or lat is None or med is None:
                continue
            e = np.asarray(elbow, dtype=float)
            a = np.asarray(lat, dtype=float)
            b = np.asarray(med, dtype=float)
            mid = 0.5 * (a + b)
            axis = mid - e
            length = float(np.linalg.norm(axis))
            if length < 1e-6:
                continue
            u = axis / length
            forearm.append(length)
            width.append(float(np.linalg.norm(a - b)))
            elbow_to_lat.append(float(np.linalg.norm(a - e)))
            elbow_to_med.append(float(np.linalg.norm(b - e)))
            # along-axis projection of each tip relative to mid
            lat_bias_along.append(float(np.dot(a - mid, u)))
            med_bias_along.append(float(np.dot(b - mid, u)))
        out[side] = {
            "forearm_elbow_to_mid": _stats(forearm),
            "width": _stats(width),
            "elbow_to_lat": _stats(elbow_to_lat),
            "elbow_to_med": _stats(elbow_to_med),
            "lat_along_vs_mid": _stats(lat_bias_along),
            "med_along_vs_mid": _stats(med_bias_along),
            "width_over_forearm": (
                None
                if not forearm
                else float(np.median(width) / np.median(forearm))
            ),
        }
    return out


def fitted_forearm(trc_path: Path, loc_path: Path) -> dict:
    frames = read_trc(trc_path)["frames"]
    names, mrows = _model_rows(read_mot(loc_path))
    out = {}
    for side, (elbow_n, lat_n, med_n) in SIDES.items():
        if not all(n in names for n in (elbow_n, lat_n, med_n)):
            out[side] = {"n": 0, "note": "markers absent from model locations"}
            continue
        rec_len = []
        mod_len = []
        mid_miss = []
        for fr, mr in zip(frames, mrows):
            p = fr["points"]
            if any(p.get(n) is None for n in (elbow_n, lat_n, med_n)):
                continue
            if any(n not in mr for n in (elbow_n, lat_n, med_n)):
                continue
            e_r = np.asarray(p[elbow_n], dtype=float)
            mid_r = 0.5 * (np.asarray(p[lat_n]) + np.asarray(p[med_n]))
            e_m = mr[elbow_n]
            mid_m = 0.5 * (mr[lat_n] + mr[med_n])
            rec_len.append(float(np.linalg.norm(mid_r - e_r)))
            mod_len.append(float(np.linalg.norm(mid_m - e_m)))
            mid_miss.append(float(np.linalg.norm(mid_r - mid_m)))
        out[side] = {
            "rec_forearm": _stats(rec_len),
            "mod_forearm": _stats(mod_len),
            "mid_miss": _stats(mid_miss),
        }
    return out


def model_station_geometry(scaled: Path) -> dict:
    sm = spacing_mapping(scaled)
    by = {r["target"]: np.array(r["location_m"], dtype=float) for r in sm["mapped"]}
    parents = {r["target"]: r["body"] for r in sm["mapped"]}
    rows = {}
    for side, (_e, lat, med) in SIDES.items():
        a, b = by[lat], by[med]
        rows[side] = {
            "lat_parent": parents[lat],
            "med_parent": parents[med],
            "lat_loc_m": a.tolist(),
            "med_loc_m": b.tolist(),
            "width_m": float(np.linalg.norm(a - b)),
            "both_on_radius": parents[lat].startswith("radius") and parents[med].startswith("radius"),
        }
    return {
        "forearm_reference_m": float(forearm_reference_distance(scaled)),
        "generic_forearm_reference_m": float(forearm_reference_distance(GENERIC)),
        "sides": rows,
    }


def participant_meta(trial_dir: Path) -> dict:
    for name in ("static_pose.json", "session_meta.json", "trial_meta.json"):
        path = trial_dir / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "source": name,
                "participant_height_m": data.get("participant_height_m"),
                "participant_mass_kg": data.get("participant_mass_kg"),
            }
    # markers.trc comments sometimes unused; check analysis report
    report = ROOT / "output" / "opensim_upper" / "analysis" / trial_dir.name / "analysis_report.md"
    return {"source": None, "analysis_report_exists": report.exists()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 4,
        "clinical_accuracy": False,
        "biomechanical_reliability": False,
        "constraints": {
            "no_trc_modification": True,
            "no_model_geometry_change": True,
            "no_dof_unlock": True,
            "no_lstm_retrain": True,
            "phase2_roll_preserved": True,
        },
        "trials": {},
    }
    for trial in TRIALS:
        tdir = ROOT / "output" / "opensim_upper" / "trials" / trial
        trc = tdir / "markers.trc"
        scaled = tdir / "Model_upper_scaled.osim"
        scale = measure_body_scales(trc, GENERIC)
        stations = model_station_geometry(scaled)
        pred = predicted_geometry(trc)
        # compare with phase3 include/exclude if present
        phase3 = ROOT / "output" / "opensim_upper" / "phase3" / trial
        ik_compare = {}
        for label, sub, trc_name in (
            ("A_exclude", "A_wrists_excluded", "markers_ik_without_wrists.trc"),
            ("B_include_w5", "B_wrists_included_w5", "markers_ik_with_wrists.trc"),
        ):
            loc_dir = phase3 / sub
            if not loc_dir.exists():
                continue
            locs = [
                p
                for p in sorted(loc_dir.glob("*model_marker_locations.sto"))
                if "axial" not in p.name
            ]
            if not locs:
                continue
            ik_compare[label] = fitted_forearm(phase3 / trc_name, locs[0])

        # all-frame widths for 103341 even when spacing rejects
        width_all = {}
        for side, (_e, lat, med) in SIDES.items():
            vals = []
            for fr in read_trc(trc)["frames"]:
                a, b = fr["points"].get(lat), fr["points"].get(med)
                if a is None or b is None:
                    continue
                vals.append(float(np.linalg.norm(np.asarray(a) - np.asarray(b))))
            width_all[side] = _stats(vals)

        payload["trials"][trial] = {
            "participant": participant_meta(tdir),
            "scale": {
                "forearm": scale["forearm"],
                "humerus": scale["humerus"],
                "torso": scale["torso"],
                "height_used_for_scale": False,
                "mass_used_for_scale": False,
            },
            "scaled_model_stations": stations,
            "predicted": pred,
            "predicted_width_all_frames": width_all,
            "ik_forearm_compare": ik_compare,
            "notes": {
                "scale_matches_lstm_mid": True,
                "width_not_scaled": True,
                "phase3_model_forearm_is_ik_fitted_not_scale_target": True,
            },
        }

    # representable-error matrix
    payload["representable_errors"] = {
        "along_forearm_midpoint_length": {
            "representable": "partially",
            "how": "forearm Y-scale matches LSTM elbow-to-mid by construction; IK may still stretch fitted length when competing markers win",
        },
        "pair_width_mismatch": {
            "representable": False,
            "how": "both stations on radius; X/Z not scaled with forearm; rigid ~75.4 mm",
        },
        "pair_roll_about_forearm": {
            "representable": "partially",
            "how": "pro_sup free in [0, 2.09] rad but competes with elbow study markers; wrist_flex/dev locked in IK",
        },
        "independent_lat_med_placement": {
            "representable": False,
            "how": "fixed local stations; no ulna-attached medial styloid DOF",
        },
        "wrist_flexion_plane": {
            "representable": False,
            "how": "CAMERA_MODEL_LOCKED locks wrist_flex and wrist_dev during IK",
        },
    }

    out_path = OUT / "phase4_report.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "trials": list(payload["trials"])}, indent=2))


if __name__ == "__main__":
    main()
