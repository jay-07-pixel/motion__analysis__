# Read-only diagnostic investigation

**Recording:** `data/diagnostics/20261012-150535/`  
**UTC:** 2026-10-12T09:35:35 to 09:36:28 (~53 s)  
**Frames:** 406 in `frames.csv`; 0 dropped. Person landmarks from frame 35.  
**This folder:** plots + `analysis.json`. Production code was not changed.

`landmarks.csv` does **not** store `u_px`/`v_px`. 2D presence is inferred from a landmark row plus `visibility`. Depth validity is `depth_valid` / `xyz_missing` / empty `raw_*_m`.

---

## A. Foot and ankle depth failure

### Evidence from the recording

| Landmark | Rows | Valid XYZ | Invalid % | Vis median | Vis when XYZ missing and vis≥0.5 |
|---|---:|---:|---:|---:|---:|
| left_ankle | 367 | 4 | 98.91 | 0.545 | 252 |
| left_heel | 367 | 4 | 98.91 | 0.244 | 0 |
| left_foot_index | 367 | 3 | 99.18 | 0.318 | 0 |
| right_ankle | 367 | 3 | 99.18 | 0.476 | 130 |
| right_heel | 367 | 3 | 99.18 | 0.202 | 0 |
| right_foot_index | 367 | 3 | 99.18 | 0.293 | 0 |
| left_knee (control) | 367 | 366 | 0.27 | 0.935 | — |
| left_hip (control) | 367 | 367 | 0.00 | — | — |

- **2D is present when XYZ is invalid.** Every missing-XYZ foot row has `visibility > 0`. Ankles are often above `pose.min_visibility` 0.5 (left 252/363 missing frames; right 130/364) and still have empty metres.
- Heels and `foot_index` never reach vis 0.5 in this take. MediaPipe is uncertain there, but deprojection does **not** use that threshold (see code).
- Knees/hips have almost complete XYZ, so aligned depth **works** on this stream for torso/thigh.
- The only valid foot XYZ is frames **40–43**, vis 0.10–0.25 (below 0.5). Z is 0.79–1.18 m (inside `min_m` 0.3 … `max_m` 6.0), similar to hip/knee Z on those frames. Y jumps ~0.11–0.21 m per frame (e.g. left_ankle Y 0.055 → 0.164 → 0.326 → 0.475 m). Those four frames are **not** a stable foot measurement; they look like accidental hits on some depth, then holes again.

### How XYZ is accepted or rejected (code)

1. `src/ui/session.py` `process_frame` (~317–323) calls `attach_camera_xyz` with `config.yaml` `camera.depth`: `min_m: 0.3`, `max_m: 6.0`, `sample_window: 5`. Diagnostics write **raw** keypoints **before** `CameraXyzSmoother` (~328–330). Smoother hold/blend did not create the 99% holes.
2. `src/geometry/deproject.py` `attach_camera_xyz` (47–50): `z_m = source.depth_distance_m(...)`. If `z_m < min_depth_m or z_m > max_depth_m`, XYZ is set to `None`. **No visibility check.** `get_distance` returning **0** is treated as invalid (`0 < 0.3`).
3. `src/capture/live_realsense.py` (90–98, 120–122, 137–139): colour 1280×720 + depth 1280×720 @ 30, `rs.align(rs.stream.color)`, then `median_depth_m` on the aligned depth frame.
4. `src/capture/depth_util.py` `median_depth_m` (16–38): integer `(u,v)`, 5×5 window, skip out-of-bounds, skip `z_m <= 0`. If **no positive samples**, return **0.0**.

Most likely mechanism: at foot pixels, the 5×5 window has **no positive aligned depth**, so `median_depth_m` returns 0, then `attach_camera_xyz` rejects it. Not a missing 2D landmark, not a 2 m clip of a good Z, not DOF code.

This recording cannot prove out-of-image `v_px` (pixels were not logged). It also cannot prove floor IR vs stereo hole vs occlusion; those are the physical explanations consistent with “2D in frame, depth 0, knees OK”.

### Ranked causes (A)

1. **Aligned depth hole at foot pixels** (`get_distance` = 0 in the 5×5 window) — floor/grazing angle / IR / occlusion / bottom of FOV. Strongest: 2D present, vis often ≥0.5 for ankles, knees valid, rare valid Z in-range but unstable.
2. **Rejection of z=0 as below `min_m` 0.3** (`deproject.py` 48–50; `depth_util.py` 33–36). Confirmed by code path; does not distinguish hole vs true <0.3 m.
3. **Low MediaPipe foot confidence** (heels/`foot_index` vis always <0.5). Contributes to bad 2D location, not to the ankle-with-vis≥0.5 failures.
4. **Out-of-bounds pixel** after rounding. Possible but **not measured** (no `u_px`/`v_px` in this CSV).
5. Alignment off — **unlikely**: hips/knees/wrists have ~0% invalid XYZ on the same aligned frame.

---

## B. Elbow and wrist predictions

Upper LSTM only (8 study markers + 4 geometric centres). No lower LSTM in this file. Predictions start frame **47** (12 frames after first pose; matches live min buffer 8 + a few). Observed elbows/wrists: 366–369 valid XYZ of 367–370 rows (~0–1 missing).

### Counts and motion scale

| Point | Valid | Missing | std XYZ (m) | range X (m) | mean f2f (mm) | p95 f2f (mm) | jumps >5 / 10 / 20 cm |
|---|---:|---:|---|---:|---:|---:|---|
| left_elbow obs | 367 | 0 | 0.026 / 0.016 / 0.056 | 0.445 | 8.1 | 27.8 | 11 / 2 / 1 |
| right_elbow obs | 366 | 1 | 0.018 / 0.012 / 0.043 | 0.237 | 8.9 | 27.2 | 9 / 0 / 0 |
| left_wrist obs | 368 | 1 | 0.027 / 0.014 / 0.067 | 0.376 | 9.8 | 29.7 | 8 / 2 / 1 |
| right_wrist obs | 369 | 1 | 0.040 / 0.010 / 0.067 | 0.577 | 11.4 | 26.1 | 10 / 3 / 2 |
| L/r elbow centres (LSTM) | 359 | 0 | ~0.018 / 0.008 / 0.027 | ~0.19–0.22 | **4.3–4.6** | **8.4–8.4** | 0 / 0 / 0 |
| L/r wrist centres (LSTM) | 359 | 0 | ~0.020 / 0.013 / 0.010 | ~0.16–0.20 | **4.2–4.6** | ~11 | 0 / 0 / 0 |

Observed median f2f ~3–6 mm (still-ish) with p95 ~27 mm and a few 10–64 cm spikes (left_elbow max f2f **0.638 m**, Z range **0.99 m**). Those spikes are **measurement jumps**, not plausible arm motion at ~8 fps.

LSTM f2f is about **half** the observed mean and has **no** >5 cm jumps except 2 frames on `L_lelbow_study` (max 5.7 cm). Smoother / smaller range than MediaPipe+depth. **Smooth ≠ anatomically correct.**

### Predicted vs observed (same frame, camera metres)

| Pair | n | mean dist (cm) | corr X/Y/Z | mean offset pred−obs (cm) dX,dY,dZ |
|---|---:|---:|---|---|
| left_elbow vs L_elbow_centre | 359 | 4.2 | 0.95 / 0.90 / 0.95 | +1.5, +2.8, −2.6 |
| right_elbow vs r_elbow_centre | 359 | 3.7 | 0.91 / 0.92 / 0.95 | −0.8, +2.1, −2.7 |
| left_wrist vs L_wrist_centre | 358 | 2.8 | 0.96 / 0.94 / 0.76 | −0.6, +1.9, +1.8 |
| right_wrist vs r_wrist_centre | 359 | 2.1 | 0.95 / 0.84 / 0.78 | −0.5, +1.2, +1.3 |

Centres sit ~2–4 cm from the MediaPipe joint, track X/Y/Z together (corr ≳ 0.84 except wrist Z ~0.76–0.78). Lateral vs medial: left `L_lelbow_study` mean X 0.211 vs observed 0.172 vs `L_melbow_study` 0.165 (lateral more +X). Right `r_lelbow_study` mean dX vs observed **−4.3 cm**. That is consistent with **not** swapping L/R outputs. Wrist Z correlation vs `r_lwrist_study` is only **0.32** — that pair does not follow observed depth well.

### Preprocess / stale frame / coordinates

- Inputs: MediaPipe shoulders/elbows/wrists/hips, Hip-centred, /1.80 m, official mean/std (`src/pose2sim_live/upper_lstm.py`). Live window T=16.
- `yup_to_camera` is applied; predicted Z ~1.40 m vs observed elbows ~1.35–1.43 m — same camera-forward scale, not OpenSim leftover.
- `pred_lag_ms` median **0**, mean **0.55**, max **199** (one stall). Infer latency mean **8.7 ms**. Because `process_frame` runs ~117 ms after `observe()`, the async infer usually finishes **before** the CSV snapshot, so lag as defined (frame time − last LSTM **input** time) is ~0. That does **not** prove the overlay is anatomically live; it only shows the buffer’s last input is this frame when infer wins the race.
- 47 frames of pose before first prediction: buffering, expected.

---

## C. Performance

| Metric | Value |
|---|---|
| Requested camera FPS | 30 (`config.yaml`) |
| Wall FPS | **7.67** (`summary.json`) |
| Mean `process_ms` | **119.1** (median 117.4, p95 187, max 248) |
| Implied processing FPS | **8.39** |
| Median inter-frame dt | **127 ms** |
| Dropped frames | **0** |
| LSTM `infer_latency_ms` mean | **8.7** (max 96) |

LSTM is ~7% of `process_ms`. The rest of `process_frame` (`session.py` 284–450) is MediaPipe Pose **and** Hands on 1280×720, deproject all joints, angles, overlay, CSV. `config.yaml` already notes Hands is expensive at whole-body distance. Timing data do **not** split Pose vs Hands vs Tk; they do show the loop cannot keep 30 fps, and ONNX is not the main cost.

---

## Ranked causes (whole recording)

1. Foot XYZ holes: **no positive aligned depth** at 2D foot pixels (then z=0 rejected).
2. Live FPS ~8: **Pose+Hands (+draw/CSV)** in one thread, not LSTM (8.7 ms).
3. Observed elbow/wrist spikes: **depth outliers** on otherwise good joints (rare 10–64 cm f2f).
4. LSTM “too smooth / smaller range”: **model + 16-frame window**, not a swapped axis (Z scale and L/R laterality match camera metres).
5. Unmeasured: foot `(u,v)` vs image bounds / raw `get_distance` histogram.

---

## Proposed next steps (not implemented)

Kept separate from the investigation. Needs approval before any code change.

### Minimal proposed fix (diagnostics first, then optional depth)

1. **Diagnostics only:** add `u_px`, `v_px`, `depth_sample_m` (raw median before min/max), `depth_n_valid_in_window`, `in_image` to `landmarks.csv`. No behaviour change.
2. **Optional depth (only after 1 confirms holes):** slightly larger foot window and/or a second sample toward the shin; still **never** invent Z or copy `foot_index` as a toe. Do not lower `min_m` without measuring true Z.
3. Do **not** treat LSTM smoothness as a reason to add a motion filter yet; observed spikes are rare depth outliers, LSTM is already smoother than the measurements.

### Safe validation

1. Repeat a 30–60 s 3D live diagnostic standing so feet are on-screen, well lit, not under the desk.
2. Confirm new columns: when `xyz_missing=1`, is `depth_sample_m` 0 and `depth_n_valid_in_window` 0, or is Z out of [0.3, 6]?
3. If `v_px` ≥ 720 or < 0, that is bounds, not floor IR.
4. Compare knee vs ankle invalid % on the same take.
5. No DOF, mapping, or weight changes. New diagnostic folder only.

### Plots in this folder

- `fig_A_invalid_xyz_feet.png`
- `fig_A_foot_visibility_when_xyz_missing.png`
- `fig_B_obs_vs_pred_z.png` / `fig_B_obs_vs_pred_x.png`
- `fig_B_f2f_displacement.png`
- `fig_C_process_ms.png` / `fig_C_pred_lag_ms.png`
- `analysis.json`
