# Evidence-based audit — `data/diagnostics/20261012-155914`

**Method:** counts and time series from `frames.csv` and `landmarks.csv`, cross-checked to `summary.json` and to the current source. Production code was not modified. Empty metres were not filled. Earlier reports were not copied as facts.

**Recording clock:** `t_frame_sec` 0.151–76.572 s · UTC 2026-10-12T10:29:14.79Z–10:30:31.45Z · **607** frames · **0** dropped · wall FPS **7.92** (summary) · mean `process_ms` **115.47**.

Landmark classes in this file: **67 observed**, **8** `predicted_upper`, **4** `geometric_centre`. **No** `predicted_lower`, **no** `calibrated_toe`.

---

## 1. Executive summary (confirmed only)

1. This take is **not** a single stationary pose. Hip Z goes from ~1.1 m (early) to ~2.24 m (middle) to ~1.5 m (end). That is walking in the depth direction.
2. A **low trunk-motion** interval can be defined from **hip** kinematics: frames **161–548** (`t` 18.96–69.45 s). Hip median f2f **3.5 mm**; in that window left ankle / left `foot_index` have **0** OOB and **0** in-image depth holes (388/388 valid XYZ). Wrists still move ~12 cm in X. This is **not** a certified quiet-standing trial.
3. Foot XYZ misses on the **whole file** are a mix of **out-of-image pixels** (mostly `v≥720`, some `v<0`) and **in-image zero-depth windows**. Positive depth rejected by `[0.3, 6.0]` m is **rare** (0–2 rows per foot landmark).
4. Those foot misses **cluster in the close/setup start and the walk-out**, not in 161–548.
5. Observed XYZ are camera metres (X right, Y down, Z forward) at plausible standing Z ~2.2 m. Segment **medians** are anatomically plausible (~31 cm upper arm, ~28 cm forearm, ~42 cm thigh, ~41 cm shank) but **min/max explode** (cm to >2 m) — depth/tracking outliers, not bone change.
6. Upper LSTM outputs **eight arm study markers only**. They sit ~2.6–5.3 cm from MediaPipe elbow/wrist. L/R laterality of elbow pairs matches camera +X. This does **not** prove ISB epicondyles/styloids. **No full-body predictions** in this file.
7. `pred_lag_ms` is 0 on **99%** of predicted frames; LSTM `infer_latency_ms` median **5.2**. That does **not** prove the overlay matches the displayed anatomy, only that the buffer’s last **input** time usually equals this frame’s `t_frame_sec`.
8. Loop time is ~115 ms. LSTM infer is ~6% of that. **Pose vs Hands vs capture vs draw vs CSV are not timed separately. The bottleneck is unproven.**

---

## 2. How intervals were chosen (not 60–499)

| Interval | Frames | `t_frame_sec` | Evidence |
|---|---|---|---|
| Start / incomplete pose | 0–1 | 0.15–0.30 | `n_observed` 42 then 0; no mid-hip XYZ |
| Close / setup / walk-back | 2–160 | 0.43–18.9 | First hip XYZ at frame **2**. Frames 2–49: median hip f2f **55 mm**, median hip Z **1.06 m**, median max foot `v` **786** (below 720). LSTM starts frame **50**. |
| Longest low **hip** motion | **161–548** | **18.96–69.45** | Rolling 8-frame median hip f2f `<15 mm` and hip-Z std `<4 cm`, `n_observed` typically 67. Hip Z median **2.24 m**. Foot `v` median **621** (above the 640 guide). Wrist X range still **12–13 cm**. |
| Short hip-quiet blips | 151–152; 555–568 | — | Too short to use as a trial |
| Walk-in toward camera / end | 549–606 (last 40: 566–606) | ~69.4–76.6 | Last 40: median hip f2f **59 mm**, hip Z **1.51 m**, max foot `v` **786** |

**A clean whole-body stationary interval is not established.** 161–548 is the best **trunk-stable, in-frame feet** window in this file. Head X still spans 19 cm there (yaw, lean, or tracking). Do not treat it as a gold-standard quiet pose.

---

## 3. Foot depth — counts from CSV rows

`min_visibility` 0.5 is the **pose draw/LSTM gate** (`config.yaml` / `session.py`). Deprojection does **not** use visibility (`deproject.py` 47–50). Audit counts vis separately.

`depth_valid` and `xyz_missing` **never disagree** on these six landmarks. CSV valid XYZ for `left_ankle` **477** matches `summary.json`.

| Landmark | Detected rows | Frames with no row (of 607) | Valid XYZ | `in_image=0` | In-image `n_valid=0` | In-image **positive** depth outside [0.3, 6] m | vis `<0.5` | vis≥0.5 and XYZ missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| left_ankle | 520 | 87 | 477 | 21 | 22 | **0** | 12 | 33 |
| right_ankle | 520 | 87 | 498 | 4 | 17 | **1** | 14 | 10 |
| left_heel | 520 | 87 | 471 | 29 | 20 | **0** | 15 | 38 |
| right_heel | 520 | 87 | 485 | 14 | 20 | **1** | 28 | 19 |
| left_foot_index | 520 | 87 | 464 | 42 | 12 | **2** | 13 | 48 |
| right_foot_index | 520 | 87 | 457 | 45 | 18 | **0** | 14 | 53 |

OOB is **not** only the bottom edge: left ankle `in_image=0` = 21, of which `v≥720` = 17 and `v<0` = 2 (MediaPipe above the frame). `v` min about **−320** (heels) during the close period.

**Split by period (left ankle / left foot_index):**

| Window | left_ankle OOB / hole / missing XYZ | left_foot_index OOB / hole / missing XYZ |
|---|---|---|
| frames `<161` (75 rows) | 18 / 3 / 21 | 32 / 4 / 37 |
| **161–548 (388 rows)** | **0 / 0 / 0** | **0 / 0 / 0** |
| frames `>548` (57 rows) | 3 / 19 / 22 | 10 / 8 / 19 |

**Code path (cross-check, not inferred from %):**

- Pixels: MediaPipe `v_px = y * height` (`extractor_2d.py` 84–85); y may be outside 0…1.
- Sample: 5×5 `get_distance`, skip non-positive (`depth_util.py` `median_depth_m` / `inspect_depth_sample`).
- Reject: `z_m < 0.3 or z_m > 6.0` including **z=0** (`deproject.py` 47–50; `config.yaml` `camera.depth`).
- Diagnostics XYZ are **pre-smoother** (`session.py` 328–331). Empty `filt_*` columns match recorder (`recorder.py`).

**Confirmed:** when feet are in-frame at ~2.2 m (161–548), ankle/foot_index XYZ succeed. **Confirmed:** remaining whole-file misses are mostly OOB or zero-window, not min/max of a good Z. **Likely:** late-window holes = subject closer again + bottom FOV / alignment edge. **Hypothesis (unverified):** IR/floor grazing vs RGB–depth FOV mismatch on the last rows — CSV cannot distinguish those.

---

## 4. 3D quality (observed vs geometric vs estimated)

**Convention (code, not assumed):** camera metres; origin colour optical centre; X right, Y down, Z forward (`deproject.py` header; summary `coord_frame: camera_3d_metre`). Mid-take hip Z ~2.24 m is a plausible D455 working distance. Y of ankles ~+0.76 m (below the optical centre) vs nose Y ~−0.80 m (above) is consistent with Y-down.

**Observed (depth at pixel):** nose, shoulders, elbows, wrists, hips, knees, ankles, heels, `foot_index`. Hands overwrite some wrist/tip names when present.

Whole-file consecutive f2f (valid XYZ only):

| Point | n valid | median f2f | max f2f | jumps >10 cm |
|---|---:|---:|---:|---:|
| left_hip | 503 | 7 mm | 0.51 m | 5 |
| left_knee | 505 | 8 mm | 0.27 m | 4 |
| left_ankle | 477 | 9 mm | 0.35 m | 7 |
| right_ankle | 498 | 9 mm | **1.94 m** | 13 |
| left_wrist | 526 | 11 mm | **2.22 m** | 20 |
| nose | 485 | 5 mm | 0.32 m | 4 |

Wrist Z range **2.66 m** on the whole file is **not** a 2.6 m arm motion; it includes the walk in Z plus spikes. **Likely depth outliers / ID swaps** when max f2f is tens of centimetres at ~8 fps.

**Segment lengths** (both ends valid XYZ, whole file):

| Segment | n | median | p05–p95 | min / max |
|---|---:|---:|---|---|
| L shoulder–elbow | 499 | 0.311 m | 0.292–0.319 | 0.018 / **2.30** |
| L elbow–wrist | 504 | 0.280 m | 0.261–0.292 | 0.090 / **2.51** |
| L hip–knee | 503 | 0.417 m | 0.398–0.428 | 0.020 / 0.71 |
| L knee–ankle | 474 | 0.411 m | 0.392–0.421 | 0.068 / 0.47 |

Medians fit a ~1.8 m subject. Extremes prove **measurement failure**, not biology. Thigh/shank are stabler than the arm in this take.

**Geometric centres:** midpoint of LSTM medial/lateral pairs (`centres.py`). Not ISB joint centres. Only elbow/wrist centres exist here (no lower LSTM).

**Estimated:** eight `*_study` markers (next section).

---

## 5. Live Pose2Sim upper LSTM

**Mapping (code):** `RESPONSE_MARKERS` in `upper_lstm.py` 28–36, same eight names in this CSV. Inputs: MP shoulders/elbows/wrists + Neck/Hip **proxies** (`extract_upper_frame` 84–107). Hips are for centering only.

**Transforms (code):** `camera_to_yup`: `(X, -Y, Z)` (`upper_lstm.py` 64–66 / `lstm_common.py` 10–15). Hip-center, divide by **1.80 m**, append height & **61 kg**, `(x-mean)/std` (`normalize_lstm_sequence`). Output `*height + Hip`, then `yup_to_camera`. **Height/mass in this recording are not stored in the CSV** — cannot verify they match the person.

**Window (code, not CSV):** `LIVE_WINDOW = 16`, `min_frames = 8`, `max_gap = 2` (`upper_lstm.py` 53–56; `engine.py` 57, 66). This file does **not** log T. First predictions at frame **50** (~5.7 s) is **consistent with** buffering + missing close-range inputs, not proof of T=16.

**Async (code):** `observe` pushes the buffer and sets `_pending`; a thread runs ONNX in `pose2sim-offline` (`engine.py` 100–157). `pred_lag_ms` = `t_frame_sec - t_pred_input_sec` (`recorder.py`). Median lag **0**, 99% exactly 0, max **312 ms**. Infer median **5.2 ms**, mean **6.7**, max **41**. Because `process_frame` is ~115 ms, infer usually finishes **before** the diagnostic snapshot of the **same** loop — lag≈0 is expected and **does not** measure UI display delay (Tk `_tick` is uninstrumented).

**Vs observed (same frame, 477–478 overlaps):**

| Pair | mean dist | corr X/Y/Z | elbow laterality (mean X lat − med) |
|---|---:|---|---|
| L elbow vs `L_elbow_centre` | 4.8 cm | 0.99 / **0.73** / 0.99 | **+4.0 cm** (lateral more +X) |
| R elbow vs `r_elbow_centre` | 5.3 cm | 0.99 / 0.81 / 1.00 | **−5.8 cm** (lateral more −X) |
| L wrist vs `L_wrist_centre` | 3.4 cm | 0.94 / 0.94 / 0.99 | ~0 (markers almost coincide) |
| R wrist vs `r_wrist_centre` | 2.6 cm | 0.99 / **0.37** / 1.00 | +1.2 cm |

Facing the camera, person’s left = camera +X. Elbow pairs match that. **Not an L/R output swap.** Right-wrist **Y** correlation **0.37** means the estimate does not follow observed vertical motion well.

LSTM f2f is similar order to observed in this take (median ~4.5 mm) but still has >10 cm jumps (3–4 frames) — smoothness is **not** guaranteed here.

**Cannot validate from this recording:** ISB identity of study markers; height/mass correctness; T=16 actually used; overlay-vs-anatomy lag; lower body; C7/ASIS/HJC.

---

## 6. Performance

`process_ms` is `(perf_counter after work − t_loop)` with **`t_loop` before `get_frame()`** (`session.py` 294–300, 475). That interval includes **camera wait + align**, Pose, Hands, deproject, LSTM `observe` (not necessarily infer), draw, optional CSV/diagnostics.

CSV has **no** Pose / Hands / render / IO split. `infer_latency_ms` is only the ONNX worker.

| Quantity | Value | What it proves |
|---|---|---|
| mean `process_ms` | 115.5 | Whole loop including grab |
| implied processing FPS | 8.66 | 1000/mean process_ms |
| wall FPS | 7.92 | Includes UI thread |
| LSTM infer / mean process | **5.8%** | ONNX is **not** most of `process_ms` |
| Pose vs Hands vs grab | — | **Unverified** |

**Do not attribute the 8 fps to Hands or Pose from this file.** Both run in the loop (`session.py` 309–314) when Hands is enabled; that is **presence**, not a measured share.

---

## 7. Findings table

| ID | Finding | Evidence | Confidence |
|---|---|---|---|
| F1 | Take includes approach, a long in-place period, and departure | Hip Z 1.06 → 2.24 → 1.51 m; hip f2f | **Confirmed** |
| F2 | 161–548 is low **hip** motion with feet in-frame and 100% left ankle/`foot_index` XYZ | CSV split; foot `v` median 621 | **Confirmed** for those two statements. Whole-body still: **not confirmed** |
| F3 | Whole-file foot misses = OOB + zero-window; min/max of a positive Z is 0–2 rows | Foot table vs `deproject.py` 47–50 | **Confirmed** |
| F4 | Camera Y-down / Z-forward metres | Code + mid-take Z~2.2 m, ankle Y > hip Y > nose Y | **Confirmed** convention; plausibility **high** |
| F5 | Arm/leg length medians plausible; extremes are outliers | Segment table | **Confirmed** |
| F6 | Upper LSTM = 8 arm markers only; laterality consistent | CSV names + mean X | **Confirmed** mapping/laterality. Anatomy: **unverified** |
| F7 | pred_lag≈0 is same-frame input timestamp, not UI proof | recorder + engine | **Confirmed** definition |
| F8 | Bottleneck component | Only whole-loop and infer timers | **Unverified** |
| F9 | Diagnostics ≠ smoothed XYZ | `raw_for_diag` before smoother | **Confirmed** |
| F10 | Lower LSTM / toes absent | No class rows | **Confirmed** |

---

## 8. Unresolved / extra measurements needed

1. Per-stage timers (grab, Pose, Hands, deproject, draw, CSV) if FPS is in scope later.
2. Log LSTM `T` and height/mass into diagnostics.
3. UI `_tick` to camera-frame latency (not in CSV).
4. Intra-window (161–548) vs whole-file segment-length CV, to see if outliers are only walk-in.
5. Which OOB feet are `v<0` vs `v≥720` vs `u` OOB, per frame (partially counted).
6. Physical cause of in-image holes after frame 548 (FOV vs IR vs occlusion) — needs depth image dump, not just median.
7. Independent anatomy (palpation / MoCap) if study markers are to be trusted as epicondyles.
8. Whether Hands ran every frame (`n_observed` 67 vs 21 suggests Pose-only vs Pose+Hands episodes).

---

## 9. Confirmed vs likely vs hypothesis

| Class | Statement |
|---|---|
| **Confirmed** | Foot XYZ works when pixels are in 1280×720 with positive aligned depth (161–548). Failures otherwise are OOB or zero window. Upper LSTM is arms-only. Loop ~115 ms; infer ~7 ms. |
| **Likely** | Early/late failures = subject too close / feet off the bottom while moving. Wrist/ankle 0.3–2 m jumps = bad depth or tracking, not motion. |
| **Hypothesis** | Bottom-row holes = RGB/depth FOV mismatch. Hands dominate FPS. 1.80 m / 61 kg match this subject. |

---

## 10. Prioritized plan (not implemented)

1. **Use a 161–548-style take as the next baseline:** feet above v=640, subject ~2.1–2.3 m, start diagnostics **after** standing still. Do not clamp pixels.
2. **Keep observed XYZ honest** (no fill, no `foot_index` as a toe).
3. **After** that baseline: Calibrate toes → lower LSTM (already coded) → compare study knee/ankle centres to MediaPipe, still without changing DOFs.
4. **Optional instrumentation** (approval): split timers; log T/height/mass.
5. **Do not** change `sample_window` / `min_m` until a take with in-frame feet still shows holes.
6. **Do not** treat upper LSTM as full-body or as ISB.

---

## Plots in this folder

- `fig_hip_motion.png`
- `fig_foot_v.png`
- `fig_foot_depth_validity.png`
- `fig_obs_xyz_left.png`
- `fig_elbow_obs_vs_pred.png`
- `fig_timing.png`
- `analysis.json` (machine-readable counts)
