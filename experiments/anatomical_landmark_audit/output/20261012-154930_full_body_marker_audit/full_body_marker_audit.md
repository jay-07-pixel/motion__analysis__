# Full-body visibility and anatomical-marker path (read-only)

**UTC folder stamp:** 2026-10-12 15:49  
**Production code:** not modified  
**Latest foot-framing report:** `experiments/anatomical_landmark_audit/output/20261012-154200_live_diag_20261012-153538/`  
**Latest diagnostic recording on disk:** `data/diagnostics/20261012-153538/`  
**No newer diagnostic folder** exists yet (`20261012-150535`, `20261012-130544` are older). Phase 1 framing validation waits on the next take after the D455f is tilted/moved.

Coordinate frame for all observed XYZ: colour-camera metres, X right, Y down, Z forward. Invalid depth stays empty. MediaPipe world landmarks are not used.

Installed augmenter (verified on disk, not downloaded this session):

- Pose2Sim 0.10.49 in `pose2sim-offline`
- `LSTM/v0.3_upper/{model.onnx,mean.npy,std.npy}`
- `LSTM/v0.3_lower/{model.onnx,mean.npy,std.npy}`
- Live subject constants in code: height 1.80 m, mass 61 kg (`src/pose2sim_live/upper_lstm.py`)

The live upper LSTM estimates **eight arm study markers only**. It does not estimate pelvis, legs, feet, C7, or thorax ISB points.

---

## Phase 1 — Observed-landmark baseline (until a new recording arrives)

Source: `20261012-153538` person-rows (~460 frames with pose). Image 1280×720.

| Landmark | Valid XYZ | `in_image=0` (v≥720) | In-image `n_valid=0` | Median v_px |
|---|---:|---:|---:|---:|
| left_hip / right_hip | 460 / 460 | 0 | 0 | (torso, well inside) |
| left_knee / right_knee | 460 / 460 | 0 | 0 | ~506 / ~514 |
| left_ankle | 459/460 | 0 | 1 | 629 |
| right_ankle | 460/460 | 0 | 0 | 638 |
| left_heel | 458/460 | 1 | 1 | 647 |
| right_heel | 458/460 | 2 | 0 | 650 |
| left_foot_index | 452/460 | 7 | 1 | 673 |
| right_foot_index | 442/460 | 18 | 0 | 687 |

Ankle XYZ is already reliable. Heel/foot-index misses are almost all **below the native image**. Three in-image holes sit at v 704–719.

**When the next `data/diagnostics/<stamp>/` appears, check (no code change):**

1. Columns: `u_px`, `v_px`, `depth_sample_m`, `depth_n_valid_in_window`, `in_image`, `depth_valid`, `xyz_missing`.
2. Framing success: `in_image=1` for ankle, heel, foot_index on standing frames; `foot_index` v ≲ 680 (about 40 px of floor); `n_v_ge_720` near 0.
3. Compare invalid %: hip/knee stay ~0; ankle/heel/foot_index should approach hip/knee if framing worked.
4. Split remaining misses: `in_image=0` vs `in_image=1` and `depth_n_valid_in_window=0` vs depth present but outside [0.3, 6] m.
5. Do **not** clamp OOB pixels or fill XYZ. Do **not** change `sample_window` / `min_m` / `max_m` unless class-B holes dominate after framing is fixed.

---

## Phase 2 — Marker classes

### 1. Directly observed MediaPipe + RealSense XYZ

These are **model landmarks**, not ISB palpable points. XYZ exists only when aligned depth at `(u_px, v_px)` is in range.

**Head / face (Pose)**  
`nose`, `left/right_eye` (+ inner/outer), `left/right_ear`, `mouth_left`, `mouth_right`  
Approx: face mesh on RGB, not skull landmarks (nasion, tragion, etc.).

**Neck**  
No MediaPipe neck joint. Do not treat `nose` or mid-shoulder as C7.

**Shoulders**  
`left_shoulder`, `right_shoulder` — not GHJC, not acromion.

**Elbows / wrists**  
`left/right_elbow`, `left/right_wrist` — one point per joint, not epicondyles or styloids. Hands may overwrite wrist/tips.

**Torso**  
No IJ, PX, T8, C7 as observed points. Trunk in current DOF uses both shoulders and both hips (`anatomical_frames.py`; not changed here).

**Pelvis / hips**  
`left_hip`, `right_hip` — not ASIS, PSIS, or HJC.

**Knees / ankles**  
`left/right_knee`, `left/right_ankle` — not epicondyles or malleoli.

**Feet**  
`left/right_heel` — not protocol calcaneus.  
`left/right_foot_index` — **one toe-ray per foot. Not BigToe and not SmallToe.**

**Hands (Hands graph, 21×2)**  
Stems in `src/pose/hands.py` `HAND_STEMS`. Shared Pose names (`wrist`, `thumb`, `index`, `pinky`) are overwritten when Hands locks. Knuckles/tips are graph points, not ISB hand markers. Hands often drop at whole-body distance.

**On `20261012-153538`:** hips/knees 100% XYZ; ankles ≥99.8%; feet limited by framing as in Phase 1. Upper LSTM was live (446 predicted frames); lower LSTM was not (no calibrated toes in that file).

---

### 2. Existing geometric points

| Name | Recipe | Approximation | When live |
|---|---|---|---|
| `Neck` | Midpoint of MP shoulders | Not C7 / C7_study | LSTM input only |
| `Hip` | Midpoint of MP hips | Not pelvis origin / HJC | LSTM centering root |
| `L/r_elbow_centre` | Midpoint of upper-LSTM medial/lateral pair | Not ISB elbow JC | Upper LSTM live |
| `L/r_wrist_centre` | Midpoint of upper-LSTM styloid pair | Not ISB wrist JC | Upper LSTM live |
| `L/r_knee_centre` | Midpoint of lower-LSTM knee pair | Not ISB knee JC | Lower LSTM live |
| `L/r_ankle_centre` | Midpoint of lower-LSTM ankle pair | Not ISB ankle JC | Lower LSTM live |

Implemented in `src/markers/centres.py`. Knee/ankle centres **cannot** appear until lower LSTM outputs exist.

---

### 3. Existing Pose2Sim **upper** LSTM estimates (live today)

**Model:** v0.3_upper ONNX. Input `(1,T,23)`, output `(1,T,24)`. Window T=16 live. Hip-centred, /height, official mean/std. Camera Y-down ↔ OpenSim Y-up in `lstm_common` / `upper_lstm.py`.

**Inputs (must have valid camera XYZ):** MP shoulders, elbows, wrists; `Neck` and `Hip` proxies.

**Outputs (8 only):**

`r_lelbow_study`, `r_melbow_study`, `r_lwrist_study`, `r_mwrist_study`,  
`L_lelbow_study`, `L_melbow_study`, `L_lwrist_study`, `L_mwrist_study`

These are **OpenCap-style study markers**, not measured epicondyles/styloids. On `20261012-150535` they sat ~2–4 cm from MP elbow/wrist centres, smoother than depth, L/R laterality consistent with camera +X. Smooth ≠ anatomically correct.

---

### 4. Additional markers that **could** be estimated with **already installed** models

All of the following require **four distinct live toes** (session RGB-D clicks + rigid foot from ankle/heel/`foot_index`) **and** valid XYZ for shoulders, hips, knees, ankles, heels. Code path: `FootCalibrator` → `LiveLowerLstm` → `missing_lower_inputs` (`src/markers/foot_calib.py`, `coordinator.py`, `lower_lstm.py`). `foot_index` is **never** copied to BigToe/SmallToe.

**Model:** v0.3_lower ONNX. Input 47 (15×3 + height + mass), output 105 (35×3).

| Output | Region | Inputs needed | Inputs available after framing + toe clicks? | How to validate (no DOF change) |
|---|---|---|---|---|
| `r.ASIS_study`, `L.ASIS_study`, `r.PSIS_study`, `L.PSIS_study` | Pelvis | Lower LSTM 15 | Yes if hips/knees/ankles/heels/toes have XYZ | Pelvis width vs MP hip width; standing quiet; **not** palpated ASIS/PSIS |
| `RHJC_study`, `LHJC_study` | Hip JC (estimated) | same | same | Distance to MP hip (expect offset); left/right consistency; not a functional HJC trial |
| `C7_study` | Neck/thorax (estimated) | same | same | Near mid-shoulder / above; not a palpated C7 |
| `r_shoulder_study`, `L_shoulder_study` | Shoulder (estimated) | same | same | Offset vs MP shoulder; **not** GHJC/acromion |
| `r_knee_study`, `r_mknee_study`, `L_knee_study`, `L_mknee_study` | Knee condyles (estimated) | same | same | Midpoint vs MP knee (~cm); medial/lateral vs camera +X |
| `r_ankle_study`, `r_mankle_study`, `L_ankle_study`, `L_mankle_study` | Malleoli (estimated) | same | same | Midpoint vs MP ankle; same laterality check |
| `r_toe_study`, `L_toe_study`, `r_5meta_study`, `L_5meta_study`, `r_calc_study`, `L_calc_study` | Foot (estimated) | same (toes as **inputs**, not outputs copies) | same | Compare to calibrated clicks and MP heel/`foot_index`; large disagreement = bad toes or framing |
| `r/L_thigh1..3_study`, `r/L_sh1..3_study` | Cluster markers | same | same | Rigid-ish on a still thigh/shank; not skin markers |

**Calibrated toes (algorithm, not a detector):** `LBigToe`, `LSmallToe`, `RBigToe`, `RSmallToe` from four depth-valid clicks, then rigid reconstruct. Validate: click depth in [0.3, 6] m and `in_image=1`; reconstruct error vs a second still click; drop frame if ankle/heel/`foot_index` XYZ missing (already: reconstruct returns nothing).

**Geometric knee/ankle centres:** automatic once lower pairs exist. Validate with the same midpoint-vs-MP test.

None of these are live on `20261012-153538` because toes were not calibrated.

---

### 5. Markers that need new data, sensors, annotation, or an **uninstalled** model

| Marker | Why unavailable | What would be required | Do not do |
|---|---|---|---|
| IJ, PX, T8 | Not in v0.3 upper or lower outputs (`catalog.py` `UNAVAILABLE`) | New model, manual thorax annotation, or extra sensors | Relabel `C7_study` or mid-shoulder as IJ/T8 |
| Acromion L/R | Not an LSTM output | Palpation, extra RGB model, or training | Relabel MP shoulder |
| GHJC L/R as ISB | Not an LSTM output; `*_shoulder_study` is not GHJC | Functional calibration or published GHJC regression with stated error | Use MP shoulder as GHJC |
| Distinct automatic BigToe/SmallToe | No HALPE-26 / OpenPose BODY_25 / RTMPose in this project | Approved install + download, or keep session clicks | Duplicate `foot_index` |
| Palpated ASIS/PSIS, epicondyles, malleoli, styloids | LSTM study names are **estimates** from OpenCap-style training | MoCap/palpation comparison set | Treat `*_study` as measured |
| Stanford Marker Enhancer **body** `.h5` | Only `lstm/arm/` is local | Approved TF env + body weights | Mix unofficial weights |
| True ISB thorax/humerus/forearm frames beyond current 7-DOF | Missing IJ/PX/T8/GHJC/epicondyles | After markers exist, separate DOF task | Change `dof_angles_3d.py` now |

Face Mesh (config can enable) is overlay 2D, not RealSense metres, not skull protocol.

---

## Region summary

| Region | Observed now | Estimated now (upper LSTM) | Estimable with installed lower LSTM + toe clicks | Blocked |
|---|---|---|---|---|
| Head | Pose face 11 | — | — | Skull protocol |
| Neck | — | — | `C7_study` only | Palpated C7 |
| Torso | Shoulders + hips | — | `C7_study`, `*_shoulder_study` | IJ, PX, T8, acromion |
| Shoulders | MP shoulder | — | `*_shoulder_study` | GHJC |
| Elbows | MP elbow | 4 study + 2 centres | — | Palpated epicondyles |
| Wrists | MP/Hands wrist | 4 study + 2 centres | — | Palpated styloids |
| Pelvis | MP hips | — | ASIS/PSIS study, HJC study | Palpated pelvis |
| Hips | MP hip | — | HJC study | Functional HJC |
| Knees | MP knee | — | 4 study + 2 centres | Palpated epicondyles |
| Ankles | MP ankle | — | 4 study + 2 centres | Palpated malleoli |
| Feet | heel + foot_index | — | 6 study foot + 4 calibrated toes | Auto big/small toe detector |

---

## Proposed implementation stages (not started)

1. **Framing (you, hardware).** Next diagnostic take with 30–40 px of floor under the toes. Re-run Phase 1 checks. Stop if OOB remains dominant.
2. **Keep observed XYZ honest.** No clamp, no invented depth, no `foot_index` as a toe.
3. **Calibrate toes** in a 3D session once standing feet have stable XYZ. Confirm four `calibrated_toe` rows in a new diagnostics folder.
4. **Enable lower LSTM** (code already boots after 4/4 clicks). Record 35 `predicted_lower` names. Compare centres to MP joints; do not change DOFs.
5. **Optional later (needs approval):** HALPE/OpenPose toe detector; thorax/GHJC model; Stanford body weights; ISB DOF expansion.

Wait for the next diagnostic recording and for approval before any pipeline change.
