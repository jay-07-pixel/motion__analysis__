# Pose2Sim marker augmentation — source audit

**Local checkout:** none. No `pose2sim` package, ONNX weights, or `MarkerAugmenter/` tree exists under this project. Facts below are from:

- https://github.com/perfanalytics/pose2sim `Pose2Sim/markerAugmentation.py` (fetched 2026-10-11)
- `Pose2Sim/common.py` (`add_shoulder_data`, `add_neck_hip_data`, `add_smalltoe_data`)
- Pose2Sim README / Config.toml documentation (earlier audit)
- OpenCap paper: Uhlrich et al., *PLOS Computational Biology* 2023 (PLOS Comp Biol e1011462)
- Marker enhancer paper: Falisse et al., *IEEE TBME* 2025; PMC11275905
- This repo’s Marker Enhancer experiment (`tools/marker_enhancer_offline_inference.py`, `data/experiments/marker_enhancer_*`)

Augmentation is **optional** in Pose2Sim (`use_augmentation = false` is supported). It was **not** run on our recordings.

## 1. What it is

`augment_markers_all` loads a **Stanford LSTM, OpenCap v0.3**, converted to ONNX (`augmenterModelName = 'LSTM'`, `augmenter_model = 'v0.3'`).

It is two networks:

| Model | Feature (input) markers | Response (output) markers |
|---|---|---|
| `v0.3_lower` | Neck, R/L Shoulder, Hip line via R/L Hip, R/L Knee, Ankle, Heel, SmallToe, BigToe | 35 names, e.g. `r.ASIS_study`, `L.PSIS_study`, `r_knee_study`, `r_mknee_study`, `r_ankle_study`, `r_mankle_study`, `r_toe_study`, `r_5meta_study`, `r_calc_study`, thigh/shank clusters, `RHJC_study`/`LHJC_study`, `C7_study`, `r_shoulder_study` |
| `v0.3_upper` | Neck, R/L Shoulder, R/L Elbow, R/L Wrist | 8 names: `r_lelbow_study`, `r_melbow_study`, `r_lwrist_study`, `r_mwrist_study`, `L_lelbow_study`, `L_melbow_study`, `L_lwrist_study`, `L_mwrist_study` |

Pose2Sim comments: “Estimate the position of 43 additional markers.” Copyright line in the file: “Copyright 2022, OpenCap”. License on this file header: Apache-2.0 (Pose2Sim package is BSD-3-Clause; augmenter origins are OpenCap).

Our ME arm LSTM outputs the **same eight arm targets** under names `RElbow_augmenter`, `RMElbow_augmenter`, `RWrist_augmenter`, `RMWrist_augmenter`, and left analogues (`OUTPUT_MARKERS` in `tools/marker_enhancer_offline_inference.py`). Pose2Sim `*_study` names are OpenCap’s labels for that set, not a new measurement method.

## 2. Input requirements (from code)

1. A **.trc** in `pose-3d/` with OpenPose-style marker names, not MediaPipe `left_shoulder`.
2. All feature markers present, or skip that augmenter with a warning.
3. If missing, **geometric fillers** may run **before** the LSTM:
   - Shoulders: Hip + **0.53 m in Y** + 0.1 m along hip-to-hip (`add_shoulder_data`). Fixed metres, OpenSim Y-up assumption.
   - Neck / Hip: midpoints of shoulders / hips (`add_neck_hip_data`). Same Neck proxy our ME used.
   - SmallToe: formula from BigToe, Heel, hip-line (`add_smalltoe_data`). **Not** a detected 5th toe.
4. Features: subtract **Hip** (tiled XYZ), divide by **height**, append height and **mass**, then z-score with `mean.npy` / `std.npy`.
5. Tensor shape `(1, n_frames, n_features)` — the **entire trial** is one LSTM sequence.
6. OpenCap training used **60 Hz** and **0.5 s** windows with **18 mm** Gaussian noise on keypoints (paper). Our full-body CSV timestamps imply **~7.3 fps**.
7. README: augmenter “Will not work properly if missing values are not interpolated (NaN in the TRC).” Our export leaves invalid depth **empty**.

## 3. Output meaning (documented vs assumed)

OpenCap states the 43 markers “correspond to what is commonly used for marker-based motion capture” so OpenSim IK is better posed. Training targets were **synthesised anatomical markers from OpenSim-processed marker-based datasets**, not live palpation on D455f subjects.

**Verified from source names:** outputs are called `*_study` / in ME `*_augmenter`.

**Not verified here:** that `r_melbow_study` is the medial humeral epicondyle on our subject to millimetre ISB tolerance. Name similarity is not a mapping proof.

Our ME reports state explicitly: predicted lateral/medial elbow and wrist markers, **not true epicondyles or styloids**.

## 4. Type of point

| Step | Type |
|---|---|
| MediaPipe + RealSense XYZ | Observed (image keypoint + depth) |
| Neck, Hip midpoints | Geometrically derived |
| Fake shoulders (0.53 m) / SmallToe recipe | Geometrically derived, **not** our measured shoulders/toes |
| LSTM `*_study` | **Learned estimate** |
| `feet_on_floor` Y offset | Global vertical shift of all markers |

No rigid-body or constant-length constraint appears in `markerAugmentation.py`. Pair distance can change every frame.

## 5. Coordinates, units, missing data

- OpenCap/Pose2Sim TRC: metres, OpenSim-style **Y up** (augmentation subtracts Hip in that frame; `feet_on_floor` uses min **Y**).
- Our `*_camera_3d_metre.trc`: Y **down**. Using it untransformed would invert height.
- Our `*_opensim_yup.trc`: `Y_osim = -Y_cam` only. Marker **names** are still MediaPipe (`left_knee`, not `LKnee`).
- Units: metres in both our export and the augmenter.
- Missing depth: empty cells; augmenter expects dense interpolated series.

## 6. Can our single-camera RGB-D TRC be input?

**Not directly.** Gaps:

| Gap | Fact | Assumption if we ignore it |
|---|---|---|
| Names | Export uses MediaPipe names | Rename `left_shoulder` → `LShoulder` etc. |
| `foot_index` vs BigToe/SmallToe | Pose has one toe proxy | Treat `foot_index` as `RBigToe` and synthesise SmallToe |
| Neck / Hip | Not in CSV | Midpoints (same as ME) |
| Empty XYZ | 7023 empty TRC cells on 20261011 | Interpolate (forbidden as “true” depth) |
| Rate | ~7.3 fps vs 60 Hz training | LSTM still runs; temporal prior is wrong |
| 3D origin | One D455f deprojection vs OpenCap **multi-view triangulation** | Domain shift; not quantified here |
| Height/mass | Required features | Must be measured, not guessed |

So: the augmenter **consumes a TRC of 3D points**, not “must triangulate inside Pose2Sim”. That does **not** make RGB-D 3D distributionally equivalent to the training 3D.

## 7. Published validation (what it actually shows)

**Uhlrich et al. 2023 (OpenCap):** arm LSTM test RMSE **15.2 mm**, body **8.0 mm**, on a **held-out split of the synthesised** dataset (108 h mocap → fake video keypoints). OpenCap vs optical mocap also reports kinematic and selected marker errors in the live lab protocol (phones, two views). That protocol is **not** a D455f.

**Falisse et al. 2025:** original enhancer **fails on out-of-distribution** poses (lying, not facing forward). Improved enhancer reduces kinematic error on a broader set. Pose2Sim ships **v0.3** (OpenCap original LSTM), not automatically the 2025 model.

**Independent (Chaumeil et al. 2024, cited in search):** wrist marker errors on the order of **centimetres** (reported 21–37 mm class); authors note augmenter is **unlikely to recover pronation** from OpenPose-like sparsity.

**This lab (same LSTM family, D455f):**

- Offline `20261005-150538`: elbow/wrist augmenter vs MP joint ~4–5 cm; pair length 4.7–10 cm.
- Live `20261007-144738`: pair length still ~2–11 cm; quiet right elbow still ~5.8 cm pair range.

That is **evidence against** using the pair as a rigid epicondylar/styloid axis on this camera.

## 8. Pose2Sim vs “Pose2Sim pose estimation”

Base Pose2Sim: RTMPose 2D → **multi-camera triangulation**. We are not using that 3D. Augmentation is a **downstream LSTM** on already-3D markers. Enabling it does not add a second camera or fix depth holes.
