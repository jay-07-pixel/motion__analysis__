# Pose2Sim full-body marker augmentation — 20261011-144812

**Inference ran.** Official `Pose2Sim.markerAugmentation.augment_markers_all` executed in the isolated `pose2sim-offline` environment. The **upper** Stanford LSTM (v0.3) produced **8 estimated arm markers** on **722 frames**. The **lower** LSTM was **skipped by the official code** because `RBigToe`, `LBigToe`, `RSmallToe`, and `LSmallToe` were absent.

Orange overlay points are **estimated**, not verified anatomical ground truth.

## Installation

| Item | Value |
|---|---|
| Environment | conda `pose2sim-offline` at `C:\Users\jayjo\anaconda3\envs\pose2sim-offline` |
| Python | 3.11.17 |
| Package | `pose2sim==0.10.49` |
| Runtime | `onnxruntime==1.31.0` CPU (`CPUExecutionProvider`; Azure EP listed but unused) |
| GPU/CUDA | not installed |
| Isolated from | project Python; `marker-enhancer-offline` |

Models shipped in the wheel (no extra download):

- `...\Pose2Sim\MarkerAugmenter\LSTM\v0.3_upper\{model.onnx, mean.npy, std.npy}`
- `...\Pose2Sim\MarkerAugmenter\LSTM\v0.3_lower\{model.onnx, mean.npy, std.npy}`

`augment_markers_all` locates them next to `markerAugmentation.py`. Height **1.80 m** and mass **61.0 kg** were passed via `config['project']['participant_height']` / `participant_mass`, then appended as two extra LSTM features after Hip-centering and height normalisation.

Full command log: `install_record.json`. Application `config.yaml`, live pipeline, 7-DOF code, and Marker Enhancer env were not modified.

## Model input requirements (from installed source)

- **Upper features (OpenPose names):** Neck, RShoulder, LShoulder, RElbow, LElbow, RWrist, LWrist. **Hip** is also required for centering.
- **Lower features:** Neck, shoulders, hips, knees, ankles, heels, **R/L SmallToe, R/L BigToe**.
- **TRC:** metres, OpenSim-style; Pose2Sim documentation and this trial used **Y-up**.
- **Sampling:** entire trial as one LSTM sequence `(1, n_frames, n_features)`. This trial ~**7.30 Hz**. No fixed 30-frame window.
- **Missing values:** augmenter does not interpolate. Incomplete upper-feature frames were **dropped** (30 of 752). Empty depth was not filled.
- **Outputs if both models run:** 8 upper `*_study` + 35 lower `*_study`.
- **This run:** 8 upper only.

MediaPipe names were **not** fed raw. A copy of the Y-up TRC was renamed.

## Mapping (defensible only)

Observed MediaPipe → LSTM keys: shoulder, elbow, wrist, hip, knee, ankle, heel (L/R).

Proxies (labelled, not observed):

- `Neck` = midpoint of shoulders
- `Hip` = midpoint of hips

**Not mapped:** `left_foot_index` / `right_foot_index` are **not** BigToe or SmallToe. No invented medial/lateral foot, ASIS/PSIS, or HJC.

See `applied_mapping.json` and `marker_provenance.json`.

## Inference result

- Upper LSTM: **ran** (`v0.3_upper`).
- Lower LSTM: **skipped** (`WARNING: v0.3_lower ... requires ['LSmallToe', 'RSmallToe', 'RBigToe', 'LBigToe']`).
- Estimated markers: **8** — `r_lelbow_study`, `r_melbow_study`, `r_lwrist_study`, `r_mwrist_study`, `L_lelbow_study`, `L_melbow_study`, `L_lwrist_study`, `L_mwrist_study`.
- Finite estimated XYZ rows: **5776** (722 × 8).
- Pair lengths (estimated lateral–medial, metres; **unstable**, not usable as fixed anatomy):

| Pair | median | stdev | min–max |
|---|---|---|---|
| right elbow | 0.039 | 0.019 | 0.007–0.118 |
| left elbow | 0.040 | 0.020 | 0.002–0.119 |
| right wrist | 0.048 | 0.020 | 0.004–0.104 |
| left wrist | 0.038 | 0.016 | 0.003–0.083 |

The official writer left the 8 estimate columns **unnamed** in the LSTM TRC header (header-replace mismatch). Coordinates were recovered as the last 24 floats per row in the documented upper output order. Header `NumFrames=743` is also stale; **722** data rows were parsed.

`feet_on_floor=False`. Calibration, triangulation, and OpenSim IK were not run.

## Files

All under `experiments/anatomical_landmark_audit/output/20261011-144812_full_body_augmentation/`:

| Path | Role |
|---|---|
| `source_opensim_yup_copy.trc` | Copy of original Y-up TRC (original preserved) |
| `trial/pose-3d/input_mapped_yup.trc` | Mapped experimental input |
| `trial/pose-3d/input_mapped_yup_LSTM.trc` | Official augmenter output (8 unnamed trailing XYZ) |
| `augmented_markers_yup.csv` | Estimated XYZ, frame/time, metres, provenance |
| `marker_provenance.json` | observed / proxy / estimated / not produced |
| `applied_mapping.json` | Name map and drop policy |
| `inference_status.json` | Confirm augmenter returned |
| `run_summary.json` | Counts and pair stats |
| `install_record.json` | Env, versions, commands |
| `viz_observed_vs_estimated.png` | Observed skeleton vs overlay |
| `viz_estimated_viewpoints.png` | Three viewpoints |
| `viz_quiet_window.gif` | Quiet-window sequence (t ≈ 43.05–62.97 s) |
| `viz_rotatable_3d.html` | Drag-to-rotate Plotly overlay |

Coordinate convention: **OpenSim Y-up metres** (X right, Y up, Z forward).

## Remaining blocker (lower / true full body)

To run `v0.3_lower` the official augmenter needs four **named** markers: `RBigToe`, `LBigToe`, `RSmallToe`, `LSmallToe`. This recording has only MediaPipe `foot_index` (and heel). Mapping `foot_index` onto both toes, or inventing a second toe, was **not** done.

**Smallest decision needed to continue lower-body inference:**

1. **Do not map toes** — keep this upper-only result (current, honest).
2. **Approved experimental proxy** — explicitly allow a labelled proxy (e.g. `foot_index` → `BigToe` only, and a documented offset or duplicate for `SmallToe`), stored as `proxy` not observed. Limitations: MediaPipe `foot_index` is one ray toward the toes, not hallux vs 5th metatarsal; a duplicate or offset would be **fabricated geometry**; lower LSTM outputs (ASIS/PSIS/HJC/thigh/shank clusters) would inherit that error.
3. **New labelled data** — capture or hand-label distinct big/small toe 3D points.

Until (2) or (3), there is no valid full-body (35-marker) overlay.
