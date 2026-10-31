# Full-body marker coverage — 20261011-144812

This folder is a **new** experiment. The previous upper-LSTM run in the parent directory was not overwritten.

**Lower-body LSTM did not run.** The official Pose2Sim 0.10.49 augmenter still lacks four distinct 3D toe inputs. Nothing was fabricated. The visualization here is the **maximum honest set** from existing assets: observed MediaPipe+depth landmarks, two geometric proxies, and the eight already-computed upper LSTM estimates.

Orange points are **estimated**, not ground-truth anatomy.

## Phase 1 — Complete marker set (installed source)

### Observed (this recording)

MediaPipe Pose 33 (names in `marker_inventory.json`) plus Hands extras in `keypoints_2d.csv`. Camera metres converted to OpenSim Y-up only in this experiment (`Y_up = −Y_cam`).

### Upper LSTM v0.3

| Role | Names |
|---|---|
| Required inputs | Neck, RShoulder, LShoulder, RElbow, LElbow, RWrist, LWrist |
| Also required | Hip (centering root) |
| Outputs (8) | `r_lelbow_study`, `r_melbow_study`, `r_lwrist_study`, `r_mwrist_study`, `L_lelbow_study`, `L_melbow_study`, `L_lwrist_study`, `L_mwrist_study` |

Status: **ran** previously; 722 frames; copied into this folder’s combined CSV.

### Lower/torso LSTM v0.3

| Role | Names |
|---|---|
| Required inputs | Neck, R/L Shoulder, R/L Hip, R/L Knee, R/L Ankle, R/L Heel, **R/L SmallToe, R/L BigToe** |
| Outputs (35) | `r.ASIS_study`, `L.ASIS_study`, `r.PSIS_study`, `L.PSIS_study`, `r_knee_study`, `r_mknee_study`, `r_ankle_study`, `r_mankle_study`, `r_toe_study`, `r_5meta_study`, `r_calc_study`, `L_knee_study`, `L_mknee_study`, `L_ankle_study`, `L_mankle_study`, `L_toe_study`, `L_calc_study`, `L_5meta_study`, `r_shoulder_study`, `L_shoulder_study`, `C7_study`, `r_thigh1..3_study`, `L_thigh1..3_study`, `r_sh1..3_study`, `L_sh1..3_study`, `RHJC_study`, `LHJC_study` |

Status: **not produced**. Those 35 names do **not** include GHJC, sternum, IJ, PX, T8, radial/ulnar styloids, or palpated hallux/5th metatarsal.

### Input classification

| Input | Status |
|---|---|
| Shoulders, elbows, wrists, hips, knees, ankles, heels | Observed MediaPipe, renamed to OpenPose keys |
| Neck, Hip | Geometric **proxies** (mid-shoulders / mid-hips) |
| `foot_index` | Observed, **not** mapped to BigToe or SmallToe |
| R/L BigToe, R/L SmallToe | **Missing** |

Pose2Sim’s BLAZEPOSE skeleton aliases MediaPipe ids 31/32 as `LBigToe`/`RBigToe` and has **no SmallToe**. That is a name alias, not four measured toes. Installed `markerAugmentation.py` **skips** `v0.3_lower` when those four names are absent. It does not call `add_smalltoe_data` in 0.10.49.

## Phase 2 — Toes

This recording contains:

- `keypoints_2d.csv` (u,v + XYZ when depth was valid)
- `overlay.mp4` (skeleton drawn on RGB)
- **No** aligned depth images, **no** `.bag`

Feet are visible as **dark sneakers** in overlay frames 357/430/504 (see `foot_inspect/`). MediaPipe places one `foot_index` near each shoe tip. Hallux vs 5th metatarsal are not separately observed. Overlay clicks cannot be deprojected: aligned depth was not saved.

Project tooling: no foot-landmark detector. Live pipeline was not changed. A bag click-to-XYZ tool is ready (does not invent coordinates):

`experiments/anatomical_landmark_audit/scripts/label_toes_from_bag.py`

## What was produced here

| File | Role |
|---|---|
| `full_body_markers_yup.csv` | Observed + Neck/Hip proxies + 8 upper estimates; metres; provenance |
| `marker_coverage.csv` | observed / proxy / estimated / missing / unavailable |
| `marker_inventory.json` | Exact I/O names |
| `quality_checks.json` | Quiet-window segments and LSTM pair lengths |
| `viz_observed_vs_estimated.png` | Observed skeleton vs overlay |
| `viz_estimated_viewpoints.png` | Three viewpoints |
| `viz_quiet_window.gif` | Quiet sequence |
| `viz_rotatable_3d.html` | Drag to rotate |
| `capture_plan.md` | Next physical step |
| `foot_inspect/` | Overlay crops of the shoes |

Coordinate convention: **OpenSim Y-up metres**.

## Quality (quiet window 43.05–62.97 s, 148 frames)

Observed hip width and both thighs have stdev/median ≤ 0.05. Right heel–`foot_index` does not (0.053). LSTM elbow/wrist pair lengths remain variable (median ~0.04 m, stdev ~0.02 m). Stability is **not** anatomical accuracy.

## Blocker

Lower LSTM needs four **distinct observed 3D** points: `LBigToe`, `LSmallToe`, `RBigToe`, `RSmallToe`.

**Smallest action from you:** record a RealSense **.bag** with both feet clearly in view (see `capture_plan.md`), then run the labeling command in that file. After labeled XYZ exist, the lower+upper augmenter can be run on a new experimental TRC copy.
