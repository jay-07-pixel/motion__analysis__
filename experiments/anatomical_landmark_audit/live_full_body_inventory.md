# Live full-body marker inventory (installed models only)

Subject defaults used by live LSTM: height **1.80 m**, mass **61 kg**.
Camera: Intel RealSense D455f, colour + aligned depth, camera metres (X right, Y down, Z forward).

This list is what the live app can actually show. Names are **not** ISB palpable points unless the source says so.

## Observed now (no extra install)

| Source | Markers | 3D |
|---|---|---|
| MediaPipe Pose | 33: face (nose, eyes, ears, mouth), shoulders, elbows, wrists, Pose finger tips, hips, knees, ankles, heels, `foot_index` | RealSense depth at each pixel |
| MediaPipe Hands | 21 × 2: wrist, CMC/MCP/PIP/DIP/tips | RealSense depth |

`foot_index` is **one toe-ray per foot**. It is not BigToe and not SmallToe.

## Geometric proxies (labelled as such)

| Name | Recipe |
|---|---|
| `Neck` | Midpoint of MediaPipe shoulders. Not C7. |
| `Hip` | Midpoint of MediaPipe hips. Not ASIS/HJC. |
| `*_elbow_centre`, `*_wrist_centre` | Midpoint of upper-LSTM medial/lateral pair |
| `*_knee_centre`, `*_ankle_centre` | Midpoint of lower-LSTM medial/lateral pair, only if lower is live |

## Pose2Sim v0.3 upper LSTM (installed, live)

Inputs (7 + Hip for centering): `Neck, RShoulder, LShoulder, RElbow, LElbow, RWrist, LWrist`  
Outputs (8): `r_lelbow_study, r_melbow_study, r_lwrist_study, r_mwrist_study, L_lelbow_study, L_melbow_study, L_lwrist_study, L_mwrist_study`

## Pose2Sim v0.3 lower LSTM (installed; needs four live toes)

Inputs (15): `Neck, RShoulder, LShoulder, RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel, RSmallToe, LSmallToe, RBigToe, LBigToe`  
Outputs (35): ASIS/PSIS study, HJC study, C7 study, shoulder study, medial/lateral knee and ankle, toe / 5th meta / calcaneus study, thigh and shank cluster markers (see `src/markers/catalog.py`).

**Not produced:** IJ, PX, T8, acromion, GHJC as ISB points.

## Toes

No locally installed RGB model outputs distinct big vs small toe. OpenPose/HALPE/RTMPose/YOLOv8-pose are **not** installed.

Live path: **Calibrate toes** — four RGB-D clicks, then rigid-foot tracking from current ankle + heel + `foot_index`. Provenance is session calibration, not MediaPipe and not a duplicated `foot_index`.

Installing a HALPE-26 / OpenPose BODY_25 detector would need explicit approval (new package, license, download size).

## Overlay styles

- Cyan / orange filled dots: observed MediaPipe + valid depth
- Magenta diamonds: Pose2Sim LSTM estimates
- Green squares: calibrated toes
- Cyan crosses: geometric joint centres
