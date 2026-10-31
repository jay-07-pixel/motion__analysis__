# Rest-of-body visualization (toes postponed)

No new lower/torso/pelvis estimates. Installed augmenter has only `v0.3_upper` and `v0.3_lower`. Lower ONNX input is **47** features (15 markers × 3 + height + mass), including BigToe/SmallToe. Official code skips lower if those names are absent. There is no documented subset and no other installed model.

## Included

**Observed (MediaPipe + RealSense, Y-up metres):** shoulders, elbows, wrists, hips, knees, ankles, heels, `foot_index`, head/face Pose points, and Hands extras when XYZ is finite.

**Proxy (green squares):** `Neck` = mid-shoulders; `Hip` = mid-hips. Not C7 / not ASIS / not HJC.

**Estimated (orange, 8):** `r_lelbow_study, r_melbow_study, r_lwrist_study, r_mwrist_study, L_lelbow_study, L_melbow_study, L_lwrist_study, L_mwrist_study` from the existing upper LSTM run.

## Not generated

All 35 lower LSTM outputs (ASIS, PSIS, C7, HJC, thigh/shank clusters, medial/lateral knees and ankles, toe/5meta/calc, shoulder_study).

## Files

- `viz_overview.png`
- `viz_viewpoints.png`
- `viz_sequence.gif`
- `viz_rotatable_3d.html`
- `combined_markers_yup.csv` (43361 observed XYZ rows, 1492 proxy, 5136 estimated)
- `marker_coverage.csv`
- `model_path_check.json`

Convention: OpenSim Y-up metres. Estimates are not verified anatomy.
