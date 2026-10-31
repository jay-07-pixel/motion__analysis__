# Lower-limb 3D landmark stability — quiet window

Observed MediaPipe Pose points + RealSense camera metres. **Not** ISB hip, knee, ankle or foot landmarks. No interpolation, augmenter, or DOF changes.

Source (read-only): `C:\Projects\motion__analysis__\data\output\20261011-144812\keypoints_2d.csv`

## Window (from previous quality audit, re-checked)

- Frames **357–504** (43.053–62.966 s)
- **148** CSV frames in that closed range; matches `frame_quality.csv` `in_quiet_window=1`
- Previous median mid-hip step **0.0015 m**

Previous quality audit (stdev/median ≤ 0.05): hip width, both thighs, both shanks. This run **recomputes** those lengths on the same window and adds per-joint XYZ steps, heels, foot_index, ankle–heel and heel–foot_index distances, and left/right symmetry.

## Joint coverage and jitter (quiet window)

| Joint | valid XYZ | missing row | invalid depth | coverage | conf median | step median m | step p95 m | step max m | steps > 20 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| left_hip | 148 | 0 | 0 | 1.000 | 1.000 | 0.0000 | 0.0073 | 0.0163 | 0 |
| right_hip | 148 | 0 | 0 | 1.000 | 1.000 | 0.0000 | 0.0048 | 0.0102 | 0 |
| left_knee | 148 | 0 | 0 | 1.000 | 0.996 | 0.0000 | 0.0050 | 0.0076 | 0 |
| right_knee | 148 | 0 | 0 | 1.000 | 0.995 | 0.0032 | 0.0080 | 0.0100 | 0 |
| left_ankle | 148 | 0 | 0 | 1.000 | 0.990 | 0.0000 | 0.0045 | 0.0069 | 0 |
| right_ankle | 148 | 0 | 0 | 1.000 | 0.993 | 0.0000 | 0.0038 | 0.0055 | 0 |
| left_heel | 148 | 0 | 0 | 1.000 | 0.860 | 0.0000 | 0.0031 | 0.0041 | 0 |
| right_heel | 148 | 0 | 0 | 1.000 | 0.922 | 0.0000 | 0.0053 | 0.0074 | 0 |
| left_foot_index | 148 | 0 | 0 | 1.000 | 0.980 | 0.0000 | 0.0035 | 0.0046 | 0 |
| right_foot_index | 148 | 0 | 0 | 1.000 | 0.987 | 0.0000 | 0.0061 | 0.0101 | 0 |

Joint “stable” rule: coverage ≥ 0.99, median step ≤ 5 mm, no step > 20 mm.

- **Meets joint-stable rule:** left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_heel, right_heel, left_foot_index, right_foot_index
- **Does not:** (none)

## Segment lengths (quiet window)

Go: stdev/median ≤ 0.05 (same as `decision.md` / quality audit).

| Segment | n valid | missing | median m | stdev m | p05 | p95 | stdev/median | go |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hip_width | 148 | 0 | 0.1985 | 0.0019 | 0.1942 | 0.2009 | 0.0097 | yes |
| left_thigh | 148 | 0 | 0.4261 | 0.0040 | 0.4169 | 0.4306 | 0.0094 | yes |
| right_thigh | 148 | 0 | 0.4401 | 0.0065 | 0.4281 | 0.4474 | 0.0147 | yes |
| left_shank | 148 | 0 | 0.3678 | 0.0096 | 0.3585 | 0.3903 | 0.0262 | yes |
| right_shank | 148 | 0 | 0.3751 | 0.0124 | 0.3667 | 0.4069 | 0.0331 | yes |
| left_ankle_heel | 148 | 0 | 0.0586 | 0.0017 | 0.0563 | 0.0623 | 0.0290 | yes |
| right_ankle_heel | 148 | 0 | 0.1181 | 0.0089 | 0.1021 | 0.1319 | 0.0756 | no |
| left_heel_foot_index | 148 | 0 | 0.1463 | 0.0046 | 0.1374 | 0.1506 | 0.0312 | yes |
| right_heel_foot_index | 148 | 0 | 0.2351 | 0.0124 | 0.2085 | 0.2527 | 0.0525 | no |

## Left/right median symmetry

| Pair | left m | right m | |diff| m | rel |
|---|---:|---:|---:|---:|
| thigh | 0.4261 | 0.4401 | 0.0141 | 0.0325 |
| shank | 0.3678 | 0.3751 | 0.0073 | 0.0195 |
| ankle_heel | 0.0586 | 0.1181 | 0.0595 | 0.6736 |
| heel_foot_index | 0.1463 | 0.2351 | 0.0888 | 0.4655 |

Relative difference > 0.05 is a warning (model-point geometry, not bone inequality).

## Decision

- **Segment go:** hip_width, left_thigh, right_thigh, left_shank, right_shank, left_ankle_heel, left_heel_foot_index
- **Segment no-go:** right_ankle_heel, right_heel_foot_index

Reconfirm vs previous audit: hip width, both thighs, both shanks should still pass. Foot segments (ankle–heel, heel–foot_index) were not in that go-list; they are judged here on the same numeric rule.

Reliable enough to **keep as observed lower-limb model landmarks** on quiet/filtered frames: hips, knees, ankles (and the thigh/shank distances they form). 
Heels and foot_index: use only with the joint-stable / segment-go result; they are weaker model points (lower confidence, possible depth outliers historically).

**Do not** build ISB knee/ankle frames from these points (single model knee ≠ two epicondyles; single ankle ≠ two malleoli; foot_index ≠ hallux and 5th metatarsal).

**Do not** enable augmenter or edit `dof_angles_3d.py` / `anatomical_frames.py`.

### Next

- If thighs/shanks/hips still go: a later experiment may **report** observed lower-limb segment lengths over time (filtered), still as model distances.
- Anatomical coordinate frames for the lower limb remain **blocked** until bony landmarks exist.
- Upper limb remains on the previous quality-audit no-go list for standing stability.

## Plots

- `lower_limb_segment_lengths.png`
- `lower_limb_joint_steps.png`

