# 3D landmark quality audit — 20261011-144812

Observed MediaPipe Pose/Hands keypoints with RealSense camera metres. **Not** ISB anatomical landmarks. No interpolation. No Marker Enhancer. No Pose2Sim augmenter.

Source (read-only): `C:\Projects\motion__analysis__\data\output\20261011-144812\keypoints_2d.csv`
Person-tracked CSV frames: **752** (index 89–846, 6.04–109.50 s).

## Selection criteria (reproducible)

Quiet standing is **not guessed**. It is the longest consecutive run of frames that pass all of:

1. Segment length in `[0.5, 1.5] ×` that segment’s **session median** (all valid lengths).
2. Mid-hip step between consecutive frame IDs ≤ **0.05 m** (else `hip_jump`).
3. Max step of any of left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle ≤ **0.08 m** (else `core_joint_jump`).
4. At least **6/10** segments have a valid in-band length.
5. Mid-hip step ≤ **0.02 m** (quiet motion gate).
6. Run duration ≥ **4.0 s**.

A missing CSV row is `missing_row` (not detected / not written). Empty XYZ with a row present is `invalid_depth`. Gaps in frame index are not interpolated.

Quality-filtered frames (gates 1–4, not requiring quiet hip): **526 / 752**.
Implausible-length frames: 211. Hip jumps: 14. Core-joint jumps: 42.

## Pose vs Hands coverage

- Pose-33 rows present, median per frame: 33.0 (max 33).
- Pose-33 valid XYZ, median: 33.0; frames with all 33 XYZ: 686.
- Hands-extra rows (34 names, not Pose-33), median: 34.0.
- Hands-extra valid XYZ, median: 34.0; any Hands XYZ: 660; all 34: 434.

Wrist/index/pinky/thumb names can come from Pose or Hands overwrite. Hands extras are the unambiguous Hands-only set.

## Quiet interval

**Selected:** frames 357–504 (43.053–62.966 s), 148 frames, 19.91 s, median mid-hip step 0.0015 m.

Other runs ≥ 4.0 s: 4 besides the selected one. All consecutive quiet runs: 15.

## Segment metrics — all person-frames

| Segment | n valid | n missing | median m | stdev m | p05 | p95 | stdev/median |
|---|---:|---:|---:|---:|---:|---:|---:|
| shoulder_width | 749 | 3 | 0.3527 | 0.1102 | 0.0577 | 0.3710 | 0.3125 |
| hip_width | 743 | 9 | 0.1981 | 0.0626 | 0.0323 | 0.2099 | 0.3160 |
| left_upper_arm | 749 | 3 | 0.2838 | 0.0511 | 0.1147 | 0.3059 | 0.1800 |
| right_upper_arm | 749 | 3 | 0.2891 | 0.0544 | 0.1933 | 0.3135 | 0.1881 |
| left_forearm | 728 | 24 | 0.2831 | 0.0600 | 0.1977 | 0.3226 | 0.2121 |
| right_forearm | 728 | 24 | 0.2954 | 0.0948 | 0.2232 | 0.4459 | 0.3210 |
| left_thigh | 743 | 9 | 0.4153 | 0.0361 | 0.3859 | 0.4301 | 0.0868 |
| right_thigh | 743 | 9 | 0.4283 | 0.0529 | 0.3200 | 0.4498 | 0.1235 |
| left_shank | 746 | 6 | 0.3851 | 0.0414 | 0.3588 | 0.4253 | 0.1076 |
| right_shank | 746 | 6 | 0.3797 | 0.0407 | 0.3470 | 0.4128 | 0.1072 |

## Segment metrics — quality-filtered (rejected frames excluded, not filled)

| Segment | n valid | n missing | median m | stdev m | p05 | p95 | stdev/median |
|---|---:|---:|---:|---:|---:|---:|---:|
| shoulder_width | 526 | 0 | 0.3554 | 0.0266 | 0.2928 | 0.3722 | 0.0748 |
| hip_width | 526 | 0 | 0.1997 | 0.0124 | 0.1843 | 0.2101 | 0.0623 |
| left_upper_arm | 526 | 0 | 0.2823 | 0.0164 | 0.2547 | 0.3023 | 0.0580 |
| right_upper_arm | 526 | 0 | 0.2923 | 0.0195 | 0.2449 | 0.3127 | 0.0666 |
| left_forearm | 507 | 19 | 0.2900 | 0.0257 | 0.2304 | 0.3100 | 0.0887 |
| right_forearm | 507 | 19 | 0.2986 | 0.0267 | 0.2429 | 0.3167 | 0.0894 |
| left_thigh | 526 | 0 | 0.4187 | 0.0111 | 0.3968 | 0.4308 | 0.0266 |
| right_thigh | 526 | 0 | 0.4303 | 0.0328 | 0.3486 | 0.4476 | 0.0762 |
| left_shank | 526 | 0 | 0.3798 | 0.0128 | 0.3623 | 0.4008 | 0.0337 |
| right_shank | 526 | 0 | 0.3800 | 0.0137 | 0.3677 | 0.4066 | 0.0360 |

## Segment metrics — selected quiet window

| Segment | n valid | n missing | median m | stdev m | p05 | p95 | stdev/median |
|---|---:|---:|---:|---:|---:|---:|---:|
| shoulder_width | 148 | 0 | 0.3540 | 0.0221 | 0.2928 | 0.3614 | 0.0624 |
| hip_width | 148 | 0 | 0.1985 | 0.0019 | 0.1942 | 0.2009 | 0.0097 |
| left_upper_arm | 148 | 0 | 0.2720 | 0.0144 | 0.2456 | 0.2916 | 0.0529 |
| right_upper_arm | 148 | 0 | 0.2918 | 0.0192 | 0.2466 | 0.3046 | 0.0658 |
| left_forearm | 129 | 19 | 0.2927 | 0.0339 | 0.2091 | 0.3173 | 0.1159 |
| right_forearm | 129 | 19 | 0.2975 | 0.0270 | 0.2336 | 0.3177 | 0.0906 |
| left_thigh | 148 | 0 | 0.4261 | 0.0040 | 0.4169 | 0.4306 | 0.0094 |
| right_thigh | 148 | 0 | 0.4401 | 0.0065 | 0.4281 | 0.4474 | 0.0147 |
| left_shank | 148 | 0 | 0.3678 | 0.0096 | 0.3585 | 0.3903 | 0.0262 |
| right_shank | 148 | 0 | 0.3751 | 0.0124 | 0.3667 | 0.4069 | 0.0331 |

## Plots

- `segment_lengths.png` — ten segment lengths vs time; red = rejected; green band = quiet window.
- `frame_quality.png` — mid-hip step, core-joint step, Pose vs Hands XYZ counts.

## Decision

Go rule from `decision.md`: on the quiet subset, thigh/shank **stdev / median ≤ 0.05**. The same rule is reported for the other segments for completeness.

- **Stable on quiet window:** hip_width, left_thigh, right_thigh, left_shank, right_shank
- **Not stable enough:** shoulder_width, left_upper_arm, right_upper_arm, left_forearm, right_forearm

These lengths are distances between **MediaPipe model points**, not bone lengths between palpable landmarks.

### Next test

- If thighs/shanks pass: keep using observed lower-limb model points on quiet/filtered frames; do **not** enable augmenter.
- Forearms/wrists: expect worse stability (depth + Hands). Do not treat them as ISB forearm bones.
- Hands extras remain intermittent; practical hand frames only when MCP XYZ exist.
- Do not modify `dof_angles_3d.py` from this audit.

## Limitations

- Session median used for the length band is computed on **all** valid frames, including motion and outliers; it is robust (median) but not a standing-only anthropometry.
- ~7 fps timestamps; a 5 cm/frame hip jump is a strong spike, not gait.
- Camera metres include depth noise; filtering drops frames rather than smoothing XYZ.

