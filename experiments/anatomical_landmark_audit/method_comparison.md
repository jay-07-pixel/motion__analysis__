# Method comparison

Background papers for this note: OpenCap (Uhlrich et al. 2023) and Marker Data Enhancement (Falisse et al. 2025), plus Pose2Sim’s use of the OpenCap LSTM. Local evidence: Marker Enhancer experiments on this D455f and the 20261011-144812 full-body audit.

## 1. Pose2Sim available marker augmentation

**What it can do:** from a dense OpenPose-named TRC, emit 43 extra 3D trajectories for OpenSim IK.

**What it cannot do (evidence):** certify ISB palpation on our subject; enforce bone length; ingest our CSV/TRC without conversion; replace missing depth; recover pronation from sparse keypoints (Chaumeil et al.; our `anatomical_frames.py` already states styloids are absent).

**Overlap with our ME work:** same Stanford arm/body LSTM family. We already ran the **arm** model. Repeating it through Pose2Sim’s ONNX wrapper is not a new anatomical measurement.

**Verdict for landmark acquisition:** experimental geometry only, **unsuitable** as the source of truth for epicondyles/styloids/ASIS.

## 2. Stanford Marker Enhancer (this repo)

Already isolated under `data/experiments/marker_enhancer_*` and `tools/marker_enhancer_*.py`. Arm model: 7 inputs (Neck proxy + shoulders, elbows, wrists) → 8 outputs. Body model exists in literature/weights but was not adopted as the DOF solution.

**Local result:** inference works; **pair geometry is not stable** enough for a joint-frame axis.

**Verdict:** preserve for comparison; **do not** feed predicted points into `dof_angles_3d.py`.

## 3. Direct observation (current D455f + MediaPipe)

**Strengths:** explicit pixels and depth; invalid depth stays empty; full-body Pose-33 now on disk; Hands MCPs give a **practical** palm frame when both hands detect.

**Limits:** keypoints ≠ bones; Hands drop at distance (right extras missing ~30% of 20261011 frames); depth outliers stretch segments; ~7 fps.

**Verdict:** this is the **only currently verified 3D** in the project. Suitable for “what the models + depth saw”. Unsuitable as ISB landmarks without extra hardware or a labelled study.

## 4. Direct anatomical annotation / measurement

Physical palpatory markers (or a calibrated pointer) on epicondyles, styloids, ASIS/PSIS, malleoli, with the same D455f or a mocap system.

**Strengths:** the only method that **defines** the anatomical point by protocol.

**Limits:** session time, clothing, soft-tissue artefact; still need camera calibration. Not software-only.

**Verdict:** required if the goal is scientifically labelled ISB landmarks. Not proposed as an immediate code change.

## 5. Other software options (evidence-bound)

| Option | Role | Not a solution for |
|---|---|---|
| OpenSim IK on MP keypoints | Smooth joint angles of a **scaled model**, not new landmarks | Inventing epicondyles |
| Pose2Sim filtering of TRC | Low-pass observed XYZ | Anatomy |
| Geometric offsets (mid-shoulder Neck, 50% bone) | Proxies | GHJC / styloids |
| RTMPose / HALPE toes | Different 2D model; still not styloids | Upper-limb ISB |
| Sports2D | Single-camera **planar** angles | 3D bony landmarks |

IK **cannot** recover information that is not in the markers or the model constraints. Using IK on augmenter points would **bake in** LSTM assumptions.

## 6. What software inference cannot fix

- A single RGB-D silhouette does not uniquely locate medial vs lateral epicondyles.
- Pronation is not observable from one wrist keypoint.
- LSTM “anatomy” is a **conditional average** of training mocap, biased to upright, forward, ~60 Hz, triangulated keypoints.
- Depth holes cannot be honestly filled by the augmenter without pretending interpolated points were measured.

## 7. Practical ranking for *this* project now

1. **Keep** observed RealSense–MediaPipe (and Hands) 3D; quality-filter outliers for analysis, without renaming them as bones.
2. **Keep** current DOF as documented practical/ISB-style **from visible points**.
3. **Do not** enable Pose2Sim/ME augmentation in the live pipeline.
4. If ISB bony points become mandatory: a **separate** marker/palpation protocol, not a network.
