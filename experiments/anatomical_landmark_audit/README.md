# Anatomical landmark acquisition — findings

Investigation only. No pipeline, recording, DOF calculator, or previous experiment output was modified. Pose2Sim is **not** in this repository; the augmenter was read from the public `perfanalytics/pose2sim` tree (`Pose2Sim/markerAugmentation.py`, `Pose2Sim/common.py`) plus OpenCap / Marker Enhancer publications.

The 2026-10-11 take (`data/output/20261011-144812`) is a **full-body** recording: 752 person-tracked CSV frames, all 33 MediaPipe Pose names, 34 extra Hands names, RealSense camera metres. It is not an upper-body-only file.

## What we already measure

Depth-deprojected MediaPipe points in the colour-camera frame (X right, Y down, Z forward). The current DOF path (`src/analysis/dof_angles_3d.py`, `src/geometry/anatomical_frames.py`) **already documents** that it does **not** have:

- ISB thorax markers (C7, T8, IJ, PX)
- Glenohumeral joint centre
- Medial/lateral humeral epicondyles
- Radial/ulnar styloids

Shoulder plane/elevation and elbow flexion use **shoulder, elbow, wrist, hips**. Wrist flexion/deviation use a **practical** hand frame (wrist + MCPs), not an ISB radius frame.

## What is missing for ISB-style segment frames

Those bony landmarks are **not observed**. MediaPipe `left_elbow` / `left_wrist` are model keypoints on the image, not palpable epicondyles or styloids. Hands give palm orientation when detection holds (~70–80% of this take), not forearm bone landmarks.

## Pose2Sim marker augmentation

Pose2Sim’s optional augmenter is the **Stanford OpenCap LSTM** (v0.3): learned 3D offsets for 43 extra markers from sparse keypoints, height, and mass. Upper-limb outputs (`r_lelbow_study`, `r_melbow_study`, `r_lwrist_study`, `r_mwrist_study`, and left-side analogues) are the **same class of point** as our Marker Enhancer `RElbow_augmenter` / `RMElbow_augmenter` / wrist pair.

They are **learned estimates**, not measurements. Published training RMSE is on **synthesised** mocap/video pairs (~15 mm arm model on OpenCap’s test split). Independent work reports larger wrist errors. Our own ME runs on this camera showed **pair lengths changing by centimetres** while the measured elbow was nearly still. The LSTM does **not** enforce rigid epicondyle or styloid distance.

It does **not** consume our MediaPipe-named, NaN-containing, ~7 Hz RGB-D TRC without unsupported conversions (OpenPose names, Hip/Neck, BigToe/SmallToe, interpolation, Y-up, 60 Hz training).

## Recommendation

**Do not enable Pose2Sim or Marker Enhancer augmentation for anatomical landmark acquisition.**  
**Do not change `dof_angles_3d.py` or `anatomical_frames.py`.**

Use observed RealSense–MediaPipe points (and Hands when present) as **model landmarks**. Treat ISB bony points as **unavailable** unless a later, separate, physical-marker or palpation study is approved.

Optional later experiment (isolated directory only): augmenter **geometry check** on the full-body TRC, with explicit go/no-go on pair-length stability. That experiment cannot certify anatomy even if it “looks like” elbows.

Details: `landmark_inventory.md`, `pose2sim_augmentation_audit.md`, `method_comparison.md`, `decision.md`.
