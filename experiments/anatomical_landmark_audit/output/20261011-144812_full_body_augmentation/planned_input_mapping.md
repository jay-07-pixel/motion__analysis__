# Planned input mapping (not applied)

For a future approved run. No TRC or CSV was rewritten.

## Coordinate systems

| File | Units | Axes | Use |
|---|---|---|---|
| CSV `x_m,y_m,z_m` | m | camera: X right, Y down, Z forward; origin colour optical centre | source of truth |
| `*_camera_3d_metre.trc` | m | same | keep for camera-space plots |
| `*_opensim_yup.trc` | m | X_cam, −Y_cam, Z_cam | only if the LSTM was trained Y-up (OpenCap/Pose2Sim) |

Do not mix Y-down features into a Y-up-trained network.

## Name map (same physical *model* point, not anatomy)

| MediaPipe (our TRC/CSV) | OpenPose / augmenter feature | Allowed? |
|---|---|---|
| `left_shoulder` / `right_shoulder` | `LShoulder` / `RShoulder` | Yes, as **model** keypoints |
| `left_elbow` / `right_elbow` | `LElbow` / `RElbow` | Yes, model |
| `left_wrist` / `right_wrist` | `LWrist` / `RWrist` | Yes, model |
| `left_hip` / `right_hip` | `LHip` / `RHip` | Yes, model — **not** ASIS or HJC |
| `left_knee` / `right_knee` | `LKnee` / `RKnee` | Yes, model — **not** epicondyles |
| `left_ankle` / `right_ankle` | `LAnkle` / `RAnkle` | Yes, model — **not** malleoli |
| `left_heel` / `right_heel` | `LHeel` / `RHeel` | Yes, model |
| `left_foot_index` / `right_foot_index` | `LBigToe` / `RBigToe` or SmallToe | **No silent map.** Different landmarks. Body LSTM requires both BigToe and SmallToe. |
| — | `Neck` | Geometric midpoint of shoulders (proxy), if used must be labelled `Neck_proxy` |
| — | `Hip` (midHip) | Geometric midpoint of hips (reference), not a pelvis marker |

Pose2Sim `add_smalltoe_data` synthesises SmallToe from BigToe/Heel/Hip line. That is a **geometric proxy**, not a measurement. Do not apply unless approved and labelled `LSmallToe_proxy`.

## Missing depth

Empty `x_m,y_m,z_m` stay empty. LSTM historically needs interpolated series. Proposed default if approved: **drop frames** that lack any required feature XYZ. Interpolation only with an `imputed=true` column.

## Output naming (when inference exists)

Keep observed names unchanged. Estimated markers: OpenCap `*_study` or ME `*_augmenter`, plus provenance `estimated_lstm`. Never rename to `lateral_epicondyle` or `radial_styloid`.
