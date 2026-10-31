# Landmark inventory

Coordinate convention for all observed 3D points in this project:

- Origin: colour-camera optical centre
- X right, Y down, Z forward
- Units: metres
- Source: aligned RealSense depth at `(u_px, v_px)`, `src/geometry/deproject.py` `attach_camera_xyz`
- Invalid depth: empty XYZ, not MediaPipe world landmarks

Latest full-body file: `data/output/20261011-144812/keypoints_2d.csv` (752 person-frames; all Pose-33 names present). Coverage numbers below are from `experiments/pose2sim_whole_body/output/20261011-144812/01_audit_recording.txt`.

MediaPipe names are **model landmarks**. They are not labelled ISB palpable points in this inventory.

## 1. Currently observed — MediaPipe Pose (33)

| Name | Typical use in this repo | On 20261011-144812 | Anatomical status |
|---|---|---|---|
| `left_shoulder`, `right_shoulder` | Trunk origin (mid-shoulder); arm proximal | ~749/752 valid XYZ | Skin/model point near shoulder, **not** GHJC or acromion by protocol |
| `left_elbow`, `right_elbow` | Elbow vertex; humerus/forearm long axis | ~749/752 | **Not** medial or lateral epicondyle |
| `left_wrist`, `right_wrist` | Distal forearm; often overwritten by Hands wrist | ~730/752 | **Not** radial or ulnar styloid |
| `left_hip`, `right_hip` | Trunk (mid-hip); hip width | ~743/752 | **Not** ASIS, PSIS, or hip joint centre |
| `left_knee`, `right_knee` | Written in full-body CSV; unused by current DOF | ~746/752 | **Not** femoral epicondyles |
| `left_ankle`, `right_ankle` | Written; unused by current DOF | ~746/752 | **Not** malleoli |
| `left_heel`, `right_heel` | Written; unused by current DOF | ~743–746/752; lower conf (~0.88–0.91) | Near calcaneus region; not a protocol calcaneus marker |
| `left_foot_index`, `right_foot_index` | Written; unused by current DOF | ~742–745/752; some Z outliers | **Not** hallux vs 5th metatarsal. No separate big/small toe |
| `left_pinky`, `right_pinky`, `left_index`, `right_index`, `left_thumb`, `right_thumb` | Pose finger tips; Hands may overwrite | High coverage on Pose names | Tips, not styloids or MCP protocol markers |
| Nose, eyes (incl. inner/outer), ears, mouth | Face; unused by DOF | Present (unlike the upper-body 20261005 take) | Not thorax landmarks |

## 2. Currently observed — MediaPipe Hands (separate graph)

`src/pose/hands.py`. Merged in `merge_pose_and_hands` (Hands overwrite shared wrist/tip names).

| Name pattern | Role | On 20261011-144812 | Anatomical status |
|---|---|---|---|
| `{side}_wrist` | Hands wrist | Often present via Pose even when extras drop | Not styloids |
| `{side}_index_mcp`, `{side}_pinky_mcp`, `{side}_middle_mcp` | Practical hand frame (`build_hand_frame`) | Left extras missing ~160/752; right ~225/752 | Knuckle **model** points; used as palm width, not ISB hand markers |
| Remaining CMC/PIP/DIP/tips | Finger chains | Same Hands drop rate | Not required for current DOF |

Hands fail more often than Pose at whole-body distance. That is a detection limit, not a naming issue.

## 3. Used by current DOF / frames (do not change these files)

From `anatomical_frames.py` and `dof_angles_3d.py` (read-only for this audit):

| Construct | Points used | Explicitly not available |
|---|---|---|
| Trunk | both shoulders, both hips | C7, T8, suprasternal notch, xiphoid |
| Shoulder plane / elevation | trunk + shoulder + elbow | GHJC, humeral epicondyles, thorax ISB |
| Elbow flexion | shoulder, elbow, wrist; sign from trunk Z | Epicondylar axis |
| Hand frame | wrist, middle_mcp, index_mcp, pinky_mcp | |
| Practical forearm | elbow, wrist, palm Z from hand | Radial styloid, ulnar styloid, pronation/supination |

## 4. Required for intended ISB-style upper-limb (and future lower-limb) frames

### Shoulder / upper arm

| Required anatomical point | Observed now? | Notes |
|---|---|---|
| Glenohumeral joint centre | No | Needs regression or functional calibration; MP shoulder is insufficient |
| Acromion / thorax (C7, IJ, PX, T8) | No | Needed to orient thorax vs humerus (ISB) |
| Lateral humeral epicondyle | No | |
| Medial humeral epicondyle | No | Defines elbow flexion axis if both present |

### Forearm / wrist / hand

| Required anatomical point | Observed now? | Notes |
|---|---|---|
| Radial styloid | No | |
| Ulnar styloid | No | Together define forearm distal frame / pronation |
| 2nd / 5th metacarpal heads (protocol) | Partial proxy | Hands MCP/tips are not the same protocol points |
| 3rd metacarpal / finger axis | Partial | `middle_mcp` used in the practical hand frame only |

### Pelvis / lower limb (future)

| Required anatomical point | Observed now? | Notes |
|---|---|---|
| LASIS, RASIS, LPSIS, RPSIS | No | MP hips are not ASIS/PSIS |
| Hip joint centres | No | |
| Medial/lateral femoral epicondyles | No | MP knee is one point |
| Medial/lateral malleoli | No | MP ankle is one point |
| Calcaneus, 1st/5th metatarsal, hallux | Partial | Heel + one `foot_index`; no 5th metatarsal / small toe |

## 5. Only estimable (not observed)

| Point class | How it would be obtained | Status |
|---|---|---|
| LSTM “study” / “augmenter” markers | OpenCap/Pose2Sim/ME networks | Learned; see other reports |
| Neck, midHip | Midpoint of shoulders / hips (`common.add_neck_hip_data`) | Geometric proxy; already used as ME `Neck` |
| Small toe if missing | Pose2Sim `add_smalltoe_data` from BigToe, Heel, Hip line | Geometric recipe; we do not have `RBigToe`, only `foot_index` |
| GHJC / HJC | Anthropometric regression from pelvis/shoulder | Not implemented; not validated here |

## 6. Depth / rate limits that affect every method

On `20261011-144812`: ~7.3 fps from timestamps; 395 rows invalid depth; bone-length min/max include spikes (e.g. right forearm up to 1.56 m; `left_foot_index` Z to 3.81 m). Any inferred bony landmark inherited from these 3D inputs will inherit those errors.
