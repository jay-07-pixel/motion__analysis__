# 2D and 3D Motion Analysis (one Intel RealSense D455f)

**Student:** Jay  
**Hardware:** Intel RealSense Depth Camera D455f  
**Program:** `python app.py`

This is a **student prototype**, not a medical device. It is not a clinical score for the 9 Hole Peg Test or the Box and Block Test. Angular and linear velocity describe a reach; official test scores are still time and block count.

The app does two jobs in one window:

1. **Measured motion** — MediaPipe joints plus RealSense depth, then shoulder / elbow / wrist angles and speeds.
2. **Extra 3D markers** — Pose2Sim’s Stanford LSTM (OpenCap v0.3) draws estimated study markers on top. Those estimates are **not** fed into the angle calculator.

---

## Where each piece comes from

| Piece | Source | What we use it for |
|---|---|---|
| Colour + aligned depth | Intel RealSense D455f, `pyrealsense2` 2.58 | Camera metres. Not a guessed skeleton. |
| 33 body joints | MediaPipe Pose 0.10.14 (BlazePose) | Observed 2D pixels, then 3D when depth is valid. |
| Finger joints | MediaPipe Hands 0.10.14 | Hand skeleton. Matched to the nearest Pose wrist. |
| Shoulder / elbow / wrist angles | This repo: `src/geometry/anatomical_frames.py`, `src/analysis/dof_angles_3d.py` | One angle per joint from the **observed** points. |
| Extra arm markers | Pose2Sim **0.10.49**, LSTM **v0.3 upper** (Stanford OpenCap augmenter; Falisse / Uhlrich marker-enhancement family) | 8 study points: lateral and medial elbow and wrist, left and right. |
| Extra body markers | Same install, LSTM **v0.3 lower** | 35 study points (pelvis, knee pair, ankle pair, calcaneus, 5th metatarsal, OpenCap toe, shoulder study, C7, hip joint centres, thigh/shank clusters). |
| Four toe inputs | **This session only.** RGB-D clicks on the live image (`src/markers/foot_calib.py`) | The lower model refuses to run without `LBigToe`, `LSmallToe`, `RBigToe`, `RSmallToe`. MediaPipe has one `foot_index` per foot, so we do not copy that point onto both toes. |
| Height and mass | Typed in the window before Start | Official LSTM recipe appends height (m) and mass (kg) after hip-centring and scaling. |
| Paper marker names (SAT, H-LE, ASIS, FM5, …) | Categories in Figure 1 of Bousigues et al., 2025, *Evaluation of a markerless data augmentation algorithm* | A **name map** onto LSTM outputs that already exist. Not a new detector. |

Pose2Sim itself is **not** a pip dependency of `requirements.txt`. ONNX runs in a separate conda env, `pose2sim-offline`, started by `tools/pose2sim_upper_predict_worker.py` and `tools/pose2sim_lower_predict_worker.py`. Weights stay in that env:

`...\Pose2Sim\MarkerAugmenter\LSTM\v0.3_upper` and `v0.3_lower` (`model.onnx`, `mean.npy`, `std.npy`).

Earlier Marker Enhancer experiments (`data/experiments/marker_enhancer_*`, `tools/marker_enhancer_*.py`) are the same model family, run offline. They are **not** the live angle path.

Investigation notes (read-only audits, not the app): `experiments/anatomical_landmark_audit/`.

---

## What we built, in order

1. **Steps 1–5** — practice scripts: live RGB, uploaded video, stick figure, CSV, 2D angles.
2. **Step 6 app** — one Tkinter window: live or file, region, 2D/3D, day/night, dials, velocities, report, optional CSV + overlay video.
3. **3D from aligned depth** — pixel + median depth → camera metres. Missing depth stays empty. MediaPipe “world landmarks” are not used.
4. **Joint frames** — three axes on shoulder, elbow, and wrist. The dial uses that frame in 3D. The frames are practical, not a full ISB set (no C7, T8, IJ, PX, glenohumeral centre, or true styloids).
5. **Hands** — finger graph when the person is close enough for Hands to fire.
6. **Pose2Sim upper LSTM, live** — magenta diamonds for the eight elbow/wrist study markers. Needs 3D and a filled height and weight.
7. **Full-body coordinator** — same session can also run the lower LSTM after four toe clicks.
8. **Diagnostics** — optional CSV under `data/diagnostics/<timestamp>/` (observed vs predicted, depth sample, lag). Not the same folder as Save CSV.
9. **Participant form** — Height (cm) and Weight (kg) before Start. Converted to metres and kilograms. Both LSTMs get that same pair. A live Pose2Sim session does not silently use the old built-in person (1.80 m, 61 kg). Those two numbers remain in code only as the published training defaults.
10. **Paper-set table** — `paper_landmarks.csv` records which requested anatomical names are estimated from a real LSTM output, and which are unavailable. The camera picture does **not** print those names.

The DOF files were not rewritten to consume LSTM points. Smoothing, depth windows, and model weights were not changed for the marker work.

---

## How it is going

**Working in the app**

- Live D455f and uploaded colour video (2D) or `.bag` (3D).
- Stick figure, dials, angular velocity, 3D wrist linear velocity, report after Stop.
- Upper LSTM elbow and wrist study markers when Live Pose2Sim is on, space is 3D, and height/weight pass validation.
- Lower LSTM pelvis, knee, foot, and related study markers **after** four successful toe clicks, a full body in frame, ONNX load, and an 8-frame buffer.
- Diagnostics and `run.json` store the height and mass actually used.

**Why elbow and wrist show first, and shoulder / knee / feet often do not**

The upper model’s only outputs are the eight elbow and wrist study markers. It does not predict a shoulder, a knee, or a foot.

Shoulder, knee, heel, ankle, and toe-ray **dots** are MediaPipe + depth. They appear only when that joint is inside the colour image and depth is valid. An arm filling the frame often has no full pose, so knees and feet never exist that frame.

Magenta shoulder-study, knee-pair, calcaneus, 5th-metatarsal, and toe-study markers are **lower** LSTM outputs. They wait for:

1. All four toe clicks (order: left big, left small, right big, right small).
2. The lower ONNX process to finish starting (often a few frames with toes already visible and no lower diamonds yet).
3. Eight valid frames in a row (window length is 16; the first result is allowed at 8). A gap longer than 2 dropped frames clears the buffer. Missing inputs are dropped, not interpolated.
4. The worker thread to finish. The picture shows the last **completed** result, which can be the previous frame.

On recording `data/diagnostics/20261012-162643`, toes first appeared at frames 347, 373, 410, and 436, and the 35 lower predictions first appeared at frame 448. That gap matches boot + 8-frame warm-up + one async frame, not missing toes.

**Not available, and not drawn as if they were**

- Acromion / SAT.
- Glenohumeral joint centre. `*_shoulder_study` is not that point. `LHJC_study` / `RHJC_study` are hip centres from the lower model, not shoulders.
- IJ, PX, T8.
- A palpated medial/lateral epicondyle, radial/ulnar styloid, ASIS, PSIS, femoral epicondyle, or metatarsal head. The LSTM names are the OpenCap **study** targets. They are estimates. Local checks showed centimetre-scale pair wander, so they are not treated as measured bones.

MediaPipe `shoulder`, `elbow`, `wrist`, `hip`, `knee`, `heel`, and `foot_index` keep their own names. They are not relabelled as SAT, epicondyles, styloids, ASIS, or FM1/FM5. Calibrated big-toe and small-toe points stay separate from FM1 and FM5.

**Video overlay**

No text on the camera image: no HUD, no point names, no paper tags. Observed joints are dots (shoulder, elbow, wrist, knee, ankle, heel, and foot index are drawn larger). LSTM estimates are magenta diamonds. Calibrated toes are squares. Pair midpoints are crosses. Names live in the 3D landmark window list and in the CSV files.

---

## Quick start

**Close Intel RealSense Viewer first.** Only one program can own the USB camera. Use a USB 3 port.

```text
pip install -r requirements.txt
python app.py
```

Python 3.10 on Windows is what this was run with.

1. Choose **Live camera** or **Upload file**.
2. Choose a region: Full body, Upper body, Both arms, Both hands, or Face.
3. Choose **2D** or **3D**, and **Day** or **Night**.
4. If **Live Pose2Sim** is on, type **Height (cm)** and **Weight (kg)** before Start. Example: `178` and `86` become 1.78 m and 86 kg. Empty, non-numeric, zero, or out-of-range values (height 100–250 cm, weight 25–250 kg) block Start. If Live Pose2Sim is off, the fields are not required.
5. Leave **Save CSV + video** ticked if you want `data/output/<timestamp>/`.
6. Press **Start**. Press **Stop**. The **Report** window is that take.

`data/input/sample.mp4` has **no person**. Use live, or your own clip.

**3D needs depth.** Live camera, or a `.bag` from Viewer. An mp4 is colour only — stay on **2D**. The app refuses 3D on a colour-only upload.

For lower-body study markers during a 3D + Live Pose2Sim session: stand so shoulders, hips, knees, ankles, and feet are in frame, click **Calibrate toes**, then click the video in order (left hallux, left small toe, right hallux, right small toe). Each click needs valid depth and a valid ankle, heel, and foot index on that foot in the same frame. Calibration is RAM only; Stop throws it away.

If the D455f is unplugged, the window stays open and shows **Camera not connected / not found**.

Live Pose2Sim also needs `C:\Users\jayjo\anaconda3\envs\pose2sim-offline\python.exe` and the v0.3 ONNX folders. Without that env the coordinator reports the upper model as unavailable; the MediaPipe dials still run.

---

## Install

```text
pip install -r requirements.txt
```

| Package | Why this version |
|---|---|
| `pyrealsense2==2.58.4.10922` | Matches the RealSense SDK / Viewer family we used |
| `mediapipe==0.10.14` | Local Pose + Hands. Do **not** use MediaPipe 1.x here (different API; pulled NumPy 2 and broke OpenCV) |
| `numpy>=1.23,<2` | Required by this MediaPipe + OpenCV pair |
| `opencv-python` | Video files + overlay |
| `PyYAML` | `config.yaml` |
| `Pillow` | Frames in the Tkinter window |
| `matplotlib` | 3D landmark window |

Pose2Sim 0.10.49 and its TensorFlow/ONNX stack stay in conda env `pose2sim-offline`. Do not install them into the app environment.

---

## Coordinates and angles

| Frame | What it is | When |
|---|---|---|
| **2D camera pixels** | `(u, v)` on the RGB image. Origin = **top-left**. `u` right, `v` down | **2D** button |
| **3D camera metres** | Same pixel + aligned depth, deprojected. Origin = **optical centre**. X right, Y down, Z forward | **3D** button |
| MediaPipe **world landmarks** | A guessed 3D skeleton from RGB only | **Not used** |
| OpenSim Y-up | LSTM internal frame: `Y_osim = -Y_camera` | Inside Pose2Sim only. Overlay and CSV are converted back to camera metres |

**2D angle** is the bend in the photo. About **180°** looks straight. If the arm points at the camera, the photo squashes it and the number is wrong.

**3D angle** is the bend in the joint’s own frame (camera metres). One number per joint, from observed MediaPipe points:

- **Shoulder** — angle between the trunk and the upper arm. The hip line is squared to the shoulder line.
- **Elbow** — bend between the upper arm and the forearm. The MediaPipe elbow is the vertex, not a humeral epicondyle.
- **Wrist** — bend between the forearm and the hand. Radial and ulnar styloids are not in this calculation (`dof_angles_3d.py` says so).

Each of those joints can show three perpendicular arrows. The dial does **not** split a joint into its full clinical degrees of freedom (shoulder twist, or palm up / palm down).

Joints below `analysis.min_confidence` (0.5) are skipped. Sitting close, legs are often weak. Upper body still keeps the **hips**, because the shoulder angle needs them. Face has landmarks only, no angle dials.

A still 3D point is calmed before the angle is computed: 5×5 median depth, then hold moves under **0.8 cm**, blend **35%** of the new point with **65%** of the last one, and keep the last good point for **5 frames** if depth drops. The dials themselves are not smoothed. Velocity has its own deadband so a still joint reads about **0**.

LSTM preprocess (unchanged official recipe): hip-centre, divide by height, append height and mass, `(x - mean) / std`, run ONNX, denormalise times height and add the hip back. Causal last timestep. Live window 16, minimum 8 frames, max gap 2.

---

## Window

- **Left:** video and skeleton. In 3D, short axis lines on the shoulder, elbow, and wrist. No numbers or names on the video. Wrist and nose coordinates sit in cards **above** the video (pixels in 2D, centimetres in 3D).
- **Right:** Left and Right dials for shoulder, elbow, and wrist. The number under a live dial is **angular velocity** (°/s).
- **Scale:** `0  45  90` or `0  20  40` (still runs to 180°).
- **Linear velocity:** wrist speed in m/s, under the dials. **3D only.**
- **Height (cm) / Weight (kg):** participant for Live Pose2Sim. Locked while a session is running.
- **Live Pose2Sim:** turns the LSTM coordinator on. Forces 3D if it was on 2D.
- **Calibrate toes:** four clicks on the video during a running 3D session. Ignored if the session is not running or LSTM is off.
- **Day / Night:** light or dark window. The camera picture is unchanged.
- After Stop, cyan / coral / amber marks on the dials are min, max, and mode of the **angle**.

A separate **Live landmarks** window plots valid camera-space points. The list there has names; the video does not.

---

## Report (after Stop)

A second window titled **Report**.

For each shoulder, elbow, and wrist:

- **Angle** — min, median, mode, max (degrees). Mode is the most common **10°** group.
- **Angular velocity** — same four numbers, in °/s.

In 3D, also **linear velocity** of the left and right wrist, in m/s. Mode there uses a **0.05 m/s** bin.

Each **Start** begins a new take.

---

## How to run each step

Practice scripts still read `config.yaml` (`source.mode` live vs file). The **2D / 3D** switch, regions, Day/Night, dials, Report, height/weight, and Pose2Sim are in the app only.

| Step | Command | What you get |
|---|---|---|
| 1 Live RGB | `python step1_show_rgb_live.py` | Colour window. **q** to quit |
| 2 Upload RGB | `python step2_show_uploaded.py` | Colour from `.mp4` / `.avi` / `.bag` |
| 3 Keypoints | `python step3_show_keypoints.py` | Stick figure on RGB |
| 4 Save | `python step4_save_keypoints.py` | CSV + overlay.mp4 + run.json |
| 5 Analyse | `python step5_analyze_2d.py` | 2D image-plane angles + `angles_2d.csv` |
| **6 App** | **`python app.py`** | Full window |

```yaml
source:
  mode: "live"          # or "file"
  file_path: "data/input/yourclip.mp4"
```

Tests that do not open the camera:

```text
python -m unittest tests.test_paper_set tests.test_full_body_markers tests.test_live_diagnostics tests.test_participant_lstm tests.test_pose2sim_live_upper -v
```

---

## Saved files

### `data/output/<timestamp>/` (Save CSV + video)

| File | Contents |
|---|---|
| `keypoints_2d.csv` | One row per joint per frame: `u_px`, `v_px`, and in 3D `x_m`, `y_m`, `z_m` (empty in 2D) |
| `angles_2d.csv` | One row per angle per frame. `degrees` is the dial value (photo angle in 2D, body-frame angle in 3D) |
| `overlay.mp4` | Stick figure. In 3D, the joint-frame arrows are on the video too |
| `run.json` | Settings snapshot, including `participant_height_m` and `participant_mass_kg` when Live Pose2Sim was on |

### `data/diagnostics/<timestamp>/` (diagnostics mode)

| File | Contents |
|---|---|
| `frames.csv` | One row per frame: timing, `n_observed`, `n_predicted` |
| `landmarks.csv` | Every observed, predicted, calibrated, and centre point. Class is `observed`, `predicted_upper`, `predicted_lower`, `calibrated_toe`, or `geometric_centre`. Missing XYZ is blank, never 0 |
| `paper_landmarks.csv` | One row per requested paper name per frame: side, method (`estimated` or `unavailable`), source LSTM name, XYZ only when that source exists |
| `summary.json` / `summary.txt` | Counts, participant height and mass, paper-set presence counts |

`n_predicted` is the **overlay list length** (upper + lower + centres + toes), not “number of lower markers”. Count lower markers with `class == predicted_lower` (35 names when the lower model is live).

Recordings are **gitignored**. Do not commit personal videos.

---

## Paper names vs what the code actually has

| Requested point | Status in this project | Source name if any |
|---|---|---|
| Shoulder SAT (acromion), left and right | **Unavailable.** Not drawn | none |
| Glenohumeral joint centre | **Unavailable.** Not drawn | none |
| H-LE / H-ME (humeral epicondyles) | **Estimated** when upper LSTM is live | `*_lelbow_study`, `*_melbow_study` |
| RS / US (radial / ulnar styloid family) | **Estimated** when upper LSTM is live | `*_lwrist_study`, `*_mwrist_study` |
| ASIS / PSIS | **Estimated** when lower LSTM is live | `*.ASIS_study`, `*.PSIS_study` |
| Hip | **Observed** MediaPipe hip, kept separate. Lower model also estimates `*HJC_study`, which is not the MediaPipe hip and not ASIS | `left_hip`, `right_hip` |
| M/L-FE (femoral epicondyles) | **Estimated** when lower LSTM is live | `*_knee_study`, `*_mknee_study` |
| FCC (calcaneus) | **Estimated** `*_calc_study`. MediaPipe heel stays its own observed point | `*_calc_study` |
| FM5 | **Estimated** | `*_5meta_study` |
| FM1 | **Estimated correspondence only** to OpenCap `*_toe_study`. Not certified as the first metatarsal head. Not `foot_index` and not the calibrated big toe | `*_toe_study` |
| Ankle, foot index | **Observed** MediaPipe | `*_ankle`, `*_foot_index` |
| Big toe / small toe | **Calibrated** by the four clicks | `LBigToe`, `LSmallToe`, `RBigToe`, `RSmallToe` |

---

## `config.yaml`

Change a value here and run again. The GUI **2D / 3D** button overrides `analysis.space` at Start. Height and weight are **not** stored in this file; they are typed each session.

| Section | Controls |
|---|---|
| `project` | Title, student, camera model |
| `source` | `live` or `file`, path, loop |
| `camera` | Colour size/fps, and `camera.depth` (size, range, median window, XYZ smooth) |
| `pose` | Pose thresholds, and Hands |
| `overlay` | Dot size and skeleton colours (OpenCV BGR) |
| `output` | Folder, CSV/video names |
| `analysis` | Angles, joint frames, regions, gauge scale, velocity deadbands, `live_pose2sim_upper` |
| `display` | Window title, quit key for Steps 1–5, foot-safety guide line |

Angles are three named joints. The middle name is the vertex:

```yaml
- name: left_elbow
  points: [left_shoulder, left_elbow, left_wrist]
```

In 3D, `analysis.joint_frames` replaces that photo angle with the frame angle of the same name. Names must match MediaPipe (`src/pose/skeleton.py`).

---

## Project layout

```text
app.py                      launch the app
config.yaml                 settings (not per-person height/weight)
requirements.txt
step1_show_rgb_live.py … step5_analyze_2d.py
docs/Motion_Analysis_Pipeline.pptx
src/
  capture/       live RealSense, mp4/avi, bag, aligned depth
  pose/          MediaPipe Pose + Hands, skeleton, draw
  geometry/      deproject, joint frames, anatomical frames
  analysis/      2D angles, 3D DOF angles, regions, velocity
  markers/       catalog, toe calibration, LSTM coordinator, paper-name map
  pose2sim_live/ upper/lower LSTM I/O, ONNX engine, participant checks
  diagnostics/   live CSV recorder
  io/            CSV and overlay writers
  ui/            Tkinter window, dials, report, live marker view
  utils/         load config, resolve paths
tools/           Pose2Sim ONNX workers; older Marker Enhancer scripts
experiments/anatomical_landmark_audit/   investigation notes
data/input/      put clips here
data/output/     timestamped saved takes (not in git)
data/diagnostics/ timestamped diagnostic takes (not in git)
```

| File | Role |
|---|---|
| `src/capture/live_realsense.py` | Live colour; depth when 3D is on |
| `src/capture/bag_file.py` | `.bag` colour, and aligned depth in 3D |
| `src/capture/video_file.py` | Colour movies (2D only) |
| `src/capture/depth_util.py` | Median depth and the diagnostic depth sample |
| `src/geometry/deproject.py` | Pixel + depth → camera X, Y, Z |
| `src/geometry/joint_frame.py` | Three axes at shoulder, elbow, wrist, and the one angle |
| `src/geometry/anatomical_frames.py` | Frame construction used by the 3D DOF path |
| `src/analysis/dof_angles_3d.py` | 3D angles from those frames. Does not read LSTM points |
| `src/analysis/angles_2d.py` | Angle in the image |
| `src/analysis/velocity.py` | Angular °/s and linear m/s |
| `src/pose/hands.py` | Finger skeleton |
| `src/markers/coordinator.py` | Upper LSTM at start; lower LSTM after four toe clicks |
| `src/markers/foot_calib.py` | In-memory rigid foot from ankle, heel, and foot index |
| `src/markers/paper_set.py` | Bousigues name map. Does not invent coordinates |
| `src/pose2sim_live/engine.py` | Background ONNX. Requires explicit height and mass |
| `src/pose2sim_live/participant.py` | cm→m, kg checks, no silent default person |
| `src/ui/app.py` | Main window |
| `src/ui/session.py` | One Start/Stop take |
| `src/ui/live_markers.py` | Video dots and the 3D list. No on-image text |
| `src/diagnostics/recorder.py` | `frames.csv`, `landmarks.csv`, `paper_landmarks.csv` |

---

## File types

| File | Inside | 2D | 3D |
|---|---|---|---|
| `.mp4` / `.avi` / `.mov` / `.mkv` | Colour only | Yes | No |
| `.bag` | RealSense colour + depth | Yes | Yes |
| Live USB | Colour + depth | Yes | Yes |

---

## Limits

- One angle per joint, not the full clinical set.
- 2D angles are wrong when the limb points at the camera.
- 3D angles need depth on every joint that frame uses. Missing depth skips that angle. It is never filled with 0.
- Linear velocity is metres per second from the wrist’s camera X, Y, Z. It is not computed in 2D.
- LSTM study markers are learned averages from another lab’s training set (triangulated keypoints, different camera, about 60 Hz, a default body near 1.80 m / 61 kg unless you override height and mass). They are not palpation.
- The lower model cannot start from `foot_index` alone.
- Toe calibration does not survive Stop.
- `LOWER LIVE` means the last finished lower inference succeeded. It can still be showing the previous frame.
- Prototype only — **not for clinical use**.

---

## Related work

- **Marker mocap** (Vicon / Qualisys): lab standard. This project is one USB camera.
- **Kinect** (Shotton et al.): skeleton from depth. We use MediaPipe on colour, then RealSense depth at the pixel.
- **OpenPose** and **MediaPipe / BlazePose**: 2D joints. This project uses MediaPipe 0.10.14.
- **OpenCap** (Uhlrich et al.) and **Marker Enhancer** (Falisse et al.): the LSTM this app runs through Pose2Sim 0.10.49. Live overlay only. Not the DOF input.
- **Pose2Sim** (`perfanalytics/pose2sim`, Pagnon et al.): the package that ships those ONNX files and the hip-centre preprocess we kept.
- **Bousigues et al., 2025**: the marker-category list used for `paper_landmarks.csv`. The paper does not make MediaPipe detect those skin points.

Slides: `docs/Motion_Analysis_Pipeline.pptx`. `docs/Progress_Update_2D_full.pptx` is the earlier 2D talk.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Camera not connected / not found | Unplugged, USB 2, or Viewer already open |
| Camera is in use | Close RealSense Viewer |
| 3D needs depth | An mp4/avi was opened with **3D**. Use Live or a `.bag`, or switch to **2D** |
| Participant measurements dialog | Height or weight empty, not a number, ≤ 0, or outside 100–250 cm / 25–250 kg |
| Cards say no depth | Colour joint was found; aligned depth at that pixel was missing, too close, or too far |
| Shoulder dial blank | Hip hidden or below 0.5 confidence. Elbow and wrist do not need the hip |
| Magenta dots only at elbow and wrist | Upper LSTM only. Shoulder/knee/foot **study** markers need Calibrate toes (4/4) and the whole body in frame |
| Toes clicked but no lower diamonds yet | ONNX still loading, or fewer than 8 valid frames, or a foot input lost depth |
| Toe click ignored | Session not started, Live Pose2Sim off, 2D mode, click in the letterbox, or no depth |
| Angle jumps when the arm points at the camera | **2D** photo angle. Use **3D** |
| UPPER unavailable | `pose2sim-offline` python or v0.3 weights not at the path in `src/pose2sim_live/upper_lstm.py` |
| Import / OpenCV errors | MediaPipe 1.x or NumPy 2; reinstall `requirements.txt` |
| File not found on Upload | Browse the file |

Repo: [github.com/jay-07-pixel/motion__analysis__](https://github.com/jay-07-pixel/motion__analysis__)
