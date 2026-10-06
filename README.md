# 2D and 3D Motion Analysis (single RealSense D455f)

**Student:** Jay  
**Hardware:** Intel RealSense Depth Camera D455f  
**Current phase:** **2D and 3D** in one window. The **2D / 3D** buttons choose the path at Start.

Assigned work is motion analysis with **one** RGB-D camera, live **and** uploaded video, settings in `config.yaml` (no hardcoding).

The user-facing program is **Step 6** (`python app.py`). Steps 1–5 are practice scripts that built the same engine one piece at a time.

This is a **student prototype**, not a medical device.

---

## What the app does

1. Read colour from the **live D455f**, or from an uploaded **`.mp4` / `.avi` / `.mov` / `.mkv` / `.bag`**.
2. Find **33 body joints** with MediaPipe Pose, plus **finger joints** with MediaPipe Hands (except the Face region).
3. **2D:** each joint is camera pixels `(u, v)`. Origin is the **top-left** of the image. `u` right, `v` down.
4. **3D:** align depth to colour, take a small median around the joint, and deproject to camera **metres**. Origin is the **optical centre**. `X` right, `Y` down, `Z` forward. This is RealSense depth, **not** MediaPipe world landmarks.
5. Draw the stick figure. In 3D, draw a **three-axis frame** on each shoulder, elbow, and wrist.
6. Compute **shoulder, elbow, and wrist** angles (left and right). One angle per joint.
7. Show **angular velocity** (°/s) under each dial, and **linear velocity** (m/s) of each wrist in 3D.
8. Optionally **save** CSV + overlay video.
9. After **Stop**, open a **Report** window: min, median, mode, and max.

---

## Quick start

**Close Intel RealSense Viewer first.** Only one program can own the USB camera.

```text
python app.py
```

1. Choose **Live camera** or **Upload file**.
2. Choose a region: Full body, Upper body, Both arms, Both hands, or Face.
3. Choose **2D** or **3D**, and **Day** or **Night**.
4. Leave **Save CSV + video** ticked if you want files.
5. Press **Start**. Press **Stop**.
6. The **Report** window opens for that take.
7. If Save was on, files are in `data/output/<timestamp>/`.

`data/input/sample.mp4` has **no person**. Use live, or your own clip.

**3D needs depth.** Live camera, or a `.bag` recorded in Viewer. An mp4 has colour only — stay on **2D** for that file. The app refuses 3D on a colour-only upload.

If the D455f is unplugged, the window stays open and shows **Camera not connected / not found**.

---

## Install

Python 3.10 is what we used on Windows.

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
| `Pillow` | Show frames in the Tkinter window |

USB **3** for the D455f. USB 2 often fails or is unstable.

---

## Coordinates

| Frame | What it is | When |
|---|---|---|
| **2D camera pixels** | `(u, v)` on the RGB image. Origin = **top-left** | **2D** button |
| **3D camera metres** | Same pixel + aligned depth, deprojected. Origin = **optical centre**. X right, Y down, Z forward | **3D** button |
| MediaPipe **world landmarks** | A guessed 3D skeleton from RGB only | **Not used** |

**2D angle** is the bend in the photo. About **180°** looks straight. If the arm points at the camera, the photo squashes it and the number is wrong.

**3D angle** is the bend in the joint’s own frame (camera metres). One number per joint:

- **Shoulder** — angle between the trunk and the upper arm. The hip line is squared to the shoulder line, so a side raise and a front raise at the same height stay close.
- **Elbow** — bend between the upper arm and the forearm.
- **Wrist** — bend between the forearm and the index finger (a proxy; there is no metacarpal marker).

Each joint also has three perpendicular arrows (cyan, green, magenta). The dial does **not** yet split a joint into its full clinical degrees of freedom (for example shoulder twist, or palm up / palm down).

Joints below `analysis.min_confidence` (0.5) are skipped. Sitting close, legs are often weak. Upper body still keeps the **hips**, because the shoulder angle needs them. Face has landmarks only, no angle dials.

---

## Window

- **Left:** video and skeleton. In 3D, short axis lines on the shoulder, elbow, and wrist. No angle numbers on the video. Wrist and nose coordinates sit in cards **above** the video (pixels in 2D, centimetres in 3D).
- **Right:** Left and Right dials for shoulder, elbow, and wrist. The number under a live dial is **angular velocity** (°/s).
- **Scale:** `0  45  90` or `0  20  40` (still runs to 180°).
- **Linear velocity:** wrist speed in m/s, under the dials. **3D only.**
- **Day / Night:** light or dark window. The camera picture is unchanged.
- After Stop, cyan / coral / amber marks on the dials are min, max, and mode of the **angle**.

A still 3D point is calmed before the angle is computed: 5×5 median depth, then hold moves under **0.8 cm**, blend **35%** of the new point with **65%** of the last one, and keep the last good point for **5 frames** if depth drops. The dials themselves are not smoothed. Velocity has its own deadband so a still joint reads about **0**.

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

Practice scripts still read `config.yaml` (`source.mode` live vs file). The **2D / 3D** switch, regions, Day/Night, dials, and Report are in the app only.

| Step | Command | What you get |
|---|---|---|
| 1 Live RGB | `python step1_show_rgb_live.py` | Colour window. **q** to quit |
| 2 Upload RGB | `python step2_show_uploaded.py` | Colour from `.mp4` / `.avi` / `.bag` |
| 3 Keypoints | `python step3_show_keypoints.py` | Stick figure on RGB |
| 4 Save | `python step4_save_keypoints.py` | CSV + overlay.mp4 + run.json |
| 5 Analyse | `python step5_analyze_2d.py` | 2D image-plane angles + `angles_2d.csv` |
| **6 App** | **`python app.py`** | Full window: 2D/3D, regions, dials, velocities, Report |

```yaml
source:
  mode: "live"          # or "file"
  file_path: "data/input/yourclip.mp4"
```

---

## Saved files (`data/output/<timestamp>/`)

Created when **Save CSV + video** is ticked (app) or when you run Steps 4–5.

| File | Contents |
|---|---|
| `keypoints_2d.csv` | One row per joint per frame: `u_px`, `v_px`, and in 3D `x_m`, `y_m`, `z_m` (empty in 2D) |
| `angles_2d.csv` | One row per angle per frame. `degrees` is the dial value for that run (photo angle in 2D, body-frame angle in 3D) |
| `overlay.mp4` | Stick figure. In 3D, the joint-frame arrows are on the video too |
| `run.json` | Settings snapshot. `coord_frame` is `camera_2d_pixel` or `camera_3d_metre` |

Recordings are **gitignored** (`data/output/*`). Do not commit personal videos.

---

## `config.yaml`

Change a value here and run again. The GUI **2D / 3D** button overrides `analysis.space` at Start.

| Section | Controls |
|---|---|
| `project` | Title, student, camera model |
| `source` | `live` or `file`, path, loop |
| `camera` | Colour size/fps, and `camera.depth` (size, range, median window, XYZ smooth) |
| `pose` | Pose thresholds, and Hands (finger skeleton, match each hand to the nearest Pose wrist) |
| `overlay` | Dot size and skeleton colours (OpenCV BGR) |
| `output` | Folder, CSV/video names |
| `analysis` | Angles, joint frames, regions, gauge scale, velocity deadbands |
| `display` | Window title and quit key for Steps 1–5 |

Angles are three named joints. The middle name is the vertex:

```yaml
- name: left_elbow
  points: [left_shoulder, left_elbow, left_wrist]
```

In 3D, `analysis.joint_frames` replaces that photo angle with the frame angle of the same name. Names must match MediaPipe (`src/pose/skeleton.py`).

---

## Project layout

```text
app.py                      ← launch the app
config.yaml                 ← all settings
requirements.txt
step1_show_rgb_live.py … step5_analyze_2d.py
docs/Motion_Analysis_Pipeline.pptx
src/
  capture/     live RealSense, mp4/avi, bag, aligned depth
  pose/        MediaPipe Pose + Hands, skeleton, draw
  geometry/    deproject pixels+depth, joint frames
  analysis/    2D angles, 3D angles, regions, velocity
  io/          CSV and overlay writers
  ui/          Tkinter window, Day/Night, dials, Report
  utils/       load config, resolve paths
data/input/    put clips here
data/output/   timestamped runs (not in git)
```

| File | Role |
|---|---|
| `src/capture/live_realsense.py` | Live colour; depth when 3D is on |
| `src/capture/bag_file.py` | `.bag` colour, and aligned depth in 3D |
| `src/capture/video_file.py` | Colour movies (2D only) |
| `src/geometry/deproject.py` | Pixel + depth → camera X, Y, Z, then smooth |
| `src/geometry/joint_frame.py` | Three axes at shoulder, elbow, wrist, and the one angle |
| `src/pose/hands.py` | Finger skeleton, matched to the Pose wrist |
| `src/analysis/angles_2d.py` | Angle in the image |
| `src/analysis/angles_3d.py` | Angle from the joint frame, or the raw 3D bone angle if the frame cannot be built |
| `src/analysis/velocity.py` | Angular °/s and linear m/s |
| `src/analysis/regions.py` | Which joints and dials a region keeps |
| `src/ui/app.py` | Main window, Day/Night, Report |
| `src/ui/live_charts.py` | Dials, scale, min/max/mode |
| `src/ui/session.py` | One Start/Stop take |

---

## File types

| File | Inside | 2D | 3D |
|---|---|---|---|
| `.mp4` / `.avi` / `.mov` / `.mkv` | Colour only | Yes | No |
| `.bag` | RealSense colour + depth | Yes | Yes |
| Live USB | Colour + depth | Yes | Yes |

---

## Limits

- One angle per joint, not the full set of shoulder / elbow / wrist degrees of freedom.
- 2D angles are wrong when the limb points at the camera.
- 3D angles need depth at every joint that frame uses. Missing depth skips that angle.
- Linear velocity is metres per second from the wrist’s camera X, Y, Z. It is not computed in 2D.
- MediaPipe world landmarks are not RealSense 3D.
- Smoothing is on the 3D points (and on the velocity numbers), not on the angle dials.
- Prototype only — **not for clinical use**. The 9 Hole Peg Test and Box and Block Test are not scored here. Angular and linear velocity are there so a reach can be described; the official test scores are still time and block count.

---

## Related work

- **Marker mocap** (Vicon / Qualisys): lab gold standard; not one USB camera.
- **Kinect** (Shotton et al.): skeleton from depth.
- **OpenPose** and **MediaPipe / BlazePose**: 2D joints. This project uses MediaPipe, then RealSense depth for 3D.
- **Marker enhancer** (Falisse et al., OpenCap): predicts extra skin markers from video keypoints. It is not wired in. The frames here are built from the MediaPipe joints we already have.

Slides: `docs/Motion_Analysis_Pipeline.pptx` (pipeline). `docs/Progress_Update_2D_full.pptx` is the earlier 2D progress talk.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Camera not connected / not found | Unplugged, USB 2, or Viewer already open |
| Camera is in use | Close RealSense Viewer |
| 3D needs depth | An mp4/avi was opened with **3D**. Use Live or a `.bag`, or switch to **2D** |
| Cards say no depth | Colour joint was found; aligned depth at that pixel was missing, too close, or too far |
| Shoulder dial blank | Hip hidden or below 0.5 confidence. Elbow and wrist do not need the hip |
| Angle jumps when the arm points at the camera | **2D** photo angle. Use **3D** |
| Import / OpenCV errors | MediaPipe 1.x or NumPy 2; reinstall `requirements.txt` |
| File not found on Upload | Browse the file |

Repo: [github.com/jay-07-pixel/motion__analysis__](https://github.com/jay-07-pixel/motion__analysis__)
