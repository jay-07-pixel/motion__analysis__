# 2D and 3D Motion Analysis (single RealSense D455f)

**Student:** Jay  
**Hardware:** Intel RealSense Depth Camera D455f  
**Current phase:** **2D only** (RGB keypoints in camera pixel coordinates)

Assigned work is **2D and 3D** motion analysis with **one** RGB-D camera, live **and** uploaded video, settings in `config.yaml` (no hardcoding).  
**This repository has finished the 2D phase.** 3D (depth → metres) is not built yet.

The user-facing program is **Step 6**. Steps 1–5 are practice scripts that built the same engine one piece at a time.

---

## What this project does (2D)

1. Read a colour picture from the **live D455f** or from an uploaded **`.mp4` / `.avi` / `.mov` / `.bag`**.
2. Find **33 body joints** with MediaPipe Pose.
3. Express each joint as **camera pixels** `(u_px, v_px)` with origin at the **top-left** of the RGB image.
4. Draw a stick figure.
5. Compute **left and right elbow angles in the image** (degrees).
6. Draw one **wrist trail** and a **pixel speed** (px/s), after ignoring MediaPipe jitter.
7. Optionally **save** CSV + overlay video.
8. After **Stop**, open a **2D analysis report** for that take (angle vs time + bent/straight bars).

This is a **student prototype**, not a medical device.

---

## Quick start (the app)

**Close Intel RealSense Viewer first.** Only one program can own the USB camera.

From the project folder (after dependencies are installed):

```text
python app.py
```

1. Choose **Live camera** or **Upload file** (Browse a clip that contains a person).
2. Leave **Save CSV + video** ticked if you want files.
3. Press **Start**. Face the camera and bend both elbows.
4. Press **Stop**.
5. A **report window** opens for that take (line chart + bar chart + min/max/mean).
6. If Save was on, files are in `data/output/<timestamp>/`.

`data/input/sample.mp4` has **no person**. Live, or your own clip, is needed to see a skeleton.

If the D455f is unplugged, the window stays open and shows **Camera not connected / not found** (it should not crash).

---

## Install

Python 3.10 is what we used on Windows.

```text
pip install -r requirements.txt
```

Pinned / important packages:

| Package | Why this version |
|---|---|
| `pyrealsense2==2.58.4.10922` | Matches the RealSense SDK / Viewer family we used |
| `mediapipe==0.10.14` | Local Pose, 33 joints. Do **not** use MediaPipe 1.x here (different API; pulled NumPy 2 and broke OpenCV) |
| `numpy>=1.23,<2` | Required by this MediaPipe + OpenCV pair |
| `opencv-python` | Video files + overlay |
| `PyYAML` | `config.yaml` |
| `Pillow` | Show frames in the Tkinter window |
| `matplotlib` | Analysis report charts after Stop |

USB **3** for the D455f. USB 2 often fails or is unstable.

---

## Camera coordinates (read this before the numbers)

| Frame | What it is | This phase |
|---|---|---|
| **2D camera pixels** | `(u, v)` on the RGB image. Origin = **top-left**. `u` right, `v` down | **In use** |
| **3D camera metres** | Same pixel + RealSense **depth**, deprojected | **Not built** |
| MediaPipe **world landmarks** | A **guessed** 3D skeleton from RGB only (origin between the hips) | **Not used as RealSense 3D** |

**Elbow degrees** are the angle **in the photograph**, not a true 3D bone angle.  
~**180°** ≈ the arm looks straight in the picture. Smaller ≈ more bent.  
If the arm points **at** the camera, the 2D angle can collapse (even toward 0°). Face the camera.

**Wrist speed** is **pixels per second**, not m/s. Metres need depth.

A still wrist still **jitters** 1–4 pixels in MediaPipe. At 30 fps that fake wiggle looks like 30–120 px/s. The app treats small moves as **0** (`analysis.speed_deadband_px`).

Joints below `analysis.min_confidence` / `pose.min_visibility` are **skipped** (cards show **—**). Sitting close: hips and legs are often weak. Upper-body recording is valid.

---

## How to run each step

Practice scripts still work. They read `config.yaml` (`source.mode` live vs file).

| Step | Command | What you get |
|---|---|---|
| 1 Live RGB | `python step1_show_rgb_live.py` | Colour window, same stream as Viewer 2D. **q** to quit |
| 2 Upload RGB | `python step2_show_uploaded.py` | Colour from `.mp4` / `.avi` / `.bag` |
| 3 Keypoints | `python step3_show_keypoints.py` | Stick figure on RGB |
| 4 Save | `python step4_save_keypoints.py` | CSV + overlay.mp4 + run.json |
| 5 Analyse | `python step5_analyze_2d.py` | Elbows + wrist trail + `angles_2d.csv` |
| **6 App** | **`python app.py`** | Live/Upload GUI + report after Stop |

For Steps 1–5, set the source in YAML:

```yaml
source:
  mode: "live"          # or "file"
  file_path: "data/input/yourclip.mp4"
```

---

## Step 6 app (detail)

**Window**

- Left: live video + skeleton + elbow labels + wrist trail.
- Right: **Left elbow °**, **Right elbow °**, **left wrist px/s**.
- No charts on this screen (so the camera stays large).

**After Stop**

A second window: **2D analysis report — this take**.

- Elbow angle vs time (left / right).
- How often each elbow was bent vs straight (30° bins, % of this take).
- Min / max / mean and sample count.

Each **Start** begins a new take. The report is only for the take you just stopped.

**If Live fails**

- Viewer still open, or another app using the camera.
- Cable is USB 2, not USB 3.
- Camera unplugged → dialog **Camera not found**.

---

## Saved files (`data/output/<timestamp>/`)

Created when **Save CSV + video** is ticked (app) or when you run Steps 4–5.

| File | Contents |
|---|---|
| `keypoints_2d.csv` | `frame, time_sec, joint, u_px, v_px, confidence, coord_frame, source` — one row per joint per frame |
| `angles_2d.csv` | `frame, time_sec, angle, degrees, vertex_u_px, vertex_v_px, min_confidence, coord_frame, source` |
| `overlay.mp4` | Video with stick figure and labels |
| `run.json` | Settings snapshot for that run (`coord_frame`: `camera_2d_pixel`) |

`coord_frame` is always `camera_2d_pixel`.  
Recordings are **gitignored** (`data/output/*`). Do not commit personal videos.

---

## `config.yaml` (nothing important hardcoded)

Change a value here, run again.

| Section | Controls |
|---|---|
| `project` | Title, student name, camera model, phase (`2d`) |
| `source` | `live` or `file`, `file_path`, `loop` |
| `camera` | Live RGB width / height / fps (default 1280×720 @ 30) |
| `pose` | MediaPipe complexity and confidence thresholds |
| `overlay` | Dot size and skeleton colours (OpenCV BGR) |
| `output` | Output folder, CSV/video names, codec |
| `analysis` | Which 2D angles, trail joint, speed deadband / smooth |
| `display` | Window title and quit key for Steps 1–5 |

Elbows are defined as three named joints, for example:

```yaml
- name: left_elbow
  points: [left_shoulder, left_elbow, left_wrist]   # angle at the middle name
```

Joint names must match MediaPipe (`src/pose/skeleton.py`).

---

## Project layout

```text
app.py                      ← launch the 2D app
config.yaml                 ← all settings
requirements.txt
step1_show_rgb_live.py … step5_analyze_2d.py
docs/Progress_Update_2D_full.pptx   ← 3-slide progress talk
src/
  capture/     live RealSense, mp4/avi, bag, factory
  pose/        MediaPipe → Keypoint2D, skeleton, draw
  analysis/    2D angle, wrist trail + px/s
  io/          CSV and overlay writers
  ui/          Tkinter app, session, report charts
  utils/       load config, resolve paths
data/input/    put clips here
data/output/   timestamped runs (not in git)
```

| File | Role |
|---|---|
| `src/utils/config_loader.py` | Load YAML; resolve paths from the project root |
| `src/capture/live_realsense.py` | Live D455f RGB; friendly “camera not found” |
| `src/capture/video_file.py` | Colour movies |
| `src/capture/bag_file.py` | RealSense `.bag` (RGB now; depth later) |
| `src/capture/factory.py` | Picks live vs file vs bag |
| `src/pose/extractor_2d.py` | MediaPipe → `(u_px, v_px)` |
| `src/pose/skeleton.py` | 33 landmark names |
| `src/analysis/angles_2d.py` | Angle at three points in the image |
| `src/analysis/trajectory_2d.py` | Wrist trail; deadband so a still wrist is 0 px/s |
| `src/io/save_keypoints.py` | `keypoints_2d.csv` |
| `src/io/save_angles.py` | `angles_2d.csv` |
| `src/io/save_video.py` | `overlay.mp4` |
| `src/ui/session.py` | One Start/Stop session (pose + save) |
| `src/ui/app.py` | Main window |
| `src/ui/live_charts.py` | Collect samples; report after Stop |

---

## File types

| File | Inside | 2D now | 3D later |
|---|---|---|---|
| `.mp4` / `.avi` / `.mov` | Colour only | Yes | No (no depth) |
| `.bag` | RealSense RGB + depth | Yes (RGB) | Yes (depth already stored) |
| Live USB | RGB now (depth not started) | Yes | After we enable depth |

---

## Limits (say this in any demo)

- 2D only. No metres, no m/s, no depth yet.
- Image-plane elbow angle is wrong if the motion is toward the camera.
- Wrist px/s is not physical speed.
- MediaPipe world landmarks are **not** RealSense 3D.
- Low-confidence joints are omitted on purpose.
- Prototype only — **not for clinical use**.

---

## Related work (short context)

- **Marker mocap** (Vicon / Qualisys): lab gold standard; not one USB camera.
- **Kinect** (Shotton et al.): skeleton from **depth**.
- **OpenPose**: classic multi-person **2D** pose.
- **MediaPipe / BlazePose**: 33 joints, real-time; **this is our 2D engine**.
- Real 3D here will be **aligned RealSense depth at the 2D pixel**, not MediaPipe’s guessed world points.

Progress slides: `docs/Progress_Update_2D_full.pptx` (problem, 2D workflow, tech stack).

---

## Next (not built)

**3D:** start the depth stream, **align depth to colour**, read depth at `(u_px, v_px)`, deproject to camera **metres**, then wrist speed in **m/s**.

Do not treat MediaPipe world landmarks as that measurement.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Camera not connected / not found | Unplugged, USB 2, or Viewer already open |
| Camera is in use | Close RealSense Viewer |
| `no person` / cards show — | No body in frame, or `sample.mp4`; sit farther / face the camera |
| Elbow stuck near 0° | Arm pointing at the camera (2D collapse) |
| Wrist speed high while still | Old app; current code uses `speed_deadband_px` |
| Import / OpenCV errors | MediaPipe 1.x or NumPy 2; reinstall `requirements.txt` |
| File not found on Upload | Path wrong; Browse the file |

Repo: [github.com/jay-07-pixel/motion__analysis__](https://github.com/jay-07-pixel/motion__analysis__)
