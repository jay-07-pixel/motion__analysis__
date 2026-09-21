# 2D and 3D Motion Analysis (single RealSense D455f)

Student: **Jay**  
Hardware: **Intel RealSense Depth Camera D455f**  
Current phase: **2D only** (RGB keypoints in camera pixel coordinates)

The one-window GUI is **Step 6**. Steps 1–5 stay as practice scripts.

---

## Step 1 (done) — Live RGB from the D455f

**What:** Our own Python window showing the same colour stream you saw in RealSense Viewer.

**Why:** 2D keypoints will be extracted on this RGB image. Viewer is only a test tool.

```text
python step1_show_rgb_live.py
```

Close Viewer first. Press **q** to quit.

---

## Step 2 (done) — Uploaded file RGB (.mp4 / .avi / .bag)

**What:** Same colour frames as Step 1, but from a **file** instead of USB.

**Why:** Sir asked for live **and** uploaded. Pose must not care where the picture came from.

| File | What is inside | 2D now | 3D later |
|---|---|---|---|
| `.mp4` / `.avi` | Colour movie | Yes | No |
| `.bag` | RealSense recording (RGB + depth) | Yes (RGB) | Yes (depth already saved) |

**How to run**

1. Copy a clip into `data/input/`
2. In `config.yaml` set:

```yaml
source:
  mode: "file"
  file_path: "data/input/yourclip.mp4"   # or yourclip.bag
```

3. Run:

```text
python step2_show_uploaded.py
```

4. Press **q** to quit. If the file ends, the window closes.

**If it fails:** file path wrong, or no clip in `data/input/` yet. Record a `.bag` in RealSense Viewer (**Record**) or use any `.mp4`.

---

## Step 3 (done) — 2D keypoints in camera pixels

**What:** Find 33 body joints on each RGB frame. Each joint is `(name, u_px, v_px, confidence)` with origin at the **top-left** of the image.

**Why:** This is the 2D measurement sir asked for. Overlay lets you check the dots sit on the body. We still ignore depth.

**How to run**

1. Close RealSense Viewer.
2. In `config.yaml`, `source.mode: "live"` (or `"file"` plus a clip **with a person**).
3. Run:

```text
python step3_show_keypoints.py
```

4. You should see a stick figure on yourself. Press **q** to quit.

`data/input/sample.mp4` has **no person**, so it will say `no person`. Use live or your own video.

Thresholds: `pose.min_visibility` etc. in `config.yaml`.

---

## Step 4 (done) — Save keypoints to CSV

**What:** Same skeleton as Step 3, plus a file of every joint.

**Why:** The window is only for you. CSV is for handover and for Step 5 (angles).

Each run creates `data/output/<timestamp>/`:

- `keypoints_2d.csv` — `frame, time_sec, joint, u_px, v_px, confidence, coord_frame, source`
- `overlay.mp4` — video with the stick figure
- `run.json` — settings used for this run

`coord_frame` is always `camera_2d_pixel` (origin top-left).

**How to run**

```text
python step4_save_keypoints.py
```

Close Viewer first if `source.mode` is `live`. Press **q** to stop and finish the files.

Open the CSV in Excel to check `left_elbow` etc. have numbers.

---

## Step 5 (done) — 2D angles and path

**What:** Elbow angles in the **image** (degrees) and a wrist trail in **pixels**.

**Why:** Dots on the body are not analysis. Angles and a path are motion numbers.

**How to run**

```text
python step5_analyze_2d.py
```

Face the camera, bend your elbows. You should see `left_elbow 140 deg` on the video. Press **q**.

Saves `angles_2d.csv` plus the same keypoint CSV / overlay as Step 4.

**Limit:** 2D only. If the arm points at the camera, the angle is wrong. Weak joints (legs when sitting close) are skipped when confidence &lt; `analysis.min_confidence`.

---

## Step 6 (done) — One user-facing app

**What:** One window: **Live** or **Upload**, **Start** / **Stop**, video on the left, live elbow degrees and wrist px/s on the right. Optionally save CSV + overlay video.

**Why:** Sir asked for a user-friendly app (live + uploaded). The engine is still Steps 1–5; the window is the front door. Thresholds stay in `config.yaml`.

**How to run**

1. Close RealSense Viewer (Live cannot share the D455f).
2. From the project folder:

```text
python app.py
```

3. Choose **Live camera** or **Upload file**, then **Start**.
4. Leave **Save CSV + video** ticked if you want files.
5. **Stop**. If Save was on, files go under `data/output/<timestamp>/`.

**What is the same as Step 5:** 2D camera pixels (origin top-left), elbow angle in the image (~180° = straight in the picture), one wrist trail, skip low-confidence joints.

**What is not in this app:** metres, depth, 3D. Student prototype — not a medical device.

**If Live fails:** Viewer still open, USB 2 instead of USB 3, or another program using the camera.

---

## Files

| File | Role |
|---|---|
| `config.yaml` | Settings including `source.file_path` |
| `src/capture/live_realsense.py` | Live D455f RGB |
| `src/capture/video_file.py` | `.mp4` / `.avi` |
| `src/capture/bag_file.py` | RealSense `.bag` |
| `src/capture/factory.py` | Picks live vs file from config |
| `step1_show_rgb_live.py` | Step 1 window |
| `step2_show_uploaded.py` | Step 2 window |
| `src/pose/extractor_2d.py` | MediaPipe → 2D keypoints `(u_px, v_px)` |
| `src/pose/draw.py` | Skeleton overlay |
| `step3_show_keypoints.py` | Step 3 window |
| `src/io/save_keypoints.py` | CSV writer |
| `src/io/save_video.py` | Overlay mp4 writer |
| `step4_save_keypoints.py` | Step 4 window + save |
| `src/analysis/angles_2d.py` | 2D angle at three joints |
| `src/analysis/trajectory_2d.py` | Wrist trail + px/s |
| `step5_analyze_2d.py` | Step 5 window + angle CSV |
| `src/ui/session.py` | One Start/Stop session (pose + angles + save) |
| `src/ui/app.py` | Tkinter window |
| `app.py` | Launch: `python app.py` |

---

## Next step (not built yet)

**3D:** use RealSense depth at the 2D pixels (metres in camera coordinates). Do not treat MediaPipe world landmarks as RealSense 3D.
