# 2D and 3D Motion Analysis (single RealSense D455f)

Student: **Jay**  
Hardware: **Intel RealSense Depth Camera D455f**  
Current phase: **2D only** (RGB keypoints in camera pixel coordinates)

We build this **one step at a time**. Pose is not in the code yet.

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

---

## Next step (not built yet)

**Step 3:** extract **2D keypoints** `(u, v)` in camera pixel coordinates on these RGB frames.
