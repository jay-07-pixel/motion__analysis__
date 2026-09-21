# 2D and 3D Motion Analysis (single RealSense D455f)

Student: **Jay**  
Hardware: **Intel RealSense Depth Camera D455f**  
Current phase: **2D only** (RGB keypoints in camera pixel coordinates)

We build this **one step at a time**. Later steps are not in the code yet.

---

## Step 1 (done) — Live RGB from the D455f

**What:** Our own Python window showing the same colour stream you saw in RealSense Viewer.

**Why:** 2D keypoints will be extracted on this RGB image. If Python cannot see RGB, pose will never work. Depth / 3D is later.

**How to run**

1. Plug the D455f into a **USB 3** port.
2. **Close RealSense Viewer** (only one app can use the camera).
3. In this folder:

```text
python step1_show_rgb_live.py
```

4. Press **q** in the video window to quit.

**If it fails:** close Viewer, try another USB 3 port, or in `config.yaml` set `width: 640` and `height: 480`.

**What to change without editing Python:** `config.yaml` (`camera.width`, `camera.height`, `camera.fps`, window title, quit key).

---

## Files added in Step 1

| File | Role |
|---|---|
| `config.yaml` | All settings (no hardcoding) |
| `src/utils/config_loader.py` | Reads that YAML |
| `src/capture/live_realsense.py` | Talks to the D455f RGB sensor |
| `step1_show_rgb_live.py` | Loop: grab frame → show window |

---

## Next step (not built yet)

**Step 2:** same RGB frame, but also from an **uploaded video file**, so live and file share one pipeline.
