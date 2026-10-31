# Foot depth vs framing — `20261012-153538`

**Recording:** `data/diagnostics/20261012-153538/` (524 frames, 0 dropped, wall FPS 8.13)  
**Image / depth stream:** 1280 × 720 (from `config.yaml` and pixel bounds)  
**New columns present:** `u_px`, `v_px`, `depth_sample_m`, `depth_n_valid_in_window`, `in_image`, `depth_valid`, `xyz_missing`. No extra columns needed.

Production code was **not** changed. Depth window, min/max Z, mappings, LSTM, DOFs, and smoothing are unchanged.

---

## 1. Root cause

Ankles are already reliable on this take. Remaining misses are **not** a crop, GUI resize, or u/v conversion bug.

| Landmark | Rows | Valid XYZ | `in_image=0` | In-image `n_valid=0` | Depth present but rejected |
|---|---:|---:|---:|---:|---:|
| left_ankle | 460 | 459 | 0 | 1 | 0 |
| right_ankle | 460 | 460 | 0 | 0 | 0 |
| left_heel | 460 | 458 | 1 | 1 | 0 |
| right_heel | 460 | 458 | 2 | 0 | 0 |
| left_foot_index | 460 | 452 | 7 | 1 | 0 |
| right_foot_index | 460 | 442 | 18 | 0 | 0 |
| left/right knee | 460 | 460 | 0 | 0 | 0 |

**Out of bounds (class A):** every `in_image=0` row has `v_px ≥ 720` and `u_px` inside 0…1279. Max `v_px` is 803 (left_foot_index). `depth_n_valid_in_window=0` and `depth_sample_m=0` there because the rounded pixel is outside the aligned depth image — same path as `median_depth_m` (`src/capture/depth_util.py`). Those frames cluster at **walk-in/out** (≈40–51 and ≈489–494). Median standing `v_px` is well inside the frame (ankles ~629–638, heels ~647–650, foot_index ~673–687).

MediaPipe is allowed to emit normalized y > 1 when it guesses a joint below the RGB image (`src/pose/extractor_2d.py`: `v_px = landmark.y * height`). That is not a display-coordinate error.

**In-image holes (class B):** three leftover samples, all left side, last ~16 rows of the image:

- left_ankle frame with `v=704.673`, `n_valid=0`
- left_heel `v=707.366`
- left_foot_index `v=718.854`

Aligned depth can be empty on the bottom strip (RGB vs stereo FOV / floor grazing). This is **not** the 99% failure from `20261012-150535`.

**Not the GUI:** `_show_frame` in `src/ui/app.py` letterboxes the **full** 1280×720 with uniform scale (`scale = min(avail_w/w, avail_h/h)`). It does not crop. Pose, depth, and CSV all use the native colour frame. Click mapping uses `_native_wh` / `_drawn_wh` with the same letterbox origin.

Previous 99% invalid ankles were the same mechanism with feet mostly off the bottom or in a depth hole; this take has feet on-screen for almost every pose frame.

---

## 2. Files changed

None in the live pipeline. This folder only:

- `_analyze_feet.py`, `analysis.json`
- `fig_v_px_vs_bounds.png`, `fig_oob_vs_hole.png`
- this report

Diagnostics fields were already sufficient.

---

## 3. Tests

No production change, so no new tests. Existing inspect/recorder tests from the previous diagnostics-only change remain the contract: OOB → `in_image=0`, zero-depth window → `depth_sample_m=0` (not invented XYZ).

---

## 4. Framing vs depth-sampling

**Manual camera placement is sufficient** for observed ankle XYZ and for almost all heel/foot_index XYZ. Do **not** change `sample_window`, `min_m`/`max_m`, or clamp OOB pixels onto the last row (that would sample the wrong place).

A 1280×800 colour mode was **not** applied: it is unverified on this D455f with aligned 1280×720 depth, and this task forbids starting the camera to try it.

A later **optional** depth-edge experiment (larger window / shin-ward sample) would only target the three class-B rows. It is not required for ankles.

---

## 5. Manual validation

1. Keep the current 1280×720 streams. Tilt the camera down or step back until **toes sit above the bottom of the live image**, with ~30–40 px of floor visible (`v` for `foot_index` ≲ 680).
2. Start 3D + Diagnostics. Stand still 10 s, then walk so feet stay fully in view (do not fill the frame head-to-toe).
3. New folder under `data/diagnostics/` (never overwrite `20261012-153538` or `20261012-150535`).
4. Check: `in_image=1` for ankle/heel/foot_index; `xyz_missing=0` except true holes; when missing, either `in_image=0` and `v_px≥720`, or `in_image=1` and `depth_n_valid_in_window=0`. Empty metres, never `0,0,0`.
5. Knees/hips should stay ~100% valid (alignment sanity).
6. Restart `python app.py` after any future code change so diagnostics columns stay current.

**Smallest safe change:** more floor in the colour image by camera tilt / distance / height — not a software FOV or depth invention.
