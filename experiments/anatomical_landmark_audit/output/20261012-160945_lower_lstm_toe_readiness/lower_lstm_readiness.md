# Toe calibration and lower LSTM readiness (read-only)

**Stamp:** 2026-10-12 16:09  
**Production code:** not modified  
**Latest live diagnostic:** `data/diagnostics/20261012-155914/` (no `calibrated_toe`, no `predicted_lower`)  
**Safe tests run:** `tests.test_full_body_markers` — 7/7 OK (no camera)

Ordinary 3D live tracking (Pose + depth + **upper** LSTM) does **not** need toe calibration. Lower-body Pose2Sim augmentation **does**.

---

## 1. What **Calibrate toes** actually does

### UI (`src/ui/app.py`)

| Action | Behaviour |
|---|---|
| Click **Calibrate toes** while **not** running | Info dialog only: start a **3D** live session, then click four toes. **No file**, no camera start. (`_on_calibrate_toes` 652–661) |
| Click **Calibrate toes** while running | Sets status to `Calibrate toes: click LEFT big toe on the video`. Does **not** write a file. Does **not** arm a separate mode flag. |
| **Left mouse click** on the live video (session running) | Always queued as `{"kind": "toe_click", "u", "v"}` in native colour pixels after undoing letterbox (`_on_video_click` 665–683). |

**No keyboard** is used in the live app for toes. (The experiment script `label_toes_from_bag.py` uses keys 1–4; that is **not** the live button.)

Clicks are processed only if `LiveBodyCoordinator` exists: **3D** and **Live Pose2Sim** checked (`session.py` 189–196). If LSTM is off, clicks are discarded (`_drain_marker_events` 254–256).

### Click processing (`session.py` 221–263, `foot_calib.py`)

1. Deproject click with the **same** depth window/min/max as joints (`_deproject_uv`). Invalid depth → banner, **no** invented XYZ.
2. Next unused name in order **`LBigToe`, `LSmallToe`, `RBigToe`, `RSmallToe`** (`catalog.py` `TOE_INPUTS`).
3. **Same frame** must have valid camera XYZ on that foot’s `ankle`, `heel`, and `foot_index`. Those three are the **rigid foot frame**, not the toe identity.
4. Store the toe in the foot-local frame: `R^T (click − ankle)` (`foot_calib.py` 96). RAM only: `FootCalibrator.local`.
5. Each later frame **reconstructs** `ankle + R @ local` if ankle/heel/`foot_index` still have vis ≥ `min_visibility` and valid XYZ. Missing → that toe omitted that frame (no fill).

**`foot_index` is not copied to BigToe/SmallToe.** Tests assert `extract_lower_frame` source has no `foot_index` string and `TOE_INPUTS` exclude those names.

Overlay caveat: reconstructed `Keypoint2D.u_px/v_px` are copied from **`foot_index`**, not from the click (`foot_calib.py` 125–126). **XYZ** is the calibrated toe. The green square may sit on the toe-ray pixel, not the hallux pixel.

### What file it writes

**None.** Calibration is in-memory for the current `MotionSession2D`. Stopping the session discards it. Diagnostics `landmarks.csv` will contain `class=calibrated_toe` **only if** diagnostics is on **after** clicks succeed (`coordinator.overlay_records`). No `toes.json` exists in the repo. No `*toe*.csv` on disk.

The separate bag labeller can write a CSV if you run it on a `.bag`; that has **not** been done here (no such output files).

### Can those clicks feed the installed lower model?

**Yes, if all four reconstructions are present with valid XYZ on a frame**, they are merged into the keypoint list and named exactly `LBigToe` / `LSmallToe` / `RBigToe` / `RSmallToe`, which `extract_lower_frame` requires. They are **session-calibrated estimates**, not MediaPipe landmarks and not palpated toes.

---

## 2. Lower-body model readiness

**Weights:** installed Pose2Sim 0.10.49 `v0.3_lower/{model.onnx,mean.npy,std.npy}` under `pose2sim-offline`. Worker: `tools/pose2sim_lower_predict_worker.py` (`inputs` shape `(T, 47)`, output last step length 105). Live class: `LiveLowerLstm` (`engine.py` 223–236). **Ready as software.** **Not ready on any current diagnostic recording.**

### Required inputs (15 names, order = `POSE2SIM_LOWER_IN`)

`Neck` (mid-shoulders), `RShoulder`, `LShoulder`, `RHip`, `LHip`, `RKnee`, `LKnee`, `RAnkle`, `LAnkle`, `RHeel`, `LHeel`, `RSmallToe`, `LSmallToe`, `RBigToe`, `LBigToe`.

| Input | Live source | Available without calibration? |
|---|---|---|
| Shoulders, hips, knees, ankles, heels | MediaPipe + RealSense XYZ | Yes, when in-frame (audit 161–548) |
| `Neck` | Geometric mid-shoulder | Yes if both shoulders have XYZ |
| Four distinct toes | **Calibrated clicks only** | **No.** MediaPipe has one `foot_index` per foot. |

**Distinct big/small toes are required.** They are **not** observed. Mapping `foot_index` onto both is **not** implemented (and must not be).

### Preprocess / sequence (same engine as upper)

- Window **T = 16** live, min **8** frames, max gap **2** (`LIVE_WINDOW` / `UpperSequenceBuffer` reused). Incomplete frame → drop, never interpolate.
- Camera → OpenSim Y-up: `(X, -Y, Z)`.
- Hip-centre (mid MP hips), `÷ 1.80 m`, append height **1.80** and mass **61**, then `(x − mean) / std` with lower `mean.npy`/`std.npy` length 47.
- Height/mass are **hardcoded**, not in diagnostics CSV.
- Causal last timestep; `yup_to_camera` for overlay/CSV.

### Outputs (35), camera metres

ASIS/PSIS study, HJC study, C7 study, shoulder study, medial/lateral knee and ankle, toe / 5th meta / calcaneus study, thigh/shank clusters (`POSE2SIM_LOWER_OUT`). Then geometric `*_knee_centre` / `*_ankle_centre` if both sides of a pair exist.

**Not produced:** IJ, PX, T8, acromion, GHJC.

Boot: `LiveLowerLstm` starts on a **background thread** only after `feet.ready()` (4/4 clicks) (`coordinator.py` 39–60). Until then HUD: `LOWER missing input: LBigToe,... · click Calibrate toes`.

---

## 3. Missing inputs — what you need to do in the UI

No recording on disk has the four toe names. Offline lower inference on `20261012-155914` **cannot** be run honestly.

**Exact live steps (no code change):**

1. Restart `python app.py` if it predates current code.
2. **3D**, **Live Pose2Sim** on, Start. Stand in the **161–548 style** pose: ~2.2 m, feet above the red guide, ~30–40 px of floor.
3. Start **Diagnostics** if you want `calibrated_toe` / `predicted_lower` rows saved.
4. Click **Calibrate toes** (status prompt).
5. On the **video** (not the 3D window), left-click in order: **left hallux**, **left 5th toe / small toe**, **right hallux**, **right 5th**. Each click needs depth in [0.3, 6] m **and** valid XYZ on that foot’s ankle, heel, `foot_index` in that frame. HUD says if depth or the foot frame is missing — click again, do not guess.
6. After `Toe calibration complete (4/4)`, wait until HUD shows `LOWER LIVE` (ONNX load + 8-frame buffer).
7. Stand still ~30 s, then Stop Diagnostics. New folder under `data/diagnostics/` — do not overwrite `20261012-155914`.

If a click is ignored: LSTM off, 2D mode, click in the letterbox margin, or invalid depth.

---

## 4. Safe offline validation performed

| Check | Result |
|---|---|
| Unit tests `test_full_body_markers` | **7 OK**: missing toes → `extract_lower_frame` is `None` and does not fake; four named toes → feat `(45,)`; rigid-foot click round-trips; `foot_index` not a toe input |
| Search `data/` landmarks for `LBigToe` / `calibrated_toe` | **None** |
| `20261012-155914` classes | observed + upper predicted + elbow/wrist centres only |
| Experiment bag toe CSV | **None** |
| Live ONNX on that recording | **Not run** (toes absent; would require fabricating inputs) |

---

## 5. Proposed next implementation stage (not started)

**No pipeline change yet.** Next approved action should be **your live 4-click calibration** during an in-frame standing diagnostic, then a read-only audit of the new `landmarks.csv` for:

- four `calibrated_toe` names with non-empty XYZ;
- 35 `predicted_lower` names after buffering;
- knee/ankle study midpoints vs MediaPipe (distance, not identity);
- still **no** DOF / mapping / `foot_index` alias.

Optional later (needs a separate approval): persist `FootCalibrator.local` to a JSON so calibration survives Stop — **not** done now.

---

## Distinction

| Mode | Toes | Lower LSTM |
|---|---|---|
| Live 3D + upper LSTM + diagnostics | Not required | Off |
| Full-body study markers (pelvis, HJC, C7, condyles, foot study, clusters) | Four calibrated toes **required** | On after 4/4 + valid XYZ on the 10 MP foot/leg/shoulder inputs |
