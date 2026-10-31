# Missing assets (do not download until approved)

## Option A — Pose2Sim (broadest: 35 body + 8 arm)

After an isolated `pip install pose2sim` (do **not** add to project `requirements.txt` without approval), the package is expected to contain:

```
Pose2Sim/MarkerAugmenter/LSTM/v0.3_lower/model.onnx
Pose2Sim/MarkerAugmenter/LSTM/v0.3_lower/mean.npy
Pose2Sim/MarkerAugmenter/LSTM/v0.3_lower/std.npy
Pose2Sim/MarkerAugmenter/LSTM/v0.3_upper/model.onnx
Pose2Sim/MarkerAugmenter/LSTM/v0.3_upper/mean.npy
Pose2Sim/MarkerAugmenter/LSTM/v0.3_upper/std.npy
```

Source: [perfanalytics/pose2sim](https://github.com/perfanalytics/pose2sim) `Pose2Sim/markerAugmentation.py` (`augmenter_model = 'v0.3'`).

Also needed at runtime: `onnxruntime` (already importable in project Python), height, mass, OpenPose-named dense TRC.

Install size: Pose2Sim plus its extras; ONNX files are large relative to git. Use a **separate venv**.

## Option B — Stanford Marker Enhancer body + existing arm

Already local:

```
data/experiments/marker_enhancer_20261005/reference_models/lstm/arm/
  model.json  weights.h5  mean.npy  std.npy  metadata.json
```

**Missing:**

```
reference_models/lstm/body/model.json
reference_models/lstm/body/weights.h5
reference_models/lstm/body/mean.npy
reference_models/lstm/body/std.npy
```

Upstream: [stanfordnmbl/marker-enhancer](https://github.com/stanfordnmbl/marker-enhancer) `reference_models/lstm/` (body + arm). Requires TensorFlow/Keras 2.12 as in conda env `marker-enhancer-offline`.

Body outputs (OpenCap list): ASIS/PSIS, medial/lateral knee and ankle “study” markers, toe/5th metatarsal/calcaneus, thigh/shank clusters, HJC, C7, shoulder_study — **estimated**.

Arm outputs (already known): 8 lateral/medial elbow/wrist **estimated** markers.

## Anthropometrics (both options)

Not in `data/output/20261011-144812/run.json`. LSTM features include height and mass. **Do not invent.** Supply measured values or an explicitly labelled experiment placeholder.

## What we will not fetch without approval

- `pip install pose2sim`
- git clone of marker-enhancer or pose2sim
- Hugging Face / GitHub LFS ONNX or `.h5` body weights
