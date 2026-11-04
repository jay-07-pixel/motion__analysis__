# Phase 2 — Wrist roll reference (mathematical correction)

## Invalid reference (removed)

`model_wrist_elbow_roll_deg` read marker `location` vectors from
`Markers_LSTM.xml` / the `.osim` MarkerSet. Elbow study markers are expressed
in the **humerus** body frame; wrist study markers in the **radius** body frame.
Mixing those two local vectors as if they were ground points produced a
spurious ~7.5° “model roll.”

OpenSim `getLocationInGround` at default pose (`pro_sup` default ≈ 90°) gives
~99° unsigned elbow–wrist ML roll. LSTM rolls on take 140115 (~85° / ~106°) sit
near that default, not near 7.5°.

## Correct reference

For side S, with elbow lateral/medial and wrist lateral/medial in **one** frame
(camera / ground):

1. `e_mid = (e_lat + e_med) / 2`, `w_mid = (w_lat + w_med) / 2`
2. `forearm = unit(w_mid − e_mid)`
3. Project `e_lat − e_med` and `w_lat − w_med` into the plane ⟂ forearm
4. Unsigned roll = angle between those projected directions

**Model reachable band:** load the sized (or camera) `.osim` in the isolated
OpenSim interpreter. Hold all coordinates at defaults except `pro_sup_{r|l}`,
sweep that coordinate over its model range, and at each sample compute the
ground-frame unsigned roll above. Optionally also sample a few `elbow_flex`
values at default `pro_sup`. The accepted band for that side is
`[min_roll, max_roll]` expanded by a small margin (default 5°).

A recorded frame passes the roll check when its camera-frame roll lies in that
band. It is **not** compared to a single default-pose angle, and **not** to the
invalid mixed-local 7.5°.

## Unchanged

- Wrist pair spacing gate (± `REPORT_MISMATCH_M`)
- Elbow spacing gate
- Step gate
- Recorded TRC bytes
- Marker XYZ (no rotation, no invention)
- `clinical_accuracy` / `biomechanical_reliability` remain false
