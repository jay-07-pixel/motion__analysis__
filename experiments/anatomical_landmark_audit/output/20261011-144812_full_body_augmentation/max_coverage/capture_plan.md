# Capture / labeling plan for the four toe landmarks

Do **not** reuse this session’s `overlay.mp4` for 3D clicks. Depth is gone.

## Why this recording cannot finish lower LSTM

| Need | This take |
|---|---|
| Distinct hallux vs 5th-metatarsal pixels, left and right | One MediaPipe `foot_index` per foot on a sneaker |
| Aligned depth at those pixels | Not saved (CSV has depth only at detected keypoints) |
| Camera intrinsics for deprojection | Not stored in `run.json` |

## Preferred next capture (existing D455f, no live-pipeline code change)

1. In **Intel RealSense Viewer**, press Record and save a `.bag` (RGB + depth).
2. Framing:
   - Full body still in view if you want the same trial, **or** a short dedicated foot take.
   - Camera lower or step closer so both shoe/toe boxes fill more than a few dozen pixels.
   - Prefer **barefoot or thin socks**. These sneakers hide hallux vs 5th metatarsal.
   - Stand quietly 10–20 s, feet slightly apart, toes pointing toward the camera or slightly out so the medial and lateral toe box are both visible.
   - Keep the same person (height 1.80 m, mass 61 kg).
3. Optional: also run the existing app live as usual so you keep a Pose CSV; the **bag** is what labeling needs.

## Labeling (after the bag exists)

From the repo, with the project Python that already has `pyrealsense2`:

```
python experiments/anatomical_landmark_audit/scripts/label_toes_from_bag.py --bag PATH\to\feet.bag --out experiments/anatomical_landmark_audit/output/20261011-144812_full_body_augmentation/toe_labels
```

- `1` LBigToe, `2` LSmallToe, `3` RBigToe, `4` RSmallToe, then click.
- Space = next frame. `s` = save. Invalid depth is stored empty.
- Click the **visible** hallux and 5th-metatarsal (or shoe medial/lateral toe-box if still shod) as four **different** pixels.
- Do not click the same pixel four times.

Then tell me the bag path and label folder. Next step here: merge labeled toes into a **new** experimental TRC copy and run both official LSTMs. No IK, no calibration, no live-pipeline edits.
