# Decision

## Mapping table (required point → methods)

| Required anatomical point | Existing MediaPipe | Candidate Pose2Sim / OpenCap output | Source for the mapping | Type | Accuracy / validation | Limitations | Recommendation |
|---|---|---|---|---|---|---|---|
| GHJC | `*_shoulder` (not GHJC) | `r_shoulder_study` / `L_shoulder_study` | Name in `getOpenPoseMarkers_lowerExtremity2`; OpenCap “anatomical markers” | Learned | OpenCap body RMSE 8 mm on **synthetic** test set; not GHJC-specific here | Not a functional HJC | **Unsuitable** as GHJC |
| Thorax C7 | none | `C7_study` | Same function | Learned | OpenCap listed C7 among mocap comparison markers | LSTM; no C7 in our CSV | **Unsuitable** without a labelled study |
| Lateral humeral epicondyle | none (`*_elbow` is one point) | `r_lelbow_study` / `L_lelbow_study` (ME: `RElbow_augmenter`) | Pose2Sim upper responses; ME `OUTPUT_MARKERS` | Learned | OpenCap arm RMSE 15.2 mm synthetic; our ME pair 2–11 cm live | Name ≠ palpation; pair not rigid | **Unsuitable** for ISB elbow axis |
| Medial humeral epicondyle | none | `r_melbow_study` / `L_melbow_study` (ME: `RMElbow_augmenter`) | Same | Learned | Same local pair-length failure | Same | **Unsuitable** |
| Radial styloid | none (`*_wrist`) | `r_lwrist_study` / `L_lwrist_study` (ME: `RWrist_augmenter`) | Same | Learned | Chaumeil-class cm wrist errors; our wrist pair unstable | No pronation | **Unsuitable** |
| Ulnar styloid | none | `r_mwrist_study` / `L_mwrist_study` (ME: `RMWrist_augmenter`) | Same | Learned | Same | Same | **Unsuitable** |
| Palm / hand orientation | Hands `index_mcp`, `pinky_mcp`, `middle_mcp`, wrist | none required | `build_hand_frame` | Observed when Hands+depth exist | Intermittent Hands on 20261011 | Not ISB metacarpals | **Suitable** as **practical** orientation only |
| ASIS / PSIS | none (`*_hip`) | `r.ASIS_study`, `L.PSIS_study`, … | Lower augmenter list | Learned | Synthetic body RMSE; OOD failures (Falisse 2025) | Hips ≠ pelvis markers | **Experimental only** / **unsuitable** for ISB pelvis |
| Hip joint centre | none | `RHJC_study`, `LHJC_study` | Same | Learned | Model target, not dual-plane X-ray | | **Unsuitable** without validation |
| Femoral epicondyles | `*_knee` (one) | `r_knee_study`, `r_mknee_study` | Same | Learned | | One MP knee | **Unsuitable** as both epicondyles |
| Malleoli | `*_ankle` (one) | `r_ankle_study`, `r_mankle_study` | Same | Learned | | | **Unsuitable** |
| Calcaneus | `*_heel` (proxy) | `r_calc_study` | Same | Learned + we already observe heel | Heel conf lower | | Heel: **observed proxy**. `*_calc_study`: **experimental only** |
| Hallux / 5th metatarsal | `*_foot_index` only | `r_toe_study`, `r_5meta_study`; SmallToe filler | Augmenter + `add_smalltoe_data` | Learned / geometric | We lack BigToe vs SmallToe | Do not equate `foot_index` to both | **Unsuitable** without a toe study |

If a cell has no defensible mapping, it is marked **unsuitable**, not “close enough”.

## Next recommended experiment

**Not** augmenter-on-by-default. **Not** DOF edits.

**Recommended next (if any software experiment):** isolated **observed-landmark quality** on `20261011-144812` only:

- Directory: `experiments/anatomical_landmark_audit/output/20261011-144812_quality/` (create later, after approval)
- Input: existing CSV, read-only
- Work: standing/quiet windows; drop frames with impossible segment lengths; report fraction of frames with valid thigh/shank/foot XYZ; Hands availability vs Pose wrists
- Expected result: a usable **observed** subset, still named as MediaPipe points
- Go: median thigh/shank lengths stable (stdev small vs median) on the quiet subset  
- No-go: if only the LSTM can “look anatomical”

**Optional later (explicit approval):** `experiments/pose2sim_whole_body/` augmenter **geometry** run on a **copy** of the Y-up TRC after documented renaming — compare pair lengths to ME.  
- Go for *further study*: pair length relative range comparable to or better than ME quiet-elbow (~cm still fails anatomy).  
- No-go for *anatomy*: any remaining centimetre-scale pair wander.

## Go / no-go (this decision)

| Action | Decision |
|---|---|
| Enable Pose2Sim `markerAugmentation` in the live app | **No-go** |
| Treat `*_study` / `*_augmenter` as epicondyles or styloids | **No-go** |
| Modify `dof_angles_3d.py` / `anatomical_frames.py` | **No-go** |
| Overwrite 20261011 recording or previous TRC | **No-go** |
| Continue using observed MP+Hands+depth as model 3D | **Go** |
| Isolated quality filter on the full-body CSV | **Go after approval** |
| Palpation / physical marker study for ISB bones | **Go only if** ISB bony frames become a stated requirement |

## Direct answers

**Can Pose2Sim marker augmentation provide the missing anatomical landmarks (epicondyles, styloids, GHJC, ASIS/PSIS, malleoli) as verified anatomy?**  
**No.** It emits learned OpenCap “study” markers. That is the same family we already rejected for frames after unstable pair lengths on this camera.

**Can Pose2Sim consume our single-camera RealSense 3D directly?**  
**No, not without unsupported conversion** (names, Neck/Hip, toes, NaN fill, Y-up, height/mass, fps/domain shift).

**Should we enable augmentation now?**  
**No.**

**Should we change the existing 7-DOF calculator to use augmenter points?**  
**No.**
