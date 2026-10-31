"""Visual evidence for Marker Enhancer on recording 20261005-150538.

Reads existing inference results. Does not rerun the model and does not
write to the original CSV/JSON outputs.

Predictions:
  data/experiments/marker_enhancer_20261005/inference_predictions.csv
Recording (MediaPipe camera metres):
  data/output/20261005-150538/keypoints_2d.csv
Figures:
  data/output/marker_enhancer_analysis/
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDING_CSV = PROJECT_ROOT / "data" / "output" / "20261005-150538" / "keypoints_2d.csv"
PRED_CSV = PROJECT_ROOT / "data" / "experiments" / "marker_enhancer_20261005" / "inference_predictions.csv"
OUT_DIR = PROJECT_ROOT / "data" / "output" / "marker_enhancer_analysis"

OUTPUT_MARKERS = (
    "RElbow_augmenter",
    "RMElbow_augmenter",
    "RWrist_augmenter",
    "RMWrist_augmenter",
    "LElbow_augmenter",
    "LMElbow_augmenter",
    "LWrist_augmenter",
    "LMWrist_augmenter",
)

PAIRS = (
    ("right_elbow", "RElbow_augmenter", "RMElbow_augmenter", "right_elbow"),
    ("left_elbow", "LElbow_augmenter", "LMElbow_augmenter", "left_elbow"),
    ("right_wrist", "RWrist_augmenter", "RMWrist_augmenter", "right_wrist"),
    ("left_wrist", "LWrist_augmenter", "LMWrist_augmenter", "left_wrist"),
)

MP_ARM = (
    "right_shoulder",
    "left_shoulder",
    "right_elbow",
    "left_elbow",
    "right_wrist",
    "left_wrist",
)

MIN_CONFIDENCE = 0.5


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing required file (not generating fake data): {path}")


def load_predictions(path: Path) -> dict[int, dict]:
    """frame -> {marker: (xyz_camera, xyz_me, time_sec)}."""
    by_frame: dict[int, dict] = {}
    n_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n_rows += 1
            marker = row["marker"]
            if marker not in OUTPUT_MARKERS:
                continue
            frame = int(row["frame"])
            cam = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], dtype=float)
            me = np.array([float(row["x_me_m"]), float(row["y_me_m"]), float(row["z_me_m"])], dtype=float)
            slot = by_frame.setdefault(frame, {"time_sec": float(row["time_sec"]), "cam": {}, "me": {}})
            slot["cam"][marker] = cam
            slot["me"][marker] = me
    return {"by_frame": by_frame, "n_rows": n_rows}


def load_mediapipe_arm(path: Path, frames: set[int]) -> dict[int, dict[str, np.ndarray]]:
    stored: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["coord_frame"] != "camera_3d_metre":
                continue
            joint = row["joint"]
            if joint not in MP_ARM:
                continue
            frame = int(row["frame"])
            if frame not in frames:
                continue
            if not row["x_m"] or not row["y_m"] or not row["z_m"]:
                continue
            if float(row["confidence"]) < MIN_CONFIDENCE:
                continue
            point = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], dtype=float)
            if np.all(np.isfinite(point)):
                stored[frame][joint] = point
    return stored


def dist_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n_valid": 0,
            "min_m": math.nan,
            "max_m": math.nan,
            "mean_m": math.nan,
            "median_m": math.nan,
            "std_m": math.nan,
        }
    return {
        "n_valid": int(values.size),
        "min_m": float(np.min(values)),
        "max_m": float(np.max(values)),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "std_m": float(np.std(values, ddof=0)),
    }


def fmt(stats: dict) -> str:
    if stats["n_valid"] == 0:
        return "n=0  (no valid frames)"
    return (
        f"n={stats['n_valid']}  min={stats['min_m']:.4f} m  max={stats['max_m']:.4f} m  "
        f"mean={stats['mean_m']:.4f} m  median={stats['median_m']:.4f} m  std={stats['std_m']:.4f} m"
    )


def pair_lengths(pred_frames: dict, a: str, b: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    times = []
    lengths = []
    for frame in sorted(pred_frames):
        cam = pred_frames[frame]["cam"]
        if a not in cam or b not in cam:
            continue
        pa, pb = cam[a], cam[b]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        frames.append(frame)
        times.append(pred_frames[frame]["time_sec"])
        lengths.append(float(np.linalg.norm(pa - pb)))
    return np.array(frames), np.array(times), np.array(lengths)


def mp_travel(mp_frames: dict, joint: str, frames: np.ndarray) -> np.ndarray:
    points = []
    for frame in frames:
        joints = mp_frames.get(int(frame), {})
        if joint not in joints:
            points.append(np.array([math.nan, math.nan, math.nan]))
        else:
            points.append(joints[joint])
    pts = np.vstack(points)
    finite = np.all(np.isfinite(pts), axis=1)
    if not np.any(finite):
        return np.full(len(frames), math.nan)
    centre = np.median(pts[finite], axis=0)
    return np.linalg.norm(pts - centre, axis=1)


def plot_pair_distances(out_path: Path, series: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for label, times, lengths in series:
        ax.plot(times, lengths * 1000.0, lw=1.4, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pair distance (mm)")
    ax.set_title("Marker Enhancer predicted pair distance")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_elbow_pair_vs_mp_motion(
    out_path: Path,
    times: np.ndarray,
    pair_m: np.ndarray,
    mp_travel_m: np.ndarray,
    side: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    axes[0].plot(times, pair_m * 1000.0, color="#0D9488", lw=1.5)
    axes[0].set_ylabel("Predicted pair distance (mm)")
    axes[0].set_title(f"{side} predicted elbow-pair length vs MediaPipe elbow movement")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, mp_travel_m * 1000.0, color="#334155", lw=1.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("MediaPipe elbow distance\nfrom its median position (mm)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_3d(out_path: Path, pred_frames: dict, mp_frames: dict, frame: int) -> None:
    pred = pred_frames[frame]["cam"]
    mp = mp_frames.get(frame, {})
    fig = plt.figure(figsize=(10.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")

    def scatter(points, **kwargs):
        pts = np.vstack(points)
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], **kwargs)

    def line(a, b, **kwargs):
        ax.plot([a[0], b[0]], [a[2], b[2]], [-a[1], -b[1]], **kwargs)

    if all(name in mp for name in MP_ARM):
        scatter(
            [mp["right_shoulder"], mp["left_shoulder"], mp["right_elbow"], mp["left_elbow"], mp["right_wrist"], mp["left_wrist"]],
            c="#64748B",
            s=42,
            label="MediaPipe shoulder / elbow / wrist",
        )
        line(mp["right_shoulder"], mp["right_elbow"], color="#94A3B8", lw=1.6)
        line(mp["right_elbow"], mp["right_wrist"], color="#94A3B8", lw=1.6)
        line(mp["left_shoulder"], mp["left_elbow"], color="#94A3B8", lw=1.6)
        line(mp["left_elbow"], mp["left_wrist"], color="#94A3B8", lw=1.6)
        line(mp["left_shoulder"], mp["right_shoulder"], color="#CBD5E1", lw=1.2)

    pred_pts = [pred[name] for name in OUTPUT_MARKERS if name in pred]
    scatter(pred_pts, c="#0D9488", s=36, marker="^", label="Marker Enhancer predictions")
    for _, a, b, _ in PAIRS:
        if a in pred and b in pred:
            line(pred[a], pred[b], color="#0F766E", lw=1.8)

    ax.set_xlabel("X camera (m, right)")
    ax.set_ylabel("Z camera (m, forward)")
    ax.set_zlabel("up (m)  = -Y camera")
    ax.set_title(f"Camera-metre landmarks, frame {frame}  (Y inverted for display only)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    require_file(PRED_CSV)
    require_file(RECORDING_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    packed = load_predictions(PRED_CSV)
    pred_frames = packed["by_frame"]
    n_pred_frames = len(pred_frames)
    n_markers = len(OUTPUT_MARKERS)
    sample_frame = sorted(pred_frames)[0]
    sample_xyz = pred_frames[sample_frame]["cam"][OUTPUT_MARKERS[0]]

    print("=== Marker Enhancer prediction file ===")
    print(f"path: {PRED_CSV}")
    print(f"csv_rows: {packed['n_rows']}")
    print(f"frames: {n_pred_frames}")
    print(f"markers_per_frame: {n_markers}")
    print(f"model_output_shape_from_runtime: (464, 24)  # 8 markers x 3")
    print(f"loaded_array_equivalent: ({n_pred_frames}, {n_markers}, 3)")
    print("coordinate format:")
    print("  x_m, y_m, z_m     = camera_3d_metre  (X right, Y down, Z forward)")
    print("  x_me_m, y_me_m, z_me_m = Marker Enhancer lab frame after the fixed camera map")
    print(f"example frame {sample_frame} {OUTPUT_MARKERS[0]} camera XYZ: {sample_xyz.tolist()}")

    mp_frames = load_mediapipe_arm(RECORDING_CSV, set(pred_frames))

    pair_series = []
    pair_stats = {}
    for label, a, b, mp_joint in PAIRS:
        frames, times, lengths = pair_lengths(pred_frames, a, b)
        pair_series.append((label, times, lengths, frames, mp_joint, a, b))
        pair_stats[label] = dist_stats(lengths)
        print(f"pair {label}: {fmt(pair_stats[label])}")

    plot_pair_distances(
        OUT_DIR / "elbow_pair_distance.png",
        [(s[0], s[1], s[2]) for s in pair_series if "elbow" in s[0]],
    )
    plot_pair_distances(
        OUT_DIR / "wrist_pair_distance.png",
        [(s[0], s[1], s[2]) for s in pair_series if "wrist" in s[0]],
    )

    mid_frame = sorted(pred_frames)[len(pred_frames) // 2]
    plot_3d(OUT_DIR / "marker_enhancer_3d.png", pred_frames, mp_frames, mid_frame)

    comparison_lines = []
    for label, times, lengths, frames, mp_joint, a, b in pair_series:
        if "elbow" not in label:
            continue
        travel = mp_travel(mp_frames, mp_joint, frames)
        plot_elbow_pair_vs_mp_motion(
            OUT_DIR / f"{label}_pair_vs_mediapipe_elbow.png",
            times,
            lengths,
            travel,
            label.replace("_", " "),
        )
        finite = np.isfinite(travel) & np.isfinite(lengths)
        travel_f = travel[finite]
        lengths_f = lengths[finite]
        quiet = travel_f < 0.02
        n_quiet = int(np.sum(quiet))
        pair_range_all = float(np.max(lengths_f) - np.min(lengths_f)) if lengths_f.size else math.nan
        pair_range_quiet = (
            float(np.max(lengths_f[quiet]) - np.min(lengths_f[quiet])) if n_quiet else math.nan
        )
        mp_range = float(np.max(travel_f) - np.min(travel_f)) if travel_f.size else math.nan
        corr = (
            float(np.corrcoef(travel_f, lengths_f)[0, 1])
            if travel_f.size >= 3 and np.std(travel_f) > 0 and np.std(lengths_f) > 0
            else math.nan
        )
        comparison_lines.append(
            f"{label}: MediaPipe {mp_joint} travel from its median position "
            f"range={mp_range:.4f} m. Predicted pair-length range={pair_range_all:.4f} m. "
            f"Pearson r(travel, pair length)={corr:.3f}. "
            f"While MediaPipe elbow stayed within 20 mm of its median "
            f"(n={n_quiet} frames), predicted pair-length range was {pair_range_quiet:.4f} m."
        )

    repo_explicit = (
        "Repository (tools/marker_enhancer_offline_inference.py):\n"
        "- OUTPUT_MARKERS lists the eight augmenter names in this order:\n"
        "  RElbow_augmenter, RMElbow_augmenter, RWrist_augmenter, RMWrist_augmenter,\n"
        "  LElbow_augmenter, LMElbow_augmenter, LWrist_augmenter, LMWrist_augmenter.\n"
        "- PAIRS groups (R*, RM*) as right/left elbow_pair and wrist_pair.\n"
        "- MEASURED_COMPARISON compares RElbow_augmenter with measured RElbow (same for L/wrist),\n"
        "  and the text report labels that comparison 'predicted lateral marker'.\n"
        "- validation_report.json biomechanical_note: 'These are predicted lateral and medial\n"
        "  elbow and wrist markers. They are not true epicondyles or styloids.'\n"
        "- Neck input is labeled shoulder_midpoint_neck_proxy.\n"
        "- Coordinates x_m,y_m,z_m are camera_3d_metre; x_me_m,y_me_m,z_me_m are the mapped lab frame.\n"
    )
    inferred = (
        "Inferred from geometry in this analysis (not claimed as anatomy):\n"
        "- Pair distance is the Euclidean length between the two predicted points of each PAIRS entry.\n"
        "- Variation of that length while the MediaPipe elbow centre barely moves is evidence about\n"
        "  whether the pair is a stable local geometry for a joint coordinate frame.\n"
        "- We do not treat RElbow_augmenter / RMElbow_augmenter as measured epicondyles.\n"
    )

    stats_path = OUT_DIR / "statistics.txt"
    lines = [
        "Marker Enhancer evidence  --  recording 20261005-150538",
        "Status: inference completed successfully. This file reviews reliability for anatomical frames.",
        "",
        "Prediction file (do not modify):",
        f"  {PRED_CSV}",
        "Recording MediaPipe 3D (do not modify):",
        f"  {RECORDING_CSV}",
        "",
        f"Frames with predictions: {n_pred_frames}",
        f"Markers: {n_markers}  (8 names x 3 camera metres = 24 values / frame)",
        f"CSV rows: {packed['n_rows']}",
        "Coordinate format: camera_3d_metre in x_m,y_m,z_m; also ME-frame x_me_m,y_me_m,z_me_m.",
        "",
        "Pair distances (existing predictions, all finite frames)",
    ]
    for label, _, _, _, _, a, b in pair_series:
        lines.append(f"  {a} <-> {b}")
        lines.append(f"    {fmt(pair_stats[label])}")
    lines.append("")
    lines.append("Predicted elbow-pair length vs MediaPipe elbow-centre movement")
    lines.extend(f"  {item}" for item in comparison_lines)
    lines.append("")
    lines.append(repo_explicit)
    lines.append(inferred)
    lines.append("Figures")
    lines.append("  elbow_pair_distance.png")
    lines.append("  wrist_pair_distance.png")
    lines.append("  marker_enhancer_3d.png")
    lines.append("  right_elbow_pair_vs_mediapipe_elbow.png")
    lines.append("  left_elbow_pair_vs_mediapipe_elbow.png")
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {stats_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
