"""Record live marker XYZ and timing. Missing values stay empty, never zero-filled."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.capture.depth_util import DepthSampleInspect
from src.markers.paper_set import PAPER_LANDMARKS, resolve_paper_landmarks
from src.pose.keypoints import Keypoint2D
from src.pose.wrist_identity import WRIST_IDENTITY_FIELDS, WristIdentityDecision
from src.utils.config_loader import resolve_project_path

FRAME_FIELDS = (
    "frame",
    "t_frame_sec",
    "t_unix",
    "t_utc",
    "process_ms",
    "dropped_before",
    "n_observed",
    "n_observed_xyz",
    "n_predicted",
    "t_pred_input_sec",
    "t_pred_done_sec",
    "infer_latency_ms",
    "pred_lag_ms",
    "fps",
    "capture_wait_ms",
    "pose_ms",
    "hands_ms",
    "pair_ms",
    "depth_ms",
    "lstm_prep_ms",
    "filter_ms",
    "draw_ms",
    "loop_ms",
)

LANDMARK_FIELDS = (
    "frame",
    "t_frame_sec",
    "t_unix",
    "landmark",
    "class",
    "raw_x_m",
    "raw_y_m",
    "raw_z_m",
    "filt_x_m",
    "filt_y_m",
    "filt_z_m",
    "visibility",
    "u_px",
    "v_px",
    "depth_sample_m",
    "depth_n_valid_in_window",
    "in_image",
    "depth_valid",
    "xyz_missing",
    "t_pred_input_sec",
    "t_pred_done_sec",
    "infer_latency_ms",
    "pred_lag_ms",
    "paper_name",
    "anatomical_status",
)

PAPER_FIELDS = (
    "frame",
    "t_frame_sec",
    "paper_name",
    "side",
    "region",
    "method",
    "source_landmark",
    "raw_x_m",
    "raw_y_m",
    "raw_z_m",
    "present",
    "reason",
)


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(int(value))


def _fmt_flag(value: bool | None) -> str:
    if value is None:
        return ""
    return "1" if value else "0"


def _pixel_uv(kp: Keypoint2D) -> tuple[str, str]:
    u = kp.u_px if math.isfinite(kp.u_px) else None
    v = kp.v_px if math.isfinite(kp.v_px) else None
    return _fmt(u, 3), _fmt(v, 3)


def _xyz_ok(x, y, z) -> bool:
    return (
        x is not None
        and y is not None
        and z is not None
        and math.isfinite(x)
        and math.isfinite(y)
        and math.isfinite(z)
    )


class _LandmarkStats:
    def __init__(self) -> None:
        self.n = 0
        self.n_valid = 0
        self.n_invalid = 0
        self.sum = [0.0, 0.0, 0.0]
        self.sum_sq = [0.0, 0.0, 0.0]
        self.min = [math.inf, math.inf, math.inf]
        self.max = [-math.inf, -math.inf, -math.inf]
        self.last: tuple[float, float, float] | None = None
        self.disp_sum = 0.0
        self.disp_n = 0
        self.disp_max = 0.0

    def add(self, xyz: tuple[float, float, float] | None) -> None:
        self.n += 1
        if xyz is None:
            self.n_invalid += 1
            self.last = None
            return
        self.n_valid += 1
        for i, v in enumerate(xyz):
            self.sum[i] += v
            self.sum_sq[i] += v * v
            self.min[i] = min(self.min[i], v)
            self.max[i] = max(self.max[i], v)
        if self.last is not None:
            dx = xyz[0] - self.last[0]
            dy = xyz[1] - self.last[1]
            dz = xyz[2] - self.last[2]
            step = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.disp_sum += step
            self.disp_n += 1
            self.disp_max = max(self.disp_max, step)
        self.last = xyz

    def report(self) -> dict:
        out: dict = {
            "n_frames_seen": self.n,
            "n_valid_xyz": self.n_valid,
            "n_invalid_xyz": self.n_invalid,
            "invalid_xyz_pct": (100.0 * self.n_invalid / self.n) if self.n else 0.0,
        }
        if self.n_valid == 0:
            out["std_m"] = {"x": None, "y": None, "z": None}
            out["range_m"] = {"x": None, "y": None, "z": None}
            out["mean_frame_to_frame_m"] = None
            out["max_frame_to_frame_m"] = None
            return out
        n = self.n_valid
        std = []
        rng = []
        for i in range(3):
            mean = self.sum[i] / n
            var = max(0.0, self.sum_sq[i] / n - mean * mean)
            std.append(math.sqrt(var))
            rng.append(self.max[i] - self.min[i])
        out["std_m"] = {"x": std[0], "y": std[1], "z": std[2]}
        out["range_m"] = {"x": rng[0], "y": rng[1], "z": rng[2]}
        out["mean_frame_to_frame_m"] = (self.disp_sum / self.disp_n) if self.disp_n else None
        out["max_frame_to_frame_m"] = self.disp_max if self.disp_n else None
        return out


class LiveDiagnosticsRecorder:
    """Write diagnostics CSVs under data/diagnostics/<stamp>/. Safe to close twice."""

    def __init__(
        self,
        folder: Path,
        participant_height_m: float | None = None,
        participant_mass_kg: float | None = None,
        filter_settings: dict | None = None,
    ) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.dir = folder
        self.n_frames = 0
        self.n_dropped = 0
        self.participant_height_m = participant_height_m
        self.participant_mass_kg = participant_mass_kg
        self.filter_settings = dict(filter_settings or {})
        self._closed = False
        self._process_ms: list[float] = []
        self._stage_ms: dict[str, list[float]] = defaultdict(list)
        self._stats: dict[str, _LandmarkStats] = defaultdict(_LandmarkStats)
        self._class: dict[str, str] = {}
        self._t_start_unix = time.time()
        self._t_start_wall = datetime.now(timezone.utc).isoformat()
        frames_path = folder / "frames.csv"
        marks_path = folder / "landmarks.csv"
        paper_path = folder / "paper_landmarks.csv"
        self._frames_file = frames_path.open("w", encoding="utf-8", newline="")
        self._marks_file = marks_path.open("w", encoding="utf-8", newline="")
        self._paper_file = paper_path.open("w", encoding="utf-8", newline="")
        self._frames = csv.DictWriter(self._frames_file, fieldnames=list(FRAME_FIELDS))
        self._marks = csv.DictWriter(self._marks_file, fieldnames=list(LANDMARK_FIELDS))
        self._paper = csv.DictWriter(self._paper_file, fieldnames=list(PAPER_FIELDS))
        self._frames.writeheader()
        self._marks.writeheader()
        self._paper.writeheader()
        self.frames_path = frames_path
        self.landmarks_path = marks_path
        self.paper_path = paper_path
        self._paper_present: dict[str, int] = defaultdict(int)
        self._wrist_counts: dict[str, int] = defaultdict(int)
        identity_path = folder / "wrist_identity.csv"
        self._identity_file = identity_path.open("w", encoding="utf-8", newline="")
        self._identity = csv.DictWriter(self._identity_file, fieldnames=list(WRIST_IDENTITY_FIELDS))
        self._identity.writeheader()
        self.wrist_identity_path = identity_path

    @classmethod
    def create_session(
        cls,
        root: str = "data/diagnostics",
        participant_height_m: float | None = None,
        participant_mass_kg: float | None = None,
        filter_settings: dict | None = None,
    ) -> "LiveDiagnosticsRecorder":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = resolve_project_path(root) / stamp
        n = 1
        while folder.exists():
            folder = resolve_project_path(root) / f"{stamp}_{n}"
            n += 1
        return cls(
            folder,
            participant_height_m=participant_height_m,
            participant_mass_kg=participant_mass_kg,
            filter_settings=filter_settings,
        )

    def note_dropped(self) -> None:
        if self._closed:
            return
        self.n_dropped += 1

    def write_frame(
        self,
        frame_index: int,
        t_frame_sec: float,
        process_ms: float,
        fps: float,
        observed: list[Keypoint2D],
        predicted: list[dict],
        depth_inspect: dict[str, DepthSampleInspect] | None = None,
        stage_ms: dict | None = None,
    ) -> None:
        """Append one frames.csv row plus one landmarks.csv row per landmark.

        Predicted dict keys: name, x_m, y_m, z_m, class, t_pred_input_sec,
        t_pred_done_sec, infer_latency_ms. Missing XYZ must be None, not 0.
        depth_inspect is optional per observed name; unavailable fields stay empty.
        """
        if self._closed:
            return
        t_unix = time.time()
        t_utc = datetime.fromtimestamp(t_unix, timezone.utc).isoformat()
        n_obs = len(observed)
        n_obs_xyz = 0
        for kp in observed:
            valid = _xyz_ok(kp.x_m, kp.y_m, kp.z_m)
            if valid:
                n_obs_xyz += 1
            xyz = (float(kp.x_m), float(kp.y_m), float(kp.z_m)) if valid else None
            self._class[kp.name] = "observed"
            self._stats[kp.name].add(xyz)
            u_txt, v_txt = _pixel_uv(kp)
            inspect = (depth_inspect or {}).get(kp.name)
            self._marks.writerow(
                {
                    "frame": frame_index,
                    "t_frame_sec": _fmt(t_frame_sec, 6),
                    "t_unix": _fmt(t_unix, 6),
                    "landmark": kp.name,
                    "class": "observed",
                    "raw_x_m": _fmt(xyz[0] if xyz else None),
                    "raw_y_m": _fmt(xyz[1] if xyz else None),
                    "raw_z_m": _fmt(xyz[2] if xyz else None),
                    "filt_x_m": _fmt(kp.filt_x_m),
                    "filt_y_m": _fmt(kp.filt_y_m),
                    "filt_z_m": _fmt(kp.filt_z_m),
                    "visibility": _fmt(kp.confidence, 4),
                    "u_px": u_txt,
                    "v_px": v_txt,
                    "depth_sample_m": _fmt(None if inspect is None else inspect.depth_sample_m),
                    "depth_n_valid_in_window": _fmt_int(None if inspect is None else inspect.n_valid),
                    "in_image": _fmt_flag(None if inspect is None else inspect.in_image),
                    "depth_valid": "1" if valid else "0",
                    "xyz_missing": "0" if valid else "1",
                    "t_pred_input_sec": "",
                    "t_pred_done_sec": "",
                    "infer_latency_ms": "",
                    "pred_lag_ms": "",
                    "paper_name": "",
                    "anatomical_status": "observed",
                }
            )

        pred_input = None
        pred_done = None
        infer_ms = None
        for rec in predicted:
            x, y, z = rec.get("x_m"), rec.get("y_m"), rec.get("z_m")
            valid = _xyz_ok(x, y, z)
            xyz = (float(x), float(y), float(z)) if valid else None
            name = str(rec["name"])
            kind = str(rec.get("class") or "predicted")
            t_in = rec.get("t_pred_input_sec")
            t_done = rec.get("t_pred_done_sec")
            infer = rec.get("infer_latency_ms")
            lag = None
            if t_in is not None and math.isfinite(float(t_in)):
                lag = (t_frame_sec - float(t_in)) * 1000.0
                pred_input = float(t_in) if pred_input is None else pred_input
            if t_done is not None and math.isfinite(float(t_done)):
                pred_done = float(t_done) if pred_done is None else pred_done
            if infer is not None and math.isfinite(float(infer)):
                infer_ms = float(infer) if infer_ms is None else infer_ms
            self._class[name] = kind
            self._stats[name].add(xyz)
            self._marks.writerow(
                {
                    "frame": frame_index,
                    "t_frame_sec": _fmt(t_frame_sec, 6),
                    "t_unix": _fmt(t_unix, 6),
                    "landmark": name,
                    "class": kind,
                    "raw_x_m": _fmt(xyz[0] if xyz else None),
                    "raw_y_m": _fmt(xyz[1] if xyz else None),
                    "raw_z_m": _fmt(xyz[2] if xyz else None),
                    "filt_x_m": "",
                    "filt_y_m": "",
                    "filt_z_m": "",
                    "visibility": _fmt(rec.get("visibility"), 4),
                    "u_px": "",
                    "v_px": "",
                    "depth_sample_m": "",
                    "depth_n_valid_in_window": "",
                    "in_image": "",
                    "depth_valid": "1" if valid else "0",
                    "xyz_missing": "0" if valid else "1",
                    "t_pred_input_sec": _fmt(t_in, 6),
                    "t_pred_done_sec": _fmt(t_done, 6),
                    "infer_latency_ms": _fmt(infer, 3),
                    "pred_lag_ms": _fmt(lag, 3),
                    "paper_name": str(rec.get("paper_name") or ""),
                    "anatomical_status": str(rec.get("anatomical_status") or kind),
                }
            )

        lag_frame = None
        if pred_input is not None:
            lag_frame = (t_frame_sec - pred_input) * 1000.0
        self._frames.writerow(
            {
                "frame": frame_index,
                "t_frame_sec": _fmt(t_frame_sec, 6),
                "t_unix": _fmt(t_unix, 6),
                "t_utc": t_utc,
                "process_ms": _fmt(process_ms, 3),
                "dropped_before": self.n_dropped,
                "n_observed": n_obs,
                "n_observed_xyz": n_obs_xyz,
                "n_predicted": len(predicted),
                "t_pred_input_sec": _fmt(pred_input, 6),
                "t_pred_done_sec": _fmt(pred_done, 6),
                "infer_latency_ms": _fmt(infer_ms, 3),
                "pred_lag_ms": _fmt(lag_frame, 3),
                "fps": _fmt(fps, 3),
                "capture_wait_ms": _fmt(None if stage_ms is None else stage_ms.get("capture_wait"), 3),
                "pose_ms": _fmt(None if stage_ms is None else stage_ms.get("pose"), 3),
                "hands_ms": _fmt(None if stage_ms is None else stage_ms.get("hands"), 3),
                "pair_ms": _fmt(None if stage_ms is None else stage_ms.get("pair"), 3),
                "depth_ms": _fmt(None if stage_ms is None else stage_ms.get("depth"), 3),
                "lstm_prep_ms": _fmt(None if stage_ms is None else stage_ms.get("lstm_prep"), 3),
                "filter_ms": _fmt(None if stage_ms is None else stage_ms.get("filter"), 3),
                "draw_ms": _fmt(None if stage_ms is None else stage_ms.get("draw"), 3),
                "loop_ms": _fmt(None if stage_ms is None else stage_ms.get("loop"), 3),
            }
        )
        if stage_ms:
            for key in ("capture_wait", "pose", "hands", "pair", "depth", "lstm_prep", "filter", "draw", "loop"):
                value = stage_ms.get(key)
                if value is not None and math.isfinite(float(value)):
                    self._stage_ms[key].append(float(value))
        source_xyz: dict[str, tuple[float, float, float]] = {}
        for kp in observed:
            if _xyz_ok(kp.x_m, kp.y_m, kp.z_m):
                source_xyz[kp.name] = (float(kp.x_m), float(kp.y_m), float(kp.z_m))
        for rec in predicted:
            x, y, z = rec.get("x_m"), rec.get("y_m"), rec.get("z_m")
            if _xyz_ok(x, y, z):
                source_xyz[str(rec["name"])] = (float(x), float(y), float(z))
        for row in resolve_paper_landmarks(source_xyz):
            if row["present"]:
                self._paper_present[row["paper_name"]] += 1
            self._paper.writerow(
                {
                    "frame": frame_index,
                    "t_frame_sec": _fmt(t_frame_sec, 6),
                    "paper_name": row["paper_name"],
                    "side": row["side"],
                    "region": row["region"],
                    "method": row["method"],
                    "source_landmark": row["source_landmark"],
                    "raw_x_m": _fmt(row["x_m"]),
                    "raw_y_m": _fmt(row["y_m"]),
                    "raw_z_m": _fmt(row["z_m"]),
                    "present": "1" if row["present"] else "0",
                    "reason": row["reason"],
                }
            )
        self._process_ms.append(float(process_ms))
        self._stage_ms["process"].append(float(process_ms))
        if infer_ms is not None:
            self._stage_ms["lstm_infer"].append(float(infer_ms))
        if lag_frame is not None:
            self._stage_ms["prediction_age"].append(float(lag_frame))
        self.n_frames += 1

    def write_wrist_identity(
        self,
        frame_index: int,
        t_frame_sec: float,
        decision: WristIdentityDecision,
    ) -> None:
        """Record raw Hands detections and the overwrite decision for one frame.

        Accepted landmarks stay in landmarks.csv. This file keeps the raw
        MediaPipe detections, including the ones whose overwrite was rejected.
        """
        if self._closed:
            return
        self._identity.writerow(decision.to_csv_row(frame_index, t_frame_sec))
        self._wrist_counts[decision.classification] += 1

    def close(self) -> dict:
        if self._closed:
            return {"dir": str(self.dir), "frames": self.n_frames, "dropped": self.n_dropped}
        self._closed = True
        try:
            self._frames_file.flush()
            self._marks_file.flush()
            self._paper_file.flush()
            self._identity_file.flush()
        except Exception:
            pass
        try:
            self._frames_file.close()
            self._marks_file.close()
            self._paper_file.close()
            self._identity_file.close()
        except Exception:
            pass
        summary = self._build_summary()
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.dir / "summary.txt").write_text(_summary_text(summary), encoding="utf-8")
        return summary

    def _build_summary(self) -> dict:
        elapsed = max(1e-9, time.time() - self._t_start_unix)
        mean_ms = sum(self._process_ms) / len(self._process_ms) if self._process_ms else None
        landmarks = {}
        for name, stat in sorted(self._stats.items()):
            row = stat.report()
            row["class"] = self._class.get(name, "")
            landmarks[name] = row
        return {
            "coord_frame": "camera_3d_metre",
            "units": "metres",
            "filtered": bool(self.filter_settings.get("enabled")),
            "filter": dict(self.filter_settings),
            "stage_ms": {name: _duration_stats(values) for name, values in sorted(self._stage_ms.items())},
            "t_start_utc": self._t_start_wall,
            "t_end_utc": datetime.now(timezone.utc).isoformat(),
            "dir": str(self.dir),
            "frames": self.n_frames,
            "dropped_frames": self.n_dropped,
            "wall_elapsed_sec": elapsed,
            "processing_fps": (1000.0 / mean_ms) if mean_ms and mean_ms > 0 else None,
            "wall_fps": self.n_frames / elapsed,
            "mean_process_ms": mean_ms,
            "participant_height_m": self.participant_height_m,
            "participant_mass_kg": self.participant_mass_kg,
            "wrist_identity": dict(self._wrist_counts),
            "paper_set": {
                spec.paper_name: {
                    "side": spec.side,
                    "region": spec.region,
                    "method": spec.method,
                    "source_landmark": spec.source_name,
                    "notes": spec.notes,
                    "n_frames_present": int(self._paper_present.get(spec.paper_name, 0)),
                }
                for spec in PAPER_LANDMARKS
            },
            "landmarks": landmarks,
        }


def _summary_text(summary: dict) -> str:
    lines = [
        "Live marker diagnostics",
        f"folder: {summary['dir']}",
        f"units: {summary['units']}  coord_frame: {summary['coord_frame']}",
        (
            "filtered coordinates: recorded separately from raw"
            if summary.get("filtered")
            else "filtered coordinates: filter off"
        ),
        f"frames: {summary['frames']}   dropped: {summary['dropped_frames']}",
        (
            f"participant: {summary['participant_height_m']:.2f} m  "
            f"{summary['participant_mass_kg']:.1f} kg"
            if summary.get("participant_height_m") is not None
            and summary.get("participant_mass_kg") is not None
            else "participant: not set (Live Pose2Sim off or not started)"
        ),
        f"wall FPS: {summary['wall_fps']:.3f}" if summary.get("wall_fps") is not None else "wall FPS: n/a",
        (
            f"processing FPS: {summary['processing_fps']:.3f}  "
            f"(mean process {summary['mean_process_ms']:.2f} ms)"
            if summary.get("processing_fps") is not None
            else "processing FPS: n/a"
        ),
        "",
        "paper set (Bousigues correspondences; LSTM ≠ palpation)",
    ]
    for name, row in (summary.get("paper_set") or {}).items():
        lines.append(
            f"{name:10s}  {row['side']:5s}  {row['method']:12s}  "
            f"src={row['source_landmark'] or '—':18s}  "
            f"present {row['n_frames_present']}"
        )
    lines.extend(
        [
        "",
        "landmark                  class            valid  invalid%   stdX_m    stdY_m    stdZ_m   rangeX_m  f2f_m",
    ]
    )
    for name, row in summary.get("landmarks", {}).items():
        std = row.get("std_m") or {}
        rng = row.get("range_m") or {}
        f2f = row.get("mean_frame_to_frame_m")
        lines.append(
            f"{name:24s}  {str(row.get('class', '')):14s}  "
            f"{row.get('n_valid_xyz', 0):5d}  {row.get('invalid_xyz_pct', 0):7.2f}  "
            f"{_fmt(std.get('x'), 5):>8s}  {_fmt(std.get('y'), 5):>8s}  {_fmt(std.get('z'), 5):>8s}  "
            f"{_fmt(rng.get('x'), 5):>8s}  {_fmt(f2f, 5):>8s}"
        )
    lines.append("")
    if summary.get("filter"):
        lines.append(f"filter settings: {json.dumps(summary['filter'], sort_keys=True)}")
    lines.append("stage durations (ms):")
    for name, row in (summary.get("stage_ms") or {}).items():
        lines.append(
            f"  {name:16s}  n={row['n']}  mean={row['mean']:.2f}  median={row['median']:.2f}  "
            f"p90={row['p90']:.2f}  p95={row['p95']:.2f}"
        )
    lines.append("Missing XYZ is blank, not 0. A rejected depth is not replaced by the previous point.")
    return "\n".join(lines) + "\n"


def _duration_stats(values: list[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}

    def pick(percent: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (percent / 100.0) * (len(ordered) - 1)
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return ordered[low]
        weight = pos - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": pick(50),
        "p90": pick(90),
        "p95": pick(95),
    }
