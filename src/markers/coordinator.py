"""Assemble observed + calibrated + LSTM markers for one live frame."""

from __future__ import annotations

import threading

from src.markers.centres import pair_centres
from src.markers.foot_calib import FootCalibrator
from src.markers.paper_set import annotate_overlay_records
from src.pose.keypoints import Keypoint2D
from src.pose2sim_live.lower_lstm import missing_lower_inputs
from src.pose2sim_live.participant import require_participant_metres_kg


class LiveBodyCoordinator:
    """Non-blocking upper/lower LSTM plus session toe calibration."""

    def __init__(
        self,
        enabled: bool,
        height_m: float | None = None,
        mass_kg: float | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.feet = FootCalibrator()
        self.upper = None
        self.lower = None
        self.upper_error = None
        self.lower_error = None
        self.height_m: float | None = None
        self.mass_kg: float | None = None
        self._lower_starting = False
        self._toes: list[Keypoint2D] = []
        self._missing_lower: tuple[str, ...] = ()
        if self.enabled:
            self.height_m, self.mass_kg = require_participant_metres_kg(height_m, mass_kg)
            try:
                from src.pose2sim_live.engine import LiveUpperLstm

                self.upper = LiveUpperLstm(height_m=self.height_m, mass_kg=self.mass_kg)
            except Exception as exc:
                self.upper_error = str(exc)

    def start_toe_prompt(self) -> str:
        return self.feet.prompt()

    def add_toe_click(self, u, v, xyz, keypoints: list[Keypoint2D]) -> str:
        msg = self.feet.add_click(u, v, xyz, keypoints)
        if self.feet.ready():
            self._boot_lower()
        return msg

    def _boot_lower(self) -> None:
        if not self.enabled or self.lower is not None or self._lower_starting:
            return
        if self.lower_error:
            return
        self._lower_starting = True

        def _run() -> None:
            try:
                from src.pose2sim_live.engine import LiveLowerLstm

                self.lower = LiveLowerLstm(height_m=self.height_m, mass_kg=self.mass_kg)
            except Exception as exc:
                self.lower_error = str(exc)
            finally:
                self._lower_starting = False

        threading.Thread(target=_run, name="pose2sim-lower-boot", daemon=True).start()

    def observe(
        self,
        keypoints: list[Keypoint2D],
        time_sec: float,
        min_visibility: float,
        frame_index: int | None = None,
    ) -> None:
        toes = self.feet.reconstruct(keypoints, min_visibility)
        merged = list(keypoints) + toes
        if self.upper is not None:
            self.upper.observe(merged, time_sec, min_visibility, frame_index)
        if self.feet.ready():
            self._boot_lower()
        if self.lower is not None:
            self.lower.observe(merged, time_sec, min_visibility, frame_index)
        self._toes = toes
        self._missing_lower = missing_lower_inputs(merged, min_visibility)

    def overlay_points(self) -> list[tuple[str, float, float, float]]:
        return [(r["name"], r["x_m"], r["y_m"], r["z_m"]) for r in self.overlay_records() if r.get("x_m") is not None]

    def overlay_records(self) -> list[dict]:
        """Predicted/calibrated points with prediction timestamps. No zero fill."""
        recs: list[dict] = []
        for engine, kind in ((self.upper, "predicted_upper"), (self.lower, "predicted_lower")):
            if engine is None:
                continue
            snap = engine.snapshot()
            for name, x, y, z in engine.latest_camera_xyz():
                recs.append(
                    {
                        "name": name,
                        "x_m": x,
                        "y_m": y,
                        "z_m": z,
                        "class": kind,
                        "visibility": 1.0,
                        "t_pred_input_sec": snap.get("input_time_sec"),
                        "source_frame": snap.get("source_frame"),
                        "t_pred_done_perf": snap.get("result_perf"),
                        "infer_latency_ms": snap.get("latency_ms"),
                    }
                )
        centres = pair_centres(
            [(r["name"], r["x_m"], r["y_m"], r["z_m"]) for r in recs if r.get("x_m") is not None]
        )
        parent_t = recs[0] if recs else {}
        for name, x, y, z in centres:
            recs.append(
                {
                    "name": name,
                    "x_m": x,
                    "y_m": y,
                    "z_m": z,
                    "class": "geometric_centre",
                    "visibility": 1.0,
                    "t_pred_input_sec": parent_t.get("t_pred_input_sec"),
                    "source_frame": parent_t.get("source_frame"),
                    "t_pred_done_perf": parent_t.get("t_pred_done_perf"),
                    "infer_latency_ms": parent_t.get("infer_latency_ms"),
                }
            )
        for kp in getattr(self, "_toes", []):
            recs.append(
                {
                    "name": kp.name,
                    "x_m": kp.x_m,
                    "y_m": kp.y_m,
                    "z_m": kp.z_m,
                    "class": "calibrated_toe",
                    "visibility": kp.confidence,
                    "t_pred_input_sec": None,
                    "t_pred_done_perf": None,
                    "infer_latency_ms": None,
                }
            )
        return annotate_overlay_records(recs)

    def banners(self) -> list[str]:
        lines = []
        if not self.enabled:
            lines.append("Live Pose2Sim unchecked")
            return lines
        if self.height_m is not None and self.mass_kg is not None:
            lines.append(f"Participant  {self.height_m:.2f} m   {self.mass_kg:.1f} kg")
        if self.upper_error:
            lines.append(f"UPPER unavailable: {self.upper_error}")
        elif self.upper is None:
            lines.append("UPPER not started")
        else:
            snap = self.upper.snapshot()
            lines.append(_status_line("UPPER", snap, 8))
        if self.lower_error:
            lines.append(f"LOWER unavailable: {self.lower_error}")
        elif self._lower_starting:
            lines.append("LOWER warming up (loading ONNX)")
        elif self.lower is None:
            miss = getattr(self, "_missing_lower", ("LBigToe", "LSmallToe", "RBigToe", "RSmallToe"))
            toes = [n for n in miss if "Toe" in n]
            extra = [n for n in miss if "Toe" not in n]
            toe_txt = ",".join(toes) if toes else "toes ok"
            extra_txt = f"  also missing {','.join(extra)}" if extra else ""
            lines.append(f"LOWER missing input: {toe_txt}{extra_txt}  ·  click Calibrate toes")
        else:
            snap = self.lower.snapshot()
            lines.append(_status_line("LOWER", snap, 35))
        lines.append(self.feet.prompt())
        return lines

    def close(self) -> None:
        for engine in (self.upper, self.lower):
            if engine is None:
                continue
            try:
                engine.close()
            except Exception:
                pass


def _status_line(label: str, snap: dict, expected: int) -> str:
    status = snap.get("status") or "unknown"
    if status == "buffering":
        return f"{label} buffering {snap.get('buffered')}/{snap.get('min_frames')}"
    if status == "live":
        lat = snap.get("latency_ms")
        lat_s = f"{lat:.0f} ms" if lat is not None else "…"
        return f"{label} LIVE T={snap.get('T')} {lat_s}  {snap.get('n_estimates')}/{expected} est"
    return f"{label} {status} {snap.get('error') or ''}"
