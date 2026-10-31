"""Background Pose2Sim LSTM inference on the live keypoint stream.

Uses pose2sim-offline ONNX in a subprocess. Does not open a camera.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.pose2sim_live.lstm_common import (
    denorm_lstm_outputs,
    load_norm_stats,
    normalize_lstm_sequence,
    yup_to_camera,
)
from src.pose2sim_live.participant import require_participant_metres_kg
from src.pose2sim_live.upper_lstm import (
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_FRAMES,
    ISOLATED_PYTHON,
    LIVE_WINDOW,
    MODEL_DIR as UPPER_MODEL_DIR,
    N_INPUT as UPPER_N_INPUT,
    N_OUTPUT as UPPER_N_OUTPUT,
    RESPONSE_MARKERS as UPPER_RESPONSE,
    UpperSequenceBuffer,
    extract_upper_frame,
)

WORKER_UPPER = Path(__file__).resolve().parents[2] / "tools" / "pose2sim_upper_predict_worker.py"
WORKER_LOWER = Path(__file__).resolve().parents[2] / "tools" / "pose2sim_lower_predict_worker.py"


class LiveOnnxLstm:
    """Observe live keypoints; expose last causal LSTM step in camera metres."""

    def __init__(
        self,
        *,
        model_dir: Path,
        worker: Path,
        n_input: int,
        n_output: int,
        response_markers: tuple[str, ...],
        extract_fn: Callable,
        height_m: float,
        mass_kg: float,
        window: int = LIVE_WINDOW,
        min_frames: int = DEFAULT_MIN_FRAMES,
        thread_name: str = "pose2sim-lstm",
    ) -> None:
        self.height_m, self.mass_kg = require_participant_metres_kg(height_m, mass_kg)
        self.n_output = int(n_output)
        self.response_markers = tuple(response_markers)
        self.extract_fn = extract_fn
        self.buffer = UpperSequenceBuffer(window=window, min_frames=min_frames, max_gap=DEFAULT_MAX_GAP)
        self.status = "buffering"
        self.latency_ms: float | None = None
        self.input_time_sec: float | None = None
        self.result_perf: float | None = None
        self.last_T = 0
        self.error: str | None = None
        self.missing: tuple[str, ...] = ()
        self._mean, self._std = load_norm_stats(model_dir, n_input)
        self._lock = threading.Lock()
        self._pending = False
        self._latest: list[tuple[str, float, float, float]] = []
        if not ISOLATED_PYTHON.is_file():
            raise FileNotFoundError(f"pose2sim-offline python not found: {ISOLATED_PYTHON}")
        if not worker.is_file():
            raise FileNotFoundError(f"Pose2Sim worker not found: {worker}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [str(ISOLATED_PYTHON), "-u", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=env,
        )
        ready = json.loads((self._proc.stdout.readline() or b"{}").decode("utf-8"))
        if not ready.get("ok"):
            raise RuntimeError(ready.get("error") or "Pose2Sim worker failed to start")
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=thread_name, daemon=True)
        self._thread.start()

    def observe(self, keypoints, time_sec: float, min_visibility: float) -> None:
        extracted = self.extract_fn(keypoints, min_visibility)
        with self._lock:
            state = self.buffer.push(extracted, time_sec)
            if state == "buffering":
                self.status = "buffering"
                if not self.buffer.frames:
                    self._latest = []
                return
            self._pending = True
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.05)
            self._wake.clear()
            while not self._stop.is_set():
                with self._lock:
                    if not self._pending or len(self.buffer.frames) < self.buffer.min_frames:
                        self._pending = False
                        break
                    self._pending = False
                    feat, hip = self.buffer.arrays()
                    feat, hip = feat.copy(), hip.copy()
                    last_t = float(self.buffer.frames[-1].time_sec) if self.buffer.frames else None
                self._infer(feat, hip, time.perf_counter(), last_t)

    def _infer(self, feat: np.ndarray, hip: np.ndarray, t0: float, input_time_sec: float | None) -> None:
        try:
            inputs = normalize_lstm_sequence(
                feat, hip, self.height_m, self.mass_kg, self._mean, self._std
            )
            raw_in = np.ascontiguousarray(inputs, dtype=np.float32)
            header = (json.dumps({"cmd": "predict_bin", "T": int(raw_in.shape[0])}) + "\n").encode("utf-8")
            self._proc.stdin.write(header)
            self._proc.stdin.write(raw_in.tobytes())
            self._proc.stdin.flush()
            reply = json.loads((self._proc.stdout.readline() or b"{}").decode("utf-8"))
            if not reply.get("ok"):
                with self._lock:
                    self.error = str(reply.get("error"))
                    self.status = "error"
                return
            raw = np.asarray(reply["raw_last"], dtype=np.float64)
            n_mark = self.n_output // 3
            yup = denorm_lstm_outputs(raw, hip[-1], self.height_m, self.n_output)[0].reshape(n_mark, 3)
            points = []
            for i, name in enumerate(self.response_markers):
                x, y, z = yup_to_camera(float(yup[i, 0]), float(yup[i, 1]), float(yup[i, 2]))
                points.append((name, x, y, z))
            latency = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._latest = points
                self.latency_ms = latency
                self.input_time_sec = input_time_sec
                self.result_perf = time.perf_counter()
                self.last_T = int(feat.shape[0])
                self.status = "live"
                self.error = None
        except Exception as exc:
            with self._lock:
                self.error = repr(exc)
                self.status = "error"

    def latest_camera_xyz(self) -> list[tuple[str, float, float, float]]:
        with self._lock:
            return list(self._latest)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "latency_ms": self.latency_ms,
                "T": self.last_T,
                "buffered": len(self.buffer.frames),
                "window": self.buffer.window,
                "min_frames": self.buffer.min_frames,
                "n_estimates": len(self._latest),
                "error": self.error,
                "height_m": self.height_m,
                "mass_kg": self.mass_kg,
                "causal": True,
                "input_time_sec": self.input_time_sec,
                "result_perf": self.result_perf,
            }

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        try:
            if self._proc.stdin:
                self._proc.stdin.write(b'{"cmd":"quit"}\n')
                self._proc.stdin.flush()
        except Exception:
            pass
        try:
            self._proc.kill()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=2)
        except Exception:
            pass
        try:
            self._thread.join(timeout=1)
        except Exception:
            pass


class LiveUpperLstm(LiveOnnxLstm):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model_dir=UPPER_MODEL_DIR,
            worker=WORKER_UPPER,
            n_input=UPPER_N_INPUT,
            n_output=UPPER_N_OUTPUT,
            response_markers=UPPER_RESPONSE,
            extract_fn=extract_upper_frame,
            thread_name="pose2sim-upper",
            **kwargs,
        )


class LiveLowerLstm(LiveOnnxLstm):
    def __init__(self, **kwargs) -> None:
        from src.pose2sim_live.lower_lstm import MODEL_DIR, N_INPUT, N_OUTPUT, RESPONSE_MARKERS, extract_lower_frame

        super().__init__(
            model_dir=MODEL_DIR,
            worker=WORKER_LOWER,
            n_input=N_INPUT,
            n_output=N_OUTPUT,
            response_markers=RESPONSE_MARKERS,
            extract_fn=extract_lower_frame,
            thread_name="pose2sim-lower",
            **kwargs,
        )
