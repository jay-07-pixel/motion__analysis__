"""ONNX worker for Pose2Sim v0.3 lower LSTM. No camera."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL = Path(
    r"C:\Users\jayjo\anaconda3\envs\pose2sim-offline\Lib\site-packages"
    r"\Pose2Sim\MarkerAugmenter\LSTM\v0.3_lower\model.onnx"
)
N_FEAT = 47


def _emit(payload: dict) -> None:
    sys.stdout.buffer.write((json.dumps(payload) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def _read_request() -> dict | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    request = json.loads(line.decode("utf-8"))
    if request.get("cmd") == "predict_bin":
        count = int(request["T"]) * N_FEAT
        blob = sys.stdin.buffer.read(count * 4)
        if len(blob) != count * 4:
            request["error"] = f"short binary read: {len(blob)} bytes"
            request["inputs"] = None
            return request
        request["inputs"] = np.frombuffer(blob, dtype=np.float32).reshape(int(request["T"]), N_FEAT)
    return request


def main() -> None:
    try:
        session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    except Exception as exc:
        _emit({"ok": False, "error": str(exc)})
        raise SystemExit(1)
    _emit({"ok": True, "event": "ready", "model": str(MODEL)})
    while True:
        try:
            request = _read_request()
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"bad json: {exc}"})
            continue
        if request is None:
            return
        if request.get("cmd") == "quit":
            _emit({"ok": True, "event": "bye"})
            return
        if request.get("inputs") is None:
            _emit({"ok": False, "error": request.get("error") or "missing inputs"})
            continue
        features = np.asarray(request.get("inputs"), dtype=np.float32)
        if features.ndim != 2 or features.shape[-1] != N_FEAT:
            _emit({"ok": False, "error": f"inputs must be (T, {N_FEAT}), got {list(features.shape)}"})
            continue
        if not np.all(np.isfinite(features)):
            _emit({"ok": False, "error": "non-finite inputs"})
            continue
        try:
            raw = session.run(["output_0"], {"inputs": features.reshape(1, features.shape[0], N_FEAT)})[0]
        except Exception as exc:
            _emit({"ok": False, "error": str(exc)})
            continue
        last = np.reshape(raw, (raw.shape[1], raw.shape[2]))[-1]
        _emit({"ok": True, "raw_last": last.astype(float).tolist(), "T": int(features.shape[0])})


if __name__ == "__main__":
    main()
