"""Load the arm LSTM and answer one-sequence predict requests on stdin.

Used by tools/live_marker_enhancer.py when this interpreter has TensorFlow
and the camera process does not. Not a second camera, not a new model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import marker_enhancer_offline_inference as me  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Usage: predict_worker.py <model_dir>"}), flush=True)
        raise SystemExit(1)
    model_dir = Path(sys.argv[1])
    try:
        tf = me._import_tensorflow()
        model = me.load_arm_model(model_dir, tf)
        in_shape = list(model.input_shape)
        out_shape = list(model.output_shape)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "ok": True,
                "event": "ready",
                "tensorflow": tf.__version__,
                "input_shape": in_shape,
                "output_shape": out_shape,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"bad json: {exc}"}), flush=True)
            continue
        if request.get("cmd") == "quit":
            print(json.dumps({"ok": True, "event": "bye"}), flush=True)
            return
        features = np.asarray(request.get("features"), dtype=np.float32)
        if features.ndim != 2 or features.shape[-1] != 23:
            print(
                json.dumps({"ok": False, "error": f"features must be (T, 23), got {list(features.shape)}"}),
                flush=True,
            )
            continue
        if not np.all(np.isfinite(features)):
            print(json.dumps({"ok": False, "error": "non-finite features"}), flush=True)
            continue
        try:
            raw = model.predict(features.reshape(1, features.shape[0], 23), verbose=0)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
            continue
        if raw.ndim == 3:
            raw = raw[0]
        last = raw[-1]
        if last.shape != (24,) or not np.all(np.isfinite(last)):
            print(json.dumps({"ok": False, "error": f"bad last-step shape {list(last.shape)}"}), flush=True)
            continue
        print(json.dumps({"ok": True, "raw_last": last.astype(float).tolist()}), flush=True)


if __name__ == "__main__":
    main()
