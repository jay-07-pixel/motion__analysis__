"""Pose2Sim v0.3 upper live preprocess, buffer, and optional ONNX worker."""

from __future__ import annotations

import json
import subprocess
import time
import unittest
from pathlib import Path

import numpy as np

from src.pose.keypoints import Keypoint2D
from src.pose2sim_live.upper_lstm import (
    DEFAULT_MAX_GAP,
    FEATURE_MARKERS,
    HEIGHT_M,
    ISOLATED_PYTHON,
    MASS_KG,
    MODEL_DIR,
    N_INPUT,
    N_OUTPUT,
    RESPONSE_MARKERS,
    UpperSequenceBuffer,
    camera_to_yup,
    denorm_outputs,
    extract_upper_frame,
    load_norm_stats,
    normalize_sequence,
    yup_to_camera,
)

WORKER = Path(__file__).resolve().parents[1] / "tools" / "pose2sim_upper_predict_worker.py"


def body(**xyz) -> list[Keypoint2D]:
    names = {
        "left_shoulder": (0.2, 0.1, 1.0),
        "right_shoulder": (-0.2, 0.1, 1.0),
        "left_elbow": (0.25, 0.3, 1.05),
        "right_elbow": (-0.25, 0.3, 1.05),
        "left_wrist": (0.28, 0.5, 1.02),
        "right_wrist": (-0.28, 0.5, 1.02),
        "left_hip": (0.1, 0.6, 1.1),
        "right_hip": (-0.1, 0.6, 1.1),
    }
    names.update(xyz)
    return [
        Keypoint2D(name=n, u_px=100, v_px=100, confidence=0.9, x_m=p[0], y_m=p[1], z_m=p[2])
        for n, p in names.items()
    ]


class PreprocessTest(unittest.TestCase):
    def test_feature_order_and_shape(self) -> None:
        extracted = extract_upper_frame(body(), 0.5)
        self.assertIsNotNone(extracted)
        feat, hip = extracted
        self.assertEqual(feat.shape, (21,))
        self.assertEqual(hip.shape, (3,))
        ls = camera_to_yup(0.2, 0.1, 1.0)
        rs = camera_to_yup(-0.2, 0.1, 1.0)
        np.testing.assert_allclose(feat[0:3], 0.5 * (ls + rs))
        np.testing.assert_allclose(feat[3:6], rs)

    def test_missing_depth_is_not_zero_filled(self) -> None:
        pts = body()
        pts[2] = Keypoint2D("left_elbow", 100, 100, 0.9)  # no xyz
        self.assertIsNone(extract_upper_frame(pts, 0.5))

    def test_low_confidence_rejected(self) -> None:
        pts = body()
        pts[0] = Keypoint2D("left_shoulder", 100, 100, 0.2, x_m=0.2, y_m=0.1, z_m=1.0)
        self.assertIsNone(extract_upper_frame(pts, 0.5))

    def test_normalize_matches_official_steps(self) -> None:
        T = 4
        feat = np.arange(T * 21, dtype=np.float64).reshape(T, 21)
        hip = np.array([[1.0, 2.0, 3.0]] * T)
        mean = np.zeros(23)
        std = np.ones(23)
        out = normalize_sequence(feat, hip, 1.8, 61.0, mean, std)
        self.assertEqual(out.shape, (T, N_INPUT))
        centered = feat - np.tile(hip, (1, 7))
        np.testing.assert_allclose(out[:, :21], centered / 1.8)
        np.testing.assert_allclose(out[:, 21], 1.8)
        np.testing.assert_allclose(out[:, 22], 61.0)

    def test_denorm_is_inverse_of_height_and_hip(self) -> None:
        raw = np.ones(24)
        hip = np.array([0.1, -0.2, 1.0])
        out = denorm_outputs(raw, hip, 1.8)
        self.assertEqual(out.shape, (1, N_OUTPUT))
        np.testing.assert_allclose(out[0, 0:3], 1.8 + hip)

    def test_yup_roundtrip(self) -> None:
        x, y, z = yup_to_camera(*camera_to_yup(0.2, 0.4, 1.1))
        self.assertAlmostEqual(x, 0.2)
        self.assertAlmostEqual(y, 0.4)
        self.assertAlmostEqual(z, 1.1)


class BufferTest(unittest.TestCase):
    def test_gap_clears_instead_of_interpolating(self) -> None:
        buf = UpperSequenceBuffer(window=10, min_frames=3, max_gap=2)
        frame = extract_upper_frame(body(), 0.5)
        self.assertEqual(buf.push(frame, 0.0), "buffering")
        self.assertEqual(buf.push(frame, 0.1), "buffering")
        self.assertEqual(buf.push(frame, 0.2), "ready")
        self.assertEqual(buf.push(None, 0.3), "buffering")
        self.assertEqual(len(buf.frames), 3)
        buf.push(None, 0.4)
        buf.push(None, 0.5)
        self.assertEqual(len(buf.frames), 0)

    def test_a_few_missing_frames_do_not_erase_the_window(self) -> None:
        buf = UpperSequenceBuffer(window=10, min_frames=3, max_gap=DEFAULT_MAX_GAP)
        frame = extract_upper_frame(body(), 0.5)
        for stamp in (0.0, 0.1, 0.2):
            buf.push(frame, stamp)
        for stamp in (0.3, 0.4, 0.5):
            self.assertEqual(buf.push(None, stamp), "buffering")
        self.assertEqual(len(buf.frames), 3)
        self.assertEqual(buf.push(frame, 0.6), "ready")


class WorkerTest(unittest.TestCase):
    def test_onnx_last_step_shape(self) -> None:
        if not ISOLATED_PYTHON.is_file() or not (MODEL_DIR / "model.onnx").is_file():
            self.skipTest("pose2sim-offline not installed")
        mean, std = load_norm_stats(MODEL_DIR)
        extracted = extract_upper_frame(body(), 0.5)
        feat = np.stack([extracted[0]] * 16, axis=0)
        hip = np.stack([extracted[1]] * 16, axis=0)
        inputs = normalize_sequence(feat, hip, HEIGHT_M, MASS_KG, mean, std)
        proc = subprocess.Popen(
            [str(ISOLATED_PYTHON), str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        ready = json.loads(proc.stdout.readline())
        self.assertTrue(ready.get("ok"), ready)
        proc.stdin.write(json.dumps({"cmd": "predict", "inputs": inputs.tolist()}) + "\n")
        proc.stdin.flush()
        reply = json.loads(proc.stdout.readline())
        proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        proc.stdin.flush()
        proc.communicate(timeout=5)
        self.assertTrue(reply.get("ok"), reply)
        self.assertEqual(len(reply["raw_last"]), 24)
        self.assertEqual(reply["T"], 16)
        self.assertEqual(len(RESPONSE_MARKERS), 8)
        self.assertEqual(len(FEATURE_MARKERS), 7)


class LiveEngineTest(unittest.TestCase):
    def test_live_engine_returns_eight_camera_points(self) -> None:
        if not ISOLATED_PYTHON.is_file() or not (MODEL_DIR / "model.onnx").is_file():
            self.skipTest("pose2sim-offline not installed")
        from src.pose2sim_live.engine import LiveUpperLstm

        engine = LiveUpperLstm(height_m=HEIGHT_M, mass_kg=MASS_KG)
        try:
            for i in range(12):
                engine.observe(body(), i * 0.033, 0.5)
            deadline = time.perf_counter() + 8.0
            points = []
            while time.perf_counter() < deadline:
                points = engine.latest_camera_xyz()
                if len(points) == 8:
                    break
                time.sleep(0.05)
            self.assertEqual(len(points), 8, engine.snapshot())
            self.assertEqual([p[0] for p in points], list(RESPONSE_MARKERS))
            for name, x, y, z in points:
                self.assertTrue(np.isfinite(x) and np.isfinite(y) and np.isfinite(z), name)
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
