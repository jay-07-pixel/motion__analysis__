"""MJPEG preview tests. No camera: frames are synthetic NumPy arrays."""

from __future__ import annotations

import ast
import http.client
import json
import socket
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import pose_game_bridge
from src.io.preview_stream import PREVIEW_HOST, PREVIEW_PORT, PreviewStream, encode_preview_jpeg

ROOT = Path(__file__).resolve().parents[1]


def frame(width: int = 1280, height: int = 720) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (0, 0, 255)  # left half red in BGR
    return image


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class EncodeTest(unittest.TestCase):
    def test_frame_is_jpeg_downscaled_with_aspect(self) -> None:
        jpeg = encode_preview_jpeg(frame(), max_width=640, quality=70)
        self.assertEqual(jpeg[:2], b"\xff\xd8")
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (360, 640))
        self.assertLess(len(jpeg), 1280 * 720 * 3 // 20)

    def test_orientation_is_not_mirrored(self) -> None:
        jpeg = encode_preview_jpeg(frame(), max_width=640)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(decoded[180, 50, 2], 200)
        self.assertLess(decoded[180, 600, 2], 50)

    def test_small_frame_is_not_upscaled(self) -> None:
        decoded = cv2.imdecode(np.frombuffer(encode_preview_jpeg(frame(320, 180)), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (180, 320))


class StreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = PreviewStream(port=0, max_fps=0)
        self.stream.start()
        self.base = f"http://{PREVIEW_HOST}:{self.stream.port}"

    def tearDown(self) -> None:
        self.stream.stop()

    def test_defaults_are_localhost_8766(self) -> None:
        self.assertEqual((PREVIEW_HOST, PREVIEW_PORT), ("127.0.0.1", 8766))
        self.assertNotEqual(PREVIEW_PORT, pose_game_bridge.POSE_BRIDGE_PORT)

    def test_snapshot_503_before_first_frame(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/snapshot", timeout=2)
        self.assertEqual(ctx.exception.code, 503)

    def test_snapshot_and_health_after_submit(self) -> None:
        self.assertTrue(self.stream.submit(frame()))
        self.assertTrue(wait_until(lambda: self.stream.frames_encoded >= 1))
        with urllib.request.urlopen(self.base + "/snapshot", timeout=2) as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertEqual(response.read(2), b"\xff\xd8")
        with urllib.request.urlopen(self.base + "/health", timeout=2) as response:
            health = json.loads(response.read())
        self.assertEqual((health["width"], health["height"]), (1280, 720))

    def test_video_is_multipart_mjpeg(self) -> None:
        conn = http.client.HTTPConnection(PREVIEW_HOST, self.stream.port, timeout=3)
        conn.request("GET", "/video")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn("multipart/x-mixed-replace", response.headers["Content-Type"])
        feeder = threading.Thread(target=lambda: [self.stream.submit(frame()) or time.sleep(0.05) for _ in range(10)])
        feeder.start()
        head = response.read(200)
        feeder.join()
        self.assertIn(b"Content-Type: image/jpeg", head)
        self.assertIn(b"\xff\xd8", head)
        conn.close()

    def test_unknown_path_is_404(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/nope", timeout=2)
        self.assertEqual(ctx.exception.code, 404)


class StartStopTest(unittest.TestCase):
    def test_stop_with_a_viewer_connected_releases_the_port(self) -> None:
        stream = PreviewStream(port=0, max_fps=0)
        stream.start()
        port = stream.port
        stream.submit(frame())
        conn = http.client.HTTPConnection(PREVIEW_HOST, port, timeout=3)
        conn.request("GET", "/video")
        conn.getresponse().read(100)
        started = time.monotonic()
        stream.stop()
        self.assertLess(time.monotonic() - started, 3.5)
        self.assertFalse(any(t.name.startswith("preview-") and t.is_alive() for t in threading.enumerate()))
        with self.assertRaises(OSError):
            socket.create_connection((PREVIEW_HOST, port), timeout=0.5).close()
        self.assertFalse(stream.submit(frame()), "submit after stop is ignored")
        conn.close()

    def test_rate_limit_skips_frames(self) -> None:
        stream = PreviewStream(port=0, max_fps=5)
        stream.start()
        try:
            kept = sum(stream.submit(frame()) for _ in range(20))
            self.assertEqual(kept, 1)
        finally:
            stream.stop()

    def test_submit_copies_the_frame(self) -> None:
        stream = PreviewStream(port=0, max_fps=0)
        stream.start()
        try:
            image = frame()
            stream.submit(image)
            image[:] = 0
            self.assertTrue(wait_until(lambda: stream.frames_encoded >= 1))
            _seq, jpeg, _size = stream.latest()
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(decoded[180, 50, 2], 200, "camera buffer reuse cannot change the preview")
        finally:
            stream.stop()


class OnePipelineTest(unittest.TestCase):
    def test_preview_module_opens_no_camera(self) -> None:
        source = (ROOT / "src" / "io" / "preview_stream.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        for banned in ("pyrealsense2", "src.capture.live_realsense", "LiveRealSenseRGB", "mediapipe"):
            self.assertNotIn(banned, imported)
        self.assertNotIn("VideoCapture", source)

    def test_bridge_creates_exactly_one_camera(self) -> None:
        source = (ROOT / "pose_game_bridge.py").read_text(encoding="utf-8")
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LiveRealSenseRGB"
        ]
        self.assertEqual(len(calls), 1)

    def test_preview_gets_the_frames_mediapipe_used(self) -> None:
        frames = [frame(), frame(), frame()]
        keep_alive = list(frames)
        stop = threading.Event()

        class Camera:
            def get_frame(self):
                if not frames:
                    stop.set()
                    return None
                return frames.pop(0)

        seen_by_mediapipe: list[int] = []
        seen_by_preview: list[int] = []

        class Extractor:
            def extract(self, image):
                seen_by_mediapipe.append(id(image))
                return []

        class Preview:
            def submit(self, image):
                seen_by_preview.append(id(image))
                return True

        class Server:
            async def publish(self, _payload):
                return None

        class Loop:
            pass

        original = pose_game_bridge.asyncio.run_coroutine_threadsafe

        def fake_run(coro, _loop):
            coro.close()

        pose_game_bridge.asyncio.run_coroutine_threadsafe = fake_run
        try:
            pose_game_bridge._pump_camera(Camera(), Extractor(), Server(), Loop(), stop, 0.5, lambda: None, Preview())
        finally:
            pose_game_bridge.asyncio.run_coroutine_threadsafe = original
        self.assertEqual(len(seen_by_mediapipe), 3)
        self.assertEqual(seen_by_preview, seen_by_mediapipe)


if __name__ == "__main__":
    unittest.main()
