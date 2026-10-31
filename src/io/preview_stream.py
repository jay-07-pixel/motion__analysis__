"""Localhost MJPEG preview of colour frames the caller already has.

This module never opens a camera. pose_game_bridge.py hands it the same BGR
frames it sends to MediaPipe, from its one LiveRealSenseRGB pipeline.

    GET /video     multipart/x-mixed-replace MJPEG (for <img src>)
    GET /snapshot  latest JPEG, 503 until the first frame
    GET /health    JSON {"frames": n, "width": w, "height": h}

Frames are downscaled and JPEG-encoded on a separate thread, at most max_fps,
so the camera / MediaPipe loop only pays for a frame copy.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# Localhost only. PatientBalanceAI PythonPoseBridge.ts uses this same port.
PREVIEW_HOST = "127.0.0.1"
PREVIEW_PORT = 8766
_BOUNDARY = "posebridgeframe"


def encode_preview_jpeg(frame_bgr: np.ndarray, max_width: int = 640, quality: int = 70) -> bytes | None:
    """Downscale to max_width (keeping aspect) and JPEG-encode. None if encoding fails."""
    height, width = frame_bgr.shape[:2]
    if width > max_width:
        scaled_height = max(1, round(height * max_width / width))
        frame_bgr = cv2.resize(frame_bgr, (max_width, scaled_height), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else None


class PreviewStream:
    """Latest-frame MJPEG server. submit() never blocks on encoding or clients."""

    def __init__(
        self,
        host: str = PREVIEW_HOST,
        port: int = PREVIEW_PORT,
        max_width: int = 640,
        quality: int = 70,
        max_fps: float = 15.0,
    ) -> None:
        self.host = host
        self._requested_port = port
        self.max_width = max_width
        self.quality = quality
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._cond = threading.Condition()
        self._pending: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._seq = 0
        self._size: tuple[int, int] | None = None
        self._last_accept = 0.0
        self._stopped = True
        self._server: ThreadingHTTPServer | None = None
        self._threads: list[threading.Thread] = []

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self._requested_port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/video"

    @property
    def frames_encoded(self) -> int:
        return self._seq

    def start(self) -> None:
        """Bind the HTTP server and start the encoder. Raises OSError if the port is taken."""
        stream = self

        class Handler(_PreviewHandler):
            preview = stream

        self._server = ThreadingHTTPServer((self.host, self._requested_port), Handler)
        self._server.daemon_threads = True
        self._stopped = False
        self._threads = [
            threading.Thread(target=self._server.serve_forever, name="preview-http", daemon=True),
            threading.Thread(target=self._encode_loop, name="preview-encode", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, frame_bgr: np.ndarray) -> bool:
        """Offer a frame. Copied only if the preview rate allows. Returns True if kept."""
        if self._stopped:
            return False
        now = time.perf_counter()
        if now - self._last_accept < self._min_interval:
            return False
        self._last_accept = now
        copy = np.array(frame_bgr, copy=True)
        with self._cond:
            self._pending = copy
            self._cond.notify_all()
        return True

    def wait_for_jpeg(self, after_seq: int, timeout: float) -> tuple[int, bytes] | None:
        """Block until a JPEG newer than after_seq exists, the stream stops, or timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._stopped and (self._jpeg is None or self._seq <= after_seq):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._stopped or self._jpeg is None:
                return None
            return self._seq, self._jpeg

    def latest(self) -> tuple[int, bytes | None, tuple[int, int] | None]:
        with self._cond:
            return self._seq, self._jpeg, self._size

    @property
    def stopped(self) -> bool:
        return self._stopped

    def stop(self) -> None:
        """Wake every client, close the socket, and join the threads."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        for thread in self._threads:
            thread.join(timeout=3)
        self._threads = []

    def _encode_loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                frame, self._pending = self._pending, None
            try:
                jpeg = encode_preview_jpeg(frame, self.max_width, self.quality)
            except Exception as error:
                print(f"Preview encode failed: {error}")
                continue
            if jpeg is None:
                continue
            with self._cond:
                self._jpeg = jpeg
                self._seq += 1
                self._size = (int(frame.shape[1]), int(frame.shape[0]))
                self._cond.notify_all()


class _PreviewHandler(BaseHTTPRequestHandler):
    preview: PreviewStream

    def log_message(self, *_args) -> None:
        pass

    def _common_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/video":
            self._video()
        elif path == "/snapshot":
            self._snapshot()
        elif path == "/health":
            seq, _jpeg, size = self.preview.latest()
            body = json.dumps(
                {"frames": seq, "width": size[0] if size else None, "height": size[1] if size else None}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._common_headers()
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def _snapshot(self) -> None:
        _seq, jpeg, _size = self.preview.latest()
        if jpeg is None:
            self.send_error(503, "No camera frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self._common_headers()
        self.end_headers()
        self.wfile.write(jpeg)

    def _video(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self._common_headers()
        self.end_headers()
        seq = 0
        try:
            while not self.preview.stopped:
                item = self.preview.wait_for_jpeg(seq, timeout=1.0)
                if item is None:
                    continue
                seq, jpeg = item
                self.wfile.write(
                    f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return
