"""Send 2D pose from the D455f to PatientBalanceAI.

Python owns the camera. The game connects as a WebSocket client and must
not open the RealSense itself.

    python pose_game_bridge.py

Listens on ws://127.0.0.1:8765. One JSON object per processed colour frame.
Serves an MJPEG preview of the same colour frames on http://127.0.0.1:8766/video.
Both come from the one LiveRealSenseRGB pipeline below; there is no second camera.
MediaPipe Hands (config pose.hands) runs on the same frame for the 2D wrist bend.
Depth stays off. This does not start the 3D session or Marker Enhancer.

Close the live 3D app and RealSense Viewer first. They also open the D455f.
Press Ctrl+C to release the camera. A game disconnect does not stop this process.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.metrics_filter_2d import GameMetricsFilter, filter_from_config
from src.analysis.pose_metrics_2d import game_metrics_2d, tracking_status
from src.analysis.wrist_metrics_2d import wrist_metrics_2d
from src.capture.live_realsense import LiveRealSenseRGB
from src.io.preview_stream import PREVIEW_HOST, PREVIEW_PORT, PreviewStream
from src.pose.extractor_2d import PoseExtractor2D
from src.pose.face_mesh import FaceMeshExtractor2D, serialize_face_mesh_2d
from src.pose.hands import HandsExtractor2D, serialize_hands_2d
from src.pose.keypoints import Keypoint2D
from src.pose.skeleton import LANDMARK_NAMES
from src.utils.config_loader import load_config

# Localhost only. PatientBalanceAI PythonPoseBridge.ts uses this same port.
POSE_BRIDGE_HOST = "127.0.0.1"
POSE_BRIDGE_PORT = 8765


def build_pose_message(
    frame_index: int,
    width: int,
    height: int,
    keypoints: list[Keypoint2D],
    min_confidence: float,
    timestamp_ms: int | None = None,
    hand_keypoints: list[Keypoint2D] | None = None,
    face_keypoints: list[Keypoint2D] | None = None,
    metrics_filter: GameMetricsFilter | None = None,
    include_raw_metrics: bool = False,
) -> dict:
    """JSON body for one frame. Pixels stay pixels. The game normalizes them.

    metrics are the authoritative 2D angles for the game, from pose_metrics_2d,
    plus the 2D wrist bend from wrist_metrics_2d (needs hand_keypoints), then
    GameMetricsFilter when the pump passes one. A metric is None (JSON null)
    when its joints are missing or below the gate. Null is never sent as 0.
    include_raw_metrics adds the unfiltered dict as raw_metrics for debug only.
    """
    by_name = {kp.name: kp for kp in keypoints}
    raw_metrics = {
        **game_metrics_2d(by_name, min_confidence),
        **wrist_metrics_2d(by_name, hand_keypoints, min_confidence),
    }
    metrics = metrics_filter.apply(raw_metrics) if metrics_filter is not None else raw_metrics
    landmarks = []
    for index, name in enumerate(LANDMARK_NAMES):
        kp = by_name.get(name)
        if kp is None:
            continue
        landmarks.append(
            {
                "index": index,
                "name": name,
                "u_px": kp.u_px,
                "v_px": kp.v_px,
                "confidence": kp.confidence,
            }
        )
    message = {
        "timestamp_ms": int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms),
        "frame": int(frame_index),
        "width": int(width),
        "height": int(height),
        "tracking": tracking_status(by_name, min_confidence),
        "landmarks": landmarks,
        "hands": serialize_hands_2d(hand_keypoints),
        "face": serialize_face_mesh_2d(face_keypoints),
        "metrics": metrics,
    }
    if include_raw_metrics:
        message["raw_metrics"] = raw_metrics
    return message


class PoseBridgeServer:
    """WebSocket fan-out. Dropped clients are removed. The camera loop stays up."""

    def __init__(self) -> None:
        self._clients: set = set()

    async def handler(self, websocket, *_path) -> None:
        self._clients.add(websocket)
        peer = getattr(websocket, "remote_address", None)
        print(f"Game connected: {peer}")
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            print(f"Game disconnected: {peer}")

    async def publish(self, payload: str) -> None:
        if not self._clients:
            return
        dead = []
        for client in list(self._clients):
            try:
                await client.send(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)


async def serve_until(server: PoseBridgeServer, host: str, port: int, stop: asyncio.Event):
    import websockets

    try:
        async with websockets.serve(server.handler, host, port):
            print(f"Pose bridge listening on ws://{host}:{port}")
            await stop.wait()
    except OSError as error:
        print(f"Could not listen on {host}:{port}. {error}")
        raise


def _pump_camera(
    camera: LiveRealSenseRGB,
    extractor: PoseExtractor2D,
    server: PoseBridgeServer,
    loop: asyncio.AbstractEventLoop,
    stop_thread: threading.Event,
    min_confidence: float,
    on_camera_failure,
    preview: PreviewStream | None = None,
    hands: HandsExtractor2D | None = None,
    face: FaceMeshExtractor2D | None = None,
    metrics_filter: GameMetricsFilter | None = None,
    include_raw_metrics: bool = False,
) -> None:
    frame_index = 0
    while not stop_thread.is_set():
        try:
            frame = camera.get_frame()
        except RuntimeError as error:
            print(f"Camera failed: {error}")
            on_camera_failure()
            return
        if frame is None:
            continue
        height, width = frame.shape[:2]
        try:
            keypoints = extractor.extract(frame)
        except Exception as error:
            print(f"MediaPipe frame failed: {error}")
            keypoints = []
        hand_keypoints = []
        if hands is not None and keypoints:
            try:
                hand_keypoints = hands.extract(frame, pose_keypoints=keypoints)
            except Exception as error:
                print(f"MediaPipe Hands frame failed: {error}")
        face_keypoints = []
        if face is not None:
            try:
                face_keypoints = face.extract(frame)
            except Exception as error:
                print(f"MediaPipe Face Mesh frame failed: {error}")
        frame_index += 1
        message = build_pose_message(
            frame_index,
            width,
            height,
            keypoints,
            min_confidence,
            hand_keypoints=hand_keypoints,
            face_keypoints=face_keypoints,
            metrics_filter=metrics_filter,
            include_raw_metrics=include_raw_metrics,
        )
        payload = json.dumps(message)
        asyncio.run_coroutine_threadsafe(server.publish(payload), loop)
        # Same frame MediaPipe just used. Metrics go out first; the preview only copies it.
        if preview is not None:
            preview.submit(frame)


def main() -> None:
    config = load_config()
    camera_cfg = config["camera"]
    pose_cfg = config["pose"]
    min_confidence = float(pose_cfg["min_visibility"])
    analysis_cfg = config.get("analysis") or {}
    metrics_filter = filter_from_config(analysis_cfg)
    include_raw_metrics = bool(
        (analysis_cfg.get("game_metrics_filter") or {}).get("publish_raw_metrics", False)
    )

    camera = LiveRealSenseRGB(
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        fps=int(camera_cfg["fps"]),
        enable_depth=False,
    )
    extractor = PoseExtractor2D(
        model_complexity=int(pose_cfg["model_complexity"]),
        min_detection_confidence=float(pose_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(pose_cfg["min_tracking_confidence"]),
    )
    hands_cfg = pose_cfg.get("hands") or {}
    hands: HandsExtractor2D | None = None
    if bool(hands_cfg.get("enabled", True)):
        hands = HandsExtractor2D(
            max_num_hands=int(hands_cfg.get("max_num_hands", 2)),
            model_complexity=int(hands_cfg.get("model_complexity", 1)),
            min_detection_confidence=float(
                hands_cfg.get("min_detection_confidence", pose_cfg["min_detection_confidence"])
            ),
            min_tracking_confidence=float(
                hands_cfg.get("min_tracking_confidence", pose_cfg["min_tracking_confidence"])
            ),
            swap_handedness=bool(hands_cfg.get("swap_handedness", True)),
            match_to_pose_wrists=bool(hands_cfg.get("match_to_pose_wrists", True)),
        )
    else:
        print("MediaPipe Hands is off in config.yaml. Wrist metrics will be null.")

    face_cfg = pose_cfg.get("face") or {}
    face: FaceMeshExtractor2D | None = None
    if bool(face_cfg.get("enabled", True)):
        face = FaceMeshExtractor2D(
            max_num_faces=int(face_cfg.get("max_num_faces", 1)),
            refine_landmarks=bool(face_cfg.get("refine_landmarks", False)),
            min_detection_confidence=float(
                face_cfg.get("min_detection_confidence", pose_cfg["min_detection_confidence"])
            ),
            min_tracking_confidence=float(
                face_cfg.get("min_tracking_confidence", pose_cfg["min_tracking_confidence"])
            ),
        )
    else:
        print("MediaPipe Face Mesh is off in config.yaml. Face overlay will be empty.")

    def _close_models() -> None:
        extractor.close()
        if hands is not None:
            hands.close()
        if face is not None:
            face.close()

    print("2D pose bridge. Depth is off. Python owns the D455f.")
    print("Close the live 3D app and RealSense Viewer first.")
    print(f"Game client: ws://{POSE_BRIDGE_HOST}:{POSE_BRIDGE_PORT}")

    try:
        camera.start()
    except RuntimeError as error:
        print("Could not start the camera.")
        print(str(error))
        _close_models()
        sys.exit(1)

    server = PoseBridgeServer()
    stop_thread = threading.Event()
    preview: PreviewStream | None = PreviewStream(PREVIEW_HOST, PREVIEW_PORT)
    try:
        preview.start()
        print(f"Camera preview: {preview.url}")
    except OSError as error:
        print(f"Camera preview off. Could not listen on {PREVIEW_HOST}:{PREVIEW_PORT}. {error}")
        preview = None

    def _stop_preview() -> None:
        if preview is not None:
            preview.stop()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        def _stop(*_args) -> None:
            stop_thread.set()
            stop.set()

        try:
            loop.add_signal_handler(signal.SIGINT, _stop)
        except NotImplementedError:
            signal.signal(signal.SIGINT, lambda *_a: loop.call_soon_threadsafe(_stop))

        def _camera_failed() -> None:
            loop.call_soon_threadsafe(_stop)

        pump = threading.Thread(
            target=_pump_camera,
            args=(camera, extractor, server, loop, stop_thread, min_confidence, _camera_failed, preview, hands, face, metrics_filter, include_raw_metrics),
            name="pose-bridge-camera",
            daemon=True,
        )
        pump.start()
        try:
            await serve_until(server, POSE_BRIDGE_HOST, POSE_BRIDGE_PORT, stop)
        finally:
            stop_thread.set()
            pump.join(timeout=6)
            _stop_preview()
            camera.stop()
            _close_models()
            print("Pose bridge stopped. Camera released.")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        stop_thread.set()
        _stop_preview()
        camera.stop()
        _close_models()


if __name__ == "__main__":
    main()
