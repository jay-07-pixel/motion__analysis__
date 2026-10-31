"""Face Mesh overlay payload. Overlay only. No camera. No angle math."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from src.pose.face_mesh import (
    FACE_LANDMARK_COUNT,
    FACE_MESH_CONTOUR_INDICES,
    FaceMeshExtractor2D,
    face_point_confidence,
    serialize_face_mesh_2d,
)
from src.pose.keypoints import Keypoint2D
from tests.test_pose_metrics_2d import GATE, body


def mesh(u0: float = 640.0, v0: float = 120.0, count: int = FACE_LANDMARK_COUNT) -> list[Keypoint2D]:
    return [
        Keypoint2D(name=f"face_{i}", u_px=u0 + i * 0.05, v_px=v0, confidence=0.9)
        for i in range(count)
    ]


class SerializeTest(unittest.TestCase):
    def test_complete_mesh_is_468_points(self) -> None:
        payload = serialize_face_mesh_2d(mesh())
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload), 468)
        self.assertEqual(payload[0]["index"], 0)
        self.assertEqual(payload[467]["index"], 467)
        self.assertAlmostEqual(payload[10]["u_px"], 640.0 + 10 * 0.05)

    def test_missing_mesh_is_null_not_zeros(self) -> None:
        self.assertIsNone(serialize_face_mesh_2d(None))
        self.assertIsNone(serialize_face_mesh_2d([]))
        self.assertIsNone(serialize_face_mesh_2d(mesh(count=100)))

    def test_contour_count_matches_mediapipe(self) -> None:
        self.assertEqual(len(FACE_MESH_CONTOUR_INDICES), 124)
        self.assertEqual(FACE_MESH_CONTOUR_INDICES[0], (0, 37))

    def test_zero_visibility_is_still_a_present_point(self) -> None:
        class _Lm:
            visibility = 0.0
            presence = 0.0

        self.assertEqual(face_point_confidence(_Lm()), 1.0)
        self.assertEqual(face_point_confidence(type("P", (), {"presence": 0.88})()), 0.88)


class ExtractorTest(unittest.TestCase):
    def test_empty_frame_returns_no_points(self) -> None:
        import numpy as np

        extractor = FaceMeshExtractor2D()
        try:
            self.assertEqual(extractor.extract(np.zeros((72, 128, 3), dtype=np.uint8)), [])
        finally:
            extractor.close()


class BridgeWiringTest(unittest.TestCase):
    def test_build_pose_message_includes_face(self) -> None:
        from pose_game_bridge import build_pose_message

        message = build_pose_message(1, 1280, 720, body(), GATE, face_keypoints=mesh())
        self.assertEqual(len(message["face"]), 468)
        self.assertEqual(message["face"][0]["u_px"], 640.0)

    def test_missing_face_is_json_null(self) -> None:
        from pose_game_bridge import build_pose_message
        import json

        encoded = json.dumps(build_pose_message(1, 1280, 720, body(), GATE))
        self.assertIn('"face": null', encoded)

    def test_bridge_constructs_face_mesh_once(self) -> None:
        import pose_game_bridge

        source = Path(pose_game_bridge.__file__).read_text(encoding="utf-8")
        names = [
            getattr(node.func, "id", None)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
        ]
        self.assertEqual(names.count("FaceMeshExtractor2D"), 1)
        self.assertEqual(source.count("face.extract("), 1)
        self.assertNotIn("mp.solutions.face_mesh.FaceMesh(", source)


if __name__ == "__main__":
    unittest.main()
