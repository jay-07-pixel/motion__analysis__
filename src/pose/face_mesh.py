"""MediaPipe Face Mesh: 468 face points in camera pixels.

Pose only has 11 coarse face landmarks (nose, eyes, ears, mouth). Face Mesh
adds the full 468-point mesh so the game overlay can draw the real face.

z from this model is NOT RealSense metres. We keep 2D (u_px, v_px) only.
This does not change pose/wrist angle definitions.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from src.pose.keypoints import Keypoint2D

FACE_LANDMARK_COUNT = 468

# MediaPipe FACEMESH_CONTOURS (eyes, lips, oval). Same order as the game overlay.
FACE_MESH_CONTOUR_INDICES: tuple[tuple[int, int], ...] = (
    (0, 37),
    (0, 267),
    (7, 33),
    (7, 163),
    (10, 109),
    (10, 338),
    (13, 82),
    (13, 312),
    (14, 87),
    (14, 317),
    (17, 84),
    (17, 314),
    (21, 54),
    (21, 162),
    (33, 246),
    (37, 39),
    (39, 40),
    (40, 185),
    (46, 53),
    (52, 53),
    (52, 65),
    (54, 103),
    (55, 65),
    (58, 132),
    (58, 172),
    (61, 146),
    (61, 185),
    (63, 70),
    (63, 105),
    (66, 105),
    (66, 107),
    (67, 103),
    (67, 109),
    (78, 95),
    (78, 191),
    (80, 81),
    (80, 191),
    (81, 82),
    (84, 181),
    (87, 178),
    (88, 95),
    (88, 178),
    (91, 146),
    (91, 181),
    (93, 132),
    (93, 234),
    (127, 162),
    (127, 234),
    (133, 155),
    (133, 173),
    (136, 150),
    (136, 172),
    (144, 145),
    (144, 163),
    (145, 153),
    (148, 152),
    (148, 176),
    (149, 150),
    (149, 176),
    (152, 377),
    (153, 154),
    (154, 155),
    (157, 158),
    (157, 173),
    (158, 159),
    (159, 160),
    (160, 161),
    (161, 246),
    (249, 263),
    (249, 390),
    (251, 284),
    (251, 389),
    (263, 466),
    (267, 269),
    (269, 270),
    (270, 409),
    (276, 283),
    (282, 283),
    (282, 295),
    (284, 332),
    (285, 295),
    (288, 361),
    (288, 397),
    (291, 375),
    (291, 409),
    (293, 300),
    (293, 334),
    (296, 334),
    (296, 336),
    (297, 332),
    (297, 338),
    (308, 324),
    (308, 415),
    (310, 311),
    (310, 415),
    (311, 312),
    (314, 405),
    (317, 402),
    (318, 324),
    (318, 402),
    (321, 375),
    (321, 405),
    (323, 361),
    (323, 454),
    (356, 389),
    (356, 454),
    (362, 382),
    (362, 398),
    (365, 379),
    (365, 397),
    (373, 374),
    (373, 390),
    (374, 380),
    (377, 400),
    (378, 379),
    (378, 400),
    (380, 381),
    (381, 382),
    (384, 385),
    (384, 398),
    (385, 386),
    (386, 387),
    (387, 388),
    (388, 466),
)


def serialize_face_mesh_2d(face_keypoints: list[Keypoint2D] | None) -> list[dict] | None:
    """Game payload: 468 Face Mesh points, or None if the mesh is incomplete.

    Missing faces are None, never invented coordinates.
    """
    if not face_keypoints or len(face_keypoints) != FACE_LANDMARK_COUNT:
        return None
    points: list[dict] = []
    for index, kp in enumerate(face_keypoints):
        if kp is None:
            return None
        points.append(
            {
                "index": index,
                "u_px": kp.u_px,
                "v_px": kp.v_px,
                "confidence": kp.confidence,
            }
        )
    return points


class FaceMeshExtractor2D:
    """Runs MediaPipe Face Mesh and returns 468 2D points in camera pixels."""

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Create the Face Mesh model. Settings come from config.yaml.

        refine_landmarks=False keeps the classic 468-point mesh (no iris extras)
        so the overlay count stays stable.
        """
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=int(max_num_faces),
            refine_landmarks=bool(refine_landmarks),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def extract(self, bgr_frame: np.ndarray) -> list[Keypoint2D]:
        """Return 468 face points, or [] if none.

        Args:
            bgr_frame: Colour image (OpenCV BGR). Same frame as Pose/Hands.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        height, width = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return []

        face = results.multi_face_landmarks[0]
        if len(face.landmark) < FACE_LANDMARK_COUNT:
            return []

        keypoints: list[Keypoint2D] = []
        for index in range(FACE_LANDMARK_COUNT):
            landmark = face.landmark[index]
            keypoints.append(
                Keypoint2D(
                    name=f"face_{index}",
                    u_px=float(landmark.x) * width,
                    v_px=float(landmark.y) * height,
                    confidence=face_point_confidence(landmark),
                )
            )
        return keypoints

    def close(self) -> None:
        """Free the MediaPipe Face Mesh graph."""
        self._mesh.close()


def face_point_confidence(landmark: object) -> float:
    """Score for overlay gating. Face Mesh does not fill Pose visibility.

    Pose landmarks use visibility ~0.9. Face Mesh typically leaves visibility
    and presence at 0 even for a good detection. Treating that as 0 hides
    every point and also hides Pose's coarse face dots. If the mesh was
    returned, the point is present: use 1.0 unless presence/visibility is > 0.
    """
    for attr in ("presence", "visibility"):
        value = getattr(landmark, attr, None)
        if value is None:
            continue
        score = float(value)
        if score > 0:
            return score
    return 1.0
