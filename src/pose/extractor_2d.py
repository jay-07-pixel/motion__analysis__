"""Extract 2D body keypoints from one RGB frame (camera pixel coordinates).

What:
    MediaPipe Pose finds 33 joints. We convert each joint from normalized
    0..1 into pixels (u_px, v_px) on the RGB image.

Why:
    Sir asked for 2D keypoints in camera coordinates. Those pixels are the
    measurement. Overlay, CSV, and later depth lookup all use (u_px, v_px).

How:
    1. Convert BGR (OpenCV) -> RGB (MediaPipe).
    2. Run the pose model (local, no internet).
    3. u_px = x * width,  v_px = y * height. Origin = top-left.

We ignore MediaPipe 'world_landmarks'. That is a guessed 3D skeleton, not
RealSense metres. Real 3D is Phase 2: depth at these pixels.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from src.pose.keypoints import Keypoint2D
from src.pose.skeleton import LANDMARK_NAMES


class PoseExtractor2D:
    """Runs MediaPipe Pose and returns camera-frame 2D keypoints."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Create the pose model. Settings come from config.yaml.

        Args:
            model_complexity: 0 = fastest / least accurate, 1 = default,
                2 = slowest / most accurate.
            min_detection_confidence: 0..1. Below this, no person is reported
                on the first look.
            min_tracking_confidence: 0..1. Below this, MediaPipe re-detects
                instead of tracking from the previous frame.
        """
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            enable_segmentation=False,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def extract(self, bgr_frame: np.ndarray) -> list[Keypoint2D]:
        """Return 33 keypoints in camera pixels, or [] if no person found.

        Args:
            bgr_frame: Colour image from live USB, mp4, or bag (OpenCV BGR).

        Returns:
            List of Keypoint2D. Empty if the model did not see a body.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        height, width = bgr_frame.shape[:2]
        # MediaPipe wants RGB. flags.writeable = False avoids an extra copy.
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return []

        keypoints: list[Keypoint2D] = []
        for index, landmark in enumerate(results.pose_landmarks.landmark):
            # landmark.x / .y are fractions of width / height (0 = left/top).
            keypoints.append(
                Keypoint2D(
                    name=LANDMARK_NAMES[index],
                    u_px=float(landmark.x) * width,
                    v_px=float(landmark.y) * height,
                    confidence=float(landmark.visibility),
                )
            )
        return keypoints

    def close(self) -> None:
        """Free the MediaPipe graph. Call when the window closes."""
        self._pose.close()
