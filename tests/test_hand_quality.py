"""Quality gate for the four hand landmarks. The frame math is not retested here."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.geometry.anatomical_frames import build_hand_frame
from src.geometry.hand_quality import HandQualityGate, hand_frame_after_quality
from src.pose.keypoints import Keypoint2D

MIN_CONFIDENCE = 0.5


def _kp(name: str, xyz, confidence: float = 1.0, coord_frame: str = "camera_3d_metre") -> Keypoint2D:
    if xyz is None:
        return Keypoint2D(name=name, u_px=0.0, v_px=0.0, confidence=confidence)
    return Keypoint2D(
        name=name,
        u_px=0.0,
        v_px=0.0,
        confidence=confidence,
        coord_frame=coord_frame,
        x_m=xyz[0],
        y_m=xyz[1],
        z_m=xyz[2],
    )


def _hand(side: str, wrist, middle, index, pinky, confidence: float = 1.0, **overrides) -> list[Keypoint2D]:
    points = {
        f"{side}_wrist": wrist,
        f"{side}_middle_mcp": middle,
        f"{side}_index_mcp": index,
        f"{side}_pinky_mcp": pinky,
    }
    keypoints = []
    for name, xyz in points.items():
        spec = overrides.get(name)
        if spec == "drop":
            continue
        if spec == "missing_xyz":
            keypoints.append(_kp(name, None, confidence))
            continue
        if isinstance(spec, tuple) and spec and spec[0] == "raw":
            keypoints.append(spec[1])
            continue
        keypoints.append(_kp(name, xyz, confidence))
    return keypoints


def _stable(side: str = "right", shift=(0.0, 0.0, 0.0)) -> list[Keypoint2D]:
    sx, sy, sz = shift
    return _hand(
        side,
        (0.15 + sx, 0.00 + sy, 1.00 + sz),
        (0.15 + sx, 0.08 + sy, 1.00 + sz),
        (0.19 + sx, 0.03 + sy, 1.01 + sz),
        (0.11 + sx, 0.03 + sy, 0.99 + sz),
    )


class HandQualityTests(unittest.TestCase):
    def test_stable_hand_is_accepted(self) -> None:
        result = hand_frame_after_quality(_stable(), "right", MIN_CONFIDENCE, HandQualityGate(), 0.0)
        self.assertTrue(result.available)
        self.assertIsNone(result.reason)
        self.assertAlmostEqual(float(np.linalg.det(result.frame.rotation_matrix)), 1.0, places=6)

    def test_metre_jump_is_rejected(self) -> None:
        gate = HandQualityGate()
        first = hand_frame_after_quality(_stable(), "right", MIN_CONFIDENCE, gate, 0.0)
        self.assertTrue(first.available)
        jumped = _hand(
            "right",
            (0.15, 0.00, 1.00),
            (0.15, 0.08, 4.50),
            (0.19, 0.03, 1.01),
            (0.11, 0.03, 0.99),
        )
        result = hand_frame_after_quality(jumped, "right", MIN_CONFIDENCE, gate, 0.07)
        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertEqual(result.reason, "depth_jump:right_middle_mcp")

    def test_nonfinite_xyz_is_rejected(self) -> None:
        bad = Keypoint2D(
            name="right_index_mcp",
            u_px=0.0,
            v_px=0.0,
            confidence=0.9,
            coord_frame="camera_3d_metre",
            x_m=float("nan"),
            y_m=0.03,
            z_m=float("inf"),
        )
        keypoints = _hand(
            "right",
            (0.15, 0.00, 1.00),
            (0.15, 0.08, 1.00),
            (0.19, 0.03, 1.01),
            (0.11, 0.03, 0.99),
            right_index_mcp=("raw", bad),
        )
        result = hand_frame_after_quality(keypoints, "right", MIN_CONFIDENCE, HandQualityGate(), 0.0)
        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertEqual(result.reason, "invalid_3d:right_index_mcp")
        self.assertTrue(result.min_confidence is None or math.isfinite(result.min_confidence))

    def test_missing_xyz_is_rejected(self) -> None:
        keypoints = _hand(
            "right",
            (0.15, 0.00, 1.00),
            (0.15, 0.08, 1.00),
            (0.19, 0.03, 1.01),
            (0.11, 0.03, 0.99),
            right_pinky_mcp="missing_xyz",
        )
        result = hand_frame_after_quality(keypoints, "right", MIN_CONFIDENCE, HandQualityGate(), 0.0)
        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertEqual(result.reason, "missing_xyz:right_pinky_mcp")

    def test_low_confidence_is_rejected(self) -> None:
        keypoints = _hand(
            "right",
            (0.15, 0.00, 1.00),
            (0.15, 0.08, 1.00),
            (0.19, 0.03, 1.01),
            (0.11, 0.03, 0.99),
            confidence=0.2,
        )
        result = hand_frame_after_quality(keypoints, "right", MIN_CONFIDENCE, HandQualityGate(), 0.0)
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "low_confidence:right_wrist")

    def test_small_motion_is_accepted(self) -> None:
        gate = HandQualityGate()
        self.assertTrue(
            hand_frame_after_quality(_stable(), "right", MIN_CONFIDENCE, gate, 0.0).available
        )
        moved = hand_frame_after_quality(
            _stable(shift=(0.005, 0.0, 0.002)),
            "right",
            MIN_CONFIDENCE,
            gate,
            0.07,
        )
        self.assertTrue(moved.available)
        self.assertIsNone(moved.reason)

    def test_unrealistic_step_is_rejected(self) -> None:
        gate = HandQualityGate()
        hand_frame_after_quality(_stable(), "right", MIN_CONFIDENCE, gate, 1.0)
        shifted = _stable(shift=(0.35, 0.0, 0.0))
        result = hand_frame_after_quality(shifted, "right", MIN_CONFIDENCE, gate, 1.07)
        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertTrue(result.reason.startswith("xyz_jump:"))

    def test_implausible_segment_is_rejected(self) -> None:
        giant = _hand(
            "right",
            (0.15, 0.00, 1.00),
            (0.15, 0.80, 1.00),
            (0.19, 0.03, 1.01),
            (0.11, 0.03, 0.99),
        )
        result = hand_frame_after_quality(giant, "right", MIN_CONFIDENCE, HandQualityGate(), 0.0)
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "implausible_hand_geometry")
        self.assertIsNone(result.frame)

    def test_one_mcp_far_from_the_hand_is_rejected(self) -> None:
        stray = _hand(
            "right",
            (0.15, 0.00, 1.00),
            (0.15, 0.08, 1.00),
            (0.19, 0.03, 1.01),
            (2.50, 0.03, 1.00),
        )
        result = hand_frame_after_quality(stray, "right", MIN_CONFIDENCE, HandQualityGate(), 0.0)
        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertEqual(result.reason, "implausible_hand_geometry")

    def test_moderate_motion_is_not_rejected(self) -> None:
        gate = HandQualityGate()
        for index in range(8):
            result = hand_frame_after_quality(
                _stable(shift=(0.03 * index, 0.0, 0.0)),
                "right",
                MIN_CONFIDENCE,
                gate,
                0.07 * index,
            )
            self.assertTrue(result.available, result.reason)
            self.assertAlmostEqual(float(np.linalg.norm(result.frame.y_axis)), 1.0, places=6)

    def test_gate_does_not_change_frame_math(self) -> None:
        keypoints = _stable("left", shift=(-0.30, 0.0, 0.0))
        direct = build_hand_frame(keypoints, "left", MIN_CONFIDENCE)
        gated = hand_frame_after_quality(keypoints, "left", MIN_CONFIDENCE, HandQualityGate(), 0.0)
        self.assertTrue(direct.available and gated.available)
        np.testing.assert_allclose(gated.frame.origin, direct.frame.origin)
        np.testing.assert_allclose(gated.frame.x_axis, direct.frame.x_axis, atol=1e-12)
        np.testing.assert_allclose(gated.frame.y_axis, direct.frame.y_axis, atol=1e-12)
        np.testing.assert_allclose(gated.frame.z_axis, direct.frame.z_axis, atol=1e-12)
        np.testing.assert_allclose(gated.frame.rotation_matrix, direct.frame.rotation_matrix, atol=1e-12)

    def test_rejection_does_not_replace_the_last_good_point(self) -> None:
        gate = HandQualityGate()
        hand_frame_after_quality(_stable(), "right", MIN_CONFIDENCE, gate, 0.0)
        rejected = hand_frame_after_quality(
            _stable(shift=(0.0, 0.0, 3.0)),
            "right",
            MIN_CONFIDENCE,
            gate,
            0.07,
        )
        self.assertFalse(rejected.available)
        restored = hand_frame_after_quality(
            _stable(shift=(0.01, 0.0, 0.0)),
            "right",
            MIN_CONFIDENCE,
            gate,
            0.14,
        )
        self.assertTrue(restored.available)
        self.assertIsNone(restored.reason)


if __name__ == "__main__":
    unittest.main()
