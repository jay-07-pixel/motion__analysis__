"""Hand-frame checks. Points are RealSense camera metres (X right, Y down, Z forward)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.geometry.anatomical_frames import _MIN_PALM_WIDTH_M, build_hand_frame
from src.pose.keypoints import Keypoint2D

MIN_CONFIDENCE = 0.5


def _kp(
    name: str,
    xyz: tuple[float, float, float] | None,
    confidence: float = 1.0,
) -> Keypoint2D:
    if xyz is None:
        return Keypoint2D(name=name, u_px=0.0, v_px=0.0, confidence=confidence)
    return Keypoint2D(
        name=name,
        u_px=0.0,
        v_px=0.0,
        confidence=confidence,
        coord_frame="camera_3d_metre",
        x_m=xyz[0],
        y_m=xyz[1],
        z_m=xyz[2],
    )


def _hand(
    side: str,
    wrist: tuple[float, float, float],
    middle: tuple[float, float, float],
    index: tuple[float, float, float],
    pinky: tuple[float, float, float],
    confidence: float = 1.0,
    drop: str | None = None,
    clear_xyz: str | None = None,
) -> list[Keypoint2D]:
    points = {
        "wrist": wrist,
        "middle_mcp": middle,
        "index_mcp": index,
        "pinky_mcp": pinky,
    }
    keypoints: list[Keypoint2D] = []
    for stem, xyz in points.items():
        if drop == stem:
            continue
        stored = None if clear_xyz == stem else xyz
        keypoints.append(_kp(f"{side}_{stem}", stored, confidence))
    return keypoints


def _open_palm(side: str, origin: tuple[float, float, float]) -> list[Keypoint2D]:
    """Fingers along camera +Y. Index is toward camera +X from the pinky."""
    ox, oy, oz = origin
    return _hand(
        side,
        (ox, oy, oz),
        (ox, oy + 0.08, oz),
        (ox + 0.04, oy + 0.03, oz),
        (ox - 0.04, oy + 0.03, oz),
    )


def _axes(result):
    frame = result.frame
    assert frame is not None
    return (
        np.asarray(frame.x_axis, dtype=float),
        np.asarray(frame.y_axis, dtype=float),
        np.asarray(frame.z_axis, dtype=float),
        np.asarray(frame.rotation_matrix, dtype=float),
    )


class HandFrameTests(unittest.TestCase):
    def assert_right_handed(self, result) -> None:
        self.assertTrue(result.available)
        self.assertIsNone(result.reason)
        x_axis, y_axis, z_axis, rotation = _axes(result)
        self.assertTrue(np.isfinite(rotation).all())
        for axis in (x_axis, y_axis, z_axis):
            self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=6)
        self.assertAlmostEqual(float(np.dot(x_axis, y_axis)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(x_axis, z_axis)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(y_axis, z_axis)), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)
        stacked = np.column_stack((x_axis, y_axis, z_axis))
        np.testing.assert_allclose(rotation, stacked, atol=1e-8)

    def test_right_hand_is_right_handed(self) -> None:
        result = build_hand_frame(_open_palm("right", (0.15, 0.0, 1.0)), "right", MIN_CONFIDENCE)
        self.assert_right_handed(result)
        self.assertEqual(result.frame.name, "right_hand")
        np.testing.assert_allclose(result.frame.origin, (0.15, 0.0, 1.0))
        np.testing.assert_allclose(result.frame.y_axis, (0.0, 1.0, 0.0), atol=1e-8)
        np.testing.assert_allclose(result.frame.z_axis, (-1.0, 0.0, 0.0), atol=1e-8)
        np.testing.assert_allclose(result.frame.x_axis, (0.0, 0.0, 1.0), atol=1e-8)
        self.assertAlmostEqual(result.min_confidence, 1.0)

    def test_left_hand_uses_the_same_construction(self) -> None:
        right = build_hand_frame(_open_palm("right", (0.15, 0.0, 1.0)), "right", MIN_CONFIDENCE)
        left = build_hand_frame(_open_palm("left", (-0.15, 0.0, 1.0)), "left", MIN_CONFIDENCE)
        self.assert_right_handed(left)
        self.assertEqual(left.frame.name, "left_hand")
        np.testing.assert_allclose(left.frame.x_axis, right.frame.x_axis, atol=1e-8)
        np.testing.assert_allclose(left.frame.y_axis, right.frame.y_axis, atol=1e-8)
        np.testing.assert_allclose(left.frame.z_axis, right.frame.z_axis, atol=1e-8)
        self.assertGreater(left.frame.origin[0] * -1.0, 0.0)
        self.assertAlmostEqual(float(np.linalg.det(left.frame.rotation_matrix)), 1.0, places=6)

    def test_rigid_camera_transform(self) -> None:
        original = build_hand_frame(_open_palm("right", (0.15, 0.02, 1.1)), "right", MIN_CONFIDENCE)
        angle = math.radians(35.0)
        rotation = np.array(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        translation = np.array([0.05, -0.02, 0.3])

        def move(point):
            moved = rotation @ np.asarray(point, dtype=float) + translation
            return (float(moved[0]), float(moved[1]), float(moved[2]))

        wrist = (0.15, 0.02, 1.1)
        moved = build_hand_frame(
            _hand(
                "right",
                move(wrist),
                move((0.15, 0.10, 1.1)),
                move((0.19, 0.05, 1.1)),
                move((0.11, 0.05, 1.1)),
            ),
            "right",
            MIN_CONFIDENCE,
        )
        self.assert_right_handed(moved)
        np.testing.assert_allclose(
            moved.frame.origin,
            rotation @ np.asarray(original.frame.origin) + translation,
            atol=1e-8,
        )
        for before, after in (
            (original.frame.x_axis, moved.frame.x_axis),
            (original.frame.y_axis, moved.frame.y_axis),
            (original.frame.z_axis, moved.frame.z_axis),
        ):
            np.testing.assert_allclose(after, rotation @ np.asarray(before), atol=1e-8)

    def test_missing_landmarks(self) -> None:
        origin = (0.15, 0.0, 1.0)
        expected = {
            "wrist": "missing_landmark:right_wrist",
            "middle_mcp": "missing_landmark:right_middle_mcp",
            "index_mcp": "missing_landmark:right_index_mcp",
            "pinky_mcp": "missing_landmark:right_pinky_mcp",
        }
        for stem, reason in expected.items():
            result = build_hand_frame(
                _hand(
                    "right",
                    origin,
                    (0.15, 0.08, 1.0),
                    (0.19, 0.03, 1.0),
                    (0.11, 0.03, 1.0),
                    drop=stem,
                ),
                "right",
                MIN_CONFIDENCE,
            )
            self.assertFalse(result.available, stem)
            self.assertIsNone(result.frame, stem)
            self.assertEqual(result.reason, reason)

    def test_missing_xyz(self) -> None:
        result = build_hand_frame(
            _hand(
                "left",
                (-0.15, 0.0, 1.0),
                (-0.15, 0.08, 1.0),
                (-0.11, 0.03, 1.0),
                (-0.19, 0.03, 1.0),
                clear_xyz="middle_mcp",
            ),
            "left",
            MIN_CONFIDENCE,
        )
        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertEqual(result.reason, "missing_xyz:left_middle_mcp")
        self.assertTrue(math.isfinite(result.min_confidence))

    def test_low_confidence(self) -> None:
        result = build_hand_frame(
            _hand(
                "right",
                (0.15, 0.0, 1.0),
                (0.15, 0.08, 1.0),
                (0.19, 0.03, 1.0),
                (0.11, 0.03, 1.0),
                confidence=0.2,
            ),
            "right",
            MIN_CONFIDENCE,
        )
        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertEqual(result.reason, "low_confidence:right_wrist")
        self.assertAlmostEqual(result.min_confidence, 0.2)

    def test_degenerate_long_axis(self) -> None:
        wrist = (0.15, 0.0, 1.0)
        result = build_hand_frame(
            _hand("right", wrist, wrist, (0.19, 0.0, 1.0), (0.11, 0.0, 1.0)),
            "right",
            MIN_CONFIDENCE,
        )
        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertEqual(result.reason, "degenerate_long_axis")

    def test_degenerate_palm_width(self) -> None:
        knuckle = (0.15, 0.03, 1.0)
        result = build_hand_frame(
            _hand("right", (0.15, 0.0, 1.0), (0.15, 0.08, 1.0), knuckle, knuckle),
            "right",
            MIN_CONFIDENCE,
        )
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "degenerate_palm_width")
        self.assertIsNone(result.frame)

    def test_palm_width_parallel_to_long_axis(self) -> None:
        result = build_hand_frame(
            _hand(
                "right",
                (0.15, 0.0, 1.0),
                (0.15, 0.08, 1.0),
                (0.15, 0.02, 1.0),
                (0.15, 0.06, 1.0),
            ),
            "right",
            MIN_CONFIDENCE,
        )
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "degenerate_palm_width")
        self.assertIsNone(result.frame)

    def test_palm_width_threshold(self) -> None:
        """Usable width is the index-to-pinky line after removing the finger axis."""
        limit = _MIN_PALM_WIDTH_M
        cases = (
            (0.050, True, None),
            (limit, True, None),
            (limit - 0.0005, False, "insufficient_palm_width"),
            (0.003, False, "insufficient_palm_width"),
        )
        for width, expect_ok, reason in cases:
            result = build_hand_frame(
                _hand(
                    "right",
                    (0.00, 0.00, 1.0),
                    (0.00, 0.08, 1.0),
                    (0.00, 0.03, 1.0),
                    (-width, 0.03, 1.0),
                ),
                "right",
                MIN_CONFIDENCE,
            )
            self.assertEqual(result.available, expect_ok, width)
            self.assertEqual(result.reason, reason, width)
            if expect_ok:
                self.assert_right_handed(result)
            else:
                self.assertIsNone(result.frame)

    def test_invalid_input_has_no_nonfinite_axes(self) -> None:
        keypoints = _open_palm("right", (0.15, 0.0, 1.0))
        keypoints[0] = Keypoint2D(
            name="right_wrist",
            u_px=0.0,
            v_px=0.0,
            confidence=float("nan"),
            x_m=float("nan"),
            y_m=float("inf"),
            z_m=1.0,
        )
        result = build_hand_frame(keypoints, "right", MIN_CONFIDENCE)
        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertEqual(result.reason, "low_confidence:right_wrist")
        self.assertIsNone(result.min_confidence)


if __name__ == "__main__":
    unittest.main()
