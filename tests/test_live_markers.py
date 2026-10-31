"""Live landmark inventory and RGB overlay. No camera."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from src.pose.keypoints import Keypoint2D
from src.ui.live_markers import draw_live_landmarks, has_camera_xyz, inventory, short_label


def kp(name: str, u: float, v: float, conf: float = 0.9, xyz=None) -> Keypoint2D:
    if xyz is None:
        return Keypoint2D(name=name, u_px=u, v_px=v, confidence=conf)
    x, y, z = xyz
    return Keypoint2D(
        name=name,
        u_px=u,
        v_px=v,
        confidence=conf,
        coord_frame="camera_3d_metre",
        x_m=x,
        y_m=y,
        z_m=z,
    )


class InventoryTest(unittest.TestCase):
    def test_counts_pose_and_hands_and_skips_invalid_depth(self) -> None:
        points = [
            kp("left_shoulder", 100, 100, xyz=(0.1, 0.2, 1.0)),
            kp("right_shoulder", 200, 100),
            kp("left_wrist", 80, 180, xyz=(0.2, 0.3, 1.1)),
            kp("left_index_mcp", 70, 200, xyz=(0.21, 0.32, 1.12)),
            kp("left_index_pip", 68, 210),
            kp("nose", 150, 40, conf=0.2),
        ]
        stats = inventory(points, min_visibility=0.5, fps=12.0, space="3d")
        self.assertEqual(stats.n_pose_detected, 3)
        self.assertEqual(stats.n_pose_xyz, 2)
        self.assertEqual(stats.n_hand_detected, 3)
        self.assertEqual(stats.n_hand_xyz, 2)
        self.assertNotIn("nose", stats.pose_names)
        self.assertEqual(stats.tracking, "person + L hand")
        self.assertFalse(has_camera_xyz(points[1]))
        self.assertTrue(has_camera_xyz(points[0]))

    def test_short_label_keeps_left_right(self) -> None:
        self.assertEqual(short_label("left_foot_index"), "L foot index")
        self.assertEqual(short_label("right_index_mcp"), "R index mcp")
        self.assertEqual(short_label("nose"), "nose")


class DrawTest(unittest.TestCase):
    def test_overlay_draws_without_inventing_xyz(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        points = [
            kp("left_elbow", 40, 80, xyz=(0.1, 0.2, 1.0)),
            kp("left_wrist", 50, 120),
        ]
        out = draw_live_landmarks(frame, points, min_visibility=0.5, fps=10.0, space="3d")
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)
        self.assertFalse(np.array_equal(out, frame))
        source = inspect.getsource(draw_live_landmarks)
        self.assertNotIn("short_label", source)
        self.assertIn("fps", source)

    def test_live_rgb_does_not_draw_joint_frame_axes(self) -> None:
        from src.ui.session import MotionSession2D

        source = inspect.getsource(MotionSession2D.process_frame)
        self.assertEqual(source.count("draw_joint_frames("), 1)
        self.assertIn("save_canvas", source.split("draw_joint_frames(")[1][:80])


if __name__ == "__main__":
    unittest.main()
