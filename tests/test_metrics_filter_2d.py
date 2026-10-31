"""Temporal filter for game 2D angles. Synthetic sequences only. No camera."""

from __future__ import annotations

import statistics
import unittest

from src.analysis.metrics_filter_2d import FILTERED_METRIC_NAMES, GameMetricsFilter


def run(values, filt=None) -> list[float | None]:
    filt = filt or GameMetricsFilter()
    return [filt.apply({"right_elbow_angle_deg": v})["right_elbow_angle_deg"] for v in values]


class StationaryTest(unittest.TestCase):
    JITTER = [170.0, 171.8, 168.9, 172.1, 169.7, 171.2, 170.3, 173.0, 168.8]

    def test_first_sample_is_the_raw_value(self) -> None:
        out = run([170.4])
        self.assertEqual(out, [170.4])

    def test_small_jitter_is_reduced(self) -> None:
        raw = self.JITTER
        filtered = run(raw)
        self.assertTrue(all(v is not None for v in filtered))
        raw_std = statistics.pstdev(raw)
        filt_std = statistics.pstdev([v for v in filtered if v is not None])
        self.assertGreater(raw_std, 1.2)
        self.assertLess(filt_std, 0.55 * raw_std)
        self.assertLess(max(filtered) - min(filtered), 0.55 * (max(raw) - min(raw)))

    def test_stable_input_stays_on_the_same_value(self) -> None:
        out = run([171.0] * 12)
        self.assertTrue(all(v == 171.0 for v in out))


class OutlierAndMotionTest(unittest.TestCase):
    def test_one_frame_glitch_does_not_jump(self) -> None:
        raw = [170.0, 171.0, 169.0, 35.0, 170.0, 172.0]
        out = run(raw)
        self.assertLess(abs(out[3] - out[2]), 2.0)
        self.assertGreater(out[3], 160.0)
        self.assertLess(abs(out[-1] - 171.0), 3.0)

    def test_two_agreed_jumps_are_treated_as_real_motion(self) -> None:
        # A real drop that lasts more than one frame must not be stuck at 170.
        out = run([170.0, 171.0, 90.0, 88.0, 87.0])
        self.assertLess(out[-1], 120.0)

    def test_gradual_raise_follows_the_arm(self) -> None:
        raw = [30, 45, 65, 90, 115, 140, 160, 175]
        out = run(raw)
        for earlier, later in zip(out, out[1:]):
            self.assertGreater(later, earlier)
        self.assertGreater(out[-1], 155.0)
        self.assertLess(out[-1], 176.0)
        # Lag is a couple of samples, not the whole raise.
        self.assertGreater(out[4], 90.0)

    def test_fast_raise_is_not_hundreds_of_milliseconds_behind(self) -> None:
        # 8 frames at 20 fps = 400 ms. Output should already be near overhead.
        out = run([30, 50, 75, 100, 125, 150, 170, 175])
        self.assertGreater(out[-1], 155.0)
        self.assertGreater(out[-2], 140.0)


class NullAndResetTest(unittest.TestCase):
    def test_null_is_not_turned_into_zero(self) -> None:
        filt = GameMetricsFilter(hold_frames=0)
        out = filt.apply({"right_wrist_angle_deg": None})
        self.assertIsNone(out["right_wrist_angle_deg"])
        self.assertNotEqual(out["right_wrist_angle_deg"], 0)

    def test_brief_gap_keeps_the_last_value(self) -> None:
        filt = GameMetricsFilter(hold_frames=4)
        filt.apply({"right_wrist_angle_deg": 150.0})
        held = [filt.apply({"right_wrist_angle_deg": None})["right_wrist_angle_deg"] for _ in range(4)]
        self.assertTrue(all(v == 150.0 for v in held))

    def test_hand_dropout_becomes_null_after_the_hold(self) -> None:
        filt = GameMetricsFilter(hold_frames=4)
        filt.apply({"right_wrist_angle_deg": 150.0})
        for _ in range(4):
            filt.apply({"right_wrist_angle_deg": None})
        gone = filt.apply({"right_wrist_angle_deg": None})["right_wrist_angle_deg"]
        self.assertIsNone(gone)

    def test_wrist_returning_starts_from_the_new_raw_value(self) -> None:
        filt = GameMetricsFilter(hold_frames=2)
        filt.apply({"right_wrist_angle_deg": 178.0})
        for _ in range(3):
            filt.apply({"right_wrist_angle_deg": None})
        again = filt.apply({"right_wrist_angle_deg": 128.0})["right_wrist_angle_deg"]
        self.assertEqual(again, 128.0)

    def test_reset_does_not_keep_the_previous_person(self) -> None:
        filt = GameMetricsFilter()
        for _ in range(8):
            filt.apply({"right_arm_angle_deg": 170.0})
        filt.reset()
        self.assertEqual(filt.apply({"right_arm_angle_deg": 12.0})["right_arm_angle_deg"], 12.0)


class IndependenceAndLeanTest(unittest.TestCase):
    def test_left_and_right_are_independent(self) -> None:
        filt = GameMetricsFilter()
        left = []
        right = []
        for i, value in enumerate([30, 50, 80, 110, 140, 170]):
            out = filt.apply({"left_elbow_angle_deg": 175.0, "right_elbow_angle_deg": value})
            left.append(out["left_elbow_angle_deg"])
            right.append(out["right_elbow_angle_deg"])
        self.assertTrue(all(v == 175.0 for v in left))
        self.assertGreater(right[-1], 150.0)
        self.assertLess(right[-1], 171.0)

    def test_torso_lean_keeps_its_sign(self) -> None:
        filt = GameMetricsFilter()
        left = [filt.apply({"torso_lean_deg": v})["torso_lean_deg"] for v in [8.4, 8.1, 8.6, 7.9, 8.3]]
        self.assertTrue(all(v > 0 for v in left))
        filt.reset()
        right = [filt.apply({"torso_lean_deg": v})["torso_lean_deg"] for v in [-8.4, -8.1, -8.6, -7.9, -8.3]]
        self.assertTrue(all(v < 0 for v in right))

    def test_torso_lean_is_not_clamped_to_0_180(self) -> None:
        filt = GameMetricsFilter()
        out = filt.apply({"torso_lean_deg": -12.4})
        self.assertEqual(out["torso_lean_deg"], -12.4)

    def test_unlisted_metrics_pass_through_raw(self) -> None:
        filt = GameMetricsFilter()
        out = filt.apply({"right_knee_angle_deg": 139.7, "right_elbow_angle_deg": 170.0})
        self.assertEqual(out["right_knee_angle_deg"], 139.7)
        self.assertEqual(out["right_elbow_angle_deg"], 170.0)

    def test_does_not_mutate_the_raw_dict(self) -> None:
        filt = GameMetricsFilter()
        raw = {"right_elbow_angle_deg": 170.0}
        filt.apply(raw)
        filt.apply({"right_elbow_angle_deg": 173.0})
        self.assertEqual(raw["right_elbow_angle_deg"], 170.0)

    def test_nan_is_treated_as_missing(self) -> None:
        filt = GameMetricsFilter(hold_frames=0)
        filt.apply({"right_elbow_angle_deg": 170.0})
        self.assertIsNone(filt.apply({"right_elbow_angle_deg": float("nan")})["right_elbow_angle_deg"])

    def test_every_listed_metric_has_its_own_channel(self) -> None:
        self.assertEqual(len(FILTERED_METRIC_NAMES), 7)
        filt = GameMetricsFilter()
        first = {name: 10.0 + i for i, name in enumerate(FILTERED_METRIC_NAMES)}
        out = filt.apply(first)
        for name, value in first.items():
            self.assertEqual(out[name], value)


class BridgeWiringTest(unittest.TestCase):
    def test_build_pose_message_can_apply_the_filter(self) -> None:
        from pose_game_bridge import build_pose_message
        from tests.test_pose_metrics_2d import GATE, body

        filt = GameMetricsFilter()
        first = build_pose_message(1, 1280, 720, body(), GATE, metrics_filter=filt)
        elbow = first["metrics"]["right_elbow_angle_deg"]
        self.assertIsNotNone(elbow)
        jittered = dict(first["metrics"])
        # Same pose again: filter holds a still person even if we only send metrics
        # through apply; the message path is what the pump uses.
        second = build_pose_message(2, 1280, 720, body(), GATE, metrics_filter=filt)
        self.assertEqual(second["metrics"]["right_elbow_angle_deg"], elbow)
        self.assertNotIn("raw_metrics", second)
        del jittered

    def test_debug_payload_keeps_raw_beside_filtered_metrics(self) -> None:
        from pose_game_bridge import build_pose_message
        from tests.test_pose_metrics_2d import GATE, body

        filt = GameMetricsFilter()
        msg = build_pose_message(
            1, 1280, 720, body(), GATE, metrics_filter=filt, include_raw_metrics=True
        )
        self.assertIn("raw_metrics", msg)
        self.assertEqual(
            msg["metrics"]["right_elbow_angle_deg"],
            msg["raw_metrics"]["right_elbow_angle_deg"],
        )


if __name__ == "__main__":
    unittest.main()
