"""LSTM results stay tied to the camera frame that produced them."""

from __future__ import annotations

import unittest

from src.pose2sim_live.sync import accept_newer_result, classify_prediction
from src.ui.session import _lstm_display_decision


class PredictionClockTest(unittest.TestCase):
    def test_source_frame_is_reported_with_the_current_frame(self) -> None:
        decision = classify_prediction(
            1.00,
            1.02,
            delay_ms=80,
            max_age_ms=200,
            source_frame=10,
            current_frame=10,
        )
        self.assertEqual(decision["status"], "current")
        self.assertEqual(decision["source_frame"], 10)
        self.assertEqual(decision["current_frame"], 10)
        self.assertTrue(decision["draw"])
        self.assertAlmostEqual(decision["age_ms"], 20.0)

    def test_delayed_prediction_stays_visible_and_labeled(self) -> None:
        decision = classify_prediction(1.00, 1.12, delay_ms=80, max_age_ms=200, source_frame=4, current_frame=6)
        self.assertEqual(decision["status"], "delayed")
        self.assertTrue(decision["draw"])
        self.assertAlmostEqual(decision["age_ms"], 120.0)

    def test_stale_prediction_is_not_drawn(self) -> None:
        decision = classify_prediction(1.00, 1.25, delay_ms=80, max_age_ms=200, source_frame=1, current_frame=8)
        self.assertEqual(decision["status"], "stale")
        self.assertFalse(decision["draw"])

    def test_missing_prediction_is_unavailable(self) -> None:
        decision = classify_prediction(None, 1.0, delay_ms=80, max_age_ms=200, current_frame=3)
        self.assertEqual(decision["status"], "unavailable")
        self.assertFalse(decision["draw"])
        self.assertIsNone(decision["age_ms"])

    def test_older_worker_result_is_rejected(self) -> None:
        self.assertFalse(accept_newer_result(1.20, 1.05))
        self.assertTrue(accept_newer_result(1.05, 1.20))
        self.assertTrue(accept_newer_result(None, 1.05))
        self.assertFalse(accept_newer_result(1.05, None))
        ahead = classify_prediction(1.30, 1.00, delay_ms=80, max_age_ms=200, source_frame=9, current_frame=7)
        self.assertEqual(ahead["status"], "unavailable")
        self.assertFalse(ahead["draw"])

    def test_display_decision_uses_configured_limits(self) -> None:
        records = [
            {
                "class": "predicted_upper",
                "t_pred_input_sec": 2.0,
                "source_frame": 15,
                "x_m": 0.1,
                "y_m": 0.2,
                "z_m": 1.4,
            }
        ]
        decision = _lstm_display_decision(records, 2.30, 20, {"lstm_sync": {"delay_ms": 50, "max_age_ms": 100}})
        self.assertEqual(decision["status"], "stale")
        self.assertEqual(decision["source_frame"], 15)
        self.assertEqual(decision["current_frame"], 20)
        self.assertFalse(decision["draw"])


if __name__ == "__main__":
    unittest.main()
