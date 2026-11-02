"""The fit picture uses a full frame that sits close to the skeleton."""

from __future__ import annotations

import unittest

from src.opensim_upper.visualize import choose_fit_frame


def _frame(time_sec: float, points: dict) -> dict:
    return {"time_sec": time_sec, "points": points}


def _model(point: tuple[float, float, float], extra: tuple[float, float, float] | None = None) -> dict:
    row = {"RElbow_tx": point[0], "RElbow_ty": point[1], "RElbow_tz": point[2]}
    if extra is not None:
        row.update({"LElbow_tx": extra[0], "LElbow_ty": extra[1], "LElbow_tz": extra[2]})
    return row


class FitFrameTest(unittest.TestCase):
    def test_a_reach_at_the_end_is_not_the_picture(self) -> None:
        still = (0.10, -0.20, 1.20)
        frames = [
            _frame(1.0, {"RElbow": still, "LElbow": (0.2, -0.2, 1.2)}),
            _frame(2.0, {"RElbow": (0.11, -0.20, 1.21), "LElbow": (0.2, -0.2, 1.2)}),
            _frame(3.0, {"RElbow": (0.40, -0.20, 1.20), "LElbow": (0.2, -0.2, 1.2)}),
        ]
        model_rows = [_model(still, (0.2, -0.2, 1.2)) for _ in frames]
        chosen = choose_fit_frame(frames, model_rows)
        self.assertEqual(frames[chosen]["time_sec"], 1.0)
        self.assertEqual(frames[2]["points"]["RElbow"], (0.40, -0.20, 1.20))

    def test_a_frame_missing_a_point_does_not_replace_a_full_one(self) -> None:
        still = (0.10, -0.20, 1.20)
        frames = [
            _frame(1.0, {"RElbow": still, "LElbow": None}),
            _frame(2.0, {"RElbow": (0.20, -0.20, 1.20), "LElbow": (0.2, -0.2, 1.2)}),
        ]
        model_rows = [
            _model(still, (0.2, -0.2, 1.2)),
            _model(still, (0.2, -0.2, 1.2)),
        ]
        chosen = choose_fit_frame(frames, model_rows)
        self.assertEqual(frames[chosen]["time_sec"], 2.0)
