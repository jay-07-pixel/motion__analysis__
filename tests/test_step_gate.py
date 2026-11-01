"""A one-frame jump is left out of IK and is not replaced."""

from __future__ import annotations

import unittest

from src.opensim_upper.step_gate import omit_impossible_steps


def _frames(samples: list[tuple[float, tuple[float, float, float] | None]]) -> list[dict]:
    return [
        {"frame": index + 1, "time_sec": time_sec, "points": {"r_lwrist_study": point}}
        for index, (time_sec, point) in enumerate(samples)
    ]


class StepGateTest(unittest.TestCase):
    def test_a_real_step_stays_and_a_jump_is_left_empty(self) -> None:
        original = _frames(
            [
                (0.00, (0.0, 0.0, 1.0)),
                (0.05, (0.12, 0.0, 1.0)),
                (0.10, (0.32, 0.0, 1.0)),
                (0.15, (0.13, 0.0, 1.0)),
                (0.20, (0.14, 0.0, 1.0)),
            ]
        )
        gated, summary = omit_impossible_steps(original)
        self.assertEqual(gated[1]["points"]["r_lwrist_study"], (0.12, 0.0, 1.0))
        self.assertIsNone(gated[2]["points"]["r_lwrist_study"])
        self.assertIsNone(gated[3]["points"]["r_lwrist_study"])
        self.assertEqual(gated[4]["points"]["r_lwrist_study"], (0.14, 0.0, 1.0))
        self.assertEqual(original[2]["points"]["r_lwrist_study"], (0.32, 0.0, 1.0))
        self.assertFalse(summary["replaced_with_previous"])
        self.assertFalse(summary["coordinates_moved"])
        self.assertEqual(summary["omitted"], {"r_lwrist_study": 2})

    def test_the_same_distance_over_a_longer_gap_stays(self) -> None:
        gated, summary = omit_impossible_steps(
            _frames([(0.0, (0.0, 0.0, 1.0)), (0.25, (0.20, 0.0, 1.0))])
        )
        self.assertEqual(gated[1]["points"]["r_lwrist_study"], (0.20, 0.0, 1.0))
        self.assertEqual(summary["omitted"], {})


if __name__ == "__main__":
    unittest.main()
