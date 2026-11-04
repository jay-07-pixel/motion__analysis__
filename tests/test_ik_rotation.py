"""Side-pair twist scoring stays available for fit checks."""

from __future__ import annotations

import unittest

from src.opensim_upper.ik_rotation import score_forearm_rotation, wrist_pair_twist_angles_deg
from src.opensim_upper.ik_tasks import (
    AXIAL_COORDINATES,
    LENGTH_TRACK_COORDINATES,
    axial_refinement_marker_weights,
    distal_rotation_weights,
)


class IkRotationTest(unittest.TestCase):
    def test_distal_weights_raise_wrist_over_shoulder(self) -> None:
        weights = distal_rotation_weights()
        self.assertGreater(weights["r_lwrist_study"], weights["RShoulder"])
        self.assertGreater(weights["L_mwrist_study"], weights["LElbow"])
        axial = axial_refinement_marker_weights()
        self.assertGreater(axial["r_lwrist_study"], weights["r_lwrist_study"])
        self.assertGreater(axial["r_lwrist_study"], axial["r_lelbow_study"])
        self.assertEqual(len(AXIAL_COORDINATES), 4)
        self.assertIn("elbow_flex_r", LENGTH_TRACK_COORDINATES)
        self.assertNotIn("pro_sup_r", LENGTH_TRACK_COORDINATES)

    def test_aligned_pairs_score_near_zero(self) -> None:
        frames = [
            {
                "time_sec": 0.0,
                "points": {
                    "r_lwrist_study": (0.0, 0.0, 0.0),
                    "r_mwrist_study": (0.07, 0.0, 0.0),
                    "L_lwrist_study": (0.0, 0.1, 0.0),
                    "L_mwrist_study": (0.07, 0.1, 0.0),
                    "r_lelbow_study": (0.0, 0.2, 0.0),
                    "r_melbow_study": (0.07, 0.2, 0.0),
                    "L_lelbow_study": (0.0, 0.3, 0.0),
                    "L_melbow_study": (0.07, 0.3, 0.0),
                },
            }
        ]
        rows = [
            {
                "r_lwrist_study_tx": 0.0,
                "r_lwrist_study_ty": 0.0,
                "r_lwrist_study_tz": 0.0,
                "r_mwrist_study_tx": 0.07,
                "r_mwrist_study_ty": 0.0,
                "r_mwrist_study_tz": 0.0,
                "L_lwrist_study_tx": 0.0,
                "L_lwrist_study_ty": 0.1,
                "L_lwrist_study_tz": 0.0,
                "L_mwrist_study_tx": 0.07,
                "L_mwrist_study_ty": 0.1,
                "L_mwrist_study_tz": 0.0,
                "r_lelbow_study_tx": 0.0,
                "r_lelbow_study_ty": 0.2,
                "r_lelbow_study_tz": 0.0,
                "r_melbow_study_tx": 0.07,
                "r_melbow_study_ty": 0.2,
                "r_melbow_study_tz": 0.0,
                "L_lelbow_study_tx": 0.0,
                "L_lelbow_study_ty": 0.3,
                "L_lelbow_study_tz": 0.0,
                "L_melbow_study_tx": 0.07,
                "L_melbow_study_ty": 0.3,
                "L_melbow_study_tz": 0.0,
            }
        ]
        angles = wrist_pair_twist_angles_deg(frames, rows)
        self.assertEqual(angles, [0.0, 0.0])
        scored = score_forearm_rotation(frames, rows, 0.03)
        self.assertLess(scored["median_wrist_pair_twist_deg"], 1.0)
        self.assertLess(scored["score"], score_forearm_rotation(frames, rows, 0.04)["score"])

    def test_twisted_pair_scores_worse(self) -> None:
        frames = [
            {
                "time_sec": 0.0,
                "points": {
                    "r_lwrist_study": (0.0, 0.0, 0.0),
                    "r_mwrist_study": (0.07, 0.0, 0.0),
                    "L_lwrist_study": None,
                    "L_mwrist_study": None,
                    "r_lelbow_study": None,
                    "r_melbow_study": None,
                    "L_lelbow_study": None,
                    "L_melbow_study": None,
                },
            }
        ]
        aligned = [
            {
                "r_lwrist_study_tx": 0.0,
                "r_lwrist_study_ty": 0.0,
                "r_lwrist_study_tz": 0.0,
                "r_mwrist_study_tx": 0.07,
                "r_mwrist_study_ty": 0.0,
                "r_mwrist_study_tz": 0.0,
            }
        ]
        twisted = [
            {
                "r_lwrist_study_tx": 0.0,
                "r_lwrist_study_ty": 0.0,
                "r_lwrist_study_tz": 0.0,
                "r_mwrist_study_tx": 0.0,
                "r_mwrist_study_ty": 0.07,
                "r_mwrist_study_tz": 0.0,
            }
        ]
        good = score_forearm_rotation(frames, aligned, 0.04)
        bad = score_forearm_rotation(frames, twisted, 0.04)
        self.assertLess(good["median_wrist_pair_twist_deg"], 1.0)
        self.assertGreater(bad["median_wrist_pair_twist_deg"], 80.0)
        self.assertLess(good["score"], bad["score"])


if __name__ == "__main__":
    unittest.main()
