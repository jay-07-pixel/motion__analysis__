"""Participant height/mass for live Pose2Sim. No camera."""

from __future__ import annotations

import inspect
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.diagnostics.recorder import LiveDiagnosticsRecorder
from src.markers.coordinator import LiveBodyCoordinator
from src.pose2sim_live.engine import LiveOnnxLstm
from src.pose2sim_live.participant import (
    ParticipantError,
    parse_ui_height_cm,
    parse_ui_mass_kg,
    require_participant_metres_kg,
)


class ParseTest(unittest.TestCase):
    def test_cm_and_kg_convert_for_this_subject(self) -> None:
        self.assertAlmostEqual(parse_ui_height_cm("178"), 1.78)
        self.assertAlmostEqual(parse_ui_height_cm("178,0"), 1.78)
        self.assertAlmostEqual(parse_ui_mass_kg("86"), 86.0)

    def test_empty_and_non_numeric_rejected(self) -> None:
        with self.assertRaises(ParticipantError):
            parse_ui_height_cm("")
        with self.assertRaises(ParticipantError):
            parse_ui_height_cm("  ")
        with self.assertRaises(ParticipantError):
            parse_ui_mass_kg("abc")
        with self.assertRaises(ParticipantError):
            parse_ui_mass_kg(None)

    def test_non_positive_and_implausible_rejected(self) -> None:
        with self.assertRaises(ParticipantError):
            parse_ui_height_cm("0")
        with self.assertRaises(ParticipantError):
            parse_ui_height_cm("-170")
        with self.assertRaises(ParticipantError):
            parse_ui_height_cm("50")
        with self.assertRaises(ParticipantError):
            parse_ui_height_cm("300")
        with self.assertRaises(ParticipantError):
            parse_ui_mass_kg("0")
        with self.assertRaises(ParticipantError):
            parse_ui_mass_kg("10")
        with self.assertRaises(ParticipantError):
            parse_ui_mass_kg("400")

    def test_require_rejects_missing_without_default_person(self) -> None:
        with self.assertRaises(ParticipantError):
            require_participant_metres_kg(None, 86.0)
        with self.assertRaises(ParticipantError):
            require_participant_metres_kg(1.78, None)


class EngineRequiresMeasurementsTest(unittest.TestCase):
    def test_live_onnx_lstm_requires_height_and_mass(self) -> None:
        params = inspect.signature(LiveOnnxLstm.__init__).parameters
        self.assertIs(params["height_m"].default, inspect.Parameter.empty)
        self.assertIs(params["mass_kg"].default, inspect.Parameter.empty)


class CoordinatorAnthroTest(unittest.TestCase):
    def test_enabled_without_measurements_does_not_start_lstm(self) -> None:
        with self.assertRaises(ParticipantError):
            LiveBodyCoordinator(enabled=True)
        with self.assertRaises(ParticipantError):
            LiveBodyCoordinator(enabled=True, height_m=None, mass_kg=86)

    def test_upper_and_lower_receive_the_same_entered_measurements(self) -> None:
        with patch("src.pose2sim_live.engine.LiveUpperLstm") as mock_upper:
            with patch("src.pose2sim_live.engine.LiveLowerLstm") as mock_lower:
                mock_upper.return_value = MagicMock()
                mock_lower.return_value = MagicMock()
                body = LiveBodyCoordinator(enabled=True, height_m=1.78, mass_kg=86.0)
                mock_upper.assert_called_once_with(height_m=1.78, mass_kg=86.0)
                self.assertEqual(body.height_m, 1.78)
                self.assertEqual(body.mass_kg, 86.0)
                body._boot_lower()
                deadline = time.time() + 2.0
                while time.time() < deadline and mock_lower.call_count == 0:
                    time.sleep(0.02)
                mock_lower.assert_called_with(height_m=1.78, mass_kg=86.0)
                self.assertEqual(mock_upper.call_args.kwargs, mock_lower.call_args.kwargs)

    def test_hud_shows_metres_and_kilograms(self) -> None:
        with patch("src.pose2sim_live.engine.LiveUpperLstm") as mock_upper:
            mock_upper.return_value = MagicMock()
            mock_upper.return_value.snapshot.return_value = {
                "status": "buffering",
                "buffered": 0,
                "min_frames": 8,
            }
            body = LiveBodyCoordinator(enabled=True, height_m=1.78, mass_kg=86.0)
            text = " ".join(body.banners())
            self.assertIn("1.78 m", text)
            self.assertIn("86.0 kg", text)


class DiagnosticsAnthroTest(unittest.TestCase):
    def test_summary_records_height_and_mass_used(self) -> None:
        folder = Path(tempfile.mkdtemp()) / "diag"
        rec = LiveDiagnosticsRecorder(folder, participant_height_m=1.78, participant_mass_kg=86.0)
        rec.close()
        payload = json.loads((rec.dir / "summary.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(payload["participant_height_m"], 1.78)
        self.assertAlmostEqual(payload["participant_mass_kg"], 86.0)
        txt = (rec.dir / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("1.78 m", txt)
        self.assertIn("86.0 kg", txt)


if __name__ == "__main__":
    unittest.main()
