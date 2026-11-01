"""Capture readiness, recording start/stop, and analysis refusal. No camera."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pose.keypoints import Keypoint2D
from src.ui.capture_analysis import analyze_capture, describe_results
from src.ui.readiness import capture_readiness
from src.ui.session import MotionSession2D
from src.utils.config_loader import load_config


def _point(name: str, z_m: float | None, confidence: float = 0.9) -> Keypoint2D:
    return Keypoint2D(
        name=name,
        u_px=100,
        v_px=100,
        confidence=confidence,
        coord_frame="camera_3d_metre" if z_m is not None else "camera_2d_pixel",
        x_m=0.0 if z_m is not None else None,
        y_m=0.0 if z_m is not None else None,
        z_m=z_m,
    )


class ReadinessTest(unittest.TestCase):
    def test_uses_configured_depth_range(self) -> None:
        near = capture_readiness(
            [_point("left_shoulder", 0.2), _point("right_shoulder", 0.2)],
            0.5,
            0.3,
            6.0,
        )
        self.assertFalse(near["ready"])
        self.assertIn("Too close", near["instruction"])
        far = capture_readiness(
            [_point("left_shoulder", 7.0), _point("right_shoulder", 7.0)],
            0.5,
            0.3,
            6.0,
        )
        self.assertIn("Too far", far["instruction"])
        ok = capture_readiness(
            [_point("left_shoulder", 1.5), _point("right_shoulder", 1.7)],
            0.5,
            0.3,
            6.0,
        )
        self.assertTrue(ok["ready"])
        self.assertAlmostEqual(ok["distance_m"], 1.6)
        hidden = capture_readiness([_point("left_shoulder", 1.5, confidence=0.1)], 0.5, 0.3, 6.0)
        self.assertFalse(hidden["shoulders_visible"])
        self.assertNotEqual(hidden["instruction"], "0")


class RecordingTest(unittest.TestCase):
    def test_capture_does_not_open_a_second_session_or_overwrite(self) -> None:
        folder = Path(tempfile.mkdtemp())
        cfg = load_config()
        cfg["output"] = dict(cfg["output"])
        cfg["output"]["folder"] = str(folder / "runs")
        cfg["output"]["diagnostics_folder"] = str(folder / "diag")
        cfg["output"]["save_overlay_video"] = False
        session = MotionSession2D(cfg, save_files=False)

        class _Source:
            label = "live"
            device_name = "D455f"
            depth_streaming = True

        session.source = _Source()
        session.begin_recording()
        first = session.run_dir
        session.begin_recording()
        self.assertEqual(session.run_dir, first)
        self.assertTrue(session.recording)
        summary = session.end_recording()
        self.assertFalse(session.recording)
        self.assertTrue(session.source is not None)
        self.assertTrue((Path(summary["run_dir"]) / "run.json").is_file())
        self.assertTrue(Path(summary["diag_dir"]).is_dir())
        session.end_recording()
        self.assertEqual(session.last_capture["run_dir"], summary["run_dir"])


class AnalysisGateTest(unittest.TestCase):
    def test_analysis_does_not_ask_for_the_old_registration(self) -> None:
        folder = Path(tempfile.mkdtemp())
        diag = folder / "capture"
        diag.mkdir()
        (diag / "landmarks.csv").write_text("frame,landmark\n", encoding="utf-8")
        result = analyze_capture(diag, folder)
        self.assertFalse(result["ok"])
        self.assertFalse(result["opensim_ran"])
        self.assertFalse(result["clinical_accuracy"])
        self.assertFalse(result["registration_applied"])
        self.assertNotIn("registration.json", result["message"])
        self.assertFalse((folder / "output").exists())
        text = describe_results(result)
        self.assertIn("Right shoulder flexion: unavailable", text)
        self.assertNotIn("Right shoulder flexion: 0.00", text)

    def test_zero_degrees_stays_zero(self) -> None:
        folder = Path(tempfile.mkdtemp())
        analysis = folder / "analysis"
        analysis.mkdir()
        payload = {
            "frames": [
                {
                    "time_sec": 0.0,
                    "solved": {
                        "arm_flex_r": {"degrees": 0.0, "reported_as_measurement": True},
                    },
                }
            ]
        }
        (analysis / "joint_angles.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
        text = describe_results(
            {
                "ok": True,
                "message": "Inverse kinematics finished.",
                "analysis_dir": str(analysis),
                "clinical_accuracy": False,
            }
        )
        self.assertIn("Right shoulder flexion: 0.00°", text)
        self.assertIn("Right humeral rotation: unavailable", text)
        self.assertIn("Joint-angle timeline: unavailable", text)


class DashboardSmokeTest(unittest.TestCase):
    def test_window_has_the_capture_controls(self) -> None:
        import tkinter as tk

        from src.ui.app import MotionAnalysisApp

        root = tk.Tk()
        root.withdraw()
        try:
            app = MotionAnalysisApp(root)
            root.update_idletasks()
            self.assertEqual(app.connect_btn["text"], "CONNECT CAMERA")
            self.assertEqual(app.capture_btn["text"], "START CAPTURE")
            self.assertEqual(app.stop_capture_btn["text"], "STOP CAPTURE")
            self.assertEqual(app.analyze_btn["text"], "RUN 3D ANALYSIS")
            self.assertEqual(app.results_btn["text"], "VIEW RESULTS")
            self.assertEqual(str(app.capture_btn["state"]), "disabled")
            self.assertEqual(str(app.analyze_btn["state"]), "disabled")
            self.assertEqual(str(app.connect_btn["state"]), "normal")
            self.assertEqual(app.camera_status_var.get(), "Camera disconnected")
            texts = []

            def _walk(widget) -> None:
                try:
                    texts.append(str(widget.cget("text")))
                except tk.TclError:
                    pass
                for child in widget.winfo_children():
                    _walk(child)

            _walk(app.root)
            joined = "\n".join(texts)
            for removed in ("Day", "Night", "Start Diagnostics", "Upload", "Pose2Sim"):
                self.assertNotIn(removed, joined)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
