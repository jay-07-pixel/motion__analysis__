"""Capture dashboard.

Connect the RealSense, position the participant, record RGB, depth, and
landmarks, then run the existing upper-body analysis on that capture.
"""

from __future__ import annotations

import copy
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.regions import apply_region, region_id
from src.opensim_upper.frame_calibration import FrameCalibration
from src.ui.capture_analysis import analyze_capture, describe_results
from src.ui.frame_calibration_dialog import FrameCalibrationDialog
from src.ui.live_charts import JOINT_LABELS, MotionDashboard
from src.ui.live_markers import LiveMarkerWindow
from src.ui.pose2sim_playback import Pose2SimPlaybackWindow
from src.pose2sim_live.participant import ParticipantError, parse_ui_height_cm, parse_ui_mass_kg
from src.ui.session import MotionSession2D
from src.ui.theme import PAL, use_night
from src.utils.config_loader import load_config, resolve_project_path


class MotionAnalysisApp:
    """Main window. Worker thread owns MediaPipe; this class only draws widgets."""

    def __init__(self, root: tk.Tk) -> None:
        """Build the layout from config (title, default live/file)."""
        self.root = root
        self.config = load_config()
        project = self.config["project"]
        self.root.title(str(project["title"]))
        self.root.minsize(1360, 800)
        self.root.geometry("1520x900")
        self.root.configure(bg=PAL.bg)

        self._worker: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_bgr = None
        self._latest_angles: dict[str, float | None] = {}
        self._latest_gauge_display: dict[str, float | None] = {}
        self._latest_angular: dict[str, float | None] = {}
        self._latest_linear: dict[str, float | None] = {}
        self._latest_readout: list[dict] = []
        self._latest_time = 0.0
        self._latest_index = -1
        self._latest_keypoints = []
        self._latest_fps = 0.0
        self._latest_space = "2d"
        self._latest_min_vis = 0.5
        self._latest_lstm_estimates = []
        self._session_events: queue.Queue = queue.Queue()
        self._native_wh = (1280, 720)
        self._drawn_wh = (1280, 720)
        self._marker_window: LiveMarkerWindow | None = None
        self._p2s_window: Pose2SimPlaybackWindow | None = None
        self.frame_calibration = FrameCalibration()
        self._frame_dialog: FrameCalibrationDialog | None = None
        self._frame_click_role: str | None = None
        self._pending_frame_click: dict | None = None
        self._chart_index = -1
        self._want_summary = False
        self._report: tk.Toplevel | None = None
        self._report_stats: dict | None = None
        self._status_from_worker = ""
        self._photo = None
        self._running = False
        self._error_dialog_shown = False
        self._diag_recording = False
        self._diag_wanted = False
        self._diag_from_worker = ""
        self._live_cfg: dict | None = None
        self._live_snap: dict = {}
        self._saved_capture: dict = {}
        self._analysis_result: dict | None = None
        self._analyzing = False
        self._phase = "disconnected"
        self.dashboard = None

        self.mode_var = tk.StringVar(value=str(self.config["source"]["mode"]))
        self.region_var = tk.StringVar(value=region_id(self.config))
        self.space_var = tk.StringVar(
            value=str((self.config.get("analysis") or {}).get("space", "2d")).lower()
        )
        self.path_var = tk.StringVar(value=str(self.config["source"]["file_path"]))
        self.save_var = tk.BooleanVar(value=True)
        self.lstm_var = tk.BooleanVar(value=True)
        self.height_cm_var = tk.StringVar(value="")
        self.mass_kg_var = tk.StringVar(value="")
        display_cfg = self.config.get("display") or {}
        self.foot_safety_v = tk.IntVar(value=int(display_cfg.get("foot_safety_line_v_px", 640)))
        self.status_var = tk.StringVar(value="Camera disconnected")
        self.badge_var = tk.StringVar(value="DISCONNECTED")
        self.meta_var = tk.StringVar(value="")
        self.theme_var = tk.StringVar(value="night")
        self._badge_role = "idle"
        self.gauge_step = float((self.config.get("analysis") or {}).get("angle_gauge_step_deg", 45))

        use_night(True)
        self._build_workflow()
        self._build_status()
        self._refresh_header_meta()
        self._paint_badge("idle")
        self._refresh_actions()
        self._tick()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_workflow(self) -> None:
        """Buttons across the top, camera on the left, angles on the right."""
        project = self.config["project"]
        header = tk.Frame(self.root, bg=PAL.panel, height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        self._header = header
        left = tk.Frame(header, bg=PAL.panel)
        left.pack(side=tk.LEFT, padx=20, pady=8)
        self._header_left = left
        self._title_label = tk.Label(
            left,
            text=str(project["title"]),
            bg=PAL.panel,
            fg=PAL.text,
            font=("Segoe UI", 16, "bold"),
        )
        self._title_label.pack(anchor="w")
        self._meta_label = tk.Label(
            left,
            textvariable=self.meta_var,
            bg=PAL.panel,
            fg=PAL.muted,
            font=("Segoe UI", 9),
        )
        self._meta_label.pack(anchor="w")
        self._badge = tk.Label(
            header,
            textvariable=self.badge_var,
            bg=PAL.idle,
            fg=PAL.text,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
        )
        self._badge.pack(side=tk.RIGHT, padx=20)

        bar = tk.Frame(self.root, bg=PAL.panel)
        bar.pack(fill=tk.X, padx=16, pady=(10, 0))
        self._toolbar = bar
        actions = tk.Frame(bar, bg=PAL.panel)
        actions.pack(side=tk.LEFT, fill=tk.Y)
        self.connect_btn = self._action_button(actions, "CONNECT CAMERA", self._on_connect, PAL.accent, horizontal=True)
        self.capture_btn = self._action_button(actions, "START CAPTURE", self._on_start_capture, PAL.start, horizontal=True)
        self.stop_capture_btn = self._action_button(actions, "STOP CAPTURE", self._on_stop_capture, PAL.stop, horizontal=True)
        self.analyze_btn = self._action_button(actions, "RUN 3D ANALYSIS", self._on_analyze, PAL.accent, horizontal=True)
        self.results_btn = self._action_button(actions, "VIEW RESULTS", self._on_view_results, PAL.idle, horizontal=True)

        anthro = tk.Frame(bar, bg=PAL.panel)
        anthro.pack(side=tk.RIGHT, padx=(16, 8))
        self._anthro_frame = anthro
        self._height_label = tk.Label(anthro, text="Height (cm)", bg=PAL.panel, fg=PAL.muted, font=("Segoe UI", 9))
        self._height_label.grid(row=0, column=0, sticky="e", padx=(0, 6))
        self._height_entry = tk.Entry(anthro, textvariable=self.height_cm_var, width=6, font=("Segoe UI", 11))
        self._height_entry.grid(row=0, column=1, sticky="w")
        self._mass_label = tk.Label(anthro, text="Weight (kg)", bg=PAL.panel, fg=PAL.muted, font=("Segoe UI", 9))
        self._mass_label.grid(row=0, column=2, sticky="e", padx=(14, 6))
        self._mass_entry = tk.Entry(anthro, textvariable=self.mass_kg_var, width=6, font=("Segoe UI", 11))
        self._mass_entry.grid(row=0, column=3, sticky="w")

        self.camera_status_var = tk.StringVar(value="Camera disconnected")
        self.camera_model_var = tk.StringVar(value="Model unavailable")
        self.depth_status_var = tk.StringVar(value="Depth stream off")
        self.stream_var = tk.StringVar(value="Resolution and FPS appear after connect")
        self.position_var = tk.StringVar(value="Connect the camera, then step into view.")
        self.distance_var = tk.StringVar(value="Distance unavailable")
        self.record_var = tk.StringVar(value="Not recording")
        self.elapsed_var = tk.StringVar(value="Elapsed  00:00")
        self.frames_var = tk.StringVar(value="Frames  0")
        self.fps_var = tk.StringVar(value="FPS  —")
        self.landmark_var = tk.StringVar(value="Landmarks  not visible")
        self.analysis_var = tk.StringVar(value="Analysis has not been run.")
        info = tk.Frame(self.root, bg=PAL.bg)
        info.pack(fill=tk.X, padx=16, pady=(8, 0))
        self._info = info

        def _info_row(variables: tuple) -> None:
            row = tk.Frame(info, bg=PAL.bg)
            row.pack(fill=tk.X)
            for var in variables:
                tk.Label(row, textvariable=var, bg=PAL.bg, fg=PAL.muted, font=("Segoe UI", 9), anchor="w").pack(
                    side=tk.LEFT, padx=(0, 14)
                )

        _info_row((self.camera_status_var, self.camera_model_var, self.depth_status_var, self.stream_var))
        _info_row((
            self.position_var,
            self.distance_var,
            self.record_var,
            self.elapsed_var,
            self.frames_var,
            self.fps_var,
            self.landmark_var,
        ))

        body = tk.Frame(self.root, bg=PAL.bg)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
        self._body = body
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0, minsize=460)
        body.grid_rowconfigure(0, weight=1)

        video_frame = tk.Frame(body, bg=PAL.video, highlightbackground=PAL.line, highlightthickness=1)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._video_frame = video_frame
        self.video_label = tk.Label(
            video_frame,
            bg=PAL.video,
            fg=PAL.muted,
            text="Connect the camera to see the participant.",
            font=("Segoe UI", 12),
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind("<Button-1>", self._on_video_click)

        side = tk.Frame(body, bg=PAL.bg)
        side.grid(row=0, column=1, sticky="nsew")
        self._side = side
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=0)

        self._dash_host = tk.Frame(side, bg=PAL.bg)
        self._dash_host.grid(row=0, column=0, sticky="nsew")
        self._dashboard_compact = False
        self._dashboard_column = True
        self.dashboard = MotionDashboard(
            self._dash_host,
            self._analysis_for_ui(),
            on_step=self._remember_gauge_step,
            compact=False,
            column=True,
        )
        self.dashboard.pack(fill=tk.BOTH, expand=True)

        self._performance_enabled = bool((self.config.get("display") or {}).get("performance_diagnostics", True))
        self._display_fps = 0.0
        self._ui_display_ms = 0.0
        self._display_stamp = None
        self._display_index = -1
        foot = tk.Frame(side, bg=PAL.panel, height=44)
        foot.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        foot.grid_propagate(False)
        self._status_foot = foot
        tk.Label(
            foot,
            textvariable=self.analysis_var,
            bg=PAL.panel,
            fg=PAL.text,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=430,
        ).pack(fill=tk.X, padx=10, pady=8)
        self.perf_var = tk.StringVar(
            value="Timing appears after the camera connects." if self._performance_enabled else ""
        )

    def _action_button(
        self,
        parent: tk.Frame,
        text: str,
        command,
        colour: str,
        horizontal: bool = False,
    ) -> tk.Button:
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bd=0,
            padx=14,
            pady=8,
            font=("Segoe UI", 11, "bold"),
            bg=colour,
            fg="white" if colour != PAL.panel else PAL.text,
            activeforeground="white",
            cursor="hand2",
        )
        if horizontal:
            button.pack(side=tk.LEFT, padx=(0, 8), pady=8)
        else:
            button.pack(fill=tk.X, padx=12, pady=4)
        return button

    def _build_header(self, project: dict) -> None:
        """Top strip: project name and camera model from config.yaml."""
        header = tk.Frame(self.root, bg=PAL.panel, height=72)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        self._header = header
        left = tk.Frame(header, bg=PAL.panel)
        left.pack(side=tk.LEFT, padx=20, pady=10)
        self._header_left = left
        self._title_label = tk.Label(
            left,
            text=str(project["title"]),
            bg=PAL.panel,
            fg=PAL.text,
            font=("Segoe UI", 16, "bold"),
        )
        self._title_label.pack(anchor="w")
        self._meta_label = tk.Label(
            left,
            textvariable=self.meta_var,
            bg=PAL.panel,
            fg=PAL.muted,
            font=("Segoe UI", 9),
        )
        self._meta_label.pack(anchor="w")
        badge = tk.Label(
            header,
            textvariable=self.badge_var,
            bg=PAL.idle,
            fg=PAL.text,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
        )
        badge.pack(side=tk.RIGHT, padx=20)
        self._badge = badge
        theme_wrap = tk.Frame(header, bg=PAL.panel)
        theme_wrap.pack(side=tk.RIGHT, padx=(0, 8))
        self._theme_wrap = theme_wrap
        self._theme_btns: dict[str, tk.Button] = {}
        for key, text in (("day", "  Day  "), ("night", "  Night  ")):
            btn = tk.Button(
                theme_wrap,
                text=text,
                command=lambda k=key: self._set_theme(k),
                bd=0,
                padx=12,
                pady=6,
                font=("Segoe UI", 10, "bold"),
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=4)
            self._theme_btns[key] = btn
        space_wrap = tk.Frame(header, bg=PAL.panel)
        space_wrap.pack(side=tk.RIGHT, padx=(0, 8))
        self._space_wrap = space_wrap
        self._space_btns: dict[str, tk.Button] = {}
        for key, text in (("2d", "  2D  "), ("3d", "  3D  ")):
            btn = tk.Button(
                space_wrap,
                text=text,
                command=lambda k=key: self._set_space(k),
                bd=0,
                padx=12,
                pady=6,
                font=("Segoe UI", 10, "bold"),
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=4)
            self._space_btns[key] = btn

    def _build_controls(self) -> None:
        """Source, region picker, save, Start/Stop — one row under the header."""
        bar = tk.Frame(self.root, bg=PAL.bg)
        bar.pack(fill=tk.X, padx=16, pady=12)
        self._controls = bar

        self.live_btn = tk.Button(
            bar,
            text="  Live camera  ",
            command=lambda: self._set_mode("live"),
            bd=0,
            padx=12,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.live_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.file_btn = tk.Button(
            bar,
            text="  Upload file  ",
            command=lambda: self._set_mode("file"),
            bd=0,
            padx=12,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.file_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.p2s_btn = tk.Button(
            bar,
            text="  Pose2Sim playback  ",
            command=self._open_pose2sim_playback,
            bd=0,
            padx=12,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.p2s_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._browse_btn = tk.Button(
            bar,
            text=" Browse ",
            command=self._browse,
            bg=PAL.idle,
            fg=PAL.text,
            bd=0,
            padx=12,
            pady=8,
            font=("Segoe UI", 10),
            cursor="hand2",
            activebackground=PAL.line,
            activeforeground=PAL.text,
        )
        self._browse_btn.pack(side=tk.LEFT, padx=(0, 12))

        self._analyse_label = tk.Label(
            bar,
            text="ANALYSE",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 9, "bold"),
        )
        self._analyse_label.pack(side=tk.LEFT, padx=(0, 8))
        region_wrap = tk.Frame(bar, bg=PAL.bg)
        region_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._region_wrap = region_wrap
        self._region_btns: dict[str, tk.Button] = {}
        regions = (self.config.get("analysis") or {}).get("regions") or {}
        for key, spec in regions.items():
            label = str((spec or {}).get("label", key))
            btn = tk.Button(
                region_wrap,
                text=f"  {label}  ",
                command=lambda k=key: self._set_region(k),
                bd=0,
                padx=10,
                pady=8,
                font=("Segoe UI", 10, "bold"),
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=(0, 6))
            self._region_btns[key] = btn

        run_col = tk.Frame(bar, bg=PAL.bg)
        run_col.pack(side=tk.RIGHT, padx=(12, 0))
        self.save_check = tk.Checkbutton(
            run_col,
            text="Save CSV + video",
            variable=self.save_var,
            bg=PAL.bg,
            fg=PAL.text,
            selectcolor=PAL.card,
            activebackground=PAL.bg,
            activeforeground=PAL.text,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.save_check.pack(anchor="w")
        anthro = tk.Frame(run_col, bg=PAL.bg)
        anthro.pack(fill=tk.X, pady=(6, 0))
        self._anthro_frame = anthro
        self._height_label = tk.Label(
            anthro,
            text="Height (cm)",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self._height_label.grid(row=0, column=0, sticky="w")
        self._height_entry = tk.Entry(
            anthro,
            textvariable=self.height_cm_var,
            width=7,
            font=("Segoe UI", 9),
        )
        self._height_entry.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self._mass_label = tk.Label(
            anthro,
            text="Weight (kg)",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self._mass_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._mass_entry = tk.Entry(
            anthro,
            textvariable=self.mass_kg_var,
            width=7,
            font=("Segoe UI", 9),
        )
        self._mass_entry.grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(4, 0))
        anthro.grid_columnconfigure(0, weight=1)
        self.run_btn = tk.Button(
            run_col,
            text="  Start  ",
            command=self._on_run_toggle,
            bg=PAL.start,
            fg="white",
            bd=0,
            padx=28,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            activebackground="#26855a",
            activeforeground="white",
        )
        self.run_btn.pack(fill=tk.X, pady=(6, 0))
        self.start_btn = self.run_btn
        self.stop_btn = self.run_btn
        self.diag_btn = tk.Button(
            run_col,
            text="  Start Diagnostics  ",
            command=self._on_diag_toggle,
            bg=PAL.card,
            fg=PAL.text,
            bd=0,
            padx=12,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            activebackground=PAL.line,
            activeforeground=PAL.text,
        )
        self.diag_btn.pack(fill=tk.X, pady=(6, 0))
        foot_row = tk.Frame(run_col, bg=PAL.bg)
        foot_row.pack(fill=tk.X, pady=(6, 0))
        self._foot_safety_row = foot_row
        self._foot_safety_label = tk.Label(
            foot_row,
            text="Foot safety v",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self._foot_safety_label.pack(side=tk.LEFT)
        self._foot_safety_spin = tk.Spinbox(
            foot_row,
            from_=0,
            to=719,
            width=5,
            textvariable=self.foot_safety_v,
            font=("Segoe UI", 9),
            command=self._sync_foot_safety_v,
        )
        self._foot_safety_spin.pack(side=tk.RIGHT)
        self.foot_safety_v.trace_add("write", lambda *_: self._sync_foot_safety_v())
        self.lstm_check = tk.Checkbutton(
            bar,
            text="Live Pose2Sim",
            variable=self.lstm_var,
            bg=PAL.bg,
            fg=PAL.text,
            selectcolor=PAL.card,
            activebackground=PAL.bg,
            activeforeground=PAL.text,
            font=("Segoe UI", 10),
        )
        self.lstm_check.pack(side=tk.RIGHT, padx=8)
        self.toe_btn = tk.Button(
            bar,
            text="  Calibrate toes  ",
            command=self._on_calibrate_toes,
            bg=PAL.card,
            fg=PAL.text,
            bd=0,
            padx=10,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.toe_btn.pack(side=tk.RIGHT, padx=8)
        self.frame_btn = tk.Button(
            bar,
            text="  Frame calibration  ",
            command=self._open_frame_calibration,
            bg=PAL.card,
            fg=PAL.text,
            bd=0,
            padx=10,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.frame_btn.pack(side=tk.RIGHT, padx=8)

    def _set_space(self, space: str) -> None:
        """Switch 2D pixels vs 3D camera metres. Locked while running."""
        if self._running:
            return
        self.space_var.set(space)
        self._refresh_space_buttons()
        self._refresh_header_meta()
        self._rebuild_dashboard()

    def _refresh_space_buttons(self) -> None:
        """Highlight 2D or 3D."""
        current = self.space_var.get()
        for key, btn in self._space_btns.items():
            on = key == current
            btn.configure(
                bg=PAL.accent if on else PAL.idle,
                fg="white" if on else PAL.text,
                activebackground=PAL.accent if on else PAL.line,
                activeforeground="white" if on else PAL.text,
                state=tk.DISABLED if self._running else tk.NORMAL,
            )

    def _refresh_header_meta(self) -> None:
        """Subtitle: student, camera, and whether we are in 2D pixels or 3D metres."""
        project = self.config.get("project") or {}
        if self.space_var.get() == "3d":
            coord = "3D camera metres  ·  origin = optical centre"
        else:
            coord = "2D camera pixels"
        self.meta_var.set(
            f"{project.get('student', '')}  ·  {project.get('camera_model', '')}  ·  {coord}"
        )
        self.root.title(f"{project.get('title', 'Motion Analysis')}  —  {self.space_var.get().upper()}")

    def _set_region(self, region: str) -> None:
        """Switch region and rebuild gauges. Ignored during a take."""
        if self._running:
            return
        self.region_var.set(region)
        self._refresh_region_buttons()
        self._rebuild_dashboard()

    def _refresh_region_buttons(self) -> None:
        """Highlight the selected region pill."""
        current = self.region_var.get()
        for key, btn in self._region_btns.items():
            on = key == current
            btn.configure(
                bg=PAL.accent if on else PAL.idle,
                fg="white" if on else PAL.text,
                activebackground=PAL.accent if on else PAL.line,
                activeforeground="white" if on else PAL.text,
                state=tk.DISABLED if self._running else tk.NORMAL,
            )

    def _analysis_for_ui(self) -> dict:
        """Analysis dict after applying the selected region."""
        cfg = apply_region(copy.deepcopy(self.config), self.region_var.get())
        cfg["analysis"]["space"] = self.space_var.get()
        cfg["analysis"]["angle_gauge_step_deg"] = self.gauge_step
        return cfg["analysis"]

    def _rebuild_dashboard(self, keep_summary: bool = False) -> None:
        """Swap the right-hand dials to match the selected region."""
        held = self.dashboard.capture_summary() if keep_summary else None
        self.dashboard.destroy()
        self.dashboard = MotionDashboard(
            self._dash_host,
            self._analysis_for_ui(),
            on_step=self._remember_gauge_step,
            compact=getattr(self, "_dashboard_compact", False),
            column=getattr(self, "_dashboard_column", False),
        )
        self.dashboard.pack(fill=tk.BOTH, expand=True)
        if held is not None:
            self.dashboard.restore_summary(held)

    def _set_theme(self, theme: str) -> None:
        """Switch the window between the light Day theme and the dark Night theme."""
        if theme == self.theme_var.get():
            return
        self.theme_var.set(theme)
        use_night(theme == "night")
        self._apply_theme()

    def _paint_badge(self, role: str | None = None) -> None:
        """Status pill. role is idle, accent, start, or stop."""
        if role is not None:
            self._badge_role = role
        colours = {
            "idle": PAL.idle,
            "accent": PAL.accent,
            "start": PAL.start,
            "stop": PAL.stop,
        }
        self._badge.configure(bg=colours.get(self._badge_role, PAL.idle), fg=PAL.text)

    def _refresh_theme_buttons(self) -> None:
        """Highlight Day or Night, whichever is showing."""
        current = self.theme_var.get()
        for key, btn in self._theme_btns.items():
            on = key == current
            btn.configure(
                bg=PAL.accent if on else PAL.idle,
                fg="white" if on else PAL.text,
                activebackground=PAL.accent if on else PAL.line,
                activeforeground="white" if on else PAL.text,
            )

    def _apply_theme(self) -> None:
        """Repaint chrome, dials, and coordinate cards in the current palette."""
        self.root.configure(bg=PAL.bg)
        for frame in (
            self._header,
            self._header_left,
            self._theme_wrap,
            self._space_wrap,
            self._status_bar,
        ):
            frame.configure(bg=PAL.panel)
        self._title_label.configure(bg=PAL.panel, fg=PAL.text)
        self._meta_label.configure(bg=PAL.panel, fg=PAL.muted)
        self._paint_badge()
        for frame in (
            self._controls,
            self._region_wrap,
            self._body,
            self._left_col,
            self._readout_bar,
            self._dash_host,
        ):
            frame.configure(bg=PAL.bg)
        self._analyse_label.configure(bg=PAL.bg, fg=PAL.muted)
        self._status_label.configure(bg=PAL.panel, fg=PAL.muted)
        self._browse_btn.configure(
            bg=PAL.idle,
            fg=PAL.text,
            activebackground=PAL.line,
            activeforeground=PAL.text,
        )
        self.p2s_btn.configure(
            bg=PAL.idle,
            fg=PAL.text,
            activebackground=PAL.line,
            activeforeground=PAL.text,
        )
        self.save_check.configure(
            bg=PAL.bg,
            fg=PAL.text,
            selectcolor=PAL.card,
            activebackground=PAL.bg,
            activeforeground=PAL.text,
        )
        self.lstm_check.configure(
            bg=PAL.bg,
            fg=PAL.text,
            selectcolor=PAL.card,
            activebackground=PAL.bg,
            activeforeground=PAL.text,
        )
        self._foot_safety_row.configure(bg=PAL.bg)
        self._foot_safety_label.configure(bg=PAL.bg, fg=PAL.muted)
        self._anthro_frame.configure(bg=PAL.bg)
        self._height_label.configure(bg=PAL.bg, fg=PAL.muted)
        self._mass_label.configure(bg=PAL.bg, fg=PAL.muted)
        self._video_frame.configure(bg=PAL.video, highlightbackground=PAL.line)
        self.video_label.configure(bg=PAL.video, fg=PAL.muted)
        try:
            if self._readout_placeholder.winfo_exists():
                self._readout_placeholder.configure(bg=PAL.bg, fg=PAL.muted)
        except tk.TclError:
            pass
        self._readout_key = ("theme",)
        self._refresh_theme_buttons()
        self._refresh_mode_buttons()
        self._refresh_region_buttons()
        self._refresh_space_buttons()
        self._set_running_buttons(self._running)
        self._rebuild_dashboard(keep_summary=True)
        if self._report is not None and self._report_stats is not None:
            try:
                still_open = bool(self._report.winfo_exists())
            except tk.TclError:
                still_open = False
            if still_open:
                self._show_report(self._report_stats)

    def _build_body(self) -> None:
        """Split screen: camera left, Left/Right body speedometers right."""
        body = tk.Frame(self.root, bg=PAL.bg)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))
        self._body = body
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left_col = tk.Frame(body, bg=PAL.bg)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._left_col = left_col
        left_col.grid_rowconfigure(1, weight=1)
        left_col.grid_columnconfigure(0, weight=1)

        self._readout_bar = tk.Frame(left_col, bg=PAL.bg)
        self._readout_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._readout_key: tuple[str, ...] = ()
        self._readout_vars: dict[str, tk.StringVar] = {}
        self._readout_placeholder = tk.Label(
            self._readout_bar,
            text="Joint coordinates appear here after Start",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self._readout_placeholder.pack(fill=tk.X, pady=4)

        video_frame = tk.Frame(left_col, bg=PAL.video, highlightbackground=PAL.line, highlightthickness=1)
        video_frame.grid(row=1, column=0, sticky="nsew")
        self._video_frame = video_frame
        self.video_label = tk.Label(
            video_frame,
            bg=PAL.video,
            fg=PAL.muted,
            text="Press Start  ·  camera on this side, speedometers on the other",
            font=("Segoe UI", 12),
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind("<Button-1>", self._on_video_click)

        self._dash_host = tk.Frame(body, bg=PAL.bg)
        self._dash_host.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.dashboard = MotionDashboard(
            self._dash_host, self._analysis_for_ui(), on_step=self._remember_gauge_step
        )
        self.dashboard.pack(fill=tk.BOTH, expand=True)

    def _remember_gauge_step(self, step: float) -> None:
        """Keep the dial scale when the region or theme rebuilds the dials."""
        self.gauge_step = float(step)

    def _build_status(self) -> None:
        """Bottom line: idle / running / saved path."""
        bar = tk.Frame(self.root, bg=PAL.panel)
        bar.pack(fill=tk.X)
        self._status_bar = bar
        self._status_label = tk.Label(
            bar,
            textvariable=self.status_var,
            bg=PAL.panel,
            fg=PAL.muted,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self._status_label.pack(fill=tk.X, padx=16, pady=8)

    def _set_mode(self, mode: str) -> None:
        """Switch Live vs Upload from the two pill buttons."""
        self.mode_var.set(mode)
        self._refresh_mode_buttons()

    def _refresh_mode_buttons(self) -> None:
        """Highlight the selected source pill."""
        live = self.mode_var.get() == "live"
        self.live_btn.configure(
            bg=PAL.accent if live else PAL.idle,
            fg="white" if live else PAL.text,
            activebackground=PAL.accent if live else PAL.line,
            activeforeground="white" if live else PAL.text,
        )
        self.file_btn.configure(
            bg=PAL.accent if not live else PAL.idle,
            fg="white" if not live else PAL.text,
            activebackground=PAL.accent if not live else PAL.line,
            activeforeground="white" if not live else PAL.text,
        )
        self.p2s_btn.configure(
            bg=PAL.idle,
            fg=PAL.text,
            activebackground=PAL.line,
            activeforeground=PAL.text,
        )

    def _open_pose2sim_playback(self) -> None:
        """Replay recorded Pose2Sim estimates. Not live camera inference."""
        if self._p2s_window is not None:
            try:
                if self._p2s_window.win.winfo_exists():
                    self._p2s_window.win.lift()
                    return
            except Exception:
                pass
        try:
            self._p2s_window = Pose2SimPlaybackWindow(self.root)
        except FileNotFoundError as error:
            messagebox.showerror("Pose2Sim playback", f"Missing recording or estimate file:\n{error}")
        except Exception as error:
            messagebox.showerror("Pose2Sim playback", str(error))

    def _on_calibrate_toes(self) -> None:
        """Start the four-click live toe capture for the lower LSTM."""
        if not self._running:
            messagebox.showinfo(
                "Calibrate toes",
                "Start a 3D live session first.\n\n"
                "Then step closer so both feet are in view and click:\n"
                "1. Left big toe\n2. Left small toe\n3. Right big toe\n4. Right small toe\n\n"
                "Clicks use RealSense depth. MediaPipe foot_index is not used as a toe.",
            )
            return
        self.status_var.set("Calibrate toes: click LEFT big toe on the video")

    def attach_registration(self, solved: dict) -> str | None:
        """Copy a completed calibration into the capture that is already saved."""
        if not solved.get("complete"):
            return None
        diag = str((self._saved_capture or {}).get("diag_dir") or "")
        if not diag:
            return None
        from src.opensim_upper.workflow import write_capture_registration

        path = write_capture_registration(Path(diag), solved)
        return str(path)

    def _open_frame_calibration(self) -> None:
        """Open the camera-to-model calibration window. Does not label the video."""
        if self._frame_dialog is not None:
            try:
                if self._frame_dialog.win.winfo_exists():
                    self._frame_dialog.win.lift()
                    self._frame_dialog.refresh()
                    return
            except Exception:
                pass
        self._frame_dialog = FrameCalibrationDialog(self)

    def arm_frame_click(self, role: str) -> None:
        """The next video click records one calibration reference, not a toe."""
        prompts = {
            "anterior_from": "Frame calibration: click the REAR end of the facing rod",
            "anterior_to": "Frame calibration: click the FORWARD end of the facing rod",
            "origin": "Frame calibration: click the surveyed reference point",
        }
        self._frame_click_role = role
        self.status_var.set(prompts.get(role, "Frame calibration: click the video"))

    def _apply_frame_click(self, click: dict) -> None:
        role = str(click.get("role") or "")
        if click.get("error") or click.get("xyz") is None:
            self.status_var.set(str(click.get("error") or "Frame click was not recorded."))
            return
        try:
            if role == "origin":
                self.frame_calibration.set_origin_camera(click["xyz"])
            elif role in ("anterior_from", "anterior_to"):
                self.frame_calibration.set_anterior_point(role, click["xyz"])
            else:
                self.status_var.set("Frame click had no role.")
                return
        except ValueError as error:
            self.status_var.set(str(error))
            return
        self._frame_click_role = None
        self.status_var.set(f"Recorded {role.replace('_', ' ')} in camera metres.")
        if self._frame_dialog is not None:
            try:
                if self._frame_dialog.win.winfo_exists():
                    self._frame_dialog.refresh()
            except Exception:
                pass

    def _on_video_click(self, event) -> None:
        """Map a click on the displayed overlay back to native colour pixels."""
        if not self._running:
            return
        native_w, native_h = self._native_wh
        drawn_w, drawn_h = self._drawn_wh
        if drawn_w < 1 or drawn_h < 1:
            return
        label_w = max(1, self.video_label.winfo_width())
        label_h = max(1, self.video_label.winfo_height())
        origin_x = (label_w - drawn_w) // 2
        origin_y = (label_h - drawn_h) // 2
        x = event.x - origin_x
        y = event.y - origin_y
        if x < 0 or y < 0 or x >= drawn_w or y >= drawn_h:
            return
        u = x * native_w / drawn_w
        v = y * native_h / drawn_h
        if self._frame_click_role:
            self._session_events.put(
                {"kind": "frame_ref", "role": self._frame_click_role, "u": float(u), "v": float(v)}
            )
            return
        self._session_events.put({"kind": "toe_click", "u": float(u), "v": float(v)})

    def _browse(self) -> None:
        """Pick an mp4/avi/mov/bag. Switches source to file."""
        path = filedialog.askopenfilename(
            title="Upload video or RealSense .bag",
            filetypes=[
                ("Video or bag", "*.mp4 *.avi *.mov *.mkv *.bag"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.path_var.set(path)
            self._set_mode("file")
            self.status_var.set(f"File ready  ·  {Path(path).name}")

    def _on_diag_toggle(self) -> None:
        """Start/stop diagnostic CSV for the live camera without changing DOF."""
        if self._diag_recording:
            if self._running:
                self._session_events.put({"kind": "diag_stop"})
            self._diag_recording = False
            self._set_diag_button(False)
            self.status_var.set("Diagnostics stopping…")
            return
        if self.space_var.get() != "3d":
            self._set_space("3d")
        self._diag_recording = True
        self._set_diag_button(True)
        if self._running:
            self._session_events.put({"kind": "diag_start"})
            self.status_var.set("Diagnostics recording…")
            return
        self._diag_wanted = True
        self._on_start()

    def _set_diag_button(self, recording: bool) -> None:
        if recording:
            self.diag_btn.configure(
                text="  Stop Diagnostics  ",
                bg=PAL.stop,
                fg="white",
                activebackground="#a83a3a",
                activeforeground="white",
            )
        else:
            self.diag_btn.configure(
                text="  Start Diagnostics  ",
                bg=PAL.card,
                fg=PAL.text,
                activebackground=PAL.line,
                activeforeground=PAL.text,
            )

    def _sync_foot_safety_v(self) -> None:
        """Copy the foot-safety row into config. Preview-only; no landmark change."""
        try:
            v = int(self.foot_safety_v.get())
        except (tk.TclError, ValueError, TypeError):
            return
        v = max(0, min(719, v))
        self.config.setdefault("display", {})["foot_safety_line_v_px"] = v
        if self._live_cfg is not None:
            self._live_cfg.setdefault("display", {})["foot_safety_line_v_px"] = v

    def _on_connect(self) -> None:
        """Open the RealSense colour and depth streams without recording."""
        if self._running or self._analyzing:
            return
        try:
            height_m = parse_ui_height_cm(self.height_cm_var.get())
            mass_kg = parse_ui_mass_kg(self.mass_kg_var.get())
        except ParticipantError as error:
            messagebox.showerror("Participant measurements", str(error))
            return
        self.mode_var.set("live")
        self.space_var.set("3d")
        self._refresh_header_meta()
        self._rebuild_dashboard()
        self.lstm_var.set(True)
        self.save_var.set(False)
        cfg = copy.deepcopy(self.config)
        cfg["source"]["mode"] = "live"
        cfg["analysis"]["space"] = "3d"
        cfg["analysis"]["live_pose2sim_upper"] = True
        cfg["analysis"]["live_diagnostics"] = False
        cfg.setdefault("project", {})
        cfg["project"]["participant_height_m"] = height_m
        cfg["project"]["participant_mass_kg"] = mass_kg
        cfg.setdefault("display", {})["foot_safety_line_enabled"] = False
        self._live_cfg = cfg
        self._diag_wanted = False
        self._stop_flag.clear()
        self._running = True
        self._error_dialog_shown = False
        self._status_from_worker = ""
        with self._frame_lock:
            self._latest_bgr = None
            self._latest_angles = {}
            self._latest_gauge_display = {}
            self._latest_angular = {}
            self._latest_linear = {}
            self._latest_readout = []
            self._latest_time = 0.0
            self._latest_index = -1
            self._latest_keypoints = []
            self._latest_lstm_estimates = []
            self._latest_fps = 0.0
            self._latest_space = self.space_var.get()
            self._latest_min_vis = float((self.config.get("pose") or {}).get("min_visibility", 0.5))
        self._chart_index = -1
        self._want_summary = False
        self._phase = "connecting"
        self._refresh_actions()
        self.status_var.set("Connecting…")
        self.badge_var.set("CONNECTING")
        self._paint_badge("accent")
        self._worker = threading.Thread(target=self._run_session, args=(cfg,), daemon=True)
        self._worker.start()

    def _on_start_capture(self) -> None:
        """Record on the open camera. Does not open a second camera."""
        if not self._running or self._analyzing:
            messagebox.showerror("Camera", "Connect the camera before capture.")
            return
        snap = dict(self._live_snap)
        if snap.get("recording"):
            messagebox.showerror("Capture", "A capture is already running.")
            return
        subject = snap.get("subject") or {}
        if not subject.get("ready") or not snap.get("depth_streaming"):
            messagebox.showerror("Not ready", str(subject.get("instruction") or "The participant is not ready to capture."))
            return
        self._session_events.put({"kind": "capture_start"})

    def _on_stop_capture(self) -> None:
        """Finish the files and leave the camera connected."""
        if not self._running:
            return
        if not (self._live_snap or {}).get("recording"):
            return
        self._session_events.put({"kind": "capture_stop"})

    def _on_analyze(self) -> None:
        """Run the existing OpenSim path on the last finished capture."""
        if self._analyzing:
            return
        if (self._live_snap or {}).get("recording"):
            messagebox.showerror("Capture", "Stop the capture before analysis.")
            return
        diag = str((self._saved_capture or {}).get("diag_dir") or "")
        if not diag:
            messagebox.showerror("Analysis", "No finished capture yet. Record a movement first.")
            return
        self._analyzing = True
        self._phase = "analyzing"
        self.analysis_var.set("Analysis running…")
        self._refresh_actions()

        def _work() -> None:
            try:
                result = analyze_capture(Path(diag), PROJECT_ROOT)
            except Exception as error:
                result = {"ok": False, "opensim_ran": False, "message": str(error), "clinical_accuracy": False}
            self._analysis_result = result
            self._analyzing = False

        threading.Thread(target=_work, daemon=True).start()

    def _on_view_results(self) -> None:
        """Show the analysis report, or the reason analysis did not run."""
        result = self._analysis_result
        if not result and not self._saved_capture:
            messagebox.showinfo("Results", "Run 3D analysis after a capture.")
            return
        win = tk.Toplevel(self.root)
        win.title("Results")
        win.configure(bg=PAL.bg)
        win.geometry("760x640")
        if result:
            body = describe_results(result)
        else:
            diag = self._saved_capture.get("diag_dir")
            body = describe_results(
                {
                    "ok": False,
                    "message": "Analysis has not been run.",
                    "landmarks_csv": str(Path(diag) / "landmarks.csv") if diag else None,
                    "clinical_accuracy": False,
                }
            )
        plot = None
        if result and result.get("analysis_dir"):
            candidate = Path(result["analysis_dir"]) / "plots" / "timeline.png"
            if candidate.is_file():
                plot = candidate
        if plot is not None:
            image = Image.open(plot)
            image.thumbnail((700, 280))
            win._plot_photo = ImageTk.PhotoImage(image)
            tk.Label(win, image=win._plot_photo, bg=PAL.bg).pack(padx=16, pady=(12, 0))
        text = tk.Text(win, bg=PAL.panel, fg=PAL.text, font=("Segoe UI", 10), wrap=tk.WORD, bd=0)
        text.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        text.insert("1.0", body)
        text.configure(state=tk.DISABLED)
        folder = None
        if result and result.get("analysis_dir"):
            folder = result["analysis_dir"]
        elif self._saved_capture.get("diag_dir"):
            folder = self._saved_capture["diag_dir"]
        if folder:
            tk.Button(
                win,
                text="Open folder",
                command=lambda path=folder: __import__("os").startfile(path),
                bd=0,
                bg=PAL.accent,
                fg="white",
                font=("Segoe UI", 10, "bold"),
                padx=12,
                pady=6,
            ).pack(anchor="e", padx=16, pady=(0, 16))

    def _on_stop(self) -> None:
        """Ask the worker to finish and close the camera."""
        self._stop_flag.set()
        self.status_var.set("Stopping…")
        self.badge_var.set("STOPPING")

    def _refresh_actions(self) -> None:
        """Enable a button only when that step is valid."""
        recording = bool((self._live_snap or {}).get("recording"))
        ready = bool(((self._live_snap or {}).get("subject") or {}).get("ready")) and bool(
            (self._live_snap or {}).get("depth_streaming")
        )
        connected = self._running and self._phase not in ("connecting", "disconnected")
        has_capture = bool((self._saved_capture or {}).get("diag_dir"))

        def _enable(button: tk.Button, on: bool, colour: str) -> None:
            button.configure(state=tk.NORMAL if on else tk.DISABLED, bg=colour if on else PAL.idle)

        _enable(self.connect_btn, not self._running and not self._analyzing, PAL.accent)
        _enable(self.capture_btn, connected and ready and not recording and not self._analyzing, PAL.start)
        _enable(self.stop_capture_btn, recording, PAL.stop)
        _enable(self.analyze_btn, has_capture and not recording and not self._analyzing, PAL.accent)
        _enable(self.results_btn, bool(self._analysis_result) or has_capture, PAL.accent)
        entry_state = tk.DISABLED if self._running else tk.NORMAL
        self._height_entry.configure(state=entry_state)
        self._mass_entry.configure(state=entry_state)

    def _set_running_buttons(self, running: bool) -> None:
        self._refresh_actions()

    def _run_session(self, cfg: dict) -> None:
        """Worker thread: MediaPipe + save. Do not touch Tk widgets here."""
        while True:
            try:
                self._session_events.get_nowait()
            except queue.Empty:
                break
        session = MotionSession2D(cfg, save_files=False, events=self._session_events)
        try:
            session.start()
            label = session.source.label if session.source is not None else "source"
            self._status_from_worker = f"Running ({label})"
            while not self._stop_flag.is_set():
                try:
                    canvas = session.process_frame()
                except StopIteration:
                    self._status_from_worker = "End of file"
                    break
                if canvas is None:
                    continue
                with self._frame_lock:
                    self._latest_bgr = canvas
                    self._latest_angles = dict(session.last_gauge_angles)
                    self._latest_gauge_display = dict(session.last_gauge_display)
                    self._latest_angular = dict(session.last_angular_dps)
                    self._latest_linear = dict(session.last_linear_mps)
                    self._latest_readout = [
                        {**row, "lines": list(row.get("lines") or [])}
                        for row in session.last_highlight_readout
                    ]
                    self._latest_time = session.last_time_sec
                    self._latest_index = session.frame_index
                    self._latest_keypoints = list(session.last_keypoints)
                    self._latest_lstm_estimates = list(session.last_lstm_estimates)
                    self._latest_fps = float(session.last_fps)
                    self._latest_space = str((cfg.get("analysis") or {}).get("space", "2d"))
                    self._latest_min_vis = float((cfg.get("pose") or {}).get("min_visibility", 0.5))
                    self._live_snap = session.ui_snapshot()
                    if session.last_frame_click is not None:
                        self._pending_frame_click = session.last_frame_click
                        session.last_frame_click = None
                if session.last_diag_status:
                    self._diag_from_worker = session.last_diag_status
        except Exception as error:
            self._status_from_worker = f"Error: {_friendly_start_error(error)}"
        finally:
            try:
                summary = session.stop()
            except Exception:
                summary = {}
            extra = ""
            if summary.get("run_dir"):
                extra = f"  ·  saved {summary['run_dir']}"
            if summary.get("diag_dir"):
                extra += f"  ·  diagnostics {summary['diag_dir']}"
            self._diag_recording = False
            self._saved_capture = dict(summary)
            prefix = self._status_from_worker or ""
            if prefix.startswith("Error:"):
                self._want_summary = False
                self._status_from_worker = prefix + extra
            elif prefix.startswith("End of file"):
                self._status_from_worker = prefix + extra
            else:
                self._status_from_worker = "Stopped" + extra
            self._running = False

    def _paint_live_status(self) -> None:
        """Copy the worker snapshot into the side panel."""
        snap = dict(self._live_snap or {})
        subject = snap.get("subject") or {}
        if self._analyzing:
            self._phase = "analyzing"
            self.badge_var.set("ANALYZING")
            self._paint_badge("accent")
            self.analysis_var.set("Analysis running…")
        elif self._running and not snap:
            self._phase = "connecting"
            self.badge_var.set("CONNECTING")
            self._paint_badge("accent")
            self.camera_status_var.set("Connecting")
        elif self._running:
            if snap.get("recording"):
                self._phase = "recording"
                self.badge_var.set("RECORDING")
                self._paint_badge("stop")
            elif subject.get("ready") and snap.get("depth_streaming"):
                self._phase = "ready"
                self.badge_var.set("READY")
                self._paint_badge("start")
            else:
                self._phase = "connected"
                self.badge_var.set("CONNECTED")
                self._paint_badge("accent")
            capture = snap.get("last_capture") or {}
            if capture.get("diag_dir") and not snap.get("recording"):
                self._saved_capture = capture
            self.camera_status_var.set("Camera connected")
            self.camera_model_var.set(str(snap.get("device") or "RealSense"))
            self.depth_status_var.set("Depth stream on" if snap.get("depth_streaming") else "Depth stream waiting for a frame")
            self.stream_var.set(
                f"{snap.get('width')}×{snap.get('height')}   {snap.get('fps_config')} fps requested   "
                f"{float(snap.get('fps_measured') or 0):.1f} fps measured"
            )
            self.position_var.set(str(subject.get("instruction") or ""))
            distance = subject.get("distance_m")
            self.distance_var.set(
                "Distance unavailable" if distance is None else f"Shoulder distance  {float(distance):.2f} m"
            )
            if snap.get("recording"):
                elapsed = float(snap.get("capture_elapsed_sec") or 0.0)
                self.record_var.set("Recording")
                self.elapsed_var.set(f"Elapsed  {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
                self.frames_var.set(f"Frames  {int(snap.get('capture_frames') or 0)}")
            else:
                self.record_var.set("Not recording")
                self.elapsed_var.set("Elapsed  00:00")
                self.frames_var.set("Frames  0")
            self.fps_var.set(f"Processing FPS  {float(snap.get('fps_measured') or 0):.1f}")
            self._paint_performance(snap)
            self.landmark_var.set(f"Landmarks  {snap.get('landmark_status') or 'not visible'}")
            depth_label = "Depth available" if snap.get("depth_streaming") else "Depth waiting"
            self.depth_status_var.set(depth_label)
        elif not self._analyzing:
            self.camera_status_var.set("Camera disconnected")
            self.depth_status_var.set("Depth stream off")
        if self._analysis_result is not None and not self._analyzing:
            self.analysis_var.set(str(self._analysis_result.get("message") or "Analysis finished."))
        self._refresh_actions()

    def _paint_performance(self, snap: dict) -> None:
        """Show stage times. Hidden when display.performance_diagnostics is false."""
        if not self._performance_enabled:
            return
        perf = snap.get("performance") or {}
        interval = perf.get("delivered_interval_ms")
        interval_text = "—" if interval is None else f"{float(interval):.0f} ms"
        infer = perf.get("lstm_infer_ms")
        lag = perf.get("lstm_lag_ms")
        infer_text = "—" if infer is None else f"{float(infer):.0f} ms"
        lag_text = "—" if lag is None else f"{float(lag):.0f} ms"
        self.perf_var.set(
            f"Requested camera  {perf.get('requested_fps', '—')} fps\n"
            f"Capture wait  {float(perf.get('capture_wait_ms') or 0):.0f} ms\n"
            f"Delivered frame interval  {interval_text}\n"
            f"Processing  {float(perf.get('processing_fps') or 0):.1f} fps\n"
            f"Display  {self._display_fps:.1f} fps\n"
            f"Pose  {float(perf.get('pose_ms') or 0):.0f} ms\n"
            f"Hands  {float(perf.get('hands_ms') or 0):.0f} ms\n"
            f"Pose+Hands wall  {float(perf.get('pair_ms') or 0):.0f} ms\n"
            f"Depth  {float(perf.get('depth_ms') or 0):.0f} ms\n"
            f"LSTM buffer  {float(perf.get('lstm_prep_ms') or 0):.0f} ms\n"
            f"Filter  {float(perf.get('filter_ms') or 0):.0f} ms\n"
            f"UI display  {self._ui_display_ms:.0f} ms\n"
            f"Camera frame  {perf.get('camera_frame') if perf.get('camera_frame') is not None else '—'}\n"
            f"LSTM source frame  {perf.get('lstm_source_frame') if perf.get('lstm_source_frame') is not None else '—'}\n"
            f"LSTM status  {perf.get('lstm_status') or '—'}\n"
            f"LSTM infer  {infer_text}\n"
            f"LSTM frame age  {lag_text}\n"
            f"Angles  {float(perf.get('angles_ms') or 0):.0f} ms\n"
            f"Draw  {float(perf.get('draw_ms') or 0):.0f} ms\n"
            f"Loop  {float(perf.get('loop_ms') or 0):.0f} ms"
        )

    def _tick(self) -> None:
        """UI thread: show latest frame and numbers about 30 times a second."""
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return
        if self._status_from_worker:
            self.status_var.set(self._status_from_worker)
            if self._status_from_worker.startswith("Error:") and not self._error_dialog_shown:
                self._error_dialog_shown = True
                self.badge_var.set("ERROR")
                self._paint_badge("stop")
                title = "Camera not found" if "not connected" in self._status_from_worker.lower() or "not found" in self._status_from_worker.lower() else "Could not start"
                messagebox.showerror(title, self._status_from_worker.replace("Error: ", "", 1))
            if not self._running:
                self._phase = "disconnected"
                self._set_running_buttons(False)
                self._diag_recording = False
                if not self._status_from_worker.startswith("Error:"):
                    self.badge_var.set("DISCONNECTED")
                    self._paint_badge("idle")
        if self._running and self._diag_from_worker and not (self._live_snap or {}).get("recording"):
            self.status_var.set(self._diag_from_worker)
        with self._frame_lock:
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()
            angles = dict(self._latest_angles)
            gauge_display = dict(self._latest_gauge_display)
            angular = dict(self._latest_angular)
            linear = dict(self._latest_linear)
            readout = [dict(row) for row in self._latest_readout]
            frame_index = self._latest_index
            keypoints = list(self._latest_keypoints)
            estimates = list(self._latest_lstm_estimates)
            fps = self._latest_fps
            space = self._latest_space
            min_vis = self._latest_min_vis
            pending_click = self._pending_frame_click
            self._pending_frame_click = None
        if pending_click is not None:
            self._apply_frame_click(pending_click)
        if frame_index != self._display_index and frame_index >= 0:
            now = time.perf_counter()
            if self._display_stamp is not None:
                dt = now - self._display_stamp
                if dt > 1e-4:
                    inst = 1.0 / dt
                    self._display_fps = inst if self._display_fps <= 0 else (0.25 * inst + 0.75 * self._display_fps)
            self._display_stamp = now
            self._display_index = frame_index
        self._paint_live_status()
        if self.dashboard is not None and self._running:
            self.dashboard.on_frame(angles, angular, linear, gauge_display)
        if frame is not None:
            t_ui = time.perf_counter()
            self._show_frame(frame)
            self._ui_display_ms = (time.perf_counter() - t_ui) * 1000.0
        if not self._running and self._worker is not None and not self._worker.is_alive():
            self._phase = "disconnected"
            self._set_running_buttons(False)
        self.root.after(30, self._tick)

    def _open_marker_window(self) -> None:
        """3D inventory window. Same keypoints as the live RGB overlay."""
        if self._marker_window is not None and self._marker_window.alive():
            try:
                self._marker_window.win.lift()
            except Exception:
                pass
            return
        self._marker_window = LiveMarkerWindow(self.root)

    def _on_close(self) -> None:
        """Stop the camera thread, then close the window."""
        self._stop_flag.set()
        if self._marker_window is not None and self._marker_window.alive():
            self._marker_window._on_close()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.root.destroy()

    def _show_readout(self, rows: list[dict]) -> None:
        """Cards above the live video: same wrist numbers as the overlay boxes."""
        if not rows:
            if self._readout_key:
                for child in self._readout_bar.winfo_children():
                    child.destroy()
                self._readout_vars = {}
                self._readout_key = ()
                self._readout_placeholder = tk.Label(
                    self._readout_bar,
                    text="Joint coordinates appear here after Start",
                    bg=PAL.bg,
                    fg=PAL.muted,
                    font=("Segoe UI", 10),
                    anchor="w",
                )
                self._readout_placeholder.pack(fill=tk.X, pady=4)
            return

        key = tuple(str(row["name"]) for row in rows)
        if key != self._readout_key:
            for child in self._readout_bar.winfo_children():
                child.destroy()
            self._readout_vars = {}
            for row in rows:
                color = _bgr_to_hex(row.get("color_bgr") or (180, 200, 220))
                card = tk.Frame(
                    self._readout_bar,
                    bg=PAL.card,
                    highlightbackground=color,
                    highlightthickness=2,
                )
                card.pack(side=tk.LEFT, padx=(0, 8), ipadx=8, ipady=4)
                tk.Label(
                    card,
                    text=str(row.get("label", row["name"])),
                    bg=PAL.card,
                    fg=color,
                    font=("Segoe UI", 10, "bold"),
                    anchor="w",
                ).pack(fill=tk.X)
                var = tk.StringVar(value="\n".join(row.get("lines") or []))
                tk.Label(
                    card,
                    textvariable=var,
                    bg=PAL.card,
                    fg=PAL.text,
                    font=("Consolas", 11),
                    justify=tk.LEFT,
                    anchor="w",
                ).pack(fill=tk.X)
                self._readout_vars[str(row["name"])] = var
            self._readout_key = key
            return

        for row in rows:
            var = self._readout_vars.get(str(row["name"]))
            if var is not None:
                var.set("\n".join(row.get("lines") or []))

    def _show_report(self, bundle: dict) -> None:
        """Open a Report window: angles, angular speed, and linear speed."""
        self._report_stats = bundle
        if self._report is not None:
            try:
                if self._report.winfo_exists():
                    self._report.destroy()
            except tk.TclError:
                pass
        win = tk.Toplevel(self.root)
        self._report = win
        win.title("Report")
        win.configure(bg=PAL.bg)
        win.transient(self.root)

        tk.Label(
            win,
            text="Report",
            bg=PAL.bg,
            fg=PAL.text,
            font=("Segoe UI", 18, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=20, pady=(16, 2))
        tk.Label(
            win,
            text="Min, median, mode, and max from this take.",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill=tk.X, padx=20, pady=(0, 12))

        gauges = (self._analysis_for_ui().get("gauge_joints") or {})
        sides = (
            ("Left", [str(name) for name in gauges.get("left") or []]),
            ("Right", [str(name) for name in gauges.get("right") or []]),
        )
        angle_names = [name for _heading, names in sides for name in names]
        analysis = self._analysis_for_ui()
        linear_names = [str(name) for name in ((analysis.get("velocity") or {}).get("linear_joints") or [])]
        drawn = analysis.get("draw_joints") or []
        if drawn:
            linear_names = [name for name in linear_names if name in set(drawn)]
        self._report_table(win, "Angle", angle_names, bundle.get("angles") or {}, _deg)
        self._report_table(win, "Angular velocity", angle_names, bundle.get("angular") or {}, _dps)
        if str(bundle.get("space", "2d")).lower() == "3d":
            self._report_table(win, "Linear velocity", linear_names, bundle.get("linear") or {}, _mps)
        else:
            tk.Label(
                win,
                text="Linear velocity needs 3D camera metres.",
                bg=PAL.bg,
                fg=PAL.muted,
                font=("Segoe UI", 10),
                anchor="w",
            ).pack(fill=tk.X, padx=20, pady=(0, 12))

        tk.Button(
            win,
            text="  Close  ",
            command=win.destroy,
            bg=PAL.idle,
            fg=PAL.text,
            bd=0,
            padx=16,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            activebackground=PAL.line,
            activeforeground=PAL.text,
        ).pack(anchor="e", padx=20, pady=(0, 16))
        win.update_idletasks()
        win.geometry(f"+{self.root.winfo_rootx() + 80}+{self.root.winfo_rooty() + 80}")
        win.lift()
        win.focus_set()

    def _report_table(self, win: tk.Toplevel, title: str, names: list[str], stats: dict, fmt) -> None:
        """One Report card: min, median, mode, max for each name."""
        card = tk.Frame(win, bg=PAL.card, highlightbackground=PAL.line, highlightthickness=1)
        card.pack(fill=tk.X, padx=20, pady=(0, 12))
        tk.Label(
            card,
            text=title,
            bg=PAL.card,
            fg=PAL.text,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=5, sticky="w", padx=12, pady=(10, 6))
        if not names:
            tk.Label(
                card,
                text="No joints in this region.",
                bg=PAL.card,
                fg=PAL.muted,
                font=("Segoe UI", 10),
                anchor="w",
            ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))
            return
        for col, heading in enumerate(("Joint", "Min", "Median", "Mode", "Max")):
            tk.Label(
                card,
                text=heading,
                bg=PAL.card,
                fg=PAL.muted,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
            ).grid(row=1, column=col, sticky="w", padx=12, pady=(0, 4))
        for row_i, name in enumerate(names, start=2):
            row = stats.get(name) or {}
            values = (
                _side_label(name),
                fmt(row.get("min")),
                fmt(row.get("median")),
                fmt(row.get("mode")),
                fmt(row.get("max")),
            )
            for col, text in enumerate(values):
                tk.Label(
                    card,
                    text=text,
                    bg=PAL.card,
                    fg=PAL.text,
                    font=("Segoe UI", 12, "bold" if col == 0 else "normal"),
                    anchor="w",
                ).grid(row=row_i, column=col, sticky="w", padx=12, pady=4)
        tk.Frame(card, bg=PAL.card, height=8).grid(row=len(names) + 2, column=0)

    def _show_frame(self, bgr) -> None:
        """Fit the overlay into the video panel and display it."""
        h, w = bgr.shape[:2]
        self._native_wh = (w, h)
        avail_w = max(320, self.video_label.winfo_width() or 960)
        avail_h = max(240, self.video_label.winfo_height() or 540)
        if avail_w < 50 or avail_h < 50:
            return
        scale = min(avail_w / w, avail_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        self._drawn_wh = (new_w, new_h)
        if new_w != w or new_h != h:
            bgr = cv2.resize(bgr, (new_w, new_h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self._photo, text="")


def _side_label(name: str) -> str:
    """'L Shoulder' / 'R wrist' so the two arms are not both just 'Shoulder'."""
    base = JOINT_LABELS.get(name, name.replace("_", " ").title())
    if name.startswith("left_"):
        return f"L {base}"
    if name.startswith("right_"):
        return f"R {base}"
    return base


def _deg(value: float | None) -> str:
    """Whole degrees, or a dash when that joint was never seen."""
    if value is None:
        return "—"
    return f"{value:.0f}°"


def _dps(value: float | None) -> str:
    """Whole degrees per second."""
    if value is None:
        return "—"
    return f"{value:.0f} °/s"


def _mps(value: float | None) -> str:
    """Metres per second, two decimals."""
    if value is None:
        return "—"
    return f"{value:.2f} m/s"


def _bgr_to_hex(color) -> str:
    """OpenCV BGR tuple -> Tk hex colour."""
    blue, green, red = int(color[0]), int(color[1]), int(color[2])
    return f"#{red:02x}{green:02x}{blue:02x}"


def _friendly_start_error(error: BaseException) -> str:
    """Keep camera-missing errors short for the dialog; pass other errors through."""
    text = str(error).strip() or error.__class__.__name__
    lower = text.lower()
    if "not connected" in lower or "not found" in lower or "no device" in lower:
        return (
            "Camera not connected / not found.\n\n"
            "Plug in the RealSense D455f (USB 3) and close RealSense Viewer, then press Start."
        )
    return text


def main() -> None:
    """Launch the 2D app. Run from the project folder: python app.py"""
    root = tk.Tk()
    MotionAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
