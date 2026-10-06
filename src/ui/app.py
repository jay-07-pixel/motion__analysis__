"""Step 6 — 2D motion analysis window.

Split screen: camera on the left, Left/Right body angle dials on the right
as soon as Start runs. After Stop the same dials show min / max / mode.
Capture runs on a worker thread so Start/Stop stay clickable.

Still 2D only (camera pixels, image-plane elbow degrees, wrist px/s).
Close RealSense Viewer before Live.
"""

from __future__ import annotations

import copy
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.regions import apply_region, region_id
from src.ui.live_charts import JOINT_LABELS, MotionDashboard
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
        self.root.minsize(1280, 780)
        self.root.configure(bg=PAL.bg)

        self._worker: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_bgr = None
        self._latest_angles: dict[str, float | None] = {}
        self._latest_angular: dict[str, float | None] = {}
        self._latest_linear: dict[str, float | None] = {}
        self._latest_readout: list[dict] = []
        self._latest_time = 0.0
        self._latest_index = -1
        self._chart_index = -1
        self._want_summary = False
        self._report: tk.Toplevel | None = None
        self._report_stats: dict | None = None
        self._status_from_worker = ""
        self._photo = None
        self._running = False
        self._error_dialog_shown = False

        self.mode_var = tk.StringVar(value=str(self.config["source"]["mode"]))
        self.region_var = tk.StringVar(value=region_id(self.config))
        self.space_var = tk.StringVar(
            value=str((self.config.get("analysis") or {}).get("space", "2d")).lower()
        )
        self.path_var = tk.StringVar(value=str(self.config["source"]["file_path"]))
        self.save_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Idle  ·  close RealSense Viewer before Live")
        self.badge_var = tk.StringVar(value="IDLE")
        self.meta_var = tk.StringVar(value="")
        self.theme_var = tk.StringVar(value="night")
        self._badge_role = "idle"
        self.gauge_step = float((self.config.get("analysis") or {}).get("angle_gauge_step_deg", 45))

        self._build_header(project)
        self._build_controls()
        self._build_body()
        self._build_status()
        self._refresh_mode_buttons()
        self._refresh_region_buttons()
        self._refresh_space_buttons()
        self._refresh_theme_buttons()
        self._refresh_header_meta()
        self._tick()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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

        self.save_check = tk.Checkbutton(
            bar,
            text="Save CSV + video",
            variable=self.save_var,
            bg=PAL.bg,
            fg=PAL.text,
            selectcolor=PAL.card,
            activebackground=PAL.bg,
            activeforeground=PAL.text,
            font=("Segoe UI", 10),
        )
        self.save_check.pack(side=tk.LEFT, padx=8)

        self.stop_btn = tk.Button(
            bar,
            text="  Stop  ",
            command=self._on_stop,
            bg=PAL.idle,
            fg=PAL.text,
            bd=0,
            padx=16,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.stop_btn.pack(side=tk.RIGHT, padx=(6, 0))
        self.start_btn = tk.Button(
            bar,
            text="  Start  ",
            command=self._on_start,
            bg=PAL.start,
            fg="white",
            bd=0,
            padx=18,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            activebackground="#26855a",
            activeforeground="white",
        )
        self.start_btn.pack(side=tk.RIGHT)

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
            self._dash_host, self._analysis_for_ui(), on_step=self._remember_gauge_step
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
        self.save_check.configure(
            bg=PAL.bg,
            fg=PAL.text,
            selectcolor=PAL.card,
            activebackground=PAL.bg,
            activeforeground=PAL.text,
        )
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

    def _on_start(self) -> None:
        """Start the worker. Live needs the D455f free (Viewer closed)."""
        if self._running:
            return
        if self.space_var.get() == "3d" and self.mode_var.get() == "file":
            suffix = Path(self.path_var.get().strip()).suffix.lower()
            if suffix != ".bag":
                messagebox.showerror(
                    "3D needs depth",
                    "3D camera X/Y/Z (metres) needs RealSense depth.\n\n"
                    "Use Live camera, or upload a .bag recorded in Viewer.\n"
                    "An mp4 has colour only — stay on 2D for that file.",
                )
                return
        if self.mode_var.get() == "file":
            path = resolve_project_path(self.path_var.get().strip())
            if not path.is_file():
                messagebox.showerror("File not found", f"No video or bag at:\n{path}")
                return
        cfg = copy.deepcopy(self.config)
        cfg["source"]["mode"] = self.mode_var.get()
        cfg["source"]["file_path"] = self.path_var.get()
        apply_region(cfg, self.region_var.get())
        cfg["analysis"]["space"] = self.space_var.get()
        self._stop_flag.clear()
        self._running = True
        self._error_dialog_shown = False
        self._status_from_worker = ""
        with self._frame_lock:
            self._latest_bgr = None
            self._latest_angles = {}
            self._latest_angular = {}
            self._latest_linear = {}
            self._latest_readout = []
            self._latest_time = 0.0
            self._latest_index = -1
        self._chart_index = -1
        self._want_summary = True
        self.dashboard.on_start()
        self._set_running_buttons(True)
        self._refresh_region_buttons()
        self.status_var.set("Starting…")
        self.badge_var.set("STARTING")
        self._paint_badge("accent")
        self._worker = threading.Thread(target=self._run_session, args=(cfg,), daemon=True)
        self._worker.start()

    def _on_stop(self) -> None:
        """Ask the worker to finish and close the camera/file."""
        self._stop_flag.set()
        self.status_var.set("Stopping…")
        self.badge_var.set("STOPPING")

    def _set_running_buttons(self, running: bool) -> None:
        """Enable Start or Stop, not both."""
        if running:
            self.start_btn.configure(state=tk.DISABLED, bg=PAL.idle)
            self.stop_btn.configure(state=tk.NORMAL, bg=PAL.stop, fg="white")
        else:
            self.start_btn.configure(state=tk.NORMAL, bg=PAL.start, fg="white")
            self.stop_btn.configure(state=tk.DISABLED, bg=PAL.idle, fg=PAL.text)
        self._refresh_region_buttons()
        self._refresh_space_buttons()

    def _run_session(self, cfg: dict) -> None:
        """Worker thread: MediaPipe + save. Do not touch Tk widgets here."""
        session = MotionSession2D(cfg, save_files=bool(self.save_var.get()))
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
                    self._latest_angular = dict(session.last_angular_dps)
                    self._latest_linear = dict(session.last_linear_mps)
                    self._latest_readout = [
                        {**row, "lines": list(row.get("lines") or [])}
                        for row in session.last_highlight_readout
                    ]
                    self._latest_time = session.last_time_sec
                    self._latest_index = session.frame_index
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
            prefix = self._status_from_worker or ""
            if prefix.startswith("Error:"):
                self._want_summary = False
                self._status_from_worker = prefix + extra
            elif prefix.startswith("End of file"):
                self._status_from_worker = prefix + extra
            else:
                self._status_from_worker = "Stopped" + extra
            self._running = False

    def _tick(self) -> None:
        """UI thread: show latest frame and numbers about 30 times a second."""
        if self._status_from_worker:
            self.status_var.set(self._status_from_worker)
            if self._status_from_worker.startswith("Error:") and not self._error_dialog_shown:
                self._error_dialog_shown = True
                self.badge_var.set("ERROR")
                self._paint_badge("stop")
                title = "Camera not found" if "not connected" in self._status_from_worker.lower() or "not found" in self._status_from_worker.lower() else "Could not start"
                messagebox.showerror(title, self._status_from_worker.replace("Error: ", "", 1))
            if not self._running:
                self._set_running_buttons(False)
                if not self._status_from_worker.startswith("Error:"):
                    self.badge_var.set("SUMMARY")
                    self._paint_badge("idle")
                if self._want_summary:
                    self._want_summary = False
                    with self._frame_lock:
                        last_angles = dict(self._latest_angles)
                    self.dashboard.on_stop(last_angles)
                    self._show_report(
                        {
                            "angles": self.dashboard.recorder.summary(),
                            "angular": self.dashboard.angular_recorder.summary(),
                            "linear": self.dashboard.linear_recorder.summary(),
                            "space": self.dashboard.space,
                        }
                    )
        with self._frame_lock:
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()
            angles = dict(self._latest_angles)
            angular = dict(self._latest_angular)
            linear = dict(self._latest_linear)
            readout = [dict(row) for row in self._latest_readout]
            frame_index = self._latest_index
        self._show_readout(readout)
        if self._running:
            self.badge_var.set("LIVE" if self.mode_var.get() == "live" else "FILE")
            self._paint_badge("start")
        if self._running and frame_index != self._chart_index:
            self._chart_index = frame_index
            self.dashboard.on_frame(angles, angular, linear)
        if frame is not None:
            self._show_frame(frame)
        if not self._running and self._worker is not None and not self._worker.is_alive():
            self._set_running_buttons(False)
        self.root.after(30, self._tick)

    def _on_close(self) -> None:
        """Stop the camera thread, then close the window."""
        self._stop_flag.set()
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
        avail_w = max(320, self.video_label.winfo_width() or 960)
        avail_h = max(240, self.video_label.winfo_height() or 540)
        if avail_w < 50 or avail_h < 50:
            return
        scale = min(avail_w / w, avail_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
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
