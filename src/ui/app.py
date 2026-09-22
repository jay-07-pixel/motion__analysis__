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
from src.ui.live_charts import MotionDashboard
from src.ui.session import MotionSession2D
from src.utils.config_loader import load_config, resolve_project_path

# Window colours (not analysis settings — those stay in config.yaml).
BG = "#101418"
PANEL = "#171e28"
CARD = "#1e2734"
LINE = "#2a3545"
TEXT = "#eef3f8"
MUTED = "#8b98a8"
ACCENT = "#4c9aff"
START = "#2f9e6a"
STOP = "#c44c4c"
IDLE_BTN = "#2a3340"


class MotionAnalysisApp:
    """Main window. Worker thread owns MediaPipe; this class only draws widgets."""

    def __init__(self, root: tk.Tk) -> None:
        """Build the layout from config (title, default live/file)."""
        self.root = root
        self.config = load_config()
        project = self.config["project"]
        self.root.title(str(project["title"]) + " — 2D")
        self.root.minsize(1280, 780)
        self.root.configure(bg=BG)

        self._worker: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_bgr = None
        self._latest_angles: dict[str, float | None] = {}
        self._latest_time = 0.0
        self._latest_index = -1
        self._chart_index = -1
        self._want_summary = False
        self._status_from_worker = ""
        self._photo = None
        self._running = False
        self._error_dialog_shown = False

        self.mode_var = tk.StringVar(value=str(self.config["source"]["mode"]))
        self.region_var = tk.StringVar(value=region_id(self.config))
        self.path_var = tk.StringVar(value=str(self.config["source"]["file_path"]))
        self.save_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Idle  ·  close RealSense Viewer before Live")
        self.badge_var = tk.StringVar(value="IDLE")

        self._build_header(project)
        self._build_controls()
        self._build_body()
        self._build_status()
        self._refresh_mode_buttons()
        self._refresh_region_buttons()
        self._tick()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_header(self, project: dict) -> None:
        """Top strip: project name and camera model from config.yaml."""
        header = tk.Frame(self.root, bg=PANEL, height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        left = tk.Frame(header, bg=PANEL)
        left.pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(
            left,
            text=str(project["title"]),
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text=f"{project.get('student', '')}  ·  {project.get('camera_model', '')}  ·  2D camera pixels",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w")
        badge = tk.Label(
            header,
            textvariable=self.badge_var,
            bg=IDLE_BTN,
            fg=TEXT,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
        )
        badge.pack(side=tk.RIGHT, padx=20)
        self._badge = badge

    def _build_controls(self) -> None:
        """Source, region picker, save, Start/Stop — one row under the header."""
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill=tk.X, padx=16, pady=12)

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
        tk.Button(
            bar,
            text=" Browse ",
            command=self._browse,
            bg=IDLE_BTN,
            fg=TEXT,
            bd=0,
            padx=12,
            pady=8,
            font=("Segoe UI", 10),
            cursor="hand2",
            activebackground=LINE,
            activeforeground=TEXT,
        ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(
            bar,
            text="ANALYSE",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        region_wrap = tk.Frame(bar, bg=BG)
        region_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True)
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
            bg=BG,
            fg=TEXT,
            selectcolor=CARD,
            activebackground=BG,
            activeforeground=TEXT,
            font=("Segoe UI", 10),
        )
        self.save_check.pack(side=tk.LEFT, padx=8)

        self.stop_btn = tk.Button(
            bar,
            text="  Stop  ",
            command=self._on_stop,
            bg=IDLE_BTN,
            fg=TEXT,
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
            bg=START,
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
                bg=ACCENT if on else IDLE_BTN,
                fg="white" if on else TEXT,
                activebackground=ACCENT if on else LINE,
                activeforeground="white" if on else TEXT,
                state=tk.DISABLED if self._running else tk.NORMAL,
            )

    def _analysis_for_ui(self) -> dict:
        """Analysis dict after applying the selected region."""
        cfg = apply_region(copy.deepcopy(self.config), self.region_var.get())
        return cfg["analysis"]

    def _rebuild_dashboard(self) -> None:
        """Swap the right-hand dials to match the selected region."""
        self.dashboard.destroy()
        self.dashboard = MotionDashboard(self._dash_host, self._analysis_for_ui())
        self.dashboard.pack(fill=tk.BOTH, expand=True)

    def _build_body(self) -> None:
        """Split screen: camera left, Left/Right body speedometers right."""
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        video_frame = tk.Frame(body, bg="#07090c", highlightbackground=LINE, highlightthickness=1)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.video_label = tk.Label(
            video_frame,
            bg="#07090c",
            fg=MUTED,
            text="Press Start  ·  camera on this side, speedometers on the other",
            font=("Segoe UI", 12),
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self._dash_host = tk.Frame(body, bg=BG)
        self._dash_host.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.dashboard = MotionDashboard(self._dash_host, self._analysis_for_ui())
        self.dashboard.pack(fill=tk.BOTH, expand=True)

    def _build_status(self) -> None:
        """Bottom line: idle / running / saved path."""
        bar = tk.Frame(self.root, bg=PANEL)
        bar.pack(fill=tk.X)
        tk.Label(
            bar,
            textvariable=self.status_var,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill=tk.X, padx=16, pady=8)

    def _set_mode(self, mode: str) -> None:
        """Switch Live vs Upload from the two pill buttons."""
        self.mode_var.set(mode)
        self._refresh_mode_buttons()

    def _refresh_mode_buttons(self) -> None:
        """Highlight the selected source pill."""
        live = self.mode_var.get() == "live"
        self.live_btn.configure(
            bg=ACCENT if live else IDLE_BTN,
            fg="white" if live else TEXT,
            activebackground=ACCENT if live else LINE,
            activeforeground="white" if live else TEXT,
        )
        self.file_btn.configure(
            bg=ACCENT if not live else IDLE_BTN,
            fg="white" if not live else TEXT,
            activebackground=ACCENT if not live else LINE,
            activeforeground="white" if not live else TEXT,
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
        if self.mode_var.get() == "file":
            path = resolve_project_path(self.path_var.get().strip())
            if not path.is_file():
                messagebox.showerror("File not found", f"No video or bag at:\n{path}")
                return
        cfg = copy.deepcopy(self.config)
        cfg["source"]["mode"] = self.mode_var.get()
        cfg["source"]["file_path"] = self.path_var.get()
        apply_region(cfg, self.region_var.get())
        self._stop_flag.clear()
        self._running = True
        self._error_dialog_shown = False
        self._status_from_worker = ""
        with self._frame_lock:
            self._latest_bgr = None
            self._latest_angles = {}
            self._latest_time = 0.0
            self._latest_index = -1
        self._chart_index = -1
        self._want_summary = True
        self.dashboard.on_start()
        self._set_running_buttons(True)
        self._refresh_region_buttons()
        self.status_var.set("Starting…")
        self.badge_var.set("STARTING")
        self._badge.configure(bg=ACCENT)
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
            self.start_btn.configure(state=tk.DISABLED, bg=IDLE_BTN)
            self.stop_btn.configure(state=tk.NORMAL, bg=STOP, fg="white")
        else:
            self.start_btn.configure(state=tk.NORMAL, bg=START, fg="white")
            self.stop_btn.configure(state=tk.DISABLED, bg=IDLE_BTN, fg=TEXT)
        self._refresh_region_buttons()

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
                self._badge.configure(bg=STOP)
                title = "Camera not found" if "not connected" in self._status_from_worker.lower() or "not found" in self._status_from_worker.lower() else "Could not start"
                messagebox.showerror(title, self._status_from_worker.replace("Error: ", "", 1))
            if not self._running:
                self._set_running_buttons(False)
                if not self._status_from_worker.startswith("Error:"):
                    self.badge_var.set("SUMMARY")
                    self._badge.configure(bg=IDLE_BTN)
                if self._want_summary:
                    self._want_summary = False
                    with self._frame_lock:
                        last_angles = dict(self._latest_angles)
                    self.dashboard.on_stop(last_angles)
        with self._frame_lock:
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()
            angles = dict(self._latest_angles)
            frame_index = self._latest_index
        if self._running:
            self.badge_var.set("LIVE" if self.mode_var.get() == "live" else "FILE")
            self._badge.configure(bg=START)
        if self._running and frame_index != self._chart_index:
            self._chart_index = frame_index
            self.dashboard.on_frame(angles)
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
