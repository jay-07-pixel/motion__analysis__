"""Instructions and controls for the camera-to-model frame calibration.

Text stays in this window. The live video is not labeled.
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

from src.opensim_upper.frame_calibration import INSTRUCTIONS
from src.opensim_upper.leveling import LevelingError, read_static_up_color
from src.ui.theme import PAL

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FrameCalibrationDialog:
    """Side window. Measuring up uses the camera only while the session is stopped."""

    def __init__(self, app) -> None:
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Camera to OpenSim frame")
        self.win.configure(bg=PAL.bg)
        self.win.geometry("560x640")
        text = tk.Text(
            self.win,
            wrap=tk.WORD,
            height=14,
            bg=PAL.card,
            fg=PAL.text,
            bd=0,
            padx=12,
            pady=12,
            font=("Segoe UI", 10),
        )
        text.pack(fill=tk.X, padx=12, pady=(12, 8))
        text.insert("1.0", INSTRUCTIONS)
        text.configure(state=tk.DISABLED)

        self.up_var = tk.StringVar()
        self.face_var = tk.StringVar()
        self.origin_var = tk.StringVar()
        self.result_var = tk.StringVar()
        for variable in (self.up_var, self.face_var, self.origin_var, self.result_var):
            tk.Label(
                self.win,
                textvariable=variable,
                bg=PAL.bg,
                fg=PAL.text,
                font=("Segoe UI", 9),
                anchor="w",
                justify=tk.LEFT,
                wraplength=520,
            ).pack(fill=tk.X, padx=12, pady=2)

        warning = tk.Label(
            self.win,
            text=(
                "The patient must face the camera. Turning away from the lens changes the heading. "
                "Keep the camera still during the capture. Move or rotate it, and this calibration "
                "must be repeated. The reference point needs measured ground-frame coordinates. "
                "They are not filled in, and the default pelvis position is not the ground origin."
            ),
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=520,
        )
        warning.pack(fill=tk.X, padx=12, pady=(0, 8))

        row = tk.Frame(self.win, bg=PAL.bg)
        row.pack(fill=tk.X, padx=12, pady=(8, 4))
        self.up_btn = tk.Button(
            row,
            text="1. Measure camera up",
            command=self._measure_up,
            bg=PAL.card,
            fg=PAL.text,
            bd=0,
            padx=10,
            pady=6,
        )
        self.up_btn.pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            row,
            text="2. Confirm patient faces camera",
            command=self._confirm_facing,
            bg=PAL.card,
            fg=PAL.text,
            bd=0,
            padx=10,
            pady=6,
        ).pack(side=tk.LEFT)

        tk.Button(
            self.win,
            text="3. Click physical reference point",
            command=lambda: self.app.arm_frame_click("origin"),
            bg=PAL.card,
            fg=PAL.text,
            bd=0,
            padx=10,
            pady=6,
        ).pack(anchor="w", padx=12, pady=4)

        form = tk.Frame(self.win, bg=PAL.bg)
        form.pack(fill=tk.X, padx=12, pady=8)
        tk.Label(
            form,
            text="4. Measured OpenSim ground coordinates of that point (metres)",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, columnspan=6, sticky="w")
        self.model_x = tk.StringVar()
        self.model_y = tk.StringVar()
        self.model_z = tk.StringVar()
        for column, (label, variable) in enumerate(
            (("X anterior", self.model_x), ("Y up", self.model_y), ("Z right", self.model_z))
        ):
            tk.Label(form, text=label, bg=PAL.bg, fg=PAL.muted, font=("Segoe UI", 9)).grid(
                row=1, column=column * 2, sticky="w", pady=(4, 0)
            )
            tk.Entry(form, textvariable=variable, width=8, font=("Segoe UI", 9)).grid(
                row=1, column=column * 2 + 1, sticky="w", padx=(4, 10), pady=(4, 0)
            )
        tk.Label(
            form,
            text="What physical point is it?",
            bg=PAL.bg,
            fg=PAL.muted,
            font=("Segoe UI", 9),
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 0))
        self.description = tk.StringVar()
        tk.Entry(form, textvariable=self.description, width=52, font=("Segoe UI", 9)).grid(
            row=3, column=0, columnspan=6, sticky="we", pady=(4, 0)
        )

        actions = tk.Frame(self.win, bg=PAL.bg)
        actions.pack(fill=tk.X, padx=12, pady=12)
        tk.Button(
            actions,
            text="5. Compute calibration",
            command=self._compute,
            bg=PAL.accent,
            fg="white",
            bd=0,
            padx=14,
            pady=6,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            actions,
            text="6. Save registration",
            command=self._save,
            bg=PAL.card,
            fg=PAL.text,
            bd=0,
            padx=14,
            pady=6,
        ).pack(side=tk.LEFT)
        self.refresh()

    def refresh(self) -> None:
        cal = self.app.frame_calibration
        if cal.up_camera is None:
            self.up_var.set("Up: not measured")
        else:
            up = cal.up_camera
            self.up_var.set(
                f"Up in colour camera: ({up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f}), "
                f"{cal.up_magnitude_m_s2:.2f} m/s^2, {cal.up_n_samples} samples"
            )
        if cal.heading_mode == "camera_optical_axis_negative_z":
            self.face_var.set("Facing camera: confirmed. Anterior is the horizontal part of colour -Z.")
        elif cal.heading_mode == "facing_rod":
            self.face_var.set("Facing rod: both ends recorded. This is not the front-facing protocol.")
        else:
            self.face_var.set("Facing camera: not confirmed")
        if cal.origin_camera_m is None:
            self.origin_var.set("Reference point: not clicked")
        else:
            self.origin_var.set("Reference point: clicked in camera metres")
        self.up_btn.configure(state=tk.DISABLED if self.app._running else tk.NORMAL)

    def show_up_error(self, message: str) -> None:
        self.up_btn.configure(state=tk.NORMAL if not self.app._running else tk.DISABLED)
        self.up_var.set("Up: not measured")
        messagebox.showerror("Camera up", message, parent=self.win)

    def _measure_up(self) -> None:
        if self.app._running:
            messagebox.showinfo(
                "Camera up",
                "Stop the live session first. The accelerometer uses the camera on its own.",
                parent=self.win,
            )
            return
        self.up_btn.configure(state=tk.DISABLED)
        self.up_var.set("Measuring up. Leave the camera still.")

        def work() -> None:
            try:
                assessment = read_static_up_color()
            except LevelingError as error:
                self.app.root.after(0, lambda: self.show_up_error(str(error)))
                return
            except Exception as error:
                self.app.root.after(0, lambda: self.show_up_error(str(error)))
                return
            self.app.root.after(0, lambda: self._store_up(assessment))

        import threading

        threading.Thread(target=work, daemon=True).start()

    def _store_up(self, assessment: dict) -> None:
        self.app.frame_calibration.set_measured_up(assessment)
        self.refresh()
        self.result_var.set("Up stored. Facing and the reference point are still required.")

    def _confirm_facing(self) -> None:
        self.app.frame_calibration.confirm_facing_camera()
        self.refresh()
        self.result_var.set(
            "Patient-faces-camera recorded. Calibration is not complete until up, "
            "the reference point, and its measured coordinates pass the checks."
        )

    def _read_model_origin(self) -> None:
        fields = (self.model_x.get(), self.model_y.get(), self.model_z.get())
        if not any(item.strip() for item in fields):
            return
        try:
            coords = tuple(float(item) for item in fields)
        except ValueError as error:
            raise ValueError("Model X, Y, and Z must be numbers in metres.") from error
        self.app.frame_calibration.set_origin_model(coords, self.description.get())

    def _compute(self) -> None:
        try:
            self._read_model_origin()
            solved = self.app.frame_calibration.solve()
        except ValueError as error:
            messagebox.showerror("Frame calibration", str(error), parent=self.win)
            return
        self._show_solved(solved)

    def _save(self) -> None:
        try:
            self._read_model_origin()
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = PROJECT_ROOT / "output" / "opensim_frame" / stamp / "registration.json"
            solved = self.app.frame_calibration.save(path)
        except ValueError as error:
            messagebox.showerror("Frame calibration", str(error), parent=self.win)
            return
        self._show_solved(solved)
        note = f"\nArchive {path}"
        if solved.get("complete"):
            attached = self.app.attach_registration(solved)
            if attached:
                note += f"\nSaved with capture {attached}"
            else:
                note += "\nThe next capture will store this registration with its files."
        else:
            note += "\nNot attached to a capture, because the calibration is incomplete."
        self.result_var.set(self.result_var.get() + note)

    def _show_solved(self, solved: dict) -> None:
        if solved["complete"]:
            checks = solved["checks"]
            self.result_var.set(
                "Calibration complete. It is not applied to landmarks until analysis.\n"
                f"Heading {solved.get('heading_source')}. "
                f"det {checks['determinant']:.6f}, orthogonal {checks['orthogonal']}, "
                f"inverse error {checks['inverse_max_error_m']:.2e} m, "
                f"reference error {checks['reference_alignment_error_m']:.2e} m."
            )
            return
        self.result_var.set("No transform applied.\n" + "\n".join(solved["missing"]))
        self.refresh()
