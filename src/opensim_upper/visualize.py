"""Plot static landmarks and the sized-model IK fit.

``save_reference_plot`` draws camera metres before registration.
``save_model_frame_plot`` draws a pose that has already been transformed.
``save_ik_fit_plot`` draws the full OpenSim skeleton next to the recorded points
and writes ``skeleton_fit.html`` beside the picture so the same pose can be rotated.
Neither plot moves model marker locations.
"""

from __future__ import annotations

from pathlib import Path

from src.markers.centres import STUDY_SIDE_PAIRS
from src.opensim_upper.static_pose import OBSERVED, PREDICTED


def save_reference_plot(summary: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for name in OBSERVED:
        row = summary["landmarks"][name]
        ax.scatter(row["x_m"], row["y_m"], row["z_m"], c="#1f77b4", s=30)
    for name in PREDICTED:
        row = summary["landmarks"][name]
        ax.scatter(row["x_m"], row["y_m"], row["z_m"], c="#e000ff", marker="D", s=28)
    ax.set_xlabel("X m (camera right)")
    ax.set_ylabel("Y m (camera down)")
    ax.set_zlabel("Z m (camera forward)")
    ax.set_title("Reference pose in camera metres. Model not registered.")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# Parent-to-child body origins. Legs are included when the pose table has them.
_BONE_LINKS = (
    ("pelvis", "sacrum"),
    ("pelvis", "torso"),
    ("torso", "head"),
    ("torso", "humerus_r"),
    ("humerus_r", "ulna_r"),
    ("ulna_r", "radius_r"),
    ("radius_r", "hand_r"),
    ("torso", "humerus_l"),
    ("humerus_l", "ulna_l"),
    ("ulna_l", "radius_l"),
    ("radius_l", "hand_l"),
    ("pelvis", "femur_r"),
    ("femur_r", "tibia_r"),
    ("tibia_r", "talus_r"),
    ("talus_r", "calcn_r"),
    ("calcn_r", "toes_r"),
    ("pelvis", "femur_l"),
    ("femur_l", "tibia_l"),
    ("tibia_l", "talus_l"),
    ("talus_l", "calcn_l"),
    ("calcn_l", "toes_l"),
)

# Fallback sticks when body origins were not exported (older IK folders).
_MARKER_LINKS = (
    ("RShoulder", "LShoulder"),
    ("RShoulder", "RElbow"),
    ("LShoulder", "LElbow"),
    ("RElbow", "r_lwrist_study"),
    ("RElbow", "r_mwrist_study"),
    ("LElbow", "L_lwrist_study"),
    ("LElbow", "L_mwrist_study"),
)


def _point_count(frame: dict) -> int:
    return sum(point is not None for point in frame["points"].values())


def _frame_miss_m(frame: dict, model_row: dict) -> float:
    """RMS metres between recorded points and the model markers on one frame."""
    squared = []
    for name, point in frame["points"].items():
        if point is None or f"{name}_tx" not in model_row:
            continue
        dx = float(point[0]) - float(model_row[f"{name}_tx"])
        dy = float(point[1]) - float(model_row[f"{name}_ty"])
        dz = float(point[2]) - float(model_row[f"{name}_tz"])
        squared.append(dx * dx + dy * dy + dz * dz)
    if not squared:
        return float("inf")
    return (sum(squared) / len(squared)) ** 0.5


def choose_fit_frame(frames: list[dict], model_rows: list[dict]) -> int:
    """Index of one full frame whose recorded points sit closest to the model.

    A reach at the end of a recording can hold every point and still sit far
    from the skeleton. Among frames with the most points, the closest one is
    drawn. A tie stays nearer the middle of the recording. The recorded
    coordinates are not changed.
    """
    if len(frames) != len(model_rows) or not frames:
        raise ValueError("The recorded markers and the IK model markers do not have the same number of frames.")
    fullest = max(_point_count(frame) for frame in frames)
    middle = (len(frames) - 1) / 2.0
    chosen = None
    chosen_key = None
    for index, frame in enumerate(frames):
        if _point_count(frame) != fullest:
            continue
        key = (_frame_miss_m(frame, model_rows[index]), abs(index - middle))
        if chosen_key is None or key < chosen_key:
            chosen = index
            chosen_key = key
    return int(chosen)


def bodies_from_row(row: dict) -> dict[str, tuple[float, float, float]]:
    """Read one skeleton_bodies.sto row into body-name → XYZ metres."""
    bodies = {}
    for key, value in row.items():
        if not key.endswith("_tx"):
            continue
        name = key[:-3]
        bodies[name] = (
            float(row[f"{name}_tx"]),
            float(row[f"{name}_ty"]),
            float(row[f"{name}_tz"]),
        )
    return bodies


def save_ik_fit_plot(
    trc_path: Path,
    model_locations_path: Path,
    path: Path,
    skeleton_bodies_path: Path | None = None,
    model_path: Path | None = None,
    take_rms_mm: float | None = None,
    ik_trc_path: Path | None = None,
) -> Path:
    """Draw one IK pose of the loaded model against the recorded markers.

    The frame is the fullest one whose recorded points are closest to the
    model. When body rotations and the bone files are available, the full
    skeleton meshes are drawn. Otherwise the body origins are joined as a
    stick figure. A sibling HTML file is the same pose, rotatable, with the
    gap on each marker. The vertical axis on the figure is camera up. The
    stored coordinates stay Y-down. Marker locations on the bones are not moved.
    """
    from src.opensim_upper.mot import read_mot
    from src.opensim_upper.trc import read_trc

    frames = read_trc(trc_path)["frames"]
    model_rows = read_mot(model_locations_path)["rows"]
    chosen = choose_fit_frame(frames, model_rows)
    recorded = frames[chosen]
    model_row = model_rows[chosen]
    model_points = {}
    for name in model_row:
        if name.endswith("_tx"):
            marker = name[:-3]
            model_points[marker] = (
                float(model_row[f"{marker}_tx"]),
                float(model_row[f"{marker}_ty"]),
                float(model_row[f"{marker}_tz"]),
            )
    used_in_ik = None
    if ik_trc_path is not None and Path(ik_trc_path).is_file():
        ik_frames = read_trc(ik_trc_path)["frames"]
        if len(ik_frames) == len(frames):
            used_in_ik = ik_frames[chosen]["points"]
    bodies = None
    transforms = None
    if skeleton_bodies_path is not None and Path(skeleton_bodies_path).is_file():
        from src.opensim_upper.bone_figure import transforms_from_row

        skeleton_rows = read_mot(skeleton_bodies_path)["rows"]
        if len(skeleton_rows) != len(frames):
            raise ValueError("The skeleton body table and the recorded markers do not have the same number of frames.")
        bodies = bodies_from_row(skeleton_rows[chosen])
        transforms = transforms_from_row(skeleton_rows[chosen])
    return _save_fit_figure(
        recorded,
        model_points,
        path,
        bodies=bodies,
        transforms=transforms,
        model_path=None if model_path is None else Path(model_path),
        take_rms_mm=take_rms_mm,
        used_in_ik=used_in_ik,
    )


def _screen(point: tuple[float, float, float]) -> tuple[float, float, float]:
    return (point[0], -point[1], point[2])


def _draw_links(ax, points: dict, links: tuple, color: str, width: float, label: str | None) -> list:
    drawn = []
    first = True
    for start, stop in links:
        if start not in points or stop not in points:
            continue
        a = _screen(points[start])
        b = _screen(points[stop])
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color=color,
            linewidth=width,
            solid_capstyle="round",
            label=label if first and label else None,
        )
        first = False
        drawn.extend((a, b))
    return drawn


def _render_fit_scene(scene: dict):
    """Raster the posed bones and the named points already placed in the scene."""
    import numpy as np

    from src.opensim_upper.bone_figure import render_triangles

    def column(key: str) -> np.ndarray:
        rows = [marker[key] for marker in scene["markers"] if marker[key] is not None]
        if not rows:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(rows, dtype=float)

    by_model = {marker["name"]: marker["model"] for marker in scene["markers"] if marker["model"] is not None}
    by_recorded = {
        marker["name"]: marker["recorded"] for marker in scene["markers"] if marker["recorded"] is not None
    }
    lines = []
    for start, stop in STUDY_SIDE_PAIRS[:4]:
        if start in by_model and stop in by_model:
            lines.append((np.asarray([by_model[start], by_model[stop]], dtype=float), (42, 111, 151), 2.4))
        if start in by_recorded and stop in by_recorded:
            lines.append(
                (np.asarray([by_recorded[start], by_recorded[stop]], dtype=float), (196, 92, 38), 2.4)
            )
    joints = (
        np.asarray([item["position"] for item in scene["joints"]], dtype=float)
        if scene["joints"]
        else np.zeros((0, 3), dtype=float)
    )
    groups = [
        (joints, (11, 31, 58), 5.0),
        (column("model"), (42, 111, 151), 6.5),
        (column("recorded"), (196, 92, 38), 8.0),
    ]
    return render_triangles(scene["triangles"], groups, lines=lines)


def _save_mesh_picture(image, path: Path, time_sec: float) -> None:
    from PIL import Image, ImageDraw

    width = image.shape[1]
    header = 86
    footer = 42
    canvas = Image.new("RGB", (width, image.shape[0] + header + footer), (255, 255, 255))
    canvas.paste(Image.fromarray(image), (0, header))
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    small = _font(14)
    draw.text(
        (18, 12),
        f"Full OpenSim skeleton at {time_sec:.2f} s. Not a clinical measurement.",
        fill=(20, 20, 20),
        font=font,
    )
    draw.text(
        (18, 42),
        "Shown standing. Legs are the model pose. Arms and points follow this fit.",
        fill=(60, 60, 60),
        font=small,
    )
    legend = (
        ((11, 31, 58), "joints"),
        ((42, 111, 151), "model markers"),
        ((196, 92, 38), "recorded points"),
    )
    x = 18
    y = header + image.shape[0] + 12
    for color, label in legend:
        draw.ellipse((x, y, x + 12, y + 12), fill=color, outline=(255, 255, 255))
        draw.text((x + 18, y - 2), label, fill=(30, 30, 30), font=small)
        x += 18 + int(draw.textlength(label, font=small)) + 28
    canvas.save(path)


def _font(size: int):
    from PIL import ImageFont

    for candidate in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _save_fit_figure(
    recorded: dict,
    model_points: dict,
    path: Path,
    bodies: dict[str, tuple[float, float, float]] | None = None,
    transforms: dict | None = None,
    model_path: Path | None = None,
    take_rms_mm: float | None = None,
    used_in_ik: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.opensim_upper.fit_view import build_fit_scene, write_fit_html

    scene = build_fit_scene(
        recorded,
        model_points,
        transforms,
        model_path,
        bodies=bodies,
        take_rms_mm=take_rms_mm,
        used_in_ik=used_in_ik,
    )
    write_fit_html(scene, path.with_suffix(".html"))
    if scene.get("triangles") is not None and len(scene["triangles"]):
        holding = path.with_name(path.stem + "_writing.png")
        _save_mesh_picture(_render_fit_scene(scene), holding, float(recorded["time_sec"]))
        try:
            holding.replace(path)
        except OSError:
            holding.unlink(missing_ok=True)
        return path
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    drawn: list = []
    if bodies:
        drawn.extend(_draw_links(ax, bodies, _BONE_LINKS, "#0b1f3a", 4.0, "OpenSim skeleton"))
        joints = [_screen(point) for point in bodies.values()]
        if joints:
            xs, ys, zs = zip(*joints)
            ax.scatter(xs, ys, zs, c="#0b1f3a", s=36, depthshade=False, label="joints")
            drawn.extend(joints)
    else:
        drawn.extend(_draw_links(ax, model_points, _MARKER_LINKS, "#1f4e79", 2.0, "sized model"))
    recorded_named = {
        name: point for name, point in recorded["points"].items() if point is not None
    }
    drawn.extend(_draw_links(ax, model_points, STUDY_SIDE_PAIRS[:4], "#2a6f97", 2.5, "model side pairs"))
    drawn.extend(_draw_links(ax, recorded_named, STUDY_SIDE_PAIRS[:4], "#c45c26", 2.5, "recorded side pairs"))
    if model_points:
        xs, ys, zs = zip(*(_screen(point) for point in model_points.values()))
        ax.scatter(xs, ys, zs, c="#2a6f97", s=28, depthshade=False, label="model markers")
        drawn.extend(_screen(point) for point in model_points.values())
    recorded_points = [_screen(point) for point in recorded_named.values()]
    if recorded_points:
        xs, ys, zs = zip(*recorded_points)
        ax.scatter(xs, ys, zs, c="#c45c26", s=40, depthshade=False, label="recorded points")
    ax.legend(loc="upper left")
    cloud = drawn + recorded_points
    if cloud:
        xs, ys, zs = zip(*cloud)
        span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 0.2)
        cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
        half = span / 2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)
    ax.view_init(elev=18, azim=-70)
    ax.set_xlabel("X m (camera right)")
    ax.set_ylabel("camera up (stored Y is down)")
    ax.set_zlabel("Z m (camera forward)")
    kind = "OpenSim skeleton" if bodies else "Sized-model markers"
    ax.set_title(
        f"{kind} fit at {float(recorded['time_sec']):.2f} s. Not a clinical measurement."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_model_frame_plot(transformed: dict, path: Path) -> Path:
    """Scatter registered landmarks. Observed and predicted markers share the transform."""
    if transformed.get("coord_frame") != "opensim_ground_metre":
        raise ValueError("This plot requires a registered pose.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for name in OBSERVED:
        row = transformed["landmarks"][name]
        ax.scatter(row["x_m"], row["y_m"], row["z_m"], c="#1f77b4", s=30)
    for name in PREDICTED:
        row = transformed["landmarks"][name]
        ax.scatter(row["x_m"], row["y_m"], row["z_m"], c="#e000ff", marker="D", s=28)
    ax.set_xlabel("X m (anterior)")
    ax.set_ylabel("Y m (up)")
    ax.set_zlabel("Z m (model right)")
    ax.set_title("Registered landmarks. Marker locations were not adjusted.")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
