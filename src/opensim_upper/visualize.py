"""Plot static landmarks.

``save_reference_plot`` draws camera metres before registration.
``save_model_frame_plot`` draws a pose that has already been transformed.
Neither plot moves model marker locations, and neither overlays a guessed skeleton.
"""

from __future__ import annotations

from pathlib import Path

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


_FIT_LINKS = (
    ("RShoulder", "LShoulder"),
    ("RShoulder", "RElbow"),
    ("LShoulder", "LElbow"),
    ("RElbow", "r_lwrist_study"),
    ("RElbow", "r_mwrist_study"),
    ("LElbow", "L_lwrist_study"),
    ("LElbow", "L_mwrist_study"),
)


def save_ik_fit_plot(trc_path: Path, model_locations_path: Path, path: Path) -> Path:
    """Draw the last IK pose of the loaded model against the recorded markers.

    The vertical axis on the figure is camera up. The stored coordinates stay
    Y-down. Marker locations on the bones are not moved.
    """
    from src.opensim_upper.mot import read_mot
    from src.opensim_upper.trc import read_trc

    frames = read_trc(trc_path)["frames"]
    model_rows = read_mot(model_locations_path)["rows"]
    if len(frames) != len(model_rows):
        raise ValueError("The recorded markers and the IK model markers do not have the same number of frames.")
    present = [sum(point is not None for point in frame["points"].values()) for frame in frames]
    fullest = max(present)
    recorded = [frame for frame, count in zip(frames, present) if count == fullest][-1]
    model_row = model_rows[frames.index(recorded)]
    model_points = {}
    for name in model_row:
        if name.endswith("_tx"):
            marker = name[:-3]
            model_points[marker] = (
                float(model_row[f"{marker}_tx"]),
                float(model_row[f"{marker}_ty"]),
                float(model_row[f"{marker}_tz"]),
            )
    return _save_fit_figure(recorded, model_points, path)


def _screen(point: tuple[float, float, float]) -> tuple[float, float, float]:
    return (point[0], -point[1], point[2])


def _save_fit_figure(recorded: dict, model_points: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    drawn = []
    for start, stop in _FIT_LINKS:
        if start not in model_points or stop not in model_points:
            continue
        a = _screen(model_points[start])
        b = _screen(model_points[stop])
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#1f4e79", linewidth=2)
        drawn.extend((a, b))
    if drawn:
        xs, ys, zs = zip(*drawn)
        ax.scatter(xs, ys, zs, c="#1f4e79", s=18, label="sized model")
    recorded_points = []
    for name, point in recorded["points"].items():
        if point is None:
            continue
        recorded_points.append(_screen(point))
    if recorded_points:
        xs, ys, zs = zip(*recorded_points)
        ax.scatter(xs, ys, zs, c="#c45c26", s=28, depthshade=False, label="recorded points")
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
    ax.set_xlabel("X m (camera right)")
    ax.set_ylabel("camera up (stored Y is down)")
    ax.set_zlabel("Z m (camera forward)")
    ax.set_title(
        f"Sized-model fit at {float(recorded['time_sec']):.2f} s. Not a clinical measurement."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
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
