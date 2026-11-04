"""Play the bone fit for the same length of time as the capture.

The live camera recording is left as it is. This writes a separate movie of
the OpenSim bones and the marker points, one posed frame per recorded sample,
at a rate that makes the file last as long as the capture.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from src.opensim_upper.bone_figure import (
    _shade,
    depth_raster,
    fit_view_camera,
    project_view,
    read_vtp,
    stand_points,
    transforms_from_row,
)

_VIDEO_WIDTH = 640
_VIDEO_HEIGHT = 860


def playback_fps(times: list[float]) -> float:
    """Frames per second that makes ``len(times)`` frames last ``times[-1] - times[0]``."""
    if len(times) < 2:
        return 1.0
    span = float(times[-1]) - float(times[0])
    if span <= 1e-6:
        return 1.0
    return len(times) / span


def _mesh_limit(filename: str, body: str) -> int:
    """Keep the skull, ribs, and spine whole. Thin only the small finger bones."""
    lower = filename.lower()
    if any(part in lower for part in ("phalanx", "meta", "carpal", "thumb", "digit")):
        return 80
    return 20000


def _decimate(faces: np.ndarray, limit: int) -> np.ndarray:
    count = len(faces)
    if count <= limit:
        return faces
    index = np.linspace(0, count - 1, limit).astype(np.int64)
    return faces[index]


def load_playback_meshes(model_path: Path | None) -> list[tuple[str, np.ndarray]]:
    """Body name and local triangles, scaled and thinned for playback."""
    from src.opensim_upper.bone_figure import merged_body_meshes
    from src.opensim_upper.camera_model import SOURCE_MODEL
    from src.opensim_upper.environment import POSE2SIM_GEOMETRY

    if not SOURCE_MODEL.is_file() or not POSE2SIM_GEOMETRY.is_dir():
        return []
    table = merged_body_meshes(SOURCE_MODEL, model_path)
    meshes = []
    for body, items in table.items():
        for item in items:
            file_path = POSE2SIM_GEOMETRY / item["file"]
            if not file_path.is_file():
                continue
            points, triangles = read_vtp(str(file_path))
            local = np.asarray(points, dtype=np.float64) * np.asarray(item["scale"], dtype=np.float64)
            faces = local[triangles]
            faces = _decimate(faces, _mesh_limit(item["file"], body))
            if len(faces):
                meshes.append((body, np.ascontiguousarray(faces, dtype=np.float32)))
    return meshes


def pose_playback_meshes(
    meshes: list[tuple[str, np.ndarray]],
    transforms: dict,
) -> np.ndarray | None:
    """Posed triangles in the upright display frame, or None when the torso is missing."""
    if "torso" not in transforms:
        return None
    chunks = []
    for name, faces in meshes:
        if name not in transforms:
            continue
        rotation, origin = transforms[name]
        flat = np.asarray(faces, dtype=np.float64).reshape(-1, 3)
        world = flat @ np.asarray(rotation, dtype=np.float64).T + np.asarray(origin, dtype=np.float64)
        chunks.append(world.reshape((-1, 3, 3)))
    if not chunks:
        return None
    return stand_points(np.concatenate(chunks, axis=0), transforms)


def _show_points(points: list[tuple[float, float, float]], transforms: dict | None) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=float)
    array = np.asarray(points, dtype=float)
    if transforms and "torso" in transforms:
        return stand_points(array, transforms)
    array = array.copy()
    array[:, 1] *= -1.0
    return array


def render_playback_frame(
    triangles: np.ndarray | None,
    point_groups: list[tuple[np.ndarray, tuple[int, int, int], int]],
    view: dict,
    caption: str,
    lines: list[tuple[np.ndarray, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """One BGR frame. Bones first, then points, so the markers stay visible."""
    if triangles is not None and len(triangles):
        screen = project_view(triangles.reshape(-1, 3), view).reshape(-1, 3, 3)
        colors = _shade(np.asarray(triangles, dtype=np.float64))[:, ::-1]
        image = depth_raster(screen, colors, view["height"], view["width"])
    else:
        image = np.full((view["height"], view["width"], 3), 247, dtype=np.uint8)
    for segment, color in lines or ():
        if len(segment) != 2:
            continue
        projected = project_view(segment, view)
        cv2.line(
            image,
            (int(round(projected[0, 0])), int(round(projected[0, 1]))),
            (int(round(projected[1, 0])), int(round(projected[1, 1]))),
            color,
            2,
            cv2.LINE_AA,
        )
    for points, color, radius in point_groups:
        if len(points) == 0:
            continue
        projected = project_view(points, view)
        for px, py, _depth in projected:
            center = (int(round(px)), int(round(py)))
            cv2.circle(image, center, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        caption,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return image


def _marker_points(row: dict) -> dict[str, tuple[float, float, float]]:
    points = {}
    for name, value in row.items():
        if not str(name).endswith("_tx"):
            continue
        marker = str(name)[:-3]
        points[marker] = (
            float(row[f"{marker}_tx"]),
            float(row[f"{marker}_ty"]),
            float(row[f"{marker}_tz"]),
        )
    return points


def _stick_lines(transforms: dict | None, bodies: dict | None) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    from src.opensim_upper.visualize import _BONE_LINKS

    if transforms:
        points = {name: origin for name, (_rotation, origin) in transforms.items()}
        shown = {
            name: _show_points([point], transforms)[0]
            for name, point in points.items()
        }
    elif bodies:
        shown = {name: _show_points([point], None)[0] for name, point in bodies.items()}
    else:
        return []
    lines = []
    for start, stop in _BONE_LINKS:
        if start in shown and stop in shown:
            lines.append((np.vstack((shown[start], shown[stop])), (58, 31, 11)))
    return lines


def write_fit_video(
    trc_path: Path,
    model_locations_path: Path,
    path: Path,
    skeleton_bodies_path: Path | None = None,
    model_path: Path | None = None,
) -> Path:
    """Write the bone movie. Playback length matches the capture length."""
    from src.io.save_video import OverlayVideoWriter
    from src.opensim_upper.mot import read_mot
    from src.opensim_upper.trc import read_trc
    from src.opensim_upper.visualize import bodies_from_row

    frames = read_trc(trc_path)["frames"]
    model_rows = read_mot(model_locations_path)["rows"]
    if len(frames) != len(model_rows) or not frames:
        raise ValueError("The recorded markers and the IK model markers do not have the same number of frames.")
    skeleton_rows = None
    if skeleton_bodies_path is not None and Path(skeleton_bodies_path).is_file():
        skeleton_rows = read_mot(skeleton_bodies_path)["rows"]
        if len(skeleton_rows) != len(frames):
            raise ValueError("The skeleton body table and the recorded markers do not have the same number of frames.")
    meshes = load_playback_meshes(None if model_path is None else Path(model_path))
    times = [float(frame["time_sec"]) for frame in frames]
    samples = {0, len(frames) // 2, len(frames) - 1}
    bound_parts = []
    prepared = []
    for index, frame in enumerate(frames):
        transforms = transforms_from_row(skeleton_rows[index]) if skeleton_rows else None
        triangles = (
            pose_playback_meshes(meshes, transforms)
            if transforms and meshes and index in samples
            else None
        )
        recorded_shown = _show_points(
            [point for point in frame["points"].values() if point is not None],
            transforms,
        )
        model_shown = _show_points(list(_marker_points(model_rows[index]).values()), transforms)
        if transforms:
            joints = _show_points([origin for _rotation, origin in transforms.values()], transforms)
        elif skeleton_rows:
            joints = _show_points(list(bodies_from_row(skeleton_rows[index]).values()), None)
        else:
            joints = np.zeros((0, 3), dtype=float)
        if triangles is not None and index in samples:
            bound_parts.append(triangles.reshape(-1, 3))
        if len(recorded_shown):
            bound_parts.append(recorded_shown)
        if len(model_shown):
            bound_parts.append(model_shown)
        if len(joints):
            bound_parts.append(joints)
        prepared.append((transforms, triangles if index in samples else None, recorded_shown, model_shown, joints))
    if not bound_parts:
        raise ValueError("The fit video has no bones or points to draw.")
    bounds = np.concatenate(bound_parts, axis=0)
    low = bounds.min(axis=0) - 0.16
    high = bounds.max(axis=0) + 0.16
    corners = np.array(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
        dtype=float,
    )
    view = fit_view_camera(corners, _VIDEO_WIDTH, _VIDEO_HEIGHT)
    fps = playback_fps(times)
    writer = OverlayVideoWriter(Path(path), fps, "mp4v")
    try:
        for index, frame in enumerate(frames):
            transforms, sampled, recorded_shown, model_shown, joints = prepared[index]
            triangles = sampled
            if triangles is None and transforms and meshes:
                triangles = pose_playback_meshes(meshes, transforms)
            groups = [
                (joints, (58, 31, 11), 4),
                (model_shown, (151, 111, 42), 6),
                (recorded_shown, (38, 92, 196), 7),
            ]
            lines = [] if triangles is not None else _stick_lines(transforms, None)
            caption = f"Bone fit  {float(frame['time_sec']):.2f} s    Not a clinical measurement."
            writer.write(render_playback_frame(triangles, groups, view, caption, lines))
    finally:
        writer.close()
    return Path(path)


class BonePlayback:
    """Read the bone movie in step with the capture clock."""

    def __init__(self, path: Path) -> None:
        self.cap = cv2.VideoCapture(str(path))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 1.0)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.index = -1
        self.started = time.perf_counter()
        self.done = self.count <= 0
        self.last: np.ndarray | None = None

    def frame(self) -> np.ndarray | None:
        if self.done:
            return self.last
        target = int((time.perf_counter() - self.started) * max(self.fps, 1e-3))
        if target >= self.count:
            self.done = True
            return self.last
        while self.index < target:
            ok, image = self.cap.read()
            self.index += 1
            if not ok:
                self.done = True
                break
            self.last = image
        return self.last

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
