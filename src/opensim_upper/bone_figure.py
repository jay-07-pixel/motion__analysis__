"""Draw the full OpenSim bone meshes for one posed frame.

The picture uses the Rajagopal/Pose2Sim bone files (skull, ribs, arms, pelvis,
legs, feet, and the hand bones). Vertices stay in each body's frame, then the
IK body rotation and origin place them in camera metres. A missing geometry
file yields no image, and the caller keeps the stick figure.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.opensim_upper.model_xml import _local


def transforms_from_row(row: dict) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Body name to (rotation, origin) when the row stores rotations."""
    if "torso_r00" not in row or "torso_tx" not in row:
        return None
    names = [key[:-3] for key in row if key.endswith("_tx") and f"{key[:-3]}_r22" in row]
    transforms = {}
    for name in names:
        rotation = np.array(
            [
                [float(row[f"{name}_r00"]), float(row[f"{name}_r01"]), float(row[f"{name}_r02"])],
                [float(row[f"{name}_r10"]), float(row[f"{name}_r11"]), float(row[f"{name}_r12"])],
                [float(row[f"{name}_r20"]), float(row[f"{name}_r21"]), float(row[f"{name}_r22"])],
            ],
            dtype=float,
        )
        origin = np.array(
            [float(row[f"{name}_tx"]), float(row[f"{name}_ty"]), float(row[f"{name}_tz"])],
            dtype=float,
        )
        transforms[name] = (rotation, origin)
    return transforms or None


def body_meshes(path: Path) -> dict[str, list[dict]]:
    """Body name to the mesh files attached to that body, with scale factors."""
    table: dict[str, list[dict]] = {}
    for element in ET.parse(Path(path)).getroot().iter():
        if _local(element.tag) != "Body" or not element.get("name"):
            continue
        meshes = []
        for mesh in element.iter():
            if _local(mesh.tag) != "Mesh":
                continue
            filename = None
            scale = (1.0, 1.0, 1.0)
            for child in mesh:
                tag = _local(child.tag)
                if tag == "mesh_file" and child.text and child.text.strip():
                    filename = child.text.strip()
                elif tag == "scale_factors" and child.text:
                    parts = child.text.split()
                    if len(parts) == 3:
                        scale = (float(parts[0]), float(parts[1]), float(parts[2]))
            if filename:
                meshes.append({"file": filename, "scale": scale})
        table[element.get("name")] = meshes
    return table


def merged_body_meshes(source_path: Path, scaled_path: Path | None) -> dict[str, list[dict]]:
    """Full-body mesh list. A sized upper-body file replaces the bodies it still has."""
    table = body_meshes(source_path)
    if scaled_path is not None and Path(scaled_path).is_file():
        for name, meshes in body_meshes(scaled_path).items():
            if meshes:
                table[name] = meshes
    return table


@lru_cache(maxsize=128)
def read_vtp(path: str) -> tuple[np.ndarray, np.ndarray]:
    """ASCII VTK polydata as points (N, 3) and triangle indices (M, 3)."""
    root = ET.parse(path).getroot()
    points = None
    connectivity = None
    offsets = None
    for element in root.iter():
        tag = _local(element.tag)
        if tag == "Points":
            for child in element.iter():
                if _local(child.tag) == "DataArray" and child.text and child.text.strip():
                    points = np.fromstring(child.text, sep=" ", dtype=np.float64).reshape(-1, 3)
                    break
        elif tag == "Polys":
            for child in list(element):
                if _local(child.tag) != "DataArray" or not child.text:
                    continue
                name = (child.get("Name") or child.get("name") or "").lower()
                values = np.fromstring(child.text, sep=" ", dtype=np.int64)
                if name == "connectivity":
                    connectivity = values
                elif name == "offsets":
                    offsets = values
    if points is None or connectivity is None:
        raise ValueError(f"{path} has no polygon points.")
    triangles = []
    if offsets is None:
        if len(connectivity) % 3 != 0:
            raise ValueError(f"{path} polygons are not triangles.")
        triangles = connectivity.reshape(-1, 3)
        return points, triangles
    start = 0
    for end in offsets:
        polygon = connectivity[start : int(end)]
        start = int(end)
        if len(polygon) < 3:
            continue
        for index in range(1, len(polygon) - 1):
            triangles.append((int(polygon[0]), int(polygon[index]), int(polygon[index + 1])))
    if not triangles:
        raise ValueError(f"{path} has no triangles.")
    return points, np.asarray(triangles, dtype=np.int64)


def apply_body_pose(
    points: np.ndarray,
    scale: tuple[float, float, float],
    rotation: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Mesh vertices into camera metres. Scale is applied in the body frame."""
    local = np.asarray(points, dtype=float) * np.asarray(scale, dtype=float)
    return local @ np.asarray(rotation, dtype=float).T + np.asarray(origin, dtype=float)


def stand_points(points: np.ndarray, transforms: dict) -> np.ndarray:
    """Turn stored camera metres upright for the picture.

    Stored Y is down. The fitted thorax can pitch the body toward the camera,
    so the view is rotated about the torso until the body up axis is display
    up, body right is display +X, and the front faces the camera. Distances
    are unchanged. The stored coordinates are not edited.
    """
    rotation, torso_origin = transforms["torso"]
    shown = np.diag([1.0, -1.0, 1.0]) @ np.asarray(rotation, dtype=float)
    upright = np.column_stack(
        (
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
    )
    align = upright @ np.linalg.inv(shown)
    array = np.asarray(points, dtype=float)
    flat = np.array(array, dtype=float, copy=True).reshape(-1, 3)
    flat[:, 1] *= -1.0
    torso_shown = np.array(torso_origin, dtype=float, copy=True)
    torso_shown[1] *= -1.0
    stood = (flat - torso_shown) @ align.T + np.array([0.0, 0.0, 1.6])
    return stood.reshape(array.shape)


def posed_triangles(
    transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    meshes: dict[str, list[dict]],
    geometry_dir: Path,
) -> np.ndarray | None:
    """Every bone triangle in camera metres, shape (N, 3, 3)."""
    geometry_dir = Path(geometry_dir)
    if not geometry_dir.is_dir():
        return None
    chunks = []
    for name, (rotation, origin) in transforms.items():
        for mesh in meshes.get(name, ()):
            file_path = geometry_dir / mesh["file"]
            if not file_path.is_file():
                continue
            points, triangles = read_vtp(str(file_path))
            world = apply_body_pose(points, mesh["scale"], rotation, origin)
            chunks.append(world[triangles])
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _camera(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Look at the whole body from the front, a little to one side."""
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    span = float(max(np.max(high - low), 0.3))
    eye = center + np.array([-0.28, 0.12, -2.4]) * span
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    return eye, forward, right, up


def fit_view_camera(points: np.ndarray, width: int, height: int) -> dict:
    """One front view that frames ``points``. Used so a video does not reframe every frame."""
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    eye, forward, right, up = _camera(cloud)
    center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2.0
    centered = cloud - center
    half_x = max(float(np.max(np.abs(centered @ right))), 1e-3)
    half_y = max(float(np.max(np.abs(centered @ up))), 1e-3)
    scale = 0.90 * min((width / 2.0) / half_x, (height / 2.0) / half_y)
    return {
        "eye": eye,
        "forward": forward,
        "right": right,
        "up": up,
        "center": center,
        "scale": float(scale),
        "width": int(width),
        "height": int(height),
    }


def project_view(points: np.ndarray, view: dict) -> np.ndarray:
    """Project metres to pixel x, pixel y, and depth. Shape (N, 3)."""
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    relative = cloud - view["center"]
    px = view["width"] / 2.0 + (relative @ view["right"]) * view["scale"]
    py = view["height"] / 2.0 - (relative @ view["up"]) * view["scale"]
    depth = (cloud - view["eye"]) @ view["forward"]
    return np.stack([px, py, depth], axis=-1)


def depth_raster(screen: np.ndarray, colors: np.ndarray, height: int, width: int) -> np.ndarray:
    """Paint triangles with a per-pixel depth test.

    ``screen`` is (N, 3, 3) pixel x, pixel y, depth. A nearer depth wins, so a
    rib or the far side of the skull cannot draw over the front.
    """
    from array import array

    zrows = [array("f", [1.0e9]) * width for _ in range(height)]
    rows = [bytearray([247]) * (width * 3) for _ in range(height)]
    count = len(screen)
    for index in range(count):
        x0 = float(screen[index, 0, 0])
        y0 = float(screen[index, 0, 1])
        z0 = float(screen[index, 0, 2])
        x1 = float(screen[index, 1, 0])
        y1 = float(screen[index, 1, 1])
        z1 = float(screen[index, 1, 2])
        x2 = float(screen[index, 2, 0])
        y2 = float(screen[index, 2, 1])
        z2 = float(screen[index, 2, 2])
        if y0 > y1:
            x0, y0, z0, x1, y1, z1 = x1, y1, z1, x0, y0, z0
        if y1 > y2:
            x1, y1, z1, x2, y2, z2 = x2, y2, z2, x1, y1, z1
        if y0 > y1:
            x0, y0, z0, x1, y1, z1 = x1, y1, z1, x0, y0, z0
        if y2 - y0 < 1e-4 or y2 < 0.0 or y0 >= height:
            continue
        color_b = int(colors[index, 0])
        color_g = int(colors[index, 1])
        color_r = int(colors[index, 2])
        y_top = 0 if y0 <= 0.0 else int(y0) if y0 == int(y0) else int(y0) + 1
        y_bot = height - 1 if y2 >= height else int(y2) if y2 > int(y2) else int(y2) - 1
        if y_bot < y_top:
            continue
        long_span = y2 - y0
        for y in range(y_top, y_bot + 1):
            sample = y + 0.5
            if sample < y0 or sample > y2:
                continue
            along = (sample - y0) / long_span
            if along < 0.0:
                along = 0.0
            elif along > 1.0:
                along = 1.0
            left_x = x0 + (x2 - x0) * along
            left_z = z0 + (z2 - z0) * along
            if sample < y1 and y1 > y0:
                short_span = y1 - y0
                portion = 0.0 if short_span <= 1e-8 else (sample - y0) / short_span
                start_x, start_z = x0, z0
                stop_x, stop_z = x1, z1
            else:
                short_span = y2 - y1
                portion = 0.0 if short_span <= 1e-8 else (sample - y1) / short_span
                start_x, start_z = x1, z1
                stop_x, stop_z = x2, z2
            if portion < 0.0:
                portion = 0.0
            elif portion > 1.0:
                portion = 1.0
            right_x = start_x + (stop_x - start_x) * portion
            right_z = start_z + (stop_z - start_z) * portion
            if left_x > right_x:
                left_x, right_x = right_x, left_x
                left_z, right_z = right_z, left_z
            x_start = 0 if left_x <= 0.0 else int(left_x) if left_x == int(left_x) else int(left_x) + 1
            x_end = width - 1 if right_x >= width else int(right_x) if right_x > int(right_x) else int(right_x) - 1
            if x_end < x_start:
                continue
            span = right_x - left_x
            zrow = zrows[y]
            irow = rows[y]
            if span < 1e-6:
                if left_z < zrow[x_start]:
                    zrow[x_start] = left_z
                    base = x_start * 3
                    irow[base] = color_b
                    irow[base + 1] = color_g
                    irow[base + 2] = color_r
                continue
            step = (right_z - left_z) / span
            depth = left_z + (x_start + 0.5 - left_x) * step
            for x in range(x_start, x_end + 1):
                if depth < zrow[x]:
                    zrow[x] = depth
                    base = x * 3
                    irow[base] = color_b
                    irow[base + 1] = color_g
                    irow[base + 2] = color_r
                depth += step
    packed = b"".join(rows)
    return np.frombuffer(packed, dtype=np.uint8).reshape(height, width, 3).copy()


def _shade(triangles: np.ndarray) -> np.ndarray:
    """Bone grey. Lighting is two-sided so a flipped winding does not go black."""
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    normal = np.cross(edges_a, edges_b)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(length, 1e-12)
    key = np.array([0.25, 0.72, -0.65], dtype=float)
    key = key / np.linalg.norm(key)
    fill = np.array([-0.55, 0.15, -0.35], dtype=float)
    fill = fill / np.linalg.norm(fill)
    light = 0.34 + 0.48 * np.abs(normal @ key) + 0.22 * np.abs(normal @ fill)
    bone = np.array([214.0, 206.0, 196.0])
    return np.clip(bone * light[:, None], 0, 255).astype(np.uint8)


def _raster(image: np.ndarray, zbuf: np.ndarray, screen: np.ndarray, color: np.ndarray) -> None:
    height, width = zbuf.shape
    xs = screen[:, 0]
    ys = screen[:, 1]
    if xs.max() < 0 or ys.max() < 0 or xs.min() >= width or ys.min() >= height:
        return
    minx = max(int(np.floor(xs.min())), 0)
    maxx = min(int(np.ceil(xs.max())), width - 1)
    miny = max(int(np.floor(ys.min())), 0)
    maxy = min(int(np.ceil(ys.max())), height - 1)
    if minx > maxx or miny > maxy:
        return
    x0, y0 = screen[0, 0], screen[0, 1]
    x1, y1 = screen[1, 0], screen[1, 1]
    x2, y2 = screen[2, 0], screen[2, 1]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denominator)) < 1e-8:
        return
    grid_x = np.arange(minx, maxx + 1)
    grid_y = np.arange(miny, maxy + 1)
    x, y = np.meshgrid(grid_x, grid_y)
    w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
    w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
    w2 = 1.0 - w0 - w1
    mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    if not np.any(mask):
        return
    depth = w0 * screen[0, 2] + w1 * screen[1, 2] + w2 * screen[2, 2]
    view = zbuf[miny : maxy + 1, minx : maxx + 1]
    update = mask & (depth < view)
    if not np.any(update):
        return
    view[update] = depth[update]
    image[miny : maxy + 1, minx : maxx + 1][update] = color


def _paint_disk(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    color: tuple[int, int, int],
    ring: bool = True,
) -> None:
    height, width = image.shape[:2]
    reach = int(np.ceil(radius + 1.6))
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    x0 = max(0, cx_i - reach)
    x1 = min(width, cx_i + reach + 1)
    y0 = max(0, cy_i - reach)
    y1 = min(height, cy_i + reach + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = (xx - cx) ** 2 + (yy - cy) ** 2
    patch = image[y0:y1, x0:x1]
    if ring:
        patch[distance <= (radius + 1.5) ** 2] = (255, 255, 255)
    patch[distance <= radius ** 2] = color


def _paint_segment(
    image: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    radius: float,
) -> None:
    steps = max(int(np.hypot(x1 - x0, y1 - y0)), 1)
    for step in range(steps + 1):
        blend = step / steps
        _paint_disk(
            image,
            x0 + (x1 - x0) * blend,
            y0 + (y1 - y0) * blend,
            radius,
            color,
            ring=False,
        )


def render_triangles(
    triangles: np.ndarray,
    point_groups: list[tuple[np.ndarray, tuple[int, int, int], float]],
    width: int = 860,
    height: int = 1120,
    lines: list[tuple[np.ndarray, tuple[int, int, int], float]] | None = None,
) -> np.ndarray:
    """Raster the posed bones. ``triangles`` is (N, 3, 3) in display metres, Y up.

    Point groups are drawn after the bones so the markers stay visible.
    Each group is (points, colour, radius in pixels).
    """
    if len(triangles) == 0:
        raise ValueError("No bone triangles to draw.")
    cloud = triangles.reshape(-1, 3)
    eye, forward, right, up = _camera(cloud)
    centered = cloud - (cloud.min(axis=0) + cloud.max(axis=0)) / 2.0
    half_x = float(np.max(np.abs(centered @ right)))
    half_y = float(np.max(np.abs(centered @ up)))
    half_x = max(half_x, 1e-3)
    half_y = max(half_y, 1e-3)
    scale = 0.90 * min((width / 2.0) / half_x, (height / 2.0) / half_y)
    center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2.0

    def project(points: np.ndarray) -> np.ndarray:
        relative = points - center
        px = width / 2.0 + (relative @ right) * scale
        py = height / 2.0 - (relative @ up) * scale
        depth = (points - eye) @ forward
        return np.stack([px, py, depth], axis=-1)

    screen = project(cloud).reshape(-1, 3, 3)
    colors = _shade(triangles)
    image = np.full((height, width, 3), 247, dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    for index in range(len(screen)):
        _raster(image, zbuf, screen[index], colors[index])
    for segment, color, radius in lines or ():
        if len(segment) != 2:
            continue
        projected = project(np.asarray(segment, dtype=float))
        _paint_segment(
            image,
            float(projected[0, 0]),
            float(projected[0, 1]),
            float(projected[1, 0]),
            float(projected[1, 1]),
            color,
            radius,
        )
    for points, color, radius in point_groups:
        if len(points) == 0:
            continue
        projected = project(np.asarray(points, dtype=float))
        for px, py, _depth in projected:
            _paint_disk(image, float(px), float(py), radius, color)
    return image
