"""Pick the correct RGB source from config (no hardcoded file type).

Step 2: mode "file" -> .bag uses BagFileRGB, other videos use VideoFileRGB.
Live camera stays available as mode "live" (Step 1).
"""

from __future__ import annotations

from pathlib import Path

from src.capture.bag_file import BagFileRGB
from src.capture.base_source import RGBSource
from src.capture.live_realsense import LiveRealSenseRGB
from src.capture.video_file import VIDEO_SUFFIXES, VideoFileRGB
from src.utils.config_loader import resolve_project_path


def create_rgb_source(config: dict) -> RGBSource:
    """Build a live camera, video file, or bag player from config.yaml.

    Args:
        config: Full YAML dict (must contain 'source' and, for live, 'camera').

    Returns:
        An object with start / get_frame / stop.

    Raises:
        ValueError: unknown mode or unsupported file suffix.
        KeyError: required config keys missing.
    """
    source_cfg = config["source"]
    mode = str(source_cfg["mode"]).strip().lower()

    if mode == "live":
        camera_cfg = config["camera"]
        return LiveRealSenseRGB(
            width=int(camera_cfg["width"]),
            height=int(camera_cfg["height"]),
            fps=int(camera_cfg["fps"]),
        )

    if mode == "file":
        file_path = resolve_project_path(source_cfg["file_path"])
        loop = bool(source_cfg.get("loop", False))
        return _source_from_file(file_path, loop=loop)

    raise ValueError(
        f"source.mode must be 'live' or 'file' (got {mode!r}). Edit config.yaml."
    )


def _source_from_file(file_path: Path, loop: bool) -> RGBSource:
    """Choose bag vs movie from the file extension (not hardcoded in the script)."""
    suffix = file_path.suffix.lower()
    if suffix == ".bag":
        return BagFileRGB(file_path=file_path, loop=loop, realtime=True)
    if suffix in VIDEO_SUFFIXES:
        return VideoFileRGB(file_path=file_path, loop=loop)
    raise ValueError(
        f"Unsupported upload type {suffix!r} for {file_path}. "
        "Use .bag (RealSense) or a colour movie (.mp4, .avi, .mov, ...)."
    )
