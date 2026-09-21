"""Load settings from config.yaml so nothing important is hardcoded.

Why:
    Sir asked that resolution, FPS, and similar values must not be buried in
    code. This module is the single door into that YAML file.
"""

from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return the folder that contains config.yaml (the repo root).

    We walk up from this file: src/utils/config_loader.py -> src -> root.
    That way the script still finds config even if you run it from elsewhere.
    """
    # Path(__file__) is this Python file. .resolve() makes it an absolute path.
    this_file = Path(__file__).resolve()
    # parents[0] = utils, [1] = src, [2] = project root
    return this_file.parents[2]


def load_config(config_path: Path | None = None) -> dict:
    """Read YAML config and return it as a nested dictionary.

    Args:
        config_path: Optional path to a YAML file. If omitted, we use
            <project_root>/config.yaml.

    Returns:
        dict with keys such as 'project', 'camera', 'display'.

    Raises:
        FileNotFoundError: config file is missing.
        ValueError: file exists but is empty or not valid YAML mapping.
    """
    if config_path is None:
        config_path = get_project_root() / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # encoding='utf-8' so comments and names like "Jay" always read correctly.
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping (got {type(data)})")

    return data


def resolve_project_path(maybe_relative: str | Path) -> Path:
    """Turn a config path into an absolute path.

    Relative paths are from the project folder (where config.yaml lives),
    so 'data/input/clip.mp4' works no matter which folder you run from.
    Absolute paths (D:/videos/clip.mp4) are left unchanged.
    """
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return get_project_root() / path
