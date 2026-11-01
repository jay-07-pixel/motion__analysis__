"""Write an upper-body model copy. The Pose2Sim original is only read.

The copy keeps the original joint definitions and inserts the twelve
supported markers with the locations from ``Markers_LSTM.xml``. Geometry
files are not copied and not modified. C7, pelvis, ASIS, PSIS, and wrist
joint markers are not inserted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from src.opensim_upper.environment import POSE2SIM_GEOMETRY, POSE2SIM_MARKERS, POSE2SIM_MODEL
from src.opensim_upper.marker_map import target_names
from src.opensim_upper.model_xml import _local


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _marker_set(model_node: ET.Element) -> ET.Element:
    for element in model_node.iter():
        if _local(element.tag) == "MarkerSet":
            return element
    raise ValueError("The model has no MarkerSet.")


def _objects(marker_set: ET.Element) -> ET.Element:
    for child in list(marker_set):
        if _local(child.tag) == "objects":
            return child
    raise ValueError("The MarkerSet has no objects element.")


def write_upper_model(
    dest: Path,
    model_path: Path | None = None,
    markers_path: Path | None = None,
) -> dict:
    """Copy the model XML and insert only the supported marker definitions."""
    source = Path(model_path or POSE2SIM_MODEL)
    markers = Path(markers_path or POSE2SIM_MARKERS)
    dest = Path(dest)
    if not source.is_file():
        raise FileNotFoundError(f"Pose2Sim model not found: {source}")
    if not markers.is_file():
        raise FileNotFoundError(f"Pose2Sim marker file not found: {markers}")
    if dest.resolve() == source.resolve():
        raise ValueError("Refusing to overwrite the original Pose2Sim model.")
    if dest.resolve().parent == source.resolve().parent:
        raise ValueError("Refusing to write the copy beside the original model.")
    geometry = Path(POSE2SIM_GEOMETRY)
    if geometry.is_dir() and dest.resolve().is_relative_to(geometry.resolve()):
        raise ValueError("Refusing to write inside the original Geometry directory.")

    source_hash = _sha256(source)
    marker_hash = _sha256(markers)
    geometry_count = len(list(geometry.glob("*"))) if geometry.is_dir() else None

    allowed = set(target_names())
    model_tree = ET.parse(source)
    model_node = model_tree.getroot().find("Model")
    if model_node is None:
        raise ValueError(f"{source} has no Model element.")
    objects = _objects(_marker_set(model_node))
    if list(objects):
        raise ValueError("Refusing to insert markers into a model that already has some.")

    copied = []
    for marker in ET.parse(markers).getroot().iter():
        if _local(marker.tag) != "Marker":
            continue
        name = marker.get("name")
        if name not in allowed:
            continue
        objects.append(copy.deepcopy(marker))
        copied.append(name)
    missing = allowed.difference(copied)
    if missing:
        raise ValueError(f"Marker definitions missing from XML: {sorted(missing)}")

    model_node.insert(
        0,
        ET.Comment(
            " Upper-body marker subset. Locations are unchanged from Markers_LSTM.xml. "
            "Joints were not edited. Geometry files were not copied or modified. "
        ),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    model_tree.write(dest, encoding="utf-8", xml_declaration=True)

    if _sha256(source) != source_hash or _sha256(markers) != marker_hash:
        raise RuntimeError("The original model or marker file changed while writing the copy.")
    geometry_after = len(list(geometry.glob("*"))) if geometry.is_dir() else None
    if geometry_after != geometry_count:
        raise RuntimeError("The original Geometry directory changed while writing the copy.")

    note = {
        "copy": str(dest),
        "source_model": str(source),
        "source_markers": str(markers),
        "source_model_sha256": source_hash,
        "geometry_dir": str(geometry),
        "geometry_copied": False,
        "n_markers_inserted": len(copied),
        "marker_names": sorted(copied),
        "locations_adjusted": False,
        "joints_modified": False,
    }
    dest.with_suffix(".json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    return note
