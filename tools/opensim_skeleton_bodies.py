"""Write OpenSim body poses for every IK frame.

    C:\\Users\\jayjo\\anaconda3\\envs\\opensim-upper\\python.exe \\
        tools/opensim_skeleton_bodies.py <model.osim> <ik_results.mot> <out.sto>

The sized model is posed from the MOT. Each body origin in the camera ground
is written as *_tx *_ty *_tz, then a row-major rotation *_r00 .. *_r22.
The sized model is the upper body. Pelvis and both legs are taken from the
full source model at its standing pose and welded to the fitted torso, so the
picture can show the whole skeleton. Geometry mesh files are not required.
The model file is not edited.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


BODY_ORDER = (
    "torso",
    "head",
    "humerus_r",
    "ulna_r",
    "radius_r",
    "hand_r",
    "humerus_l",
    "ulna_l",
    "radius_l",
    "hand_l",
)

# Present on the full Rajagopal/Pose2Sim file. Not solved by this upper-body IK.
LOWER_BODIES = (
    "pelvis",
    "sacrum",
    "femur_r",
    "tibia_r",
    "talus_r",
    "calcn_r",
    "toes_r",
    "femur_l",
    "tibia_l",
    "talus_l",
    "calcn_l",
    "toes_l",
)

SOURCE_MODEL = (
    Path(__file__).resolve().parents[1] / "models" / "opensim_upper" / "Model_Pose2Sim_upper.osim"
)


def _quiet_opensim_log() -> None:
    import opensim

    try:
        opensim.Logger.setLevelString("Off")
    except Exception:
        return


def _transform_matrix(body, state):
    import numpy as np

    transform = body.getTransformInGround(state)
    rotation = transform.R()
    point = transform.p()
    matrix = np.eye(4)
    for i in range(3):
        matrix[i, 3] = float(point.get(i))
        for j in range(3):
            matrix[i, j] = float(rotation.get(i, j))
    return matrix


def _torso_relatives(source_path: Path) -> dict:
    """Body pose in the torso frame at the source model's default standing pose."""
    import numpy as np
    import opensim

    _quiet_opensim_log()
    model = opensim.Model(str(source_path))
    state = model.initSystem()
    model.realizePosition(state)
    body_set = model.getBodySet()
    if not body_set.contains("torso"):
        raise ValueError(f"{source_path} has no torso body.")
    torso_inv = np.linalg.inv(_transform_matrix(body_set.get("torso"), state))
    relatives = {}
    for index in range(body_set.getSize()):
        body = body_set.get(index)
        relatives[body.getName()] = torso_inv @ _transform_matrix(body, state)
    missing = [name for name in LOWER_BODIES if name not in relatives]
    if missing:
        raise ValueError(f"Full-body model is missing bodies needed for the skeleton plot: {missing}")
    return relatives


def write_skeleton_bodies(model_path: Path, mot_path: Path, out_path: Path) -> Path:
    import opensim

    model_path = Path(model_path)
    mot_path = Path(mot_path)
    out_path = Path(out_path)
    before = model_path.read_bytes()
    _quiet_opensim_log()
    relatives = _torso_relatives(SOURCE_MODEL) if SOURCE_MODEL.is_file() else {}
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    table = opensim.TimeSeriesTable(str(mot_path))
    labels = list(table.getColumnLabels())
    in_degrees = table.getTableMetaDataAsString("inDegrees").lower() == "yes"
    body_set = model.getBodySet()
    missing = [name for name in BODY_ORDER if not body_set.contains(name)]
    if missing:
        raise ValueError(f"Model is missing bodies needed for the skeleton plot: {missing}")
    export_names = list(BODY_ORDER)
    for name in LOWER_BODIES:
        if body_set.contains(name) or name in relatives:
            export_names.append(name)
    columns = ["time"]
    for name in export_names:
        columns.extend((f"{name}_tx", f"{name}_ty", f"{name}_tz"))
    for name in export_names:
        for row_index in range(3):
            for col_index in range(3):
                columns.append(f"{name}_r{row_index}{col_index}")
    rows: list[list[float]] = []
    times = table.getIndependentColumn()
    for index in range(table.getNumRows()):
        for j, label in enumerate(labels):
            if not model.getCoordinateSet().contains(label):
                continue
            coordinate = model.updCoordinateSet().get(label)
            value = float(table.getDependentColumnAtIndex(j)[index])
            if in_degrees and int(coordinate.getMotionType()) == 1:
                value = math.radians(value)
            coordinate.setValue(state, value, False)
        model.realizePosition(state)
        posed = {
            name: _transform_matrix(body_set.get(name), state)
            for name in BODY_ORDER
        }
        torso = posed["torso"]
        for name in export_names:
            if name in posed:
                continue
            if body_set.contains(name):
                posed[name] = _transform_matrix(body_set.get(name), state)
            else:
                posed[name] = torso @ relatives[name]
        row = [float(times[index])]
        for name in export_names:
            origin = posed[name][:3, 3]
            row.extend((float(origin[0]), float(origin[1]), float(origin[2])))
        for name in export_names:
            rotation = posed[name][:3, :3].reshape(-1)
            row.extend(float(value) for value in rotation)
        rows.append(row)
    if model_path.read_bytes() != before:
        raise RuntimeError("The model file changed while writing skeleton bodies.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "skeleton_bodies",
        f"nRows={len(rows)}",
        f"nColumns={len(columns)}",
        "inDegrees=no",
        "endheader",
        "\t".join(columns),
    ]
    for row in rows:
        lines.append("\t".join(f"{value:.8f}" for value in row))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        opensim.Logger.setLevelString("Info")
    except Exception:
        pass
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: opensim_skeleton_bodies.py model.osim ik_results.mot out.sto")
        return 2
    path = write_skeleton_bodies(Path(argv[1]), Path(argv[2]), Path(argv[3]))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
