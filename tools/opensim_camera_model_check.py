"""Load the camera-frame upper-body model and report its ground axes.

Run with the isolated OpenSim interpreter:

    C:\\Users\\jayjo\\anaconda3\\envs\\opensim-upper\\python.exe tools/opensim_camera_model_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.opensim_upper.camera_model import CAMERA_MODEL, FACING_CAMERA_ROTATION  # noqa: E402


def _column(rotation, index: int) -> list[float]:
    return [float(rotation.get(row, index)) for row in range(3)]


def main() -> int:
    import opensim

    model = opensim.Model(str(CAMERA_MODEL))
    state = model.initSystem()
    model.realizePosition(state)
    torso = model.getBodySet().get("torso")
    transform = torso.getTransformInGround(state)
    rotation = transform.R()
    origin = transform.p()
    measured = [_column(rotation, index) for index in range(3)]
    expected = FACING_CAMERA_ROTATION.tolist()
    mismatch = max(
        abs(measured[column][row] - expected[row][column])
        for column in range(3)
        for row in range(3)
    )
    joint = model.getJointSet().get("ground_thorax")
    gravity = model.getGravity()
    bodies = [model.getBodySet().get(i).getName() for i in range(model.getBodySet().getSize())]
    report = {
        "ok": mismatch < 1e-6 and joint.getChildFrame().findBaseFrame().getName() == "torso",
        "opensim_version": opensim.GetVersion(),
        "model_name": model.getName(),
        "n_bodies": len(bodies),
        "bodies": bodies,
        "n_markers": model.getMarkerSet().getSize(),
        "n_coordinates": model.getCoordinateSet().getSize(),
        "root_joint": joint.getName(),
        "root_parent": joint.getParentFrame().findBaseFrame().getName(),
        "root_child": joint.getChildFrame().findBaseFrame().getName(),
        "gravity": [float(gravity.get(i)) for i in range(3)],
        "torso_origin_m": [float(origin.get(i)) for i in range(3)],
        "torso_axes_in_ground": {
            "anterior_plus_x": measured[0],
            "up_plus_y": measured[1],
            "model_right_plus_z": measured[2],
        },
        "axis_mismatch": mismatch,
        "ground_axes": {"x": "camera_right", "y": "camera_down", "z": "camera_forward"},
    }
    print(json.dumps(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
