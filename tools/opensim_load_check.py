"""Load Model_Pose2Sim_simple.osim with the isolated OpenSim interpreter.

Run with opensim-upper's python, not the application python:

    C:\\Users\\jayjo\\anaconda3\\envs\\opensim-upper\\python.exe tools/opensim_load_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.opensim_upper.environment import POSE2SIM_GEOMETRY, POSE2SIM_MODEL  # noqa: E402


def main() -> int:
    import opensim

    if not POSE2SIM_MODEL.is_file():
        print(json.dumps({"ok": False, "error": f"missing model {POSE2SIM_MODEL}"}))
        return 1
    model = opensim.Model(str(POSE2SIM_MODEL))
    state = model.initSystem()
    names = [model.getCoordinateSet().get(i).getName() for i in range(model.getCoordinateSet().getSize())]
    upper = [name for name in names if name.startswith(("arm_", "elbow_", "pro_sup_", "wrist_"))]
    print(
        json.dumps(
            {
                "ok": True,
                "opensim_version": opensim.GetVersion(),
                "model_name": model.getName(),
                "n_coordinates": len(names),
                "n_bodies": model.getBodySet().getSize(),
                "n_markers_on_model": model.getMarkerSet().getSize(),
                "upper_coordinates": upper,
                "length_units": "meters",
                "state_ny": int(state.getNY()),
                "geometry_dir_exists": POSE2SIM_GEOMETRY.is_dir(),
                "model_path": str(POSE2SIM_MODEL),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
