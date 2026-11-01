"""Print default-pose ground positions for markers on an upper-body model copy.

Run with the isolated OpenSim interpreter:

    C:\\Users\\jayjo\\anaconda3\\envs\\opensim-upper\\python.exe tools/opensim_upper_marker_ground.py <model.osim>

These positions are the generic model in its default pose. They are not a
subject capture, and this script does not move marker locations or run IK.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    import opensim

    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "pass the model copy path"}))
        return 1
    model = opensim.Model(sys.argv[1])
    state = model.initSystem()
    model.realizePosition(state)
    markers = {}
    marker_set = model.getMarkerSet()
    for index in range(marker_set.getSize()):
        marker = marker_set.get(index)
        ground = marker.getLocationInGround(state)
        markers[marker.getName()] = [ground.get(0), ground.get(1), ground.get(2)]
    print(
        json.dumps(
            {
                "ok": True,
                "pose": "model default coordinates",
                "units": "metres",
                "frame": "ground",
                "markers_moved": False,
                "inverse_kinematics": False,
                "markers_m": markers,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
