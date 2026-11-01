"""Load an upper-body model and a TRC with the isolated OpenSim interpreter.

    C:\\Users\\jayjo\\anaconda3\\envs\\opensim-upper\\python.exe tools/opensim_trc_check.py <model.osim> <markers.trc> <result.json>

Does not modify the model, the geometry files, or the TRC. Does not run IK.
"""

from __future__ import annotations

import json
import math
import sys


def main() -> int:
    import opensim

    if len(sys.argv) != 4:
        print("usage: opensim_trc_check.py model.osim markers.trc result.json")
        return 1
    model_path, trc_path, result_path = sys.argv[1:]
    model = opensim.Model(model_path)
    model.initSystem()
    model_names = [model.getMarkerSet().get(i).getName() for i in range(model.getMarkerSet().getSize())]
    table = opensim.TimeSeriesTableVec3(trc_path)
    labels = list(table.getColumnLabels())
    marker_data = opensim.MarkerData(trc_path)
    units = table.getTableMetaDataString("Units")
    nan_counts = {}
    for index, label in enumerate(labels):
        column = table.getDependentColumnAtIndex(index)
        missing = 0
        for row_index in range(table.getNumRows()):
            value = column[row_index]
            if any(math.isnan(value[axis]) for axis in range(3)):
                missing += 1
        nan_counts[label] = missing
    payload = {
        "ok": True,
        "model_markers": model_names,
        "trc_markers": labels,
        "unrecognised_markers": [name for name in labels if name not in model_names],
        "missing_model_markers": [name for name in model_names if name not in labels],
        "n_frames": int(table.getNumRows()),
        "markerdata_n_frames": int(marker_data.getNumFrames()),
        "markerdata_n_markers": int(marker_data.getNumMarkers()),
        "units": units,
        "data_rate_hz": float(marker_data.getDataRate()),
        "nan_counts": nan_counts,
        "inverse_kinematics": False,
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
