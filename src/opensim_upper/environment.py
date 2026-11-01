"""Isolated OpenSim 4.4 environment. The application interpreter is not modified.

OpenSim's Windows build is a conda package (Apache-2.0), not a pip wheel.
Pose2Sim 0.10.49's own env does not import opensim. This env is separate.

Create it with the base Anaconda conda, from the project folder:

    C:\\Users\\jayjo\\anaconda3\\Scripts\\conda.exe create -n opensim-upper -y ^
        python=3.11 "numpy=1.23.*" "opensim=4.4" -c opensim-org

Check it:

    C:\\Users\\jayjo\\anaconda3\\envs\\opensim-upper\\python.exe -c "import opensim; print(opensim.GetVersion())"

The model file stays in the Pose2Sim install. It is not copied into this repo.
"""

from __future__ import annotations

import os
from pathlib import Path

OPENSIM_PYTHON = Path(
    os.environ.get(
        "OPENSIM_PYTHON",
        r"C:\Users\jayjo\anaconda3\envs\opensim-upper\python.exe",
    )
)
POSE2SIM_MODEL = Path(
    os.environ.get(
        "POSE2SIM_SIMPLE_OSIM",
        r"C:\Users\jayjo\anaconda3\envs\pose2sim-offline\Lib\site-packages"
        r"\Pose2Sim\OpenSim_Setup\Model_Pose2Sim_simple.osim",
    )
)
POSE2SIM_MARKERS = POSE2SIM_MODEL.parent / "Markers_LSTM.xml"
POSE2SIM_GEOMETRY = POSE2SIM_MODEL.parent / "Geometry"

INSTALL_COMMAND = (
    r"C:\Users\jayjo\anaconda3\Scripts\conda.exe create -n opensim-upper -y "
    r'python=3.11 "numpy=1.23.*" "opensim=4.4" -c opensim-org'
)
