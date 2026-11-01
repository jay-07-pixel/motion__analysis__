"""Offline upper-body IK. The live application is not called.

The OpenSim solve runs in the isolated opensim-upper interpreter. A MOT file
is not treated as a successful biomechanical result; the report separates
execution from reliability.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.opensim_upper.environment import OPENSIM_PYTHON
from src.opensim_upper.ik_tasks import ik_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPPER_MODEL = PROJECT_ROOT / "models" / "opensim_upper" / "Model_Pose2Sim_upper.osim"
_TOOL = PROJECT_ROOT / "tools" / "opensim_ik_run.py"


def ik_directory(project_root: Path, trial_id: str) -> Path:
    if not trial_id or any(character in trial_id for character in "\\/:"):
        raise ValueError("trial_id must be a single directory name.")
    return Path(project_root) / "output" / "opensim_upper" / "ik" / trial_id


def run_upper_body_ik(
    trc_path: Path,
    output_dir: Path,
    trial_metadata: Path | None = None,
    weight_overrides: dict[str, float] | None = None,
    model_path: Path | None = None,
) -> dict:
    """Validate the TRC, then run IK. The model file is not edited."""
    if not OPENSIM_PYTHON.is_file():
        raise FileNotFoundError(f"Isolated OpenSim interpreter was not found: {OPENSIM_PYTHON}")
    model = Path(model_path or UPPER_MODEL)
    if not model.is_file():
        raise FileNotFoundError(f"Upper-body model copy was not found: {model}")
    metadata = {}
    if trial_metadata is not None and Path(trial_metadata).is_file():
        metadata = json.loads(Path(trial_metadata).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "model": str(model.resolve()),
        "trc": str(Path(trc_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "weight_overrides": weight_overrides or {},
        "synthetic": metadata.get("synthetic"),
        "participant_capture": metadata.get("participant_capture"),
        "physical_validation": metadata.get("physical_validation", "not_performed"),
        "sampled_from_opensim_model": metadata.get("sampled_from_opensim_model"),
        "tasks_preview": ik_tasks(weight_overrides)["tasks"],
    }
    plan_path = output_dir / "ik_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    completed = subprocess.run(
        [str(OPENSIM_PYTHON), str(_TOOL), str(plan_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    log_path = output_dir / "ik_logs.txt"
    extra = (completed.stdout or "") + (completed.stderr or "")
    if extra.strip():
        previous = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        log_path.write_text(previous + "\n" + extra, encoding="utf-8")
    report_path = output_dir / "ik_report.json"
    if not report_path.is_file():
        raise RuntimeError(completed.stderr or completed.stdout or "IK did not write a report.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["return_code"] = completed.returncode
    return report
