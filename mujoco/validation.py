"""Bind demo results/checkpoints to the actual model and policy contract."""
import hashlib
import json
from pathlib import Path
from sim_robot import ROOT
from robot_interface import OBSERVATION_VERSION, JOINT_NAMES

def contract(config):
    model = Path(config["model"])
    if not model.is_absolute():
        model = ROOT / model
    paths = [model, *(ROOT / "meshes").glob("*.STL")]
    paths += [ROOT / "mujoco" / name for name in
              ("sim_robot.py", "robot_interface.py", "hexapod_rl_env.py", "rewards.py")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    cfg = dict(config, model=model.name)
    return dict(observation_version=OBSERVATION_VERSION, joint_names=JOINT_NAMES,
                robot_config=cfg, sha256=hashes)

def require_validation(config):
    expected = contract(config)
    for filename in ("inspect_robot.json", "random_test.json"):
        path = ROOT / "logs" / filename
        if not path.exists():
            raise RuntimeError(f"Run model inspection then random test first: missing {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("passed") or report.get("contract") != expected:
            raise RuntimeError(f"Validation stale or failed: rerun {filename} demo")
    return expected
