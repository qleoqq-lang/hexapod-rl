"""No policy, no RL: validate model, stance, and signed single-joint response."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from _bootstrap import ROOT
import mujoco
import numpy as np
from sim_robot import SimRobot, load_config
from validation import contract

def inspect(config_path=None, model_path=None, render=False, snapshot=None):
    config = load_config(config_path)
    if model_path:
        config["model"] = str(Path(model_path).resolve())
    robot = SimRobot(config)
    viewer = None
    try:
        assert len(robot.tibia_ids) == 6 and len(robot.joint_ids) == 18
        assert np.allclose(np.rad2deg(robot.nominal_command).reshape(6, 3), [0, -35, 110])
        if render:
            from mujoco.viewer import launch_passive
            viewer = launch_passive(robot.model, robot.data)
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.25, 135, -25
        initial = asdict(robot.get_state())
        min_height, max_height, min_upright, max_violation = float("inf"), 0., 1., 0.
        def tick(action):
            nonlocal min_height, max_height, min_upright, max_violation
            telemetry = robot.apply_action(action)
            s = robot.get_state()
            min_height = min(min_height, float(s.base_position[2]))
            max_height = max(max_height, float(s.base_position[2]))
            min_upright = min(min_upright, float(-s.projected_gravity[2]))
            max_violation = max(max_violation, telemetry["joint_violation_rad"])
            assert config["termination"]["min_height_m"] < s.base_position[2] < config["termination"]["max_height_m"]
            assert -s.projected_gravity[2] > config["termination"]["min_upright"]
            assert max_violation <= config["termination"]["joint_tolerance_rad"]
            if viewer and viewer.is_running():
                viewer.cam.lookat[:] = s.base_position
                viewer.sync()
                time.sleep(robot.dt)
        zero = np.zeros(18)
        for _ in range(500):
            tick(zero)
        settled = asdict(robot.get_state())
        assert sum(settled["tibia_contacts"]) == 6
        responses = []
        for j, name in enumerate(robot.joint_names):
            before = robot.get_state().joint_position[j]
            action = np.zeros(18)
            action[j] = 0.03 / robot.action_scale
            for _ in range(30):
                tick(action)
            response = float(robot.get_state().joint_position[j] - before)
            assert response > 0.003, (name, response)
            responses.append(dict(joint=name, target_delta_rad=0.03, measured_delta_rad=response))
            for _ in range(30):
                tick(zero)
        if snapshot:
            from PIL import Image
            robot.model.vis.global_.offwidth = 800
            robot.model.vis.global_.offheight = 600
            with mujoco.Renderer(robot.model, height=600, width=800) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = robot.data.qpos[:3]
                camera.distance, camera.azimuth, camera.elevation = 1.25, 135, -25
                renderer.update_scene(robot.data, camera=camera)
                Path(snapshot).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(renderer.render()).save(snapshot)
        return dict(passed=True, contract=contract(config), model=str(robot.model_path), legs=6, controlled_joints=18,
                    nq=robot.model.nq, nv=robot.model.nv, mass_kg=float(robot.model.body_mass.sum()),
                    joint_names=robot.joint_names, joint_limits_rad=robot.limits.tolist(),
                    initial={k: v.tolist() for k, v in initial.items()},
                    settled={k: v.tolist() for k, v in settled.items()},
                    single_joint_responses=responses, min_height=min_height, max_height=max_height,
                    min_upright=min_upright, max_joint_violation_rad=max_violation,
                    control_steps=1580, physics_steps=1580 * robot.frame_skip,
                    warnings=int(sum(w.number for w in robot.data.warning)))
    finally:
        if viewer:
            viewer.close()
        robot.close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--render", action="store_true")
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "logs/inspect_robot.json")
    args = p.parse_args()
    result = inspect(args.config, args.model, args.render, args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("contract", "initial", "settled", "joint_limits_rad")}, indent=2))
    print("Report:", args.output)

if __name__ == "__main__":
    main()
