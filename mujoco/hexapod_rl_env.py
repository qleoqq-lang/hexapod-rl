"""Adapted from the archive's Gymnasium environment. Constructs ONLY SimRobot."""
import argparse
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim_robot import SimRobot, load_config
from rewards import compute_reward
from robot_interface import OBSERVATION_VERSION

class HexapodMuJoCoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, model_path=None, render_mode=None, config_path=None,
                 frame_skip=None, episode_seconds=None, action_scale_rad=None):
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or human")
        self.config = load_config(config_path)
        for k, v in (("model", model_path), ("frame_skip", frame_skip),
                     ("episode_seconds", episode_seconds), ("action_scale_rad", action_scale_rad)):
            if v is not None:
                self.config[k] = str(v) if k == "model" else v
        self.robot = SimRobot(self.config)
        self.model, self.data = self.robot.model, self.robot.data
        self.render_mode, self.viewer = render_mode, None
        self.metadata = dict(type(self).metadata, render_fps=round(1 / self.robot.dt))
        self.max_steps = int(self.config["episode_seconds"] / self.robot.dt)
        if self.max_steps < 1:
            raise ValueError("episode_seconds must allow at least one control step")
        self.action_space = spaces.Box(-1., 1., (18,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (63,), dtype=np.float32)
        self.last_action = np.zeros(18)
        self.step_count = 0
        self.needs_reset = True

    def get_observation(self):
        obs = self.robot.get_observation(self.last_action)
        if not self.observation_space.contains(obs):
            raise FloatingPointError("Invalid observation")
        return obs

    def _info(self):
        state = self.robot.get_state()
        return dict(base_height=float(state.base_position[2]),
                    forward_velocity=float(state.world_linear_velocity[0]),
                    upright=float(-state.projected_gravity[2]),
                    foot_contacts=int(state.tibia_contacts.sum()),
                    observation_version=OBSERVATION_VERSION)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.reset(self.np_random, noise=(options or {}).get("noise", True))
        self.last_action[:] = 0
        self.step_count = 0
        self.needs_reset = False
        if self.render_mode == "human":
            self.render()
        return self.get_observation(), self._info()

    def termination_reasons(self, state, joint_violation=0.):
        cfg = self.config["termination"]
        height = float(state.base_position[2])
        upright = float(-state.projected_gravity[2])
        reasons = []
        if upright <= 0:
            reasons.append("flipped")
        elif upright < cfg["min_upright"]:
            reasons.append("excessive_tilt")
        if height < cfg["min_height_m"]:
            reasons.append("height_low")
        if height > cfg["max_height_m"]:
            reasons.append("height_high")
        if joint_violation > cfg["joint_tolerance_rad"]:
            reasons.append("joint_limit")
        return reasons

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Call reset() before step() and after an episode ends")
        telemetry = self.robot.apply_action(action)
        applied = telemetry.pop("applied_action")
        state = self.robot.get_state()
        reasons = self.termination_reasons(state, telemetry["joint_violation_rad"])
        terminated = bool(reasons)
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        reward, terms = compute_reward(state, applied, self.last_action, telemetry["energy"],
                                       terminated, self.config["reward"])
        if not np.isfinite(reward):
            raise FloatingPointError("Non-finite reward")
        self.last_action[:] = applied
        self.needs_reset = terminated or truncated
        info = self._info() | telemetry | dict(reward_terms=terms, termination_reasons=reasons)
        if self.render_mode == "human":
            self.render()
        return self.get_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        import mujoco.viewer
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 1.25
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25
        if self.viewer.is_running():
            self.viewer.cam.lookat[:] = self.data.qpos[:3]
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.robot.close()

def smoke_test(steps=500, model_path=None):
    env = HexapodMuJoCoEnv(model_path=model_path)
    try:
        obs, info = env.reset(seed=7)
        total = 0.
        terminated = truncated = False
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(np.zeros(18, np.float32))
            total += reward
            if terminated or truncated:
                break
        return dict(observation_shape=obs.shape, steps=env.step_count, total_reward=total,
                    terminated=terminated, truncated=truncated, **info)
    finally:
        env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    print(json.dumps(smoke_test(args.steps, args.model), indent=2))

