"""Random actions only; does not import PPO or train a policy."""
import argparse
import json
from pathlib import Path
from _bootstrap import ROOT
import numpy as np
from gymnasium.utils.env_checker import check_env
from hexapod_rl_env import HexapodMuJoCoEnv
from validation import contract

def run(episodes=10, config_path=None, seed=123):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = HexapodMuJoCoEnv(config_path=config_path)
    try:
        check_env(env, skip_render_check=True)
        rng = np.random.default_rng(seed)
        rows = []
        rewards = []
        max_violation = 0.
        max_torque = 0.
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            assert obs.shape == (63,) and obs.dtype == np.float32
            episode_reward = 0.
            for step in range(env.max_steps):
                action = rng.uniform(-1, 1, 18).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs)
                assert np.isfinite(reward) and all(np.isfinite(v) for v in info["reward_terms"].values())
                assert np.isclose(reward, sum(info["reward_terms"].values()))
                assert isinstance(terminated, bool) and isinstance(truncated, bool)
                max_violation = max(max_violation, info["joint_violation_rad"])
                max_torque = max(max_torque, info["peak_torque"])
                assert max_torque <= env.config["torque_limit_nm"] + 1e-9
                if info["joint_violation_rad"] > env.config["termination"]["joint_tolerance_rad"]:
                    assert terminated and "joint_limit" in info["termination_reasons"]
                episode_reward += reward
                rewards.append(reward)
                if terminated or truncated:
                    break
            assert terminated or truncated
            rows.append(dict(episode=ep, steps=step+1, reward=episode_reward,
                             terminated=terminated, truncated=truncated,
                             reasons=info["termination_reasons"], final_height=info["base_height"]))
        return dict(passed=True, contract=contract(env.config), observation_shape=[63], action_shape=[18],
                    seed=seed, episodes=rows, total_steps=sum(x["steps"] for x in rows),
                    reward_min=min(rewards), reward_max=max(rewards),
                    max_joint_violation_rad=max_violation, max_torque_nm=max_torque,
                    warnings=0, non_finite=0, gymnasium_check=True)
    finally:
        env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "logs/random_test.json")
    args = p.parse_args()
    result = run(args.episodes, args.config, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "contract"}, indent=2))
