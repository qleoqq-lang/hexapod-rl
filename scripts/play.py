"""Load a validated simulation-only PPO checkpoint and run inference."""
import argparse
import json
import time
from pathlib import Path
from _bootstrap import ROOT
from hexapod_rl_env import HexapodMuJoCoEnv
from validation import contract

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "logs/play.json")
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    env = HexapodMuJoCoEnv(config_path=args.config, render_mode=None if args.headless else "human")
    try:
        model = PPO.load(args.checkpoint, device="cpu")
        if getattr(model, "hexapod_contract", None) != contract(env.config):
            raise ValueError("Checkpoint model/observation/action/config does not match this environment")
        rows = []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            start = env.robot.get_state().base_position.copy()
            total = 0.
            while True:
                wall = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if not args.headless:
                    time.sleep(max(0, env.robot.dt - (time.perf_counter() - wall)))
                if terminated or truncated:
                    break
            end = env.robot.get_state().base_position
            rows.append(dict(episode=ep, steps=env.step_count, reward=total,
                             displacement_x_m=float(end[0] - start[0]),
                             mean_forward_m_s=float((end[0] - start[0]) / (env.step_count * env.robot.dt)),
                             terminated=terminated, truncated=truncated,
                             termination_reasons=info["termination_reasons"]))
        result = dict(checkpoint=str(args.checkpoint.resolve()), trained_timesteps=model.num_timesteps,
                      episodes=rows, inference_passed=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.close()

if __name__ == "__main__":
    main()
