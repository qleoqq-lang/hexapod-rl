"""Extend the archive's Stable-Baselines3 PPO entrypoint with validation/resume/logging."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml
from hexapod_rl_env import HexapodMuJoCoEnv
from sim_robot import ROOT, load_config
from validation import require_validation

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--config", type=Path, default=ROOT / "configs/robot.yaml")
    p.add_argument("--train-config", type=Path, default=ROOT / "configs/train.yaml")
    p.add_argument("--resume", type=Path)
    p.add_argument("--run-name", default=datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
    args = p.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        p.error("run-name must be a plain directory name")
    cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    steps = cfg["total_timesteps"] if args.steps is None else args.steps
    workers = cfg["workers"] if args.workers is None else args.workers
    if steps < 1 or workers < 1:
        p.error("steps and workers must be positive")
    # 训练前安全门 / 版本一致性检查器
    policy_contract = require_validation(load_config(args.config))
    
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecCheckNan
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(cfg["torch_threads"])
    logs = ROOT / "logs" / args.run_name
    checkpoints = ROOT / "checkpoints" / args.run_name
    if logs.exists() or checkpoints.exists():
        raise FileExistsError("Use a new run-name to preserve previous results")
    logs.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    env = make_vec_env(HexapodMuJoCoEnv, n_envs=workers, seed=cfg["seed"],
                       monitor_dir=str(logs), env_kwargs={"config_path": args.config})
    env = VecCheckNan(env, raise_exception=True)
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=cfg["device"])
            if getattr(model, "hexapod_contract", None) != policy_contract:
                raise ValueError("Checkpoint contract mismatch; model/config/observation/action changed")
        else:
            model = PPO("MlpPolicy", env, verbose=1, device=cfg["device"], seed=cfg["seed"],
                        **{k: cfg[k] for k in ("n_steps", "batch_size", "n_epochs", "learning_rate",
                                              "gamma", "gae_lambda", "clip_range")})
            model.hexapod_contract = policy_contract
        model.set_logger(configure(str(logs), ["stdout", "csv", "tensorboard"]))
        initial_steps = model.num_timesteps

        class FiniteTraining(BaseCallback):
            def __init__(self):
                super().__init__()
                self.loss_checks = []
                self.episodes = 0

            def check_training(self):
                scalars = {k: float(v) for k, v in self.model.logger.name_to_value.items()
                           if k.startswith("train/") and isinstance(v, (int, float, np.number))}
                if not all(np.isfinite(v) for v in scalars.values()):
                    raise FloatingPointError("Non-finite training loss/metric")
                if not all(torch.isfinite(v).all() for v in self.model.policy.state_dict().values()):
                    raise FloatingPointError("Non-finite policy parameter")
                for state in self.model.policy.optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                            raise FloatingPointError("Non-finite optimizer state")
                if "train/loss" in scalars:
                    self.loss_checks.append(dict(timesteps=self.num_timesteps, **scalars))

            def _on_rollout_start(self):
                self.check_training()

            def _on_step(self):
                for info in self.locals["infos"]:
                    if "episode" in info:
                        self.episodes += 1
                    for name, value in info.get("reward_terms", {}).items():
                        self.logger.record_mean("reward/" + name, value)
                    for name in ("base_height", "forward_velocity", "upright", "energy", "joint_violation_rad"):
                        if name in info:
                            self.logger.record_mean("robot/" + name, info[name])
                if self.n_calls % max(1, cfg["checkpoint_every"] // workers) == 0:
                    self.model.save(checkpoints / f"step_{self.num_timesteps}")
                return True

            def _on_training_end(self):
                self.check_training()

        callback = FiniteTraining()
        model.learn(total_timesteps=steps, callback=callback,
                    reset_num_timesteps=not bool(args.resume))
        if not callback.loss_checks:
            raise RuntimeError("No PPO optimizer update was verified")
        model.logger.dump(model.num_timesteps)
        model.save(checkpoints / "final")
        summary = dict(passed=True, initial_timesteps=initial_steps,
                       final_timesteps=model.num_timesteps, requested_additional_steps=steps,
                       completed_episodes=callback.episodes, device=str(model.device),
                       checkpoint=str(checkpoints / "final.zip"), loss_checks=callback.loss_checks)
        (logs / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (logs / "robot_config.yaml").write_text(yaml.safe_dump(load_config(args.config)), encoding="utf-8")
        (logs / "train_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (logs / "contract.json").write_text(json.dumps(policy_contract, indent=2), encoding="utf-8")
        print("Saved:", checkpoints / "final.zip")
    finally:
        env.close()

if __name__ == "__main__":
    main()

