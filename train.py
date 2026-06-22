#!/usr/bin/env python3
"""Training entry point for hexapod RL gait optimization.

Usage:
    python train.py --algo ppo --gait tripod --timesteps 5_000_000
    python train.py --algo sac --terrain rough
"""

import argparse
import os
import time
import numpy as np
from datetime import datetime

from sim.config import HexapodConfig, RLConfig
from sim.hexapod import HexapodEnv
from rl.ppo import PPO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train hexapod RL gait optimization"
    )
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["ppo", "sac"],
        help="RL algorithm"
    )
    parser.add_argument(
        "--gait", type=str, default="tripod",
        choices=["tripod", "ripple", "wave"],
        help="Gait type to optimize"
    )
    parser.add_argument(
        "--terrain", type=str, default="flat",
        choices=["flat", "slope", "rough", "stairs"],
        help="Terrain type"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable GUI rendering"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Load checkpoint to resume training"
    )
    return parser.parse_args()


def train_ppo(env, config, total_timesteps, log_dir):
    """Train hexapod using PPO."""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=config.ppo_lr,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
        clip_epsilon=config.ppo_clip_epsilon,
        entropy_coef=config.ppo_entropy_coef,
        value_coef=config.ppo_value_coef,
        max_grad_norm=config.ppo_max_grad_norm,
        update_epochs=config.ppo_update_epochs,
        batch_size=config.ppo_batch_size,
        buffer_size=config.ppo_buffer_size,
    )

    # Training loop
    obs, _ = env.reset()
    episode_rewards = []
    episode_lengths = []
    ep_reward = 0
    ep_length = 0
    best_reward = -float("inf")
    start_time = time.time()

    for step in range(total_timesteps):
        # Select action
        action, log_prob, value = agent.select_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.store_transition(obs, action, reward, done, value, log_prob)

        ep_reward += reward
        ep_length += 1
        obs = next_obs

        # Episode end
        if done:
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            obs, _ = env.reset()
            ep_reward = 0
            ep_length = 0

        # Update policy when buffer is full
        if agent.steps >= agent.buffer_size:
            # Get value of last observation for GAE
            _, _, last_val = agent.select_action(obs)
            update_info = agent.update(last_val)

            # Logging
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(
                f"[{elapsed:6.0f}s] Step {step:7d}/{total_timesteps} | "
                f"Avg Reward: {avg_reward:6.2f} | "
                f"Loss: {update_info['policy_loss']:.4f} | "
                f"KL: {update_info['approx_kl']:.4f} | "
                f"Episodes: {len(episode_rewards)}"
            )

            # Save best model
            if avg_reward > best_reward and episode_rewards:
                best_reward = avg_reward
                model_path = os.path.join(log_dir, "best_model.zip")
                agent.save(model_path)
                print(f"  -> Saved best model ({avg_reward:.2f})")

            # Save periodic checkpoint
            if step % 100000 == 0 and step > 0:
                ckpt_path = os.path.join(
                    log_dir, f"checkpoint_{step}.zip"
                )
                agent.save(ckpt_path)
                print(f"  -> Saved checkpoint at step {step}")

    # Save final model
    agent.save(os.path.join(log_dir, "final_model.zip"))
    print(f"\nTraining complete! Model saved to {log_dir}/")
    return agent


def main():
    args = parse_args()

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        "runs", f"{args.algo}_{args.gait}_{args.terrain}_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    rl_cfg = RLConfig()
    env = HexapodEnv(
        hexapod_cfg=HexapodConfig(),
        rl_cfg=rl_cfg,
        terrain_type=args.terrain,
        render_mode=args.render,
    )

    print(f"=== Hexapod RL Training ===")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Gait:      {args.gait}")
    print(f"Terrain:   {args.terrain}")
    print(f"Steps:     {args.timesteps:,}")
    print(f"Log dir:   {log_dir}")
    print(f"Obs dim:   {env.observation_space.shape[0]}")
    print(f"Act dim:   {env.action_space.shape[0]}")
    print("=" * 30)

    try:
        if args.algo == "ppo":
            train_ppo(env, rl_cfg, args.timesteps, log_dir)
        else:
            print("SAC not yet implemented - using PPO")
            train_ppo(env, rl_cfg, args.timesteps, log_dir)
    finally:
        env.close()


if __name__ == "__main__":
    main()
