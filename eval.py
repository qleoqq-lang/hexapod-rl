#!/usr/bin/env python3
"""Evaluate a trained hexapod RL model.

Usage:
    python eval.py --checkpoint runs/ppo_tripod_flat/best_model.zip
    python eval.py --checkpoint runs/ppo_tripod_flat/best_model.zip --render --episodes 10
"""

import argparse
import numpy as np
import time
from collections import deque

from sim.config import HexapodConfig, RLConfig
from sim.hexapod import HexapodEnv
from rl.ppo import PPO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained hexapod RL model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable GUI rendering"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--terrain", type=str, default="flat",
        choices=["flat", "slope", "rough", "stairs"],
        help="Terrain type for evaluation"
    )
    return parser.parse_args()


def evaluate():
    args = parse_args()

    # Create environment
    env = HexapodEnv(
        hexapod_cfg=HexapodConfig(),
        rl_cfg=RLConfig(),
        terrain_type=args.terrain,
        render_mode=args.render,
    )

    # Create agent and load checkpoint
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(obs_dim=obs_dim, action_dim=action_dim)
    agent.load(args.checkpoint)
    agent.ac.eval()  # Set to evaluation mode

    print(f"=== Evaluating {args.checkpoint} ===")
    print(f"Terrain: {args.terrain}")
    print(f"Episodes: {args.episodes}")

    episode_rewards = []
    episode_lengths = []
    episode_distances = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0
        positions = []

        while not done:
            # Deterministic action for evaluation
            action, _, _ = agent.select_action(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_len += 1
            positions.append([info.get("x_position", 0), info.get("y_position", 0)])

            if args.render:
                time.sleep(1 / 60)  # Match simulation speed

        # Compute distance traveled
        if positions:
            total_dist = sum(
                np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
                for i in range(len(positions) - 1)
            )
        else:
            total_dist = 0.0

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        episode_distances.append(total_dist)

        print(
            f"  Episode {ep+1:2d}: "
            f"Reward={ep_reward:.1f}, "
            f"Length={ep_len}, "
            f"Distance={total_dist:.2f}m"
        )

    # Summary statistics
    print(f"\n--- Summary (over {args.episodes} episodes) ---")
    print(f"Avg Reward:   {np.mean(episode_rewards):.1f} +/- {np.std(episode_rewards):.1f}")
    print(f"Avg Length:   {np.mean(episode_lengths):.0f}")
    print(f"Avg Distance: {np.mean(episode_distances):.2f}m")

    env.close()
    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
