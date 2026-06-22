#!/usr/bin/env python3
"""Visualize training results.

Usage:
    python visualize.py --logdir runs/
    python visualize.py --logdir runs/ppo_tripod_flat/
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize hexapod RL training results"
    )
    parser.add_argument(
        "--logdir", type=str, default="runs/",
        help="Directory containing training runs"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save plots to files instead of displaying"
    )
    return parser.parse_args()


def find_log_files(logdir):
    """Find training log files in directory."""
    log_files = glob.glob(os.path.join(logdir, "**", "training_log.txt"), recursive=True)
    return log_files


def plot_reward_curves(logdir, save=False):
    """Plot reward curves from training logs."""
    log_files = find_log_files(logdir)

    if not log_files:
        print(f"No training logs found in {logdir}")
        print("You can also point directly to a run directory.")
        return

    plt.figure(figsize=(12, 6))

    for log_file in log_files:
        run_name = os.path.basename(os.path.dirname(log_file))
        rewards = []
        with open(log_file, "r") as f:
            for line in f:
                if "Avg Reward" in line:
                    try:
                        reward = float(
                            line.split("Avg Reward:")[1].split("|")[0].strip()
                        )
                        rewards.append(reward)
                    except (IndexError, ValueError):
                        pass

        if rewards:
            plt.plot(rewards, label=run_name, alpha=0.8)

    plt.xlabel("Training Step")
    plt.ylabel("Average Reward")
    plt.title("Training Reward Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save:
        plt.savefig(os.path.join(logdir, "reward_curves.png"), dpi=150)
        print(f"Saved reward_curves.png")
    else:
        plt.show()


def plot_gait_comparison():
    """Compare different gait types (conceptual visualization)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    gait_names = ["Tripod", "Ripple", "Wave"]
    
    # Conceptual gait diagrams: leg lifting patterns over time
    time_points = np.linspace(0, 1, 100)
    
    for ax, name in zip(axes, gait_names):
        for leg in range(6):
            if name == "Tripod":
                # Tripod: groups of 3 alternate
                in_group_a = leg in [0, 2, 4]
                swing = np.where(
                    time_points < 0.5,
                    int(in_group_a),
                    int(not in_group_a),
                )
            elif name == "Ripple":
                phases = [0.0, 0.33, 0.67, 0.17, 0.50, 0.83]
                phase = phases[leg]
                swing = np.where(
                    (time_points - phase) % 1.0 < 0.4,
                    1, 0
                )
            else:  # Wave
                phases = [0.0, 0.5, 0.25, 0.75, 0.125, 0.625]
                phase = phases[leg]
                swing = np.where(
                    (time_points - phase) % 1.0 < 0.25,
                    1, 0
                )
            
            ax.fill_between(
                time_points, leg - 0.4, leg + 0.4,
                where=swing == 1,
                color="orange", alpha=0.7, label="Swing" if leg == 0 else ""
            )
            ax.fill_between(
                time_points, leg - 0.4, leg + 0.4,
                where=swing == 0,
                color="blue", alpha=0.3, label="Stance" if leg == 0 else ""
            )
        
        ax.set_xlabel("Gait Cycle")
        ax.set_ylabel("Leg")
        ax.set_title(f"{name} Gait")
        ax.set_yticks(range(6))
        ax.set_yticklabels(["L1", "L2", "L3", "R1", "R2", "R3"])
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    print("=== Hexapod RL Training Visualization ===\n")

    # Plot reward curves
    plot_reward_curves(args.logdir, args.save)

    # Plot gait comparison
    fig = plot_gait_comparison()
    if args.save:
        fig.savefig(os.path.join(args.logdir, "gait_comparison.png"), dpi=150)
        print("Saved gait_comparison.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
