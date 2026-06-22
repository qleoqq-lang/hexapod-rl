"""Reward functions for hexapod locomotion training."""

import numpy as np
from typing import Dict


class RewardFunction:
    """Composite reward function for hexapod gait optimization.

    Combines multiple reward terms with tunable weights:
    - Forward velocity reward
    - Energy efficiency penalty
    - Body stability penalty
    - Foot contact reward
    - Alive bonus
    - Target approach bonus
    """

    def __init__(self):
        # Reward weights
        self.w_velocity = 1.0
        self.w_energy = -0.1
        self.w_stability = -0.5
        self.w_contact = 0.2
        self.w_alive = 0.1
        self.w_target = 2.0

        # Previous position for delta computation
        self._prev_position = np.array([0.0, 0.0])
        self._prev_dist_to_target = 0.0

    def compute(self, state: Dict) -> float:
        """Compute composite reward from current state."""
        reward = 0.0

        reward += self.w_velocity * self._reward_velocity(state)
        reward += self.w_energy * self._penalty_energy(state)
        reward += self.w_stability * self._penalty_stability(state)
        reward += self.w_contact * self._reward_contact(state)
        reward += self.w_alive * self._reward_alive()
        reward += self.w_target * self._reward_target_approach(state)

        return reward

    def _reward_velocity(self, state: Dict) -> float:
        """Reward forward (x-axis) velocity."""
        vx = state["body_linear_vel"][0]
        vy = state["body_linear_vel"][1]
        forward_speed = max(0, vx)  # Only reward forward motion
        lateral_penalty = abs(vy) * 0.5
        return forward_speed - lateral_penalty

    def _penalty_energy(self, state: Dict) -> float:
        """Penalize high joint velocity / torque (energy consumption)."""
        # Simplified: penalize large joint position deviations
        joint_pos = state["joint_positions"]
        energy = np.sum(np.square(joint_pos)) / len(joint_pos)
        return energy

    def _penalty_stability(self, state: Dict) -> float:
        """Penalize excessive body rotation (roll/pitch)."""
        roll, pitch, _ = state["body_rpy"]
        return abs(roll) + abs(pitch)

    def _reward_contact(self, state: Dict) -> float:
        """Reward periodic foot contact (good gait has alternating contacts)."""
        contacts = state["foot_contacts"]
        # Penalize all feet off ground or all on ground
        n_contact = np.sum(contacts)
        if n_contact == 0 or n_contact == 6:
            return -0.5
        # Reward ~3 legs in contact (tripod gait ideal)
        return 1.0 - abs(3 - n_contact) / 3.0

    def _reward_alive(self) -> float:
        """Small alive bonus to encourage survival."""
        return 1.0

    def _reward_target_approach(self, state: Dict) -> float:
        """Reward getting closer to target position."""
        # Note: target position would need to be accessible here
        # This is a placeholder for navigation tasks
        return 0.0

    def reset(self):
        """Reset internal state between episodes."""
        self._prev_position = np.array([0.0, 0.0])
        self._prev_dist_to_target = 0.0
