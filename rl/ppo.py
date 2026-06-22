"""PPO (Proximal Policy Optimization) implementation for hexapod gait learning."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PPOBuffer:
    """Experience buffer for PPO trajectory."""

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.path_start = 0
        self.max_size = buffer_size
        self.full = False

    def store(self, obs, action, reward, done, value, log_prob):
        """Store a transition."""
        idx = self.ptr % self.max_size
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.ptr += 1
        if self.ptr >= self.max_size:
            self.full = True

    def compute_gae(self, gamma: float, gae_lambda: float, last_value: float):
        """Compute Generalized Advantage Estimation."""
        path_slice = slice(self.path_start, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        self.returns[path_slice] = returns
        self.advantages[path_slice] = advantages
        self.path_start = self.ptr

    def get_batches(self, batch_size: int) -> List[Tuple]:
        """Generate mini-batches for PPO update."""
        total = self.ptr if not self.full else self.max_size
        indices = np.random.permutation(total)
        batches = []
        for start in range(0, total, batch_size):
            batch_idx = indices[start:start + batch_size]
            batches.append((
                torch.FloatTensor(self.observations[batch_idx]),
                torch.FloatTensor(self.actions[batch_idx]),
                torch.FloatTensor(self.log_probs[batch_idx]),
                torch.FloatTensor(self.returns[batch_idx]),
                torch.FloatTensor(self.advantages[batch_idx]),
            ))
        return batches


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh(),
            ])
            in_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        
        # Actor (policy) head
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        features = self.features(obs)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        value = self.critic(features).squeeze(-1)
        return mean, std, value

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, get log prob and value."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probs, entropy, and values for given actions."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, value


class PPO:
    """PPO algorithm for hexapod gait optimization."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # Networks
        self.ac = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        # Buffer
        self.buffer = PPOBuffer(buffer_size, obs_dim, action_dim)
        self.buffer_size = buffer_size
        self.steps = 0

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Select action given observation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.ac.get_action(
                obs_t, deterministic=deterministic
            )
        return (
            action.cpu().numpy().squeeze(),
            log_prob.cpu().numpy().squeeze(),
            value.cpu().numpy().squeeze(),
        )

    def store_transition(
        self, obs, action, reward, done, value, log_prob
    ):
        """Store transition and auto-compute GAE when buffer is full."""
        self.buffer.store(obs, action, reward, done, value, log_prob)
        self.steps += 1

    def update(self, last_value: float) -> dict:
        """Update policy using collected trajectories."""
        # Compute advantages
        self.buffer.compute_gae(self.gamma, self.gae_lambda, last_value)

        # PPO update
        total = self.buffer.ptr if not self.buffer.full else self.buffer_size
        clip_fractions = []
        approx_kls = []
        loss_vals = []

        for _ in range(self.update_epochs):
            batches = self.buffer.get_batches(self.batch_size)
            for obs_b, actions_b, old_log_probs_b, returns_b, adv_b in batches:
                obs_b = obs_b.to(self.device)
                actions_b = actions_b.to(self.device)
                old_log_probs_b = old_log_probs_b.to(self.device)
                returns_b = returns_b.to(self.device)
                adv_b = adv_b.to(self.device)

                # Normalize advantages
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # Evaluate
                log_probs, entropy, values = self.ac.evaluate_actions(
                    obs_b, actions_b
                )

                # Ratio
                ratio = torch.exp(log_probs - old_log_probs_b)

                # Clipped surrogate objective
                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values, returns_b)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Tracking
                clip_frac = (ratio < 1 - self.clip_epsilon).float().mean() + \
                            (ratio > 1 + self.clip_epsilon).float().mean()
                clip_fractions.append(clip_frac.item())
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (ratio).log()).mean()
                    approx_kls.append(approx_kl.item())
                loss_vals.append(loss.item())

        # Clear buffer for next rollout
        self.buffer = PPOBuffer(self.buffer_size, self.buffer.ptr, 
                                self.buffer.observations.shape[1])

        return {
            "policy_loss": np.mean(loss_vals),
            "clip_fraction": np.mean(clip_fractions),
            "approx_kl": np.mean(approx_kls),
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.ac.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
