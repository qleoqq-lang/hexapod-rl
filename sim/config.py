"""Simulation and training configuration."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class HexapodConfig:
    """Six-legged robot kinematic parameters."""
    # Body dimensions
    body_length: float = 0.24       # m
    body_width: float = 0.12        # m
    body_height: float = 0.06       # m
    leg_count: int = 6
    
    # Coxa (hip) joint
    coxa_length: float = 0.04       # m
    coxa_range: Tuple[float, float] = (-1.57, 1.57)  # rad
    
    # Femur (upper leg) joint  
    femur_length: float = 0.10      # m
    femur_range: Tuple[float, float] = (-2.36, 0.78)  # rad
    
    # Tibia (lower leg) joint
    tibia_length: float = 0.14      # m
    tibia_range: Tuple[float, float] = (-2.36, 0.78)   # rad
    
    # Leg placement around body (angles from body center)
    leg_angles: list = field(default_factory=lambda: 
        [60, 120, 180, 240, 300, 360])  # degrees
    
    # Default standing pose (coxa, femur, tibia) in radians
    default_angles: list = field(default_factory=lambda:
        [[0.0, -1.2, 1.8] for _ in range(6)])


@dataclass
class RLConfig:
    """Reinforcement learning hyperparameters."""
    # Environment
    max_episode_steps: int = 1000
    num_envs: int = 1
    
    # PPO
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_epsilon: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_update_epochs: int = 10
    ppo_batch_size: int = 64
    ppo_buffer_size: int = 2048
    
    # SAC
    sac_lr: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_alpha: float = 0.2
    sac_buffer_size: int = 1_000_000
    sac_batch_size: int = 256
    
    # Network architecture
    hidden_dim: int = 256
    num_hidden_layers: int = 2


@dataclass
class GaitConfig:
    """Gait pattern parameters."""
    # Tripod gait (fast walking)
    tripod_cycle_time: float = 0.4     # s per stride cycle
    tripod_stance_ratio: float = 0.5   # fraction of cycle in stance
    tripod_step_height: float = 0.05   # m
    
    # Ripple gait (stability)
    ripple_cycle_time: float = 0.6
    ripple_stance_ratio: float = 0.6
    ripple_step_height: float = 0.04
    
    # Wave gait (slow, stable)
    wave_cycle_time: float = 0.8
    wave_stance_ratio: float = 0.75
    wave_step_height: float = 0.03


@dataclass
class TerrainConfig:
    """Terrain configuration."""
    terrain_type: str = "flat"  # flat, slope, rough, stairs
    slope_angle: float = 10.0   # degrees
    rough_height_range: Tuple[float, float] = (-0.02, 0.02)
    stair_height: float = 0.03
    stair_width: float = 0.15
