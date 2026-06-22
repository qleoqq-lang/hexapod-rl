"""Hexapod robot simulation environment using PyBullet.

Implements an 18-DOF (6 legs x 3 joints) hexapod with:
- Forward kinematics for foot position calculation
- Inverse kinematics for joint angle calculation
- PyBullet simulation stepping
- Observation and action space definitions
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from sim.config import HexapodConfig, RLConfig
from rl.reward import RewardFunction
from sim.terrain import TerrainGenerator


class HexapodEnv(gym.Env):
    """Custom Gym environment for hexapod robot locomotion."""

    def __init__(
        self,
        hexapod_cfg: Optional[HexapodConfig] = None,
        rl_cfg: Optional[RLConfig] = None,
        terrain_type: str = "flat",
        render_mode: bool = False,
    ):
        super().__init__()
        self.hex_cfg = hexapod_cfg or HexapodConfig()
        self.rl_cfg = rl_cfg or RLConfig()
        self.terrain_type = terrain_type
        self.render_mode = render_mode
        self.reward_fn = RewardFunction()

        # Connect to PyBullet
        self._connect_pybullet()
        self.terrain_gen = TerrainGenerator(self.physics_client)

        # Observation: body pose (3) + body vel (6) + joint angles (18) +
        #              foot contacts (6) + terrain height (9) = 42 dims
        obs_dim = 3 + 6 + 18 + 6 + 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: target joint angles for 18 joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )

        # Internal state
        self.robot_id = None
        self.joint_indices = []
        self.joint_ranges = []
        self.foot_link_indices = []
        self._timestep = 0
        self._target_vel = 0.3  # m/s
        self._target_pos = np.array([2.0, 0.0])  # target (x, y)

        self._build_robot()

    def _connect_pybullet(self):
        """Initialize PyBullet connection."""
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240.0)

    def _build_robot(self):
        """Build the hexapod robot from URDF."""
        # Using a simple box body with legs
        # In practice, replace with your custom URDF
        urdf_path = "hexapod.urdf"  # Placeholder - load your real URDF here
        try:
            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0.1],
                useFixedBase=False,
                flags=p.URDF_USE_SELF_COLLISION,
            )
        except Exception:
            # Fallback: create a simple hexapod from primitives
            self.robot_id = self._build_simple_hexapod()

        # Extract joint info
        self.joint_indices = []
        self.joint_ranges = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                self.joint_indices.append(i)
                lower = info[8]
                upper = info[9]
                self.joint_ranges.append((lower, upper))

    def _build_simple_hexapod(self) -> int:
        """Build a simple hexapod from collision shapes (fallback)."""
        # Visual-only placeholder - replace with real URDF
        body = p.createCollisionShape(p.GEOM_BOX, halfExtents=[
            self.hex_cfg.body_length / 2,
            self.hex_cfg.body_width / 2,
            self.hex_cfg.body_height / 2,
        ])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[
            self.hex_cfg.body_length / 2,
            self.hex_cfg.body_width / 2,
            self.hex_cfg.body_height / 2,
        ], rgbaColor=[0.3, 0.6, 0.9, 1])
        
        robot = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=body,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, 0.1],
        )
        return robot

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._timestep = 0

        # Reset robot position
        p.resetBasePositionAndOrientation(
            self.robot_id, [0, 0, 0.1], [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        # Reset joints to default pose
        for i, joint_idx in enumerate(self.joint_indices):
            leg_idx = i // 3
            joint_in_leg = i % 3
            default_angle = self.hex_cfg.default_angles[leg_idx][joint_in_leg]
            p.resetJointState(self.robot_id, joint_idx, default_angle)

        # Generate terrain
        self.terrain_gen.generate(self.terrain_type)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Apply action and step simulation."""
        self._timestep += 1

        # Scale action from [-1, 1] to joint ranges
        scaled_action = self._scale_action(action)

        # Apply joint target positions
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=scaled_action[i],
                positionGain=0.1,
                velocityGain=0.5,
            )

        # Step simulation
        for _ in range(4):  # 4 substeps per control step
            p.stepSimulation()

        # Get observation and reward
        obs = self._get_obs()
        reward = self.reward_fn.compute(self._get_state())
        terminated = self._check_termination()
        truncated = self._timestep >= self.rl_cfg.max_episode_steps
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        state = self._get_state()
        return np.concatenate([
            state["body_rpy"],           # 3
            state["body_angular_vel"],   # 3
            state["body_linear_vel"],    # 3
            state["joint_positions"],    # 18
            state["foot_contacts"],      # 6
            state["terrain_height"],     # 9
        ]).astype(np.float32)

    def _get_state(self) -> Dict:
        """Get full simulation state."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rpy = p.getEulerFromQuaternion(orn)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)

        # Joint positions
        joint_pos = []
        for joint_idx in self.joint_indices:
            state = p.getJointState(self.robot_id, joint_idx)
            joint_pos.append(state[0])

        # Foot contacts (simplified: check distance to ground)
        foot_contacts = np.zeros(6, dtype=np.float32)
        for i in range(6):
            # Check contact between foot and ground
            contact = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=0,  # ground plane
                linkIndexA=self.foot_link_indices[i] if self.foot_link_indices else -1,
            )
            foot_contacts[i] = 1.0 if contact else 0.0

        # Terrain height samples around robot
        terrain_height = self.terrain_gen.sample_heights(
            np.array([pos[0], pos[1]])
        )

        return {
            "body_rpy": np.array(rpy, dtype=np.float32),
            "body_angular_vel": np.array(angular_vel, dtype=np.float32),
            "body_linear_vel": np.array(linear_vel, dtype=np.float32),
            "joint_positions": np.array(joint_pos, dtype=np.float32),
            "foot_contacts": foot_contacts,
            "terrain_height": terrain_height,
        }

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action [-1, 1] to joint limits."""
        scaled = np.zeros_like(action)
        for i, (lower, upper) in enumerate(self.joint_ranges):
            mid = (upper + lower) / 2
            half = (upper - lower) / 2
            scaled[i] = mid + action[i] * half
        return scaled

    def _check_termination(self) -> bool:
        """Check if episode should terminate (robot fell)."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rpy = p.getEulerFromQuaternion(orn)
        # Robot fell if roll or pitch exceeds threshold
        if abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0:
            return True
        if pos[2] < 0.05:  # Too low = fell
            return True
        return False

    def _get_info(self) -> Dict:
        """Get additional info for logging."""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        dist_to_target = np.linalg.norm(
            np.array([pos[0], pos[1]]) - self._target_pos
        )
        return {
            "distance_to_target": float(dist_to_target),
            "x_position": float(pos[0]),
            "y_position": float(pos[1]),
            "height": float(pos[2]),
            "timestep": self._timestep,
        }

    def set_target_position(self, x: float, y: float):
        """Update target position for navigation."""
        self._target_pos = np.array([x, y])

    def close(self):
        """Clean up PyBullet."""
        if self.physics_client >= 0:
            p.disconnect(self.physics_client)
