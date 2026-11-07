"""Policy contract only. No real robot implementation or hardware imports."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np

# Preserved from dual_side_mujoco_driver.py; FR femur is different.
LEG_ORDER = ["FL", "ML", "RL", "FR", "MR", "RR"]
MJCF_PREFIX = dict(zip(LEG_ORDER, ("lf", "lm", "lr", "rf", "rm", "rr")))
COMMAND_TO_MJCF_SIGNS = {
    "FL": (-1., 1., -1.), "ML": (-1., 1., -1.), "RL": (-1., 1., -1.),
    "FR": (-1., -1., -1.), "MR": (-1., 1., -1.), "RR": (-1., 1., -1.),
}
JOINT_NAMES = [f"{MJCF_PREFIX[leg]}_{joint}_joint" for leg in LEG_ORDER
               for joint in ("coxa", "femur", "tibia")]
OBSERVATION_VERSION = "body_velocity_gravity_command_joints_previous_action_v1_63"

@dataclass
class RobotState:
    base_position: np.ndarray       # world, m
    base_quaternion: np.ndarray     # wxyz
    linear_velocity: np.ndarray     # body frame, m/s
    angular_velocity: np.ndarray    # body frame, rad/s
    world_linear_velocity: np.ndarray
    projected_gravity: np.ndarray   # body frame, unit vector
    joint_position: np.ndarray     # original MIT command signs, radians
    joint_velocity: np.ndarray     # original MIT command signs, rad/s
    tibia_contacts: np.ndarray      # diagnostics, NOT physical foot sensors

def make_observation(state, nominal, previous_action):
    return np.concatenate((state.linear_velocity, state.angular_velocity,
                           state.projected_gravity, state.joint_position - nominal,
                           state.joint_velocity, previous_action)).astype(np.float32)

class RobotInterface(Protocol):
    nominal_command: np.ndarray
    def get_state(self) -> RobotState: ...
    def get_observation(self, previous_action: np.ndarray) -> np.ndarray: ...
    def apply_action(self, action: np.ndarray) -> dict: ...
    def close(self) -> None: ...
