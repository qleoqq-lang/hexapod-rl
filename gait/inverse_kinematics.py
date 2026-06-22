"""Inverse kinematics for a single hexapod leg (3-DOF: coxa, femur, tibia)."""

import numpy as np


class HexapodIK:
    """Inverse kinematics solver for a single 3-DOF leg.

    Leg configuration (standard hexapod layout):
        Joint 1 (coxa): hip abduction/adduction - rotation around Z axis
        Joint 2 (femur): hip flexion/extension - rotation around Y axis  
        Joint 3 (tibia): knee flexion/extension - rotation around Y axis
    """

    def __init__(
        self,
        coxa_length: float = 0.04,
        femur_length: float = 0.10,
        tibia_length: float = 0.14,
    ):
        self.L1 = coxa_length  # Coxa length
        self.L2 = femur_length  # Femur length
        self.L3 = tibia_length  # Tibia length

    def solve(
        self, target_pos: np.ndarray, side: str = "left"
    ) -> np.ndarray:
        """Compute joint angles for target foot position.

        Args:
            target_pos: (3,) foot position relative to hip joint [x, y, z]
                       x = forward, y = lateral, z = up
            side: 'left' or 'right' (affects coxa angle sign)

        Returns:
            (3,) joint angles [coxa, femur, tibia] in radians
        """
        x, y, z = target_pos

        # Coxa angle (hip abduction)
        # For left legs: positive y = outward
        # For right legs: negative y = outward
        coxa = np.arctan2(y, x) if side == "left" else -np.arctan2(-y, x)

        # Distance from coxa joint to foot in the leg plane
        r = np.sqrt(x**2 + y**2) - self.L1
        
        if r < 0:
            r = 0.001  # Prevent numerical issues

        # Distance from femur joint to foot in leg plane
        D = np.sqrt(r**2 + z**2)

        # Check reachability
        if D > (self.L2 + self.L3):
            # Target unreachable - stretch to max
            D = self.L2 + self.L3 - 0.001

        # Femur angle using law of cosines
        cos_femur = (self.L2**2 + D**2 - self.L3**2) / (2 * self.L2 * D)
        cos_femur = np.clip(cos_femur, -1.0, 1.0)
        femur_angle1 = np.arctan2(z, r) - np.arccos(cos_femur)

        # Tibia angle using law of cosines
        cos_tibia = (self.L2**2 + self.L3**2 - D**2) / (2 * self.L2 * self.L3)
        cos_tibia = np.clip(cos_tibia, -1.0, 1.0)
        tibia = np.pi - np.arccos(cos_tibia)

        # Femur should be in [-2.36, 0.78], adjust if needed
        femur = np.clip(femur_angle1, -2.36, 0.78)
        coxa = np.clip(coxa, -1.57, 1.57)
        tibia = np.clip(tibia, -2.36, 0.78)

        return np.array([coxa, femur, tibia])

    def forward(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute foot position from joint angles (forward kinematics).

        Args:
            joint_angles: (3,) [coxa, femur, tibia] in radians

        Returns:
            (3,) foot position [x, y, z]
        """
        c1, f1, t1 = joint_angles
        
        x = (self.L1 + self.L2 * np.cos(f1) + self.L3 * np.cos(f1 + t1)) * np.cos(c1)
        y = (self.L1 + self.L2 * np.cos(f1) + self.L3 * np.cos(f1 + t1)) * np.sin(c1)
        z = self.L2 * np.sin(f1) + self.L3 * np.sin(f1 + t1)
        
        return np.array([x, y, z])
