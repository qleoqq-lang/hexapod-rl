"""Terrain generation for hexapod simulation."""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Tuple


class TerrainGenerator:
    """Generate different terrain types for hexapod training."""

    def __init__(self, physics_client: int):
        self.physics_client = physics_client
        self._ground_plane = None
        self._terrain_shape = None
        self._terrain_body = None

    def generate(self, terrain_type: str = "flat"):
        """Generate terrain of specified type."""
        # Remove old terrain
        self._remove_terrain()

        if terrain_type == "flat":
            self._generate_flat()
        elif terrain_type == "slope":
            self._generate_slope()
        elif terrain_type == "rough":
            self._generate_rough()
        elif terrain_type == "stairs":
            self._generate_stairs()
        else:
            self._generate_flat()

    def sample_heights(self, robot_pos: np.ndarray) -> np.ndarray:
        """Sample terrain heights at grid around robot position.

        Returns 9 height samples in a 3x3 grid around the robot.
        """
        x, y = robot_pos[0], robot_pos[1]
        grid_x = np.linspace(x - 0.3, x + 0.3, 3)
        grid_y = np.linspace(y - 0.3, y + 0.3, 3)
        
        heights = []
        for gx in grid_x:
            for gy in grid_y:
                # Ray cast to get terrain height at (gx, gy)
                result = p.rayTest(
                    [gx, gy, 2.0], [gx, gy, -0.5],
                    physicsClientId=self.physics_client
                )
                if result and result[0][0] >= 0:
                    hit_pos = result[0][3]
                    heights.append(hit_pos[2])
                else:
                    heights.append(0.0)
        
        return np.array(heights, dtype=np.float32)

    def _remove_terrain(self):
        """Remove previously generated terrain."""
        if self._ground_plane is not None:
            p.removeBody(self._ground_plane, physicsClientId=self.physics_client)
            self._ground_plane = None
        if self._terrain_body is not None:
            p.removeBody(self._terrain_body, physicsClientId=self.physics_client)
            self._terrain_body = None
            self._terrain_shape = None

    def _generate_flat(self):
        """Generate flat ground plane."""
        self._ground_plane = p.loadURDF(
            "plane.urdf", [0, 0, 0],
            physicsClientId=self.physics_client,
        )

    def _generate_slope(self, angle_deg: float = 10.0):
        """Generate a sloped terrain."""
        angle_rad = np.radians(angle_deg)
        half_extents = [2.0, 2.0, 0.05]
        
        self._terrain_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents,
            physicsClientId=self.physics_client,
        )
        self._terrain_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self._terrain_shape,
            basePosition=[0.5, 0, -0.05],
            baseOrientation=p.getQuaternionFromEuler([0, angle_rad, 0]),
            physicsClientId=self.physics_client,
        )

    def _generate_rough(self, height_range: Tuple[float, float] = (-0.02, 0.02)):
        """Generate rough terrain with random height blocks."""
        self._ground_plane = p.loadURDF(
            "plane.urdf", [0, 0, 0],
            physicsClientId=self.physics_client,
        )
        
        # Add random height blocks
        rng = np.random.default_rng(42)
        for i in range(-10, 10):
            for j in range(-10, 10):
                h = rng.uniform(*height_range)
                if abs(h) < 0.005:
                    continue
                block_shape = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[0.05, 0.05, abs(h) / 2],
                    physicsClientId=self.physics_client,
                )
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=block_shape,
                    basePosition=[i * 0.1 + 0.05, j * 0.1 + 0.05, h / 2],
                    physicsClientId=self.physics_client,
                )

    def _generate_stairs(self):
        """Generate stair terrain."""
        for i in range(5):
            step_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.1, 0.5, 0.015],
                physicsClientId=self.physics_client,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=step_shape,
                basePosition=[0.1 + i * 0.2, 0, i * 0.03 + 0.015],
                physicsClientId=self.physics_client,
            )
