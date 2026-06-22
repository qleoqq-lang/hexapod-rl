"""Navigation controller combining path planning and gait execution."""

import numpy as np
from typing import List, Tuple, Optional

from planner.path_planner import PathPlanner
from gait.gait_planner import GaitPlanner


class NavigationController:
    """High-level navigation controller for hexapod.

    Combines global path planning with local gait control.
    """

    def __init__(
        self,
        path_planner: Optional[PathPlanner] = None,
        gait_planner: Optional[GaitPlanner] = None,
    ):
        self.path_planner = path_planner or PathPlanner()
        self.gait_planner = gait_planner or GaitPlanner()
        self._waypoints: List[Tuple[float, float]] = []
        self._current_waypoint_idx = 0

    def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
        algorithm: str = "astar",
    ) -> bool:
        """Plan global path from start to goal.

        Returns:
            True if path found, False otherwise
        """
        if algorithm == "astar":
            path = self.path_planner.astar(start, goal, obstacles)
        elif algorithm == "rrt":
            path = self.path_planner.rrt(start, goal, obstacles)
        else:
            path = self.path_planner.astar(start, goal, obstacles)

        if path is None:
            self._waypoints = []
            self._current_waypoint_idx = 0
            return False

        self._waypoints = path
        self._current_waypoint_idx = 0
        return True

    def get_velocity_command(
        self, current_pos: Tuple[float, float], current_yaw: float
    ) -> Tuple[float, float]:
        """Compute velocity command to follow planned path.

        Args:
            current_pos: (x, y) robot position
            current_yaw: robot heading (radians)

        Returns:
            (forward_velocity, turn_rate) command
        """
        if not self._waypoints or self._current_waypoint_idx >= len(self._waypoints):
            return (0.0, 0.0)

        # Get current target waypoint
        target = self._waypoints[self._current_waypoint_idx]

        # Compute direction to target
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        dist_to_target = np.sqrt(dx**2 + dy**2)

        # Check if waypoint reached
        if dist_to_target < 0.05:
            self._current_waypoint_idx += 1
            if self._current_waypoint_idx >= len(self._waypoints):
                return (0.0, 0.0)  # All waypoints done
            target = self._waypoints[self._current_waypoint_idx]
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]

        # Compute desired heading
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = desired_yaw - current_yaw

        # Normalize yaw error to [-pi, pi]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi

        # Speed control
        forward_vel = min(0.3, dist_to_target * 2.0)  # Proportional to distance
        turn_rate = 2.0 * yaw_error  # P-controller for turning

        return (forward_vel, turn_rate)

    def reached_goal(self) -> bool:
        """Check if robot has reached final goal."""
        return (
            self._waypoints
            and self._current_waypoint_idx >= len(self._waypoints)
        )
