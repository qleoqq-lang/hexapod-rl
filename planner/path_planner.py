"""Path planning algorithms for hexapod navigation."""

import numpy as np
from typing import List, Tuple, Optional
import heapq


class PathPlanner:
    """Path planning for hexapod robot navigation.

    Implements:
    - A* search on a grid
    - RRT (Rapidly-exploring Random Tree) for continuous space
    """

    def __init__(self, grid_resolution: float = 0.05):
        self.resolution = grid_resolution

    def astar(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
    ) -> Optional[List[Tuple[float, float]]]:
        """A* path planning on a discretized grid.

        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            obstacles: List of (x, y, radius) obstacle circles

        Returns:
            List of (x, y) waypoints, or None if no path found
        """
        # Discretize to grid indices
        start_idx = self._world_to_grid(start)
        goal_idx = self._world_to_grid(goal)

        # A* search
        open_set = []
        heapq.heappush(open_set, (0.0, start_idx))
        came_from = {}
        g_score = {start_idx: 0.0}
        f_score = {start_idx: self._heuristic(start_idx, goal_idx)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_idx:
                # Reconstruct path
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current):
                # Check collision
                neighbor_world = self._grid_to_world(neighbor)
                if self._in_collision(neighbor_world, obstacles):
                    continue

                tentative_g = g_score[current] + self.resolution

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_idx)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        return None  # No path found

    def rrt(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
        max_iterations: int = 1000,
        step_size: float = 0.1,
        goal_sample_rate: float = 0.1,
    ) -> Optional[List[Tuple[float, float]]]:
        """RRT path planning in continuous space.

        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            obstacles: List of (x, y, radius) obstacle circles
            max_iterations: Maximum RRT iterations
            step_size: Maximum step distance between nodes
            goal_sample_rate: Probability of sampling goal directly

        Returns:
            List of (x, y) waypoints, or None if no path found
        """
        nodes = [start]
        parent = {0: -1}  # node_index -> parent_index

        for i in range(max_iterations):
            # Sample random point (biased toward goal)
            if np.random.random() < goal_sample_rate:
                random_point = goal
            else:
                random_point = (
                    np.random.uniform(-2, 2),
                    np.random.uniform(-2, 2),
                )

            # Find nearest node
            nearest_idx = self._nearest_node(nodes, random_point)
            nearest = nodes[nearest_idx]

            # Steer toward random point
            new_point = self._steer(nearest, random_point, step_size)

            # Check collision along path
            if self._path_clear(nearest, new_point, obstacles):
                nodes.append(new_point)
                parent[len(nodes) - 1] = nearest_idx

                # Check if goal reached
                dist_to_goal = np.linalg.norm(
                    np.array(new_point) - np.array(goal)
                )
                if dist_to_goal < step_size:
                    # Reconstruct path
                    return self._reconstruct_rrt_path(nodes, parent, goal)

        # Goal not reached, return best path to closest node
        if nodes:
            closest = min(
                range(len(nodes)),
                key=lambda i: np.linalg.norm(
                    np.array(nodes[i]) - np.array(goal)
                ),
            )
            return self._reconstruct_rrt_path(nodes, parent, nodes[closest])

        return None

    def _world_to_grid(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        return (
            int(point[0] / self.resolution),
            int(point[1] / self.resolution),
        )

    def _grid_to_world(self, idx: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        return (
            idx[0] * self.resolution,
            idx[1] * self.resolution,
        )

    def _heuristic(
        self, a: Tuple[int, int], b: Tuple[int, int]
    ) -> float:
        """Euclidean distance heuristic for A*."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _get_neighbors(
        self, idx: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors."""
        x, y = idx
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((x + dx, y + dy))
        return neighbors

    def _in_collision(
        self,
        point: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
    ) -> bool:
        """Check if point collides with any obstacle."""
        for ox, oy, radius in obstacles:
            if np.sqrt((point[0] - ox)**2 + (point[1] - oy)**2) < radius:
                return True
        return False

    def _path_clear(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
        num_checks: int = 10,
    ) -> bool:
        """Check if straight-line path is collision-free."""
        for t in np.linspace(0, 1, num_checks):
            point = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
            )
            if self._in_collision(point, obstacles):
                return False
        return True

    def _steer(
        self,
        from_point: Tuple[float, float],
        to_point: Tuple[float, float],
        step_size: float,
    ) -> Tuple[float, float]:
        """Move from_point toward to_point by step_size."""
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist < step_size:
            return to_point
        ratio = step_size / dist
        return (
            from_point[0] + ratio * dx,
            from_point[1] + ratio * dy,
        )

    def _nearest_node(
        self,
        nodes: List[Tuple[float, float]],
        point: Tuple[float, float],
    ) -> int:
        """Find index of nearest node to point."""
        distances = [
            np.linalg.norm(np.array(n) - np.array(point))
            for n in nodes
        ]
        return int(np.argmin(distances))

    def _reconstruct_path(
        self,
        came_from: dict,
        current: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        """Reconstruct A* path."""
        path = [self._grid_to_world(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self._grid_to_world(current))
        path.reverse()
        return path

    def _reconstruct_rrt_path(
        self,
        nodes: List[Tuple[float, float]],
        parent: dict,
        goal: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """Reconstruct RRT path."""
        path = [goal]
        current_idx = len(nodes) - 1
        while current_idx != -1:
            path.append(nodes[current_idx])
            current_idx = parent.get(current_idx, -1)
        path.reverse()
        return path
