"""
AER1516 Winter 2026 — Assignment 3: Autonomous Frontier-Based Exploration
Simulation environment, lidar sensor, and occupancy grid. DO NOT MODIFY this file.
"""

import math
import numpy as np
from config import CELL_SIZE, SENSOR_RANGE, SENSOR_NUM_RAYS, SENSOR_FOV, FREE, OCCUPIED, UNKNOWN


class Environment:
    """Wraps the true (ground-truth) binary occupancy map."""

    def __init__(self, true_map):
        """
        Args:
            true_map: np.ndarray of shape (H, W), values 0 (free) or 1 (occupied).
        """
        self.true_map = true_map
        self.height, self.width = true_map.shape
        self.total_free_cells = int(np.sum(true_map == 0))

    def world_to_grid(self, x, y):
        """Convert world coordinates (meters) to grid indices (row, col)."""
        col = int(x / CELL_SIZE)
        row = int(y / CELL_SIZE)
        return (row, col)

    def grid_to_world(self, row, col):
        """Convert grid indices to world coordinates (center of cell)."""
        x = (col + 0.5) * CELL_SIZE
        y = (row + 0.5) * CELL_SIZE
        return (x, y)

    def is_free(self, row, col):
        """Check if a cell is free in the true map."""
        if not self.is_in_bounds(row, col):
            return False
        return self.true_map[row, col] == 0

    def is_in_bounds(self, row, col):
        """Check if indices are within the map."""
        return 0 <= row < self.height and 0 <= col < self.width


class LidarSensor:
    """Simulated 2D lidar sensor with ray marching."""

    def __init__(self, environment):
        self.env = environment
        self.range = SENSOR_RANGE
        self.num_rays = SENSOR_NUM_RAYS
        self.fov = SENSOR_FOV
        self.step_size = CELL_SIZE / 2.0  # ray march step

    def scan(self, robot_x, robot_y):
        """
        Perform a lidar scan from the robot's position.

        Args:
            robot_x, robot_y: Robot position in world coordinates (meters).

        Returns:
            List of (x, y, hit) tuples where:
              - (x, y) is the endpoint of the ray in world coordinates
              - hit is True if the ray struck an obstacle, False if it reached max range
        """
        results = []
        for i in range(self.num_rays):
            angle = (i / self.num_rays) * self.fov
            dx = math.cos(angle)
            dy = math.sin(angle)

            # March along the ray
            dist = 0.0
            hit = False
            x, y = robot_x, robot_y

            while dist < self.range:
                dist += self.step_size
                x = robot_x + dx * dist
                y = robot_y + dy * dist

                row, col = self.env.world_to_grid(x, y)

                # Out-of-bounds treated as hit
                if not self.env.is_in_bounds(row, col):
                    hit = True
                    break

                # Obstacle hit
                if self.env.true_map[row, col] == 1:
                    hit = True
                    break

            results.append((x, y, hit))

        return results


class OccupancyGrid:
    """Incrementally-built occupancy grid from sensor observations."""

    def __init__(self, height, width):
        """
        Args:
            height, width: Grid dimensions in cells.
        """
        self.height = height
        self.width = width
        self.grid = np.full((height, width), UNKNOWN, dtype=np.int8)

    def update(self, robot_x, robot_y, scan_results):
        """
        Update the occupancy grid from a lidar scan using Bresenham ray tracing.

        Args:
            robot_x, robot_y: Robot position in world coordinates.
            scan_results: List of (x, y, hit) from LidarSensor.scan().
        """
        robot_row = int(robot_y / CELL_SIZE)
        robot_col = int(robot_x / CELL_SIZE)

        for (end_x, end_y, hit) in scan_results:
            end_row = int(end_y / CELL_SIZE)
            end_col = int(end_x / CELL_SIZE)

            # Bresenham line from robot to endpoint
            cells = self._bresenham(robot_row, robot_col, end_row, end_col)

            # All cells along the ray (except the last if hit) are free
            for i, (r, c) in enumerate(cells):
                if not self.is_in_bounds(r, c):
                    continue
                if i == len(cells) - 1 and hit:
                    # Last cell on a hit ray is occupied
                    self.grid[r, c] = OCCUPIED
                else:
                    # Only mark as FREE if not already known to be OCCUPIED
                    # (a ray passing through a wall cell from a different angle
                    # should not erase that wall)
                    if self.grid[r, c] != OCCUPIED:
                        self.grid[r, c] = FREE

    def get_coverage(self, total_free_cells):
        """Compute fraction of true free cells that have been observed as FREE."""
        if total_free_cells == 0:
            return 1.0
        observed_free = int(np.sum(self.grid == FREE))
        return observed_free / total_free_cells

    def is_free(self, row, col):
        if not self.is_in_bounds(row, col):
            return False
        return self.grid[row, col] == FREE

    def is_occupied(self, row, col):
        if not self.is_in_bounds(row, col):
            return False
        return self.grid[row, col] == OCCUPIED

    def is_unknown(self, row, col):
        if not self.is_in_bounds(row, col):
            return True
        return self.grid[row, col] == UNKNOWN

    def is_in_bounds(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width

    @staticmethod
    def _bresenham(r0, c0, r1, c1):
        """Bresenham's line algorithm. Returns list of (row, col) from start to end."""
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0

        while True:
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

        return cells
