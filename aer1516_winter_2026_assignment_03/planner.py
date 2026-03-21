"""
AER1516 Winter 2026 — Assignment 3: Autonomous Frontier-Based Exploration
Dijkstra path planner with obstacle inflation. DO NOT MODIFY this file.

This planner uses Dijkstra's algorithm (no heuristic), which guarantees
shortest paths but explores many more cells than necessary for distant goals.
Students can implement A* in their plan_path() to dramatically speed up
planning for long-range goals.
"""

import math
import heapq
import numpy as np
from config import CELL_SIZE, ROBOT_RADIUS, CONNECTIVITY, FREE, OCCUPIED, UNKNOWN


# Inflation radius in cells (1 cell = 0.25m buffer around known walls)
INFLATION_RADIUS = 1


def inflate_grid(occ_grid):
    """
    Create a boolean traversability mask from the occupancy grid.
    Inflates OCCUPIED cells by INFLATION_RADIUS using a circular structuring
    element, and marks UNKNOWN cells as blocked (but does NOT inflate them).
    True means the cell is blocked (robot center cannot be there).

    Args:
        occ_grid: OccupancyGrid instance.

    Returns:
        np.ndarray of shape (H, W), dtype bool. True = blocked.
    """
    H, W = occ_grid.height, occ_grid.width

    # Start with UNKNOWN cells as blocked (can't plan through unknown)
    blocked = (occ_grid.grid == UNKNOWN).copy()

    # Inflate only OCCUPIED (known wall) cells
    wall_cells = (occ_grid.grid == OCCUPIED)

    # Build circular structuring element offsets
    offsets = []
    for dr in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
        for dc in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
            if dr * dr + dc * dc <= INFLATION_RADIUS * INFLATION_RADIUS:
                offsets.append((dr, dc))

    # Inflate walls: for each occupied cell, mark surrounding cells as blocked
    obs_rows, obs_cols = np.where(wall_cells)
    for dr, dc in offsets:
        r = obs_rows + dr
        c = obs_cols + dc
        valid = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        blocked[r[valid], c[valid]] = True

    return blocked


def plan_path(occ_grid, start_rc, goal_rc):
    """
    Plan a path using Dijkstra's algorithm on the 8-connected grid with
    obstacle inflation.

    Dijkstra explores uniformly in all directions (no heuristic), which makes
    it slow for distant goals. Students should implement A* (with an admissible
    heuristic) in their own plan_path() for better performance.

    The start and goal cells are allowed even if in the inflation zone, as long
    as they are FREE cells.

    Args:
        occ_grid: OccupancyGrid instance.
        start_rc: (row, col) start position.
        goal_rc: (row, col) goal position.

    Returns:
        (path, cost) where path is a list of (row, col) from start to goal
        (inclusive), and cost is the total path cost in cell-distance units.
        Returns (None, None) if no path exists.
    """
    blocked = inflate_grid(occ_grid)
    H, W = occ_grid.height, occ_grid.width

    sr, sc = start_rc
    gr, gc = goal_rc

    # Check bounds
    if not (0 <= sr < H and 0 <= sc < W):
        return None, None
    if not (0 <= gr < H and 0 <= gc < W):
        return None, None
    # Require start and goal to be FREE cells (known open space)
    if occ_grid.grid[sr, sc] != FREE:
        return None, None
    if occ_grid.grid[gr, gc] != FREE:
        return None, None

    # Allow start and goal even if in inflation zone
    blocked[sr, sc] = False
    blocked[gr, gc] = False

    # 8-connected neighbors
    if CONNECTIVITY == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    SQRT2 = math.sqrt(2)

    # Dijkstra: priority queue ordered by g_cost only (no heuristic)
    open_set = [(0.0, sr, sc)]
    g_cost = {(sr, sc): 0.0}
    came_from = {}
    closed = set()

    while open_set:
        g, r, c = heapq.heappop(open_set)

        if (r, c) == (gr, gc):
            # Reconstruct path
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            path.reverse()
            return path, g

        if (r, c) in closed:
            continue
        closed.add((r, c))

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if blocked[nr, nc]:
                continue
            if (nr, nc) in closed:
                continue

            # Movement cost
            step_cost = SQRT2 if (dr != 0 and dc != 0) else 1.0
            new_g = g + step_cost

            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_set, (new_g, nr, nc))

    return None, None


def validate_path(occ_grid, path):
    """
    Check if every cell in the path is still traversable.
    A cell is considered blocked if it is OCCUPIED (known wall/obstacle).

    Args:
        occ_grid: OccupancyGrid instance.
        path: List of (row, col) tuples.

    Returns:
        True if the entire path is clear, False if any cell is blocked.
    """
    if not path:
        return False
    for r, c in path:
        if not (0 <= r < occ_grid.height and 0 <= c < occ_grid.width):
            return False
        if occ_grid.grid[r, c] == OCCUPIED:
            return False
    return True


def path_cost(path):
    """
    Compute the Euclidean distance along a path in cell-distance units.

    Args:
        path: List of (row, col) tuples.

    Returns:
        Total distance in cell units. Multiply by CELL_SIZE for meters.
    """
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        total += math.hypot(dr, dc)
    return total
