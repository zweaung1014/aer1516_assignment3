"""
AER1516 Winter 2026 — Assignment 3: Autonomous Frontier-Based Exploration
Test maps for exploration. DO NOT MODIFY this file.

Each map function returns (true_map, metadata) where:
  - true_map: numpy array of shape (H, W), values 0 (free) or 1 (occupied)
  - metadata: dict with 'name', 'start' (row, col), 'time_limit' (seconds), etc.
"""

import numpy as np
from config import TIME_LIMITS, DEFAULT_TIME_LIMIT


def make_open_room():
    """Open room with a few pillars. 80x80 grid (20m x 20m). ~85% free.
    Sanity-check map — any reasonable strategy should ace this."""
    H, W = 80, 80
    grid = np.zeros((H, W), dtype=np.int8)

    # Boundary walls (2 cells thick)
    grid[0:2, :] = 1
    grid[H - 2:H, :] = 1
    grid[:, 0:2] = 1
    grid[:, W - 2:W] = 1

    # Rectangular pillars
    grid[15:22, 15:22] = 1
    grid[15:22, 55:62] = 1
    grid[50:57, 35:42] = 1
    grid[35:40, 20:25] = 1

    start = (40, 40)
    return grid, {
        'name': 'open_room',
        'start': start,
        'time_limit': TIME_LIMITS.get('open_room', DEFAULT_TIME_LIMIT),
        'grid_size': (H, W),
        'physical_size': (H * 0.25, W * 0.25),
    }


def make_office():
    """Office with cubicle partitions, narrow doorways, and asymmetric rooms.
    100x80 grid (25m x 20m). Tests goal selection under limited visibility.

    Layout:
    - Central T-shaped corridor with narrow doorways to rooms
    - Room 1 (top-left): subdivided by cubicle partitions
    - Room 2 (top-right): L-shaped, wraps around a pillar block
    - Room 3 (bottom-left): two connected sub-rooms via internal doorway
    - Room 4 (bottom-right): large open area with island obstacles
    - Storage closet off the corridor (small dead-end)
    """
    H, W = 80, 100
    grid = np.ones((H, W), dtype=np.int8)  # start all walls

    # === Corridors ===
    # Horizontal main corridor (narrow: 4 cells = 1m)
    grid[37:41, 5:95] = 0
    # Vertical corridor (narrow: 4 cells)
    grid[5:75, 48:52] = 0

    # === Room 1: top-left with cubicle partitions ===
    grid[8:34, 6:44] = 0
    # Narrow doorway to corridor (4 cells wide → 2 passable after inflation)
    grid[34:37, 19:23] = 0
    # Cubicle partition walls inside room 1
    grid[14:15, 12:36] = 1   # horizontal partition
    grid[20:21, 10:30] = 1   # another horizontal partition
    grid[26:27, 16:40] = 1   # another
    grid[14:27, 24:25] = 1   # vertical partition
    # Gaps in partitions (doorway-sized openings)
    grid[14:15, 18:20] = 0   # gap in first partition
    grid[20:21, 22:24] = 0   # gap in second
    grid[26:27, 30:32] = 0   # gap in third
    grid[20:22, 24:25] = 0   # gap in vertical partition

    # === Room 2: top-right, L-shaped ===
    grid[8:34, 56:94] = 0
    # Narrow doorway to corridor
    grid[34:37, 71:75] = 0
    # Cut out a block to make it L-shaped
    grid[8:20, 56:70] = 1
    # Re-carve the L part
    grid[8:20, 74:94] = 0
    # Internal pillar
    grid[24:30, 78:84] = 1

    # === Room 3: bottom-left, two sub-rooms ===
    # Sub-room A
    grid[44:64, 6:24] = 0
    # Narrow doorway from corridor to sub-room A
    grid[41:44, 11:15] = 0
    # Sub-room B
    grid[44:64, 28:44] = 0
    # Internal doorway between A and B (narrow)
    grid[51:55, 24:28] = 0
    # Dead-end closet off sub-room B
    grid[58:64, 38:44] = 0  # already carved
    grid[64:70, 40:43] = 0  # extends south as a closet

    # === Room 4: bottom-right, open with islands ===
    grid[44:72, 56:94] = 0
    # Narrow doorway from corridor
    grid[41:44, 73:77] = 0
    # Island obstacles inside room 4
    grid[50:56, 64:70] = 1
    grid[60:66, 76:82] = 1
    grid[50:53, 80:86] = 1

    # === Storage closet off main corridor (dead-end trap) ===
    grid[37:41, 3:5] = 0  # extends corridor left
    grid[30:37, 3:6] = 0  # closet going north

    # Ensure boundary walls
    grid[0:2, :] = 1
    grid[H - 2:H, :] = 1
    grid[:, 0:2] = 1
    grid[:, W - 2:W] = 1

    start = (39, 50)  # in corridor at the T-junction
    return grid, {
        'name': 'office',
        'start': start,
        'time_limit': TIME_LIMITS.get('office', DEFAULT_TIME_LIMIT),
        'grid_size': (H, W),
        'physical_size': (H * 0.25, W * 0.25),
    }


def make_cave():
    """Winding cave with dead-end traps, narrow squeezes, and branching paths.
    120x80 grid (30m x 20m). Tests robustness and strategic path choice.

    Dead-ends are costly traps for nearest-frontier: the robot enters, explores
    a small area, then must backtrack. A smarter strategy skips small dead-ends
    in favor of the main loop which has much more area.
    """
    H, W = 80, 120
    grid = np.ones((H, W), dtype=np.int8)  # start all walls

    def carve_corridor(grid, points, width):
        """Carve a corridor along a polyline with given width."""
        for i in range(len(points) - 1):
            r1, c1 = points[i]
            r2, c2 = points[i + 1]
            steps = max(abs(r2 - r1), abs(c2 - c1)) + 1
            for t in range(steps):
                frac = t / max(steps - 1, 1)
                r = int(r1 + frac * (r2 - r1))
                c = int(c1 + frac * (c2 - c1))
                hw = width // 2
                r_lo = max(0, r - hw)
                r_hi = min(H, r + hw + 1)
                c_lo = max(0, c - hw)
                c_hi = min(W, c + hw + 1)
                grid[r_lo:r_hi, c_lo:c_hi] = 0

    # === Main loop (the "highway" — most of the free space) ===
    carve_corridor(grid, [(40, 5), (40, 25)], width=6)
    carve_corridor(grid, [(40, 25), (15, 25)], width=5)
    carve_corridor(grid, [(15, 25), (15, 70)], width=5)
    carve_corridor(grid, [(15, 70), (35, 90)], width=5)
    carve_corridor(grid, [(35, 90), (60, 90)], width=5)
    carve_corridor(grid, [(60, 90), (60, 50)], width=5)
    carve_corridor(grid, [(60, 50), (40, 35)], width=5)
    carve_corridor(grid, [(40, 35), (40, 25)], width=5)

    # Chamber at north bend (reward for taking the main loop)
    grid[6:22, 38:62] = 0
    # Chamber at east bend
    grid[26:44, 82:100] = 0
    # Chamber at south of loop
    grid[54:68, 58:78] = 0

    # === Dead-end branch 1: south from entrance (TRAP — small payoff) ===
    carve_corridor(grid, [(40, 12), (60, 12)], width=3)
    carve_corridor(grid, [(60, 12), (60, 25)], width=3)
    grid[58:64, 22:28] = 0

    # === Dead-end branch 2: north-west (TRAP — narrow, long) ===
    carve_corridor(grid, [(15, 30), (5, 30), (5, 10)], width=3)
    grid[3:8, 5:12] = 0

    # === Dead-end branch 3: south-east (TRAP — furthest from start) ===
    carve_corridor(grid, [(60, 80), (72, 80), (72, 105)], width=3)
    grid[68:76, 102:115] = 0

    # === Dead-end branch 4: narrow squeeze off the loop ===
    carve_corridor(grid, [(35, 55), (35, 65)], width=4)
    grid[30:40, 64:72] = 0

    # === Narrow squeeze on the main loop ===
    grid[58:60, 60:65] = 1
    grid[61:63, 55:60] = 1

    # Ensure boundary walls
    grid[0:2, :] = 1
    grid[H - 2:H, :] = 1
    grid[:, 0:2] = 1
    grid[:, W - 2:W] = 1

    start = (40, 8)  # near entrance
    return grid, {
        'name': 'cave',
        'start': start,
        'time_limit': TIME_LIMITS.get('cave', DEFAULT_TIME_LIMIT),
        'grid_size': (H, W),
        'physical_size': (H * 0.25, W * 0.25),
    }


# Map registry
MAPS = {
    'open_room': make_open_room,
    'office': make_office,
    'cave': make_cave,
}


def get_map(name):
    """Get a map by name. Returns (true_map, metadata)."""
    if name not in MAPS:
        raise ValueError(f"Unknown map '{name}'. Available: {list(MAPS.keys())}")
    return MAPS[name]()
