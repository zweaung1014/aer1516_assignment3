"""
AER1516 Winter 2026 — Assignment 3: Autonomous Frontier-Based Exploration
Main entry point. DO NOT MODIFY this file.

Usage:
    python run_exploration.py --map open_room --strategy nearest --no-vis
    python run_exploration.py --map office --strategy custom --seed 42
    python run_exploration.py --map cave --strategy random --speed 0.1
"""

import argparse
import math
import time
import random
import numpy as np

from config import (CELL_SIZE, MAX_EXPLORATION_STEPS, CELLS_PER_STEP,
                    VIS_INTERVAL, OCCUPIED)
from maps import get_map, MAPS
from simulator import Environment, LidarSensor, OccupancyGrid
from planner import plan_path
from exploration import (
    detect_frontiers, detect_frontiers_random,
    select_goal_nearest, select_goal_random, select_goal_custom,
    exploration_step,
)


# =============================================================================
# Exploration State
# =============================================================================

class ExplorationState:
    """Mutable state object passed to the student's exploration_step callback."""

    def __init__(self, robot_x, robot_y, time_limit):
        self.robot_x = robot_x
        self.robot_y = robot_y
        self.current_path = None          # path as [(row,col), ...] or None
        self.current_path_index = 0       # next cell to visit in current_path
        self.distance_traveled = 0.0      # meters (tracking only)
        self.blacklisted_goals = set()    # (row,col) tuples to skip
        self.exploration_complete = False  # set True to stop
        self.step_count = 0
        self.time_limit = time_limit      # seconds (informational)
        self.start_time = None            # set when exploration begins
        self.elapsed_time = 0.0           # seconds since start


# =============================================================================
# Strategy dispatch
# =============================================================================

STRATEGIES = {
    'random':  (detect_frontiers_random, select_goal_random),
    'nearest': (detect_frontiers, select_goal_nearest),
    'custom':  (detect_frontiers, select_goal_custom),
}


# =============================================================================
# Random baseline (self-contained, does NOT use exploration_step)
# =============================================================================

def _random_baseline_step(state, occ_grid, env, sensor, frontiers, select_fn,
                           vis=None, coverage_fn=None, strategy_name='random',
                           time_limit=None):
    """
    Random baseline: pick a random frontier, plan path, walk cell-by-cell
    with scanning at every cell. No error handling or replanning.
    """
    if not frontiers:
        state.exploration_complete = True
        return

    goal = select_fn(frontiers, occ_grid, state)
    if goal is None:
        state.exploration_complete = True
        return

    robot_rc = env.world_to_grid(state.robot_x, state.robot_y)
    path, cost = plan_path(occ_grid, robot_rc, goal)
    if path is None:
        state.blacklisted_goals.add(goal)
        return

    # Store path for visualization
    state.current_path = path
    state.current_path_index = 1  # skip start cell

    # Walk cell-by-cell with scanning
    for i in range(1, len(path)):
        r, c = path[i]
        if occ_grid.grid[r, c] == OCCUPIED:
            break
        x, y = env.grid_to_world(r, c)
        dist = math.hypot(x - state.robot_x, y - state.robot_y)
        state.robot_x, state.robot_y = x, y
        state.distance_traveled += dist
        state.current_path_index = i + 1

        # Scan at every cell
        scan = sensor.scan(state.robot_x, state.robot_y)
        occ_grid.update(state.robot_x, state.robot_y, scan)

        # Update elapsed time
        if state.start_time is not None:
            state.elapsed_time = time.time() - state.start_time

        # Per-cell visualization
        if vis and coverage_fn:
            remaining_path = path[i:]
            cov = coverage_fn()
            vis.update(
                occ_grid,
                (state.robot_x, state.robot_y),
                remaining_path,
                frontiers,
                cov,
                state.elapsed_time,
                time_limit or state.time_limit,
                strategy_name,
                state.step_count,
                goal_rc=goal,
            )

    state.current_path = None


# =============================================================================
# Framework path executor
# =============================================================================

def _execute_path_segment(state, occ_grid, env, sensor,
                           vis=None, frontiers=None, coverage_fn=None,
                           strategy_name='', time_limit=None):
    """Move up to CELLS_PER_STEP cells along current path with auto-scan."""
    if state.current_path is None:
        return

    goal_rc = state.current_path[-1] if state.current_path else None

    cells_moved = 0
    while (state.current_path_index < len(state.current_path)
           and cells_moved < CELLS_PER_STEP):
        r, c = state.current_path[state.current_path_index]

        # Safety: don't enter occupied cells
        if occ_grid.grid[r, c] == OCCUPIED:
            state.current_path = None
            break

        x, y = env.grid_to_world(r, c)
        dist = math.hypot(x - state.robot_x, y - state.robot_y)

        state.robot_x, state.robot_y = x, y
        state.distance_traveled += dist
        state.current_path_index += 1
        cells_moved += 1

        # Auto-scan at every cell
        scan = sensor.scan(state.robot_x, state.robot_y)
        occ_grid.update(state.robot_x, state.robot_y, scan)

        # Update elapsed time
        if state.start_time is not None:
            state.elapsed_time = time.time() - state.start_time

        # Per-cell visualization
        if vis and coverage_fn:
            remaining_path = state.current_path[state.current_path_index:]
            cov = coverage_fn()
            vis.update(
                occ_grid,
                (state.robot_x, state.robot_y),
                remaining_path,
                frontiers or [],
                cov,
                state.elapsed_time,
                time_limit or state.time_limit,
                strategy_name,
                state.step_count,
                goal_rc=goal_rc,
            )

    # Clear path if completed
    if state.current_path is not None and state.current_path_index >= len(state.current_path):
        state.current_path = None


# =============================================================================
# Main exploration loop
# =============================================================================

def run_exploration(map_name, strategy_name, visualize=True,
                    seed=None, speed=None, enforce_time=False):
    """
    Run a full exploration trial.

    Args:
        map_name: str — key in MAPS dict.
        strategy_name: str — key in STRATEGIES dict.
        visualize: bool — whether to show matplotlib visualization.
        seed: int or None — random seed for reproducibility.
        speed: float or None — seconds between vis frames (None = default).
        enforce_time: bool — if True, stop exploration when time limit is
                      reached. Default False (students run without timeout).

    Returns:
        dict with keys: coverage_history, final_coverage, total_distance,
                        num_steps, elapsed_time, terminated_reason.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Setup
    true_map, metadata = get_map(map_name)
    env = Environment(true_map)
    sensor = LidarSensor(env)
    occ_grid = OccupancyGrid(env.height, env.width)

    start_r, start_c = metadata['start']
    start_x, start_y = env.grid_to_world(start_r, start_c)
    time_limit = metadata['time_limit']

    state = ExplorationState(start_x, start_y, time_limit)
    detect_fn, select_fn = STRATEGIES[strategy_name]

    # Initial scan
    scan = sensor.scan(state.robot_x, state.robot_y)
    occ_grid.update(state.robot_x, state.robot_y, scan)

    # Check for custom exploration override (bonus)
    has_custom_loop = False
    try:
        from exploration import run_custom_exploration
        has_custom_loop = True
    except ImportError:
        pass

    # Visualization setup
    vis = None
    if visualize:
        from visualizer import ExplorationVisualizer
        vis = ExplorationVisualizer(
            true_map,
            title=f'Frontier Exploration — {map_name} / {strategy_name}',
            delay=speed,
        )

    def coverage_fn():
        return occ_grid.get_coverage(env.total_free_cells)

    coverage_history = []

    # Start the clock
    state.start_time = time.time()

    # Use custom loop if available
    if has_custom_loop and strategy_name != 'random':
        from exploration import run_custom_exploration
        result = run_custom_exploration(env, sensor, occ_grid, state, detect_fn, select_fn, vis)
        if vis:
            vis.show_complete()
        return result

    # Initial visualization frame
    if vis:
        vis.update(occ_grid, (state.robot_x, state.robot_y), None, [],
                   coverage_fn(), 0.0, time_limit, strategy_name, 0)

    # Standard exploration loop
    terminated_reason = 'max_steps'
    stale_steps = 0          # steps without coverage improvement
    last_coverage = 0.0
    STALE_LIMIT = 50         # stop after 50 steps without progress

    for step in range(MAX_EXPLORATION_STEPS):
        state.step_count = step

        # Update elapsed time
        state.elapsed_time = time.time() - state.start_time

        # Check termination
        if enforce_time and state.elapsed_time >= time_limit:
            terminated_reason = 'time_limit'
            break
        if state.exploration_complete:
            terminated_reason = 'complete'
            break
        # Stale progress detection (prevents random baseline from running forever)
        if stale_steps >= STALE_LIMIT:
            terminated_reason = 'stale'
            break

        # Detect frontiers
        frontiers = detect_fn(occ_grid)

        # Execute strategy
        if strategy_name == 'random':
            _random_baseline_step(
                state, occ_grid, env, sensor, frontiers, select_fn,
                vis=vis, coverage_fn=coverage_fn,
                strategy_name=strategy_name, time_limit=time_limit,
            )
        else:
            exploration_step(state, occ_grid, env, frontiers, select_fn)
            if not state.exploration_complete and state.current_path:
                _execute_path_segment(
                    state, occ_grid, env, sensor,
                    vis=vis, frontiers=frontiers, coverage_fn=coverage_fn,
                    strategy_name=strategy_name, time_limit=time_limit,
                )

        # Record coverage and detect stale progress
        coverage = coverage_fn()
        coverage_history.append(coverage)
        if coverage > last_coverage + 0.001:
            last_coverage = coverage
            stale_steps = 0
        else:
            stale_steps += 1

        # Update elapsed time after all work this step
        state.elapsed_time = time.time() - state.start_time

        # Per-step visualization (for steps without cell-by-cell vis, e.g. replanning)
        if vis and not state.current_path:
            vis.update(
                occ_grid,
                (state.robot_x, state.robot_y),
                None,
                frontiers,
                coverage,
                state.elapsed_time,
                time_limit,
                strategy_name,
                step,
            )

    if vis:
        vis.show_complete()

    final_coverage = occ_grid.get_coverage(env.total_free_cells)
    state.elapsed_time = time.time() - state.start_time

    return {
        'coverage_history': coverage_history,
        'final_coverage': final_coverage,
        'total_distance': state.distance_traveled,
        'num_steps': state.step_count,
        'elapsed_time': state.elapsed_time,
        'time_limit': time_limit,
        'terminated_reason': terminated_reason,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AER1516 A3: Frontier-Based Exploration'
    )
    parser.add_argument('--map', type=str, default='open_room',
                        choices=list(MAPS.keys()),
                        help='Map to explore (default: open_room)')
    parser.add_argument('--strategy', type=str, default='random',
                        choices=list(STRATEGIES.keys()),
                        help='Exploration strategy (default: random)')
    parser.add_argument('--no-vis', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--speed', type=float, default=None,
                        help='Visualization delay in seconds per frame '
                             '(e.g. 0.1 for slow, 0.01 for fast; default: 0.05)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    args = parser.parse_args()

    print(f"Running exploration: map={args.map}, strategy={args.strategy}, "
          f"seed={args.seed}")
    if args.speed:
        print(f"Visualization speed: {args.speed}s per frame")

    result = run_exploration(
        map_name=args.map,
        strategy_name=args.strategy,
        visualize=not args.no_vis,
        seed=args.seed,
        speed=args.speed,
        enforce_time=False,  # students run without timeout
    )

    print(f"\n{'=' * 50}")
    print(f"Exploration Complete")
    print(f"{'=' * 50}")
    print(f"  Final coverage:  {result['final_coverage'] * 100:.1f}%")
    print(f"  Distance used:   {result['total_distance']:.1f} m")
    print(f"  Elapsed time:    {result['elapsed_time']:.1f}s "
          f"(limit: {result['time_limit']:.0f}s)")
    print(f"  Steps:           {result['num_steps']}")
    print(f"  Termination:     {result['terminated_reason']}")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
