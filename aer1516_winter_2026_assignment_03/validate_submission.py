"""
AER1516 Winter 2026 — Assignment 3: Pre-submission validation checker.
DO NOT MODIFY this file.

Usage:
    python validate_submission.py

Checks that exploration.py:
  1. Imports correctly and has all required functions
  2. detect_frontiers returns real frontier cells (not random stubs)
  3. select_goal_nearest returns a reachable goal (not None)
  4. select_goal_custom returns a goal or None (type check)
  5. plan_path returns correct types
  6. exploration_step actually sets a path when frontiers exist
  7. A short exploration run achieves non-trivial coverage
  8. Full exploration achieves coverage thresholds on each map
  9. Custom strategy outperforms nearest on at least one map

Passing all checks is necessary but NOT sufficient for full marks.
The grading system uses additional test cases and hidden maps.
"""

import sys
import math
import time
import random
import traceback
import numpy as np


def check(label, condition, msg=''):
    status = 'PASS' if condition else 'FAIL'
    detail = f' — {msg}' if msg else ''
    print(f"  [{status}] {label}{detail}")
    return condition


def warn(label, condition, msg=''):
    status = 'PASS' if condition else 'WARN'
    detail = f' — {msg}' if msg else ''
    print(f"  [{status}] {label}{detail}")
    return True  # warnings don't fail the overall check


def _run_full_exploration(exploration_module, map_fn, goal_selector,
                          time_limit, seed=42):
    """Run a full exploration trial and return final coverage."""
    from config import CELLS_PER_STEP, OCCUPIED as OCC_VAL, MAX_EXPLORATION_STEPS
    from simulator import Environment, LidarSensor, OccupancyGrid
    from run_exploration import ExplorationState

    random.seed(seed)
    np.random.seed(seed)

    true_map, meta = map_fn()
    env = Environment(true_map)
    sensor = LidarSensor(env)
    occ = OccupancyGrid(env.height, env.width)

    sr, sc = meta['start']
    sx, sy = env.grid_to_world(sr, sc)
    scan = sensor.scan(sx, sy)
    occ.update(sx, sy, scan)

    state = ExplorationState(sx, sy, time_limit)
    state.start_time = time.time()

    for step in range(MAX_EXPLORATION_STEPS):
        state.step_count = step
        state.elapsed_time = time.time() - state.start_time

        if state.elapsed_time >= time_limit or state.exploration_complete:
            break

        frontiers = exploration_module.detect_frontiers(occ)
        exploration_module.exploration_step(
            state, occ, env, frontiers, goal_selector)

        if not state.exploration_complete and state.current_path:
            cells_moved = 0
            while (state.current_path_index < len(state.current_path)
                   and cells_moved < CELLS_PER_STEP):
                r, c = state.current_path[state.current_path_index]
                if occ.grid[r, c] == OCC_VAL:
                    state.current_path = None
                    break
                x, y = env.grid_to_world(r, c)
                dist = math.hypot(x - state.robot_x, y - state.robot_y)
                state.robot_x, state.robot_y = x, y
                state.distance_traveled += dist
                state.current_path_index += 1
                cells_moved += 1
                scan = sensor.scan(state.robot_x, state.robot_y)
                occ.update(state.robot_x, state.robot_y, scan)
            if (state.current_path is not None and
                    state.current_path_index >= len(state.current_path)):
                state.current_path = None

    return occ.get_coverage(env.total_free_cells)


def main():
    print("=" * 60)
    print("AER1516 Assignment 3 — Submission Validator")
    print("=" * 60)

    all_pass = True

    # ================================================================
    # 1. Import check
    # ================================================================
    print("\n1. Import check:")
    try:
        import exploration
        check("import exploration", True)
    except Exception as e:
        check("import exploration", False, str(e))
        print("\nCritical: Cannot import exploration.py. Fix import errors first.")
        sys.exit(1)

    # ================================================================
    # 2. Required functions
    # ================================================================
    print("\n2. Required functions:")
    required = [
        'FrontierRegion', 'detect_frontiers', 'select_goal_nearest',
        'select_goal_custom', 'exploration_step', 'plan_path'
    ]
    for name in required:
        exists = hasattr(exploration, name)
        all_pass &= check(f"{name} exists", exists)

    # ================================================================
    # 3. Frontier detection — must return real frontier cells
    # ================================================================
    print("\n3. Frontier detection:")
    try:
        from simulator import OccupancyGrid
        from config import FREE, UNKNOWN, OCCUPIED

        occ = OccupancyGrid(20, 20)
        # Create a known scenario: 10x10 free block surrounded by unknown
        occ.grid[5:15, 5:15] = FREE

        result = exploration.detect_frontiers(occ)
        all_pass &= check("detect_frontiers returns list", isinstance(result, list))

        if result:
            fr = result[0]
            all_pass &= check("FrontierRegion has .cells",
                              hasattr(fr, 'cells') and isinstance(fr.cells, list))
            all_pass &= check("FrontierRegion has .centroid",
                              hasattr(fr, 'centroid') and isinstance(fr.centroid, tuple))
            all_pass &= check("FrontierRegion has .size",
                              hasattr(fr, 'size') and isinstance(fr.size, int))

            # Verify cells are actual frontier cells (FREE + UNKNOWN neighbor)
            all_cells = []
            for region in result:
                all_cells.extend(region.cells)

            real_frontier = 0
            for r, c in all_cells:
                if occ.grid[r, c] != FREE:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 20 and 0 <= nc < 20 and occ.grid[nr, nc] == UNKNOWN:
                        real_frontier += 1
                        break

            if len(all_cells) > 0:
                ratio = real_frontier / len(all_cells)
                all_pass &= check(
                    "Returned cells are real frontier cells",
                    ratio >= 0.8,
                    f"{real_frontier}/{len(all_cells)} cells are FREE with an UNKNOWN neighbor"
                    + (" (random baseline detected — implement detect_frontiers)" if ratio < 0.3 else "")
                )
            else:
                all_pass &= check("detect_frontiers returns cells", False,
                                  "all regions have empty cells list")
        else:
            all_pass &= check("detect_frontiers returns non-empty list", False,
                              "returned empty list on a grid with clear frontiers")

    except Exception as e:
        all_pass &= check("detect_frontiers runs", False, str(e))
        traceback.print_exc()

    # ================================================================
    # 4. Goal selection — nearest must return a goal, not None
    # ================================================================
    print("\n4. Goal selection:")
    try:
        from exploration import FrontierRegion

        # Build a scenario where a goal IS reachable
        occ_goal = OccupancyGrid(20, 20)
        occ_goal.grid[:, :] = FREE  # fully free grid — any goal reachable

        class MockState:
            robot_x = 2.5
            robot_y = 2.5
            blacklisted_goals = set()

        # Frontier region at (10, 10) — clearly reachable on a fully free grid
        regions = [FrontierRegion([(10, 10), (10, 11), (10, 12)])]

        goal_n = exploration.select_goal_nearest(regions, occ_goal, MockState())
        all_pass &= check(
            "select_goal_nearest returns a goal",
            isinstance(goal_n, tuple) and len(goal_n) == 2,
            f"returned {goal_n!r}"
            + (" — stub returns None, implement select_goal_nearest" if goal_n is None else "")
        )

        goal_c = exploration.select_goal_custom(regions, occ_goal, MockState())
        # Custom can return None if not implemented yet, but warn
        all_pass &= check(
            "select_goal_custom returns tuple or None",
            goal_c is None or (isinstance(goal_c, tuple) and len(goal_c) == 2),
            f"returned {goal_c!r}"
        )
        warn("select_goal_custom returns a goal (not None)",
             goal_c is not None,
             "returns None — implement select_goal_custom before submitting")

    except Exception as e:
        all_pass &= check("goal selection runs", False, str(e))
        traceback.print_exc()

    # ================================================================
    # 5. plan_path
    # ================================================================
    print("\n5. plan_path:")
    try:
        occ_pp = OccupancyGrid(20, 20)
        occ_pp.grid[:, :] = FREE
        path, cost = exploration.plan_path(occ_pp, (5, 5), (10, 10))
        all_pass &= check("plan_path returns (path, cost) or (None, None)",
                          (path is not None and cost is not None) or
                          (path is None and cost is None))
        if path is not None:
            all_pass &= check("plan_path path is list of tuples",
                              isinstance(path, list) and len(path) > 0
                              and isinstance(path[0], tuple))
            all_pass &= check("plan_path path starts at start",
                              path[0] == (5, 5),
                              f"first cell is {path[0]}, expected (5, 5)")
            all_pass &= check("plan_path path ends at goal",
                              path[-1] == (10, 10),
                              f"last cell is {path[-1]}, expected (10, 10)")
    except Exception as e:
        all_pass &= check("plan_path runs", False, str(e))
        traceback.print_exc()

    # ================================================================
    # 6. Exploration step — must actually do something
    # ================================================================
    print("\n6. Exploration step:")
    try:
        from maps import make_open_room
        from simulator import Environment, LidarSensor
        from run_exploration import ExplorationState

        true_map, meta = make_open_room()
        env = Environment(true_map)
        sensor = LidarSensor(env)
        occ_es = OccupancyGrid(env.height, env.width)

        sr, sc = meta['start']
        sx, sy = env.grid_to_world(sr, sc)
        scan = sensor.scan(sx, sy)
        occ_es.update(sx, sy, scan)

        state = ExplorationState(sx, sy, meta['time_limit'])
        state.start_time = time.time()

        frontiers = exploration.detect_frontiers(occ_es)
        exploration.exploration_step(state, occ_es, env, frontiers,
                                     exploration.select_goal_nearest)
        all_pass &= check("exploration_step runs without error", True)

        # Check it actually did something: set a path or marked complete
        did_something = (state.current_path is not None) or state.exploration_complete
        all_pass &= check(
            "exploration_step sets a path or marks complete",
            did_something,
            "state.current_path is None and exploration_complete is False"
            + " — stub does nothing, implement exploration_step"
            if not did_something else
            f"current_path has {len(state.current_path)} cells"
            if state.current_path else "exploration marked complete"
        )

    except Exception as e:
        all_pass &= check("exploration_step runs", False, str(e))
        traceback.print_exc()

    # ================================================================
    # 7. Mini exploration run — must achieve non-trivial coverage
    # ================================================================
    print("\n7. Short exploration test (open_room, nearest, ~100 steps):")
    try:
        from config import CELLS_PER_STEP, OCCUPIED as OCC_VAL

        random.seed(42)
        np.random.seed(42)

        true_map, meta = make_open_room()
        env = Environment(true_map)
        sensor = LidarSensor(env)
        occ_run = OccupancyGrid(env.height, env.width)

        sr, sc = meta['start']
        sx, sy = env.grid_to_world(sr, sc)
        scan = sensor.scan(sx, sy)
        occ_run.update(sx, sy, scan)

        state = ExplorationState(sx, sy, meta['time_limit'])
        state.start_time = time.time()
        initial_coverage = occ_run.get_coverage(env.total_free_cells)

        MAX_TEST_STEPS = 100
        for step in range(MAX_TEST_STEPS):
            state.step_count = step
            state.elapsed_time = time.time() - state.start_time

            if state.exploration_complete:
                break

            frontiers = exploration.detect_frontiers(occ_run)
            exploration.exploration_step(state, occ_run, env, frontiers,
                                         exploration.select_goal_nearest)

            # Execute path segment (same as framework)
            if state.current_path:
                cells_moved = 0
                while (state.current_path_index < len(state.current_path)
                       and cells_moved < CELLS_PER_STEP):
                    r, c = state.current_path[state.current_path_index]
                    if occ_run.grid[r, c] == OCC_VAL:
                        state.current_path = None
                        break
                    x, y = env.grid_to_world(r, c)
                    dist = math.hypot(x - state.robot_x, y - state.robot_y)
                    state.robot_x, state.robot_y = x, y
                    state.distance_traveled += dist
                    state.current_path_index += 1
                    cells_moved += 1
                    scan = sensor.scan(state.robot_x, state.robot_y)
                    occ_run.update(state.robot_x, state.robot_y, scan)
                if (state.current_path is not None and
                        state.current_path_index >= len(state.current_path)):
                    state.current_path = None

        final_coverage = occ_run.get_coverage(env.total_free_cells)
        improved = final_coverage > initial_coverage + 0.05

        all_pass &= check(
            "Exploration achieves coverage improvement",
            improved,
            f"coverage went from {initial_coverage*100:.1f}% to {final_coverage*100:.1f}% "
            f"in {step+1} steps"
            + (" — robot is not moving, check all parts" if not improved else "")
        )

        warn("Coverage reaches >= 30% in 100 steps",
             final_coverage >= 0.30,
             f"coverage is {final_coverage*100:.1f}% — aim higher for full marks")

    except Exception as e:
        all_pass &= check("short exploration runs", False, str(e))
        traceback.print_exc()

    # ================================================================
    # 8. Full exploration coverage thresholds
    # ================================================================
    print("\n8. Full exploration coverage (nearest strategy):")
    try:
        from config import (CELLS_PER_STEP as CPS8, OCCUPIED as OCC8,
                            TIME_LIMITS)
        from maps import make_open_room as _mk_or, make_office as _mk_of, make_cave as _mk_cv

        # Minimum coverage thresholds
        VALIDATION_THRESHOLDS = {
            'open_room': (_mk_or, 0.75),
            'office': (_mk_of, 0.60),
            'cave': (_mk_cv, 0.55),
        }

        for map_name, (map_fn, threshold) in VALIDATION_THRESHOLDS.items():
            t_limit = TIME_LIMITS.get(map_name, 60)
            cov = _run_full_exploration(
                exploration, map_fn, exploration.select_goal_nearest,
                t_limit, seed=42)

            all_pass &= check(
                f"{map_name}: coverage >= {threshold:.0%}",
                cov >= threshold,
                f"coverage {cov*100:.1f}% in {t_limit}s"
                + (f" — need at least {threshold*100:.0f}%"
                   if cov < threshold else "")
            )

    except Exception as e:
        all_pass &= check("full exploration runs", False, str(e))
        traceback.print_exc()

    # ================================================================
    # 9. Custom strategy must outperform nearest on >= 1 map
    # ================================================================
    print("\n9. Custom vs. nearest comparison:")
    try:
        outperforms_any = False
        custom_results = {}
        nearest_results = {}

        for map_name, (map_fn, _) in VALIDATION_THRESHOLDS.items():
            t_limit = TIME_LIMITS.get(map_name, 60)
            cov_nearest = _run_full_exploration(
                exploration, map_fn, exploration.select_goal_nearest,
                t_limit, seed=42)
            cov_custom = _run_full_exploration(
                exploration, map_fn, exploration.select_goal_custom,
                t_limit, seed=42)
            nearest_results[map_name] = cov_nearest
            custom_results[map_name] = cov_custom

            better = cov_custom > cov_nearest + 0.005  # at least 0.5% better
            if better:
                outperforms_any = True
            status = "custom wins" if better else "nearest wins or tie"
            print(f"    {map_name}: nearest {cov_nearest*100:.1f}% vs "
                  f"custom {cov_custom*100:.1f}% ({status})")

        all_pass &= check(
            "Custom outperforms nearest on >= 1 map",
            outperforms_any,
            "custom must achieve higher coverage than nearest on at least one map"
            if not outperforms_any else "requirement met"
        )

    except Exception as e:
        all_pass &= check("custom comparison runs", False, str(e))
        traceback.print_exc()

    # ================================================================
    # 10. Time limit info
    # ================================================================
    print("\n10. Time limits (informational):")
    try:
        from config import TIME_LIMITS as TL
        for map_name, limit in TL.items():
            print(f"    {map_name}: {limit}s")
        print("    Your exploration must achieve good coverage within these limits.")
        print("    Slow planning wastes time — consider upgrading Dijkstra to A*.")
    except Exception:
        pass

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    if all_pass:
        print("All checks PASSED. Your submission appears valid.")
        print("Note: passing here is necessary but NOT sufficient for")
        print("full marks. We test with additional maps and stricter checks.")
    else:
        print("Some checks FAILED. Please fix the issues above before")
        print("submitting. Stub implementations will not pass.")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
