# AER1516 — Assignment 3: Autonomous Frontier-Based Exploration

**Course:** AER1516: Robot Motion Planning — Winter 2026  
**Due:** Friday, April 3, 2026

## Overview

A simulated ground robot equipped with a 360° lidar sensor explores unknown 2D environments using frontier-based exploration. The robot maintains an occupancy grid (free / occupied / unknown) and repeatedly selects frontier regions — free cells adjacent to unexplored space — as navigation targets until the map is fully covered.

All student work lives in `exploration.py`.

## Project Structure

```
aer1516_winter_2026_assignment_03/
├── config.py            # Constants (CELL_SIZE, FRONTIER_MIN_SIZE, etc.)
├── exploration.py       # ← Student implementation file
├── generate_plots.py    # Coverage-vs-time plot generator
├── maps.py              # Map definitions (open_room, office, cave)
├── planner.py           # Dijkstra path planner + utilities
├── run_exploration.py   # Main entry point
├── simulator.py         # 2D simulator + raycasting lidar
├── validate_submission.py  # Pre-submission sanity checks
└── visualizer.py        # Live matplotlib visualizer
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

## Running the Exploration

```bash
# Random baseline (works out of the box)
python run_exploration.py --map open_room --strategy random

# Nearest-frontier strategy (Part 2a)
python run_exploration.py --map office --strategy nearest

# Custom strategy (Part 2b)
python run_exploration.py --map cave --strategy custom
```

**Available flags:**

| Flag | Options | Description |
|------|---------|-------------|
| `--map` | `open_room`, `office`, `cave` | Environment to explore |
| `--strategy` | `random`, `nearest`, `custom` | Goal selection strategy |
| `--speed` | e.g. `0.01` (fast), `0.1` (slow) | Visualization speed |
| `--no-vis` | — | Disable visualization (faster data collection) |
| `--seed` | e.g. `42` | Fix random seed for reproducibility |

## Implementation Tasks

| Part | Function | Points | Description |
|------|----------|--------|-------------|
| 1 | `detect_frontiers()` | 12 | BFS-based frontier detection & clustering |
| 2a | `select_goal_nearest()` | 10 | Select closest reachable frontier by path cost |
| 2b | `select_goal_custom()` | 6 | Custom strategy outperforming nearest on ≥1 map |
| 3 | `exploration_step()` | 12 | Robust exploration loop (path validation, blacklisting, completion) |
| — | `plan_path()` | bonus | Optional A* upgrade for faster planning |
| — | `report.pdf` | 10 | Coverage plots + strategy discussion |
| **Total** | | **50** | |

## Coverage Thresholds (Automated Grading)

| Map | Time Limit | Min. Coverage (Nearest) |
|-----|-----------|------------------------|
| `open_room` | 30 s | 75% |
| `office` | 60 s | 60% |
| `cave` | 60 s | 55% |

## Validate Before Submitting

```bash
python validate_submission.py
```

## Submission

```bash
tar cvf handin.tar exploration.py report.pdf
```

Upload `handin.tar` through Quercus. Do **not** include any framework files.
