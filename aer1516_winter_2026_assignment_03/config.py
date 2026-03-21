"""
AER1516 Winter 2026 — Assignment 3: Autonomous Frontier-Based Exploration
Configuration parameters. DO NOT MODIFY this file.
"""

import math

# Grid resolution
CELL_SIZE = 0.25  # meters per grid cell

# Robot parameters
ROBOT_RADIUS = 0.3  # meters

# Sensor parameters
SENSOR_RANGE = 3.0  # meters
SENSOR_FOV = 2 * math.pi  # 360° omnidirectional
SENSOR_NUM_RAYS = 180  # 2° angular resolution

# Planner parameters
CONNECTIVITY = 8  # 8-connected grid

# Frontier detection
FRONTIER_MIN_SIZE = 3  # minimum cells for a valid frontier region

# Exploration loop
MAX_EXPLORATION_STEPS = 5000  # safety limit
CELLS_PER_STEP = 5  # cells moved per iteration before next decision

# Time limits (seconds) — wall-clock time from exploration start
TIME_LIMITS = {
    'open_room': 30,
    'office': 60,
    'cave': 60,
}
DEFAULT_TIME_LIMIT = 60

# Occupancy values
FREE = 0
OCCUPIED = 1
UNKNOWN = -1

# Visualization
VIS_INTERVAL = 1  # visualize every N iterations
VIS_DELAY = 0.05  # default seconds between frames (overridden by --speed)
