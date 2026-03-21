"""
AER1516 Winter 2026 — Assignment 3: Autonomous Frontier-Based Exploration
Matplotlib live visualization. DO NOT MODIFY this file.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from config import CELL_SIZE, FREE, OCCUPIED, UNKNOWN, VIS_DELAY


class ExplorationVisualizer:
    """Real-time visualization of the exploration process."""

    def __init__(self, true_map, title='Frontier Exploration', delay=None):
        """
        Args:
            true_map: Ground-truth binary occupancy map (H x W numpy array).
            title: Window title.
            delay: Seconds to pause between frames (None = use VIS_DELAY).
        """
        self.true_map = true_map
        self.delay = delay if delay is not None else VIS_DELAY
        H, W = true_map.shape

        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.fig.suptitle(title, fontsize=14)

        # Build display image: 3-channel RGB
        self.display = np.ones((H, W, 3), dtype=np.float32) * 0.5

        # True map background at low opacity
        true_bg = np.ones((H, W, 3), dtype=np.float32)
        true_bg[true_map == 1] = [0.85, 0.85, 0.85]
        true_bg[true_map == 0] = [0.95, 0.95, 0.95]
        self.true_bg = true_bg

        self.im = self.ax.imshow(self.display, origin='upper', interpolation='nearest')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        # Robot marker (red filled circle)
        self.robot_dot, = self.ax.plot([], [], 'ro', markersize=8, zorder=10)

        # Robot trail (faint red line showing where robot has been)
        self.trail_cols = []
        self.trail_rows = []
        self.trail_line, = self.ax.plot([], [], '-', color='red', alpha=0.15,
                                         linewidth=1, zorder=3)

        # Current path (blue line from robot to goal)
        self.path_line, = self.ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7, zorder=5)

        # Current goal marker (green diamond)
        self.goal_marker, = self.ax.plot([], [], 'D', color='#00cc00', markersize=6,
                                          markeredgecolor='black', markeredgewidth=1,
                                          zorder=9)

        # Frontier cells scatter (lime dots)
        self.frontier_scatter = self.ax.scatter([], [], s=8, c='lime', zorder=4, alpha=0.6)
        # Frontier centroids (yellow stars)
        self.centroid_scatter = self.ax.scatter([], [], s=50, c='yellow', marker='*',
                                                 edgecolors='black', linewidths=0.5, zorder=6)

        # Status text (above the axes, not overlapping the map)
        self.status_text = self.ax.set_title(
            '', fontsize=9, family='monospace', loc='left', pad=10
        )

        plt.ion()
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, occ_grid, robot_pos, path, frontier_regions,
               coverage, elapsed_time, time_limit, strategy_name, step,
               goal_rc=None):
        """
        Update the visualization.

        Args:
            occ_grid: OccupancyGrid instance.
            robot_pos: (x, y) in world coordinates.
            path: list of (row, col) or None.
            frontier_regions: list of FrontierRegion.
            coverage: float 0-1.
            elapsed_time: float seconds elapsed.
            time_limit: float seconds time limit.
            strategy_name: str.
            step: int.
            goal_rc: (row, col) of current goal, or None.
        """
        H, W = occ_grid.height, occ_grid.width

        # Build display from occupancy grid
        display = self.true_bg.copy() * 0.1  # faint background

        grid = occ_grid.grid
        display[grid == UNKNOWN] = [0.5, 0.5, 0.5]
        display[grid == FREE] = [1.0, 1.0, 1.0]
        display[grid == OCCUPIED] = [0.0, 0.0, 0.0]

        self.im.set_data(display)

        # Robot position (subtract 0.5 because grid_to_world returns cell center)
        rx, ry = robot_pos
        rc = rx / CELL_SIZE - 0.5
        rr = ry / CELL_SIZE - 0.5
        self.robot_dot.set_data([rc], [rr])

        # Trail
        self.trail_cols.append(rc)
        self.trail_rows.append(rr)
        self.trail_line.set_data(self.trail_cols, self.trail_rows)

        # Current path (remaining portion)
        if path:
            path_cols = [c for r, c in path]
            path_rows = [r for r, c in path]
            self.path_line.set_data(path_cols, path_rows)
        else:
            self.path_line.set_data([], [])

        # Current goal
        if goal_rc is not None:
            self.goal_marker.set_data([goal_rc[1]], [goal_rc[0]])
        elif path:
            # Show end of path as goal
            self.goal_marker.set_data([path[-1][1]], [path[-1][0]])
        else:
            self.goal_marker.set_data([], [])

        # Frontiers
        if frontier_regions:
            all_cells = []
            centroids = []
            for fr in frontier_regions:
                all_cells.extend(fr.cells)
                centroids.append(fr.centroid)
            if all_cells:
                fr_rows, fr_cols = zip(*all_cells)
                self.frontier_scatter.set_offsets(np.column_stack([fr_cols, fr_rows]))
            else:
                self.frontier_scatter.set_offsets(np.empty((0, 2)))
            if centroids:
                cr, cc = zip(*centroids)
                self.centroid_scatter.set_offsets(np.column_stack([cc, cr]))
            else:
                self.centroid_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.frontier_scatter.set_offsets(np.empty((0, 2)))
            self.centroid_scatter.set_offsets(np.empty((0, 2)))

        # Status text
        goal_str = f'({goal_rc[0]},{goal_rc[1]})' if goal_rc else ('path...' if path else 'none')
        self.status_text.set_text(
            f'Coverage: {coverage * 100:.1f}%  |  '
            f'Time: {elapsed_time:.1f}/{time_limit:.0f}s\n'
            f'Strategy: {strategy_name}  |  Step: {step}  |  '
            f'Goal: {goal_str}'
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.delay)

    def show_complete(self):
        """Show final state and block until the user closes the window."""
        self.status_text.set_text('EXPLORATION COMPLETE — Close window to exit')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.ioff()
        plt.show()

    def close(self):
        plt.close(self.fig)


def save_snapshot(occ_grid, true_map, robot_pos, path, frontier_regions,
                  coverage, elapsed_time, time_limit, strategy_name, step, filepath):
    """Save a static PNG snapshot of the exploration state."""
    H, W = occ_grid.height, occ_grid.width

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    display = np.ones((H, W, 3), dtype=np.float32) * 0.5
    grid = occ_grid.grid
    display[grid == UNKNOWN] = [0.5, 0.5, 0.5]
    display[grid == FREE] = [1.0, 1.0, 1.0]
    display[grid == OCCUPIED] = [0.0, 0.0, 0.0]

    ax.imshow(display, origin='upper', interpolation='nearest')

    rx, ry = robot_pos
    ax.plot(rx / CELL_SIZE - 0.5, ry / CELL_SIZE - 0.5, 'ro', markersize=8, zorder=10)

    if path:
        path_cols = [c for r, c in path]
        path_rows = [r for r, c in path]
        ax.plot(path_cols, path_rows, 'b-', linewidth=1.5, alpha=0.7, zorder=5)

    if frontier_regions:
        all_cells = []
        centroids = []
        for fr in frontier_regions:
            all_cells.extend(fr.cells)
            centroids.append(fr.centroid)
        if all_cells:
            fr_rows, fr_cols = zip(*all_cells)
            ax.scatter(fr_cols, fr_rows, s=8, c='lime', zorder=4, alpha=0.6)
        if centroids:
            cr, cc = zip(*centroids)
            ax.scatter(cc, cr, s=50, c='yellow', marker='*',
                       edgecolors='black', linewidths=0.5, zorder=6)

    ax.set_title(
        f'Coverage: {coverage * 100:.1f}% | '
        f'Time: {elapsed_time:.1f}/{time_limit:.0f}s | '
        f'Strategy: {strategy_name} | Step: {step}',
        fontsize=11
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
