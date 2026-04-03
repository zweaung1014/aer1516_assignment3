"""
Generate coverage-vs-step plots for the AER1516 Assignment 3 report.
Runs all 9 trials (3 maps x 3 strategies) headless and saves one figure per map.
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from run_exploration import run_exploration

MAPS = ['open_room', 'office', 'cave']
STRATEGIES = ['random', 'nearest', 'custom']
SEED = 42
STYLE = {
    'random':  {'color': 'gray',   'linestyle': '--', 'label': 'Random'},
    'nearest': {'color': 'blue',   'linestyle': '-',  'label': 'Nearest'},
    'custom':  {'color': 'red',    'linestyle': '-',  'label': 'Custom'},
}

results = {}

for map_name in MAPS:
    for strategy in STRATEGIES:
        print(f"Running {map_name} / {strategy} ...", end=' ', flush=True)
        res = run_exploration(
            map_name=map_name,
            strategy_name=strategy,
            visualize=False,
            seed=SEED,
            enforce_time=False,
        )
        results[(map_name, strategy)] = res
        print(f"coverage={res['final_coverage']*100:.1f}%  "
              f"steps={res['num_steps']}  "
              f"time={res['elapsed_time']:.1f}s")

# Generate one figure per map
for map_name in MAPS:
    fig, ax = plt.subplots(figsize=(7, 4))
    for strategy in STRATEGIES:
        res = results[(map_name, strategy)]
        history = [c * 100 for c in res['coverage_history']]  # convert to %
        steps = list(range(1, len(history) + 1))
        s = STYLE[strategy]
        ax.plot(steps, history, color=s['color'], linestyle=s['linestyle'],
                label=f"{s['label']} ({history[-1]:.1f}%)", linewidth=1.5)

    ax.set_xlabel('Step')
    ax.set_ylabel('Coverage (%)')
    ax.set_title(f'Coverage vs. Step — {map_name}')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    filename = f'coverage_{map_name}.png'
    fig.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close(fig)

print("\nDone. Use the PNG files in your report.")
