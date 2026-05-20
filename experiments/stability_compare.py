import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.grid import Grid
from src.explicit import explicit_american_put
from src.implicit import implicit_american_put
from src.rannacher import rannacher_american_put

METHOD_COLORS = {
    'Explicit': 'green',
    'Implicit': 'blue',
    'Rannacher': 'red',
}

METHODS = [
    ('Explicit', explicit_american_put),
    ('Implicit', implicit_american_put),
    ('Rannacher', rannacher_american_put),
]

def is_stable(values: np.ndarray, K: float) -> bool:
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return False
    if np.min(values) < -1e-6:
        return False
    if np.max(values) > 2 * K:
        return False
    return True

def run_explicit_oscillation_demo():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)

    # Same financial params as for the main experiment
    unstable_grid = Grid(N=75, M=50)
    stable_grid = Grid(N=80, M=4000)
    rows = []
    solutions = {}
    experiments = [
        ('Explicit unstable', explicit_american_put, unstable_grid),
        ('Explicit stable', explicit_american_put, stable_grid),
        ('Implicit stable', implicit_american_put, unstable_grid),
        ('Rannacher stable', rannacher_american_put, unstable_grid),
    ]

    for label, solver, grid in experiments:
        start = time.perf_counter()
        values = solver(grid)
        end = time.perf_counter()
        idx = np.argmin(np.abs(grid.S - grid.K))
        rows.append({
            'experiment': label,
            'N': grid.N,
            'M': grid.M,
            'dt': grid.dt,
            'price_near_K': values[idx],
            'min_value': np.min(values),
            'max_value': np.max(values),
            'runtime_seconds': end - start,
            'stable': is_stable(values, grid.K),
        })

        solutions[label] = (grid, values)

    table = pd.DataFrame(rows)
    table.to_csv(
        'results/tables/explicit_oscillation_demo.csv',
        index=False,
    )
    print(table)

    # Plot 1: replicate textbook-style unstable vs stable explicit Euler
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ax, label in zip(
        axes,
        ['Explicit unstable', 'Explicit stable'],
    ):
        grid, values = solutions[label]

        ax.plot(
            grid.S,
            values,
            marker='o',
            markersize=4,
            linewidth=2,
            color='green',
            label=label,
        )

        ax.axvline(
            grid.K,
            color='darkorange',
            linestyle='--',
            linewidth=1,
            label='Strike price K',
        )

        ax.set_ylabel('Option value')
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_title(
        f'Explicit Euler instability: N={unstable_grid.N}, M={unstable_grid.M}'
    )
    axes[1].set_title(
        f'Explicit Euler stable case: N={stable_grid.N}, M={stable_grid.M}'
    )
    axes[1].set_xlabel('Underlying asset price (S)')

    plt.tight_layout()
    plt.savefig(
        'results/figures/explicit_euler_oscillation_replication.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # Plot 2: zoom near strike for unstable explicit Euler
    grid, values = solutions['Explicit unstable']
    mask = (grid.S >= 60) & (grid.S <= 140)

    plt.figure(figsize=(8, 5))
    plt.plot(
        grid.S[mask],
        values[mask],
        marker='o',
        markersize=5,
        linewidth=2,
        color='green',
        label='Explicit Euler unstable',
    )
    plt.axvline(
        grid.K,
        color='darkorange',
        linestyle='--',
        linewidth=1,
        label='Strike price K',
    )

    plt.xlabel('Underlying asset price (S)')
    plt.ylabel('Option value')
    plt.title('Explicit Euler oscillations near the strike price')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/explicit_euler_oscillation_near_strike.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # Plot 3: compare unstable explicit with stable implicit/Rannacher on same bad grid
    plt.figure(figsize=(8, 5))

    for label in ['Explicit unstable', 'Implicit stable', 'Rannacher stable']:
        grid, values = solutions[label]

        if label.startswith('Explicit'):
            color = 'green'
        elif label.startswith('Implicit'):
            color = 'blue'
        else:
            color = 'red'

        clipped_values = np.clip(values, 0, 2 * grid.K)

        plt.plot(
            grid.S,
            clipped_values,
            linewidth=2,
            label=label,
            color=color,
        )

    plt.xlabel('Underlying asset price (S)')
    plt.ylabel('Option value, clipped to [0, 2K]')
    plt.title('Stability stress test under coarse temporal discretization')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/stability_stress_comparison_clipped.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

if __name__ == '__main__':
    run_explicit_oscillation_demo()