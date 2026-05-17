import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.grid import Grid
from src.explicit import explicit_american_put
from src.implicit import implicit_american_put
from src.rannacher import rannacher_american_put

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

def run_stress_comparison():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    # Deliberately coarse temporal grid for stability stress test
    grid = Grid(N=200, M=1000)

    rows = []
    solutions = {}

    for method_name, solver in METHODS:
        start = time.perf_counter()
        values = solver(grid)
        end = time.perf_counter()

        idx = np.argmin(np.abs(grid.S - grid.K))
        stable = is_stable(values, grid.K)

        rows.append({
            'method': method_name,
            'N': grid.N,
            'M': grid.M,
            'dt': grid.dt,
            'price_near_K': values[idx],
            'min_value': np.min(values),
            'max_value': np.max(values),
            'runtime_seconds': end - start,
            'stable': stable,
        })

        solutions[method_name] = values

    table = pd.DataFrame(rows)
    table.to_csv('results/tables/stability_stress_comparison.csv', index=False)
    print(table)

    # Plot only stable methods on normal scale
    plt.figure(figsize=(8, 5))
    for method_name, values in solutions.items():
        if is_stable(values, grid.K):
            plt.plot(grid.S, values, linewidth=2, label=method_name)

    plt.xlabel('Underlying Asset Price (S)')
    plt.ylabel('Option Value')
    plt.title('Stability Stress Test: Stable Schemes Only')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/stability_stress_stable_methods.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    for method_name, values in solutions.items():
        clipped_values = np.clip(values, 0, 2 * grid.K)
        plt.plot(grid.S, clipped_values, linewidth=2, label=method_name)

    plt.xlabel('Underlying Asset Price (S)')
    plt.ylabel('Option Value, clipped to [0, 2K]')
    plt.title('Stability Stress Test: Clipped Values')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/stability_stress_clipped.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

if __name__ == '__main__':
    run_stress_comparison()