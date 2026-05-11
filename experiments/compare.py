import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.grid import Grid
from src.explicit import explicit_american_put
from src.implicit import implicit_american_put
from src.rannacher import rannacher_american_put

def measure_method(method_name, solver, grid):
    start = time.perf_counter()
    values = solver(grid)
    end = time.perf_counter()
    idx = np.argmin(np.abs(grid.S - grid.K))
    return {
        'method': method_name,
        'N': grid.N,
        'M': grid.M,
        'dt': grid.dt,
        'price_near_K': values[idx],
        'min_value': np.min(values),
        'max_value': np.max(values),
        'runtime_seconds': end - start,
        'values': values
    }

def run_final_comparison():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)

    grid = Grid(N=50, M=500)

    results = [
        measure_method('Explicit', explicit_american_put, grid),
        measure_method('Implicit', implicit_american_put, grid),
        measure_method('Rannacher', rannacher_american_put, grid)
    ]

    table = pd.DataFrame([
        {key: value for key, value in result.items() if key != 'values'}
        for result in results
    ])
    table.to_csv('results/tables/final_method_comparison.csv', index=False)
    print(table)

    # plotting all schemes
    plt.figure(figsize=(8, 5))

    for result in results:
        plt.plot(
            grid.S,
            result['values'],
            linewidth=2,
            label=result['method']
        )

    plt.xlabel('Underlying Asset Price (S)')
    plt.ylabel('Option Value')
    plt.title('American Put Option Price: Method Comparison')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/final_method_comparison.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

    # plot zoom around strike
    plt.figure(figsize=(8, 5))

    mask = (grid.S >= 80) & (grid.S <= 120)

    for result in results:
        plt.plot(
            grid.S[mask],
            result['values'][mask],
            linewidth=2,
            label=result['method']
        )
    plt.xlabel('Underlying Asset Price (S)')
    plt.ylabel('Option Value')
    plt.title('American Put Option Price: Zoom Around Strike')
    plt.grid(True)
    plt.legend()
    plt.savefig(
        'results/figures/final_method_comparison_zoom.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

if __name__ == '__main__':
    run_final_comparison()