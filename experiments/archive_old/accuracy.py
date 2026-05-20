import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.explicit import explicit_american_put
from src.grid import Grid
from src.implicit import implicit_american_put
from src.metrics import (
    max_absolute_error,
    mean_absolute_error,
    root_mean_squared_error,
    pointwise_error,
)
from src.rannacher import rannacher_american_put

def run_accuracy_experiment() -> None:
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # benchmark
    reference_grid = Grid(N=400, M=40000)
    reference_values = rannacher_american_put(reference_grid)

    # a standard grid for comparison
    test_grid = Grid(N=200, M=10000)

    # interpolating the reference solution onto the test grid
    reference_on_test_grid = np.interp(
        test_grid.S,
        reference_grid.S,
        reference_values,
    )

    methods = [
        ('Explicit', explicit_american_put),
        ('Implicit', implicit_american_put),
        ('Rannacher', rannacher_american_put),
    ]

    results = []

    plt.figure(figsize=(8, 5))

    for method_name, solver in methods:
        start_time = time.perf_counter()
        values = solver(test_grid)
        runtime = time.perf_counter() - start_time

        errors = pointwise_error(values, reference_on_test_grid)

        idx = np.argmin(np.abs(test_grid.S - test_grid.K))

        results.append({
            'method': method_name,
            'N': test_grid.N,
            'M': test_grid.M,
            'price_near_K': values[idx],
            'reference_price_near_K': reference_on_test_grid[idx],
            'pointwise_error_near_K': errors[idx],
            'max_absolute_error': max_absolute_error(values, reference_on_test_grid),
            'mean_absolute_error': mean_absolute_error(values, reference_on_test_grid),
            'rmse': root_mean_squared_error(values, reference_on_test_grid),
            'runtime_seconds': runtime,
        })

        plt.plot(
            test_grid.S,
            errors,
            linewidth=2,
            label=method_name,
        )

    df = pd.DataFrame(results)
    df.to_csv('results/tables/accuracy_comparison.csv', index=False)
    print(df)
    plt.xlabel('Underlying Asset Price (S)')
    plt.ylabel('Absolute Error')
    plt.title('Pointwise Error Relative to Refined Rannacher Benchmark')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/accuracy_pointwise_error.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

if __name__ == '__main__':
    run_accuracy_experiment()