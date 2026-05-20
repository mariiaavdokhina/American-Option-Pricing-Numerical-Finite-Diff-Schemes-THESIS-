import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.grid import Grid
from src.explicit import explicit_american_put
from src.implicit import implicit_american_put
from src.rannacher import rannacher_american_put
from src.metrics import (
    max_absolute_error,
    mean_absolute_error,
    root_mean_squared_error,
    pointwise_error,
)

METHODS = [
    ('Explicit', explicit_american_put),
    ('Implicit', implicit_american_put),
    ('Rannacher', rannacher_american_put),
]

METHOD_COLORS = {
    'Explicit': 'green',
    'Implicit': 'blue',
    'Rannacher': 'red',
}

def is_stable(values: np.ndarray, K: float) -> bool:
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return False
    if np.min(values) < -1e-6:
        return False
    if np.max(values) > 2 * K:
        return False
    return True

def has_oscillations(
    values: np.ndarray,
    S: np.ndarray,
    K: float,
    window: float = 40.0,
    tolerance: float = 1e-6,
) -> bool:
    """
    Detects local oscillations near the strike
    """
    mask = (S >= K - window) & (S <= K + window)
    local_values = values[mask]
    if len(local_values) < 5:
        return False
    diffs = np.diff(local_values)
    # ignore tiny numerical noise
    diffs[np.abs(diffs) < tolerance] = 0.0
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 3:
        return False
    sign_changes = np.sum(signs[1:] != signs[:-1])

    return bool(sign_changes >= 2)

def complexity_counts(method_name: str, grid: Grid, damping_steps: int = 2) -> dict:
    """
    Records simple quantitative complexity indicators for the thesis table
    """
    if method_name == 'Explicit':
        linear_systems_solved = 0
        operator_evaluations = grid.M
    elif method_name == 'Implicit':
        linear_systems_solved = grid.M
        operator_evaluations = 0
    elif method_name == 'Rannacher':
        linear_systems_solved = grid.M
        operator_evaluations = max(grid.M - damping_steps, 0)
    else:
        linear_systems_solved = np.nan
        operator_evaluations = np.nan

    return {
        'grid_points_estimated': grid.N + 1,
        'time_steps': grid.M,
        'linear_systems_solved': linear_systems_solved,
        'operator_evaluations': operator_evaluations,
    }

def measure_method(method_name, solver, grid, reference_on_grid=None):
    start = time.perf_counter()
    values = solver(grid)
    end = time.perf_counter()
    idx = np.argmin(np.abs(grid.S - grid.K))
    stable = is_stable(values, grid.K)
    oscillations = has_oscillations(values, grid.S, grid.K)
    result = {
        'method': method_name,
        'N': grid.N,
        'M': grid.M,
        'dt': grid.dt,
        'price_near_K': values[idx],
        'min_value': np.min(values),
        'max_value': np.max(values),
        'runtime_seconds': end - start,
        'stable': stable,
        'oscillations': 'Yes' if oscillations else 'No',
        'values': values,
    }
    result.update(complexity_counts(method_name, grid))
    if reference_on_grid is not None:
        errors = pointwise_error(values, reference_on_grid)
        result.update({
            'reference_price_near_K': reference_on_grid[idx],
            'pointwise_error_near_K': errors[idx],
            'max_absolute_error': max_absolute_error(values, reference_on_grid),
            'mean_absolute_error': mean_absolute_error(values, reference_on_grid),
            'rmse': root_mean_squared_error(values, reference_on_grid),
        })

    return result

def run_main_comparison():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)

    grid = Grid(N=200, M=10000)
    # Fine-grid benchmark ref solution
    reference_grid = Grid(N=400, M=40000)
    reference_values = rannacher_american_put(reference_grid)
    reference_on_grid = np.interp(grid.S, reference_grid.S, reference_values)

    results = [
        measure_method(method_name, solver, grid, reference_on_grid)
        for method_name, solver in METHODS
    ]

    quantitative_columns = [
        'method',
        'N',
        'M',
        'dt',
        'price_near_K',
        'reference_price_near_K',
        'pointwise_error_near_K',
        'max_absolute_error',
        'mean_absolute_error',
        'rmse',
        'runtime_seconds',
        'grid_points_estimated',
        'time_steps',
        'linear_systems_solved',
        'operator_evaluations',
        'stable',
    ]

    qualitative_columns = [
        'method',
        'oscillations',
        'stable',
    ]

    quantitative_table = pd.DataFrame([
        {key: result[key] for key in quantitative_columns}
        for result in results
    ])

    qualitative_table = pd.DataFrame([
        {key: result[key] for key in qualitative_columns}
        for result in results
    ])

    quantitative_table.to_csv(
        'results/tables/final_method_comparison_quantitative.csv',
        index=False,
    )
    qualitative_table.to_csv(
        'results/tables/final_method_comparison_qualitative.csv',
        index=False,
    )

    print('\nQuantitative final comparison:')
    print(quantitative_table)

    print('\nQualitative final comparison:')
    print(qualitative_table)

    # Plot 1: full price plot
    plt.figure(figsize=(8, 5))
    for result in results:
        plt.plot(
            grid.S,
            result['values'],
            linewidth=2,
            label=result['method'],
            color=METHOD_COLORS[result['method']],
        )

    plt.xlabel('Underlying asset price (S)')
    plt.ylabel('Option value')
    plt.title('American put option price: method comparison')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/final_method_comparison.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # Plot 2: zoom around strike
    plt.figure(figsize=(8, 5))
    mask = (grid.S >= 80) & (grid.S <= 120)
    for result in results:
        plt.plot(
            grid.S[mask],
            result['values'][mask],
            linewidth=2,
            label=result['method'],
            color=METHOD_COLORS[result['method']],
        )
    plt.xlabel('Underlying asset price (S)')
    plt.ylabel('Option value')
    plt.title('American put option price: zoom around strike')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/final_method_comparison_zoom.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # Plot 3: absolute errors
    plt.figure(figsize=(8, 5))

    for result in results:
        errors = np.abs(result['values'] - reference_on_grid)
        plt.plot(
            grid.S,
            errors,
            linewidth=2,
            label=result['method'],
            color=METHOD_COLORS[result['method']],
        )

    plt.xlabel('Underlying asset price (S)')
    plt.ylabel('Absolute error vs benchmark')
    plt.title('Absolute error relative to refined Rannacher benchmark')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/final_method_absolute_errors.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # Plot 4: absolute errors, zoom around strike
    plt.figure(figsize=(8, 5))

    mask = (grid.S >= 80) & (grid.S <= 120)

    for result in results:
        errors = np.abs(result['values'] - reference_on_grid)
        plt.plot(
            grid.S[mask],
            errors[mask],
            linewidth=2,
            label=result['method'],
            color=METHOD_COLORS[result['method']],
        )

    plt.xlabel('Underlying asset price (S)')
    plt.ylabel('Absolute error vs benchmark')
    plt.title('Absolute error near the strike price')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/final_method_absolute_errors_zoom.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    rannacher_values = next(
    result['values'] for result in results if result['method'] == 'Rannacher'
)

    # Plot 5: difference from Rannacher
    plt.figure(figsize=(8, 5))

    for result in results:
        if result['method'] == 'Rannacher':
            continue

        difference = result['values'] - rannacher_values

        plt.plot(
            grid.S,
            difference,
            linewidth=2,
            label=f"{result['method']} - Rannacher",
            color=METHOD_COLORS[result['method']],
        )

    plt.axhline(0, linewidth=1, color='black')
    plt.xlabel('Underlying asset price (S)')
    plt.ylabel('Price difference')
    plt.title('Difference from Rannacher solution')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/difference_from_rannacher.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

def run_temporal_order_experiment():
    """
    Estimates recovered temporal order
    """
    os.makedirs('results/tables', exist_ok=True)
    N = 200
    reference_grid = Grid(N=N, M=80000)
    reference_values = rannacher_american_put(reference_grid)
    M_values = [5000, 10000, 20000, 40000]
    rows = []
    for method_name, solver in METHODS:
        previous_error = None
        previous_M = None

        for M in M_values:
            grid = Grid(N=N, M=M)
            values = solver(grid)
            # Same N, so no interpolation needed
            error = root_mean_squared_error(values, reference_values)
            if previous_error is not None and error > 0:
                recovered_order = np.log(previous_error / error) / np.log(2)
            else:
                recovered_order = np.nan
            rows.append({
                'method': method_name,
                'N': N,
                'M': M,
                'dt': grid.dt,
                'rmse_vs_fine_time_reference': error,
                'previous_M': previous_M,
                'recovered_temporal_order': recovered_order,
                'stable': is_stable(values, grid.K),
                'oscillations': 'Yes' if has_oscillations(values, grid.S, grid.K) else 'No',
            })

            previous_error = error
            previous_M = M

    order_table = pd.DataFrame(rows)
    order_table.to_csv(
        'results/tables/temporal_order_estimates.csv',
        index=False,
    )
    print('\nRecovered temporal order estimates:')
    print(order_table)

def run_coarse_grid_diagnostic():
    """
    Coarse-grid diagnostic
    """
    os.makedirs('results/tables', exist_ok=True)
    grid = Grid(N=50, M=500)
    results = [
        measure_method(method_name, solver, grid)
        for method_name, solver in METHODS
    ]

    table = pd.DataFrame([
        {key: value for key, value in result.items() if key != 'values'}
        for result in results
    ])

    table.to_csv(
        'results/tables/coarse_grid_diagnostic.csv',
        index=False,
    )

    print('\nCoarse-grid diagnostic:')
    print(table)

if __name__ == '__main__':
    run_main_comparison()
    run_temporal_order_experiment()
    run_coarse_grid_diagnostic()