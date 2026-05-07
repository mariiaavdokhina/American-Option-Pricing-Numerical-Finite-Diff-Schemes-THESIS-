import os
import numpy as np
import pandas as pd
from src.explicit import explicit_american_put
from src.grid import Grid
from src.implicit import implicit_american_put

def is_stable(values: np.ndarray, K: float) -> bool:
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return False
    if np.min(values) < -1e-6:
        return False
    if np.max(values) > 2 * K:
        return False
    return True


def run_stability_test() -> None:
    N = 200
    M_values = [10000, 5000, 2000, 1000, 500, 250, 100, 50]
    results = []
    for M in M_values:
        grid = Grid(N=N, M=M)
        idx = np.argmin(np.abs(grid.S - grid.K))

        for method_name, solver in [
            ('Explicit', explicit_american_put),
            ('Implicit', implicit_american_put),
        ]:
            try:
                values = solver(grid)

                results.append({
                    'method': method_name,
                    'N': N,
                    'M': M,
                    'dt': grid.dt,
                    'price_near_K': values[idx],
                    'min_value': np.min(values),
                    'max_value': np.max(values),
                    'stable': is_stable(values, grid.K)
                })

            except Exception as error:
                results.append({
                    'method': method_name,
                    'N': N,
                    'M': M,
                    'dt': grid.dt,
                    'price_near_K': np.nan,
                    'min_value': np.nan,
                    'max_value': np.nan,
                    'stable': False,
                    'error': str(error)
                })

    df = pd.DataFrame(results)
    os.makedirs('results/tables', exist_ok=True)
    df.to_csv('results/tables/stability_test.csv', index=False)
    print(df)

if __name__ == '__main__':
    run_stability_test()