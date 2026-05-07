import os
import time
import pandas as pd
from src.explicit import explicit_american_put
from src.grid import Grid
from src.implicit import implicit_american_put

def measure_runtime(method_name, solver, grid):
    start_time = time.perf_counter()
    values = solver(grid)
    end_time = time.perf_counter()
    idx = abs(grid.S - grid.K).argmin()

    return {
        'method': method_name,
        'N': grid.N,
        'M': grid.M,
        'price_near_K': values[idx],
        'runtime_seconds': end_time - start_time}

def run_runtime_comparison():
    grid = Grid(N=200, M=10000)
    results = [
        measure_runtime('Explicit', explicit_american_put, grid),
        measure_runtime('Implicit', implicit_american_put, grid)]

    df = pd.DataFrame(results)
    os.makedirs('results/tables', exist_ok=True)
    df.to_csv('results/tables/runtime_comparison.csv', index=False)
    print(df)

if __name__ == '__main__':
    run_runtime_comparison()