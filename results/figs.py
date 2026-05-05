import os

import matplotlib.pyplot as plt
import numpy as np

from src.explicit import explicit_american_put
from src.grid import Grid
from src.implicit import implicit_american_put


def plot_explicit_vs_implicit() -> None:
    grid = Grid(N=200, M=10000)

    explicit_values = explicit_american_put(grid)
    implicit_values = implicit_american_put(grid)

    idx = np.argmin(np.abs(grid.S - grid.K))

    print(f'Explicit price near S = K: {explicit_values[idx]}')
    print(f'Implicit price near S = K: {implicit_values[idx]}')

    os.makedirs('results/figures', exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(grid.S, explicit_values, linewidth=2, label='Explicit Method')
    plt.plot(grid.S, implicit_values, linewidth=2, linestyle='--', label='Implicit Method')
    plt.xlabel('Underlying Asset Price (S)')
    plt.ylabel('Option Value')
    plt.title('American Put Option Price: Explicit vs Implicit Method')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/explicit_vs_implicit_american_put.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

if __name__ == '__main__':
    plot_explicit_vs_implicit()