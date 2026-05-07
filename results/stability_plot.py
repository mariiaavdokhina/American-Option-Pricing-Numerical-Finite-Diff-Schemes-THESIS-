import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_stability_test() -> None:
    df = pd.read_csv('results/tables/stability_test.csv')

    os.makedirs('results/figures', exist_ok=True)

    plt.figure(figsize=(8, 5))

    for method in df['method'].unique():
        subset = df[df['method'] == method]
        plt.plot(
            subset['dt'],
            subset['price_near_K'],
            marker='o',
            linewidth=2,
            label=method
        )

    plt.xlabel('Time step size (dt)')
    plt.ylabel('Option price near S = K')
    plt.title('Stability test: explicit vs implicit')
    plt.yscale('symlog')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.savefig(
        'results/figures/stability_explicit_vs_implicit.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

if __name__ == '__main__':
    plot_stability_test()