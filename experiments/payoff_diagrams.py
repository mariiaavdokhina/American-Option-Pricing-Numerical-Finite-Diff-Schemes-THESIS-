import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('results/figures', exist_ok=True)

K = 100
S = np.linspace(0, 200, 500)
long_call = np.maximum(S - K, 0)
long_put = np.maximum(K - S, 0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    S,
    long_call,
    linewidth=2,
    label='Long Call',
)

axes[0].axhline(
    0,
    linewidth=1,
    color='black',
)

axes[0].axvline(
    K,
    linestyle='--',
    linewidth=1,
    color='grey',
    label=rf'$K = {K}$',
)

axes[0].set_title('Long call payoff')
axes[0].set_xlabel(r'Underlying price at maturity, $S_T$')
axes[0].set_ylabel('Payoff')
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(
    S,
    long_put,
    linewidth=2,
    label='Long Put',
)

axes[1].axhline(
    0,
    linewidth=1,
    color='black',
)

axes[1].axvline(
    K,
    linestyle='--',
    linewidth=1,
    color='grey',
    label=rf'$K = {K}$',
)

axes[1].set_title('Long put payoff')
axes[1].set_xlabel(r'Underlying price at maturity, $S_T$')
axes[1].set_ylabel('Payoff')
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()

plt.savefig(
    'results/figures/payoff_diagrams.png',
    dpi=300,
    bbox_inches='tight',
)
plt.show()