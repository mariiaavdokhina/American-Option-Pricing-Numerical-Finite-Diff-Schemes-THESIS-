import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.grid import Grid
from src.rannacher import rannacher_american_put
from src.metrics import (
    max_absolute_error,
    mean_absolute_error,
    root_mean_squared_error,
    pointwise_error,
)

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K = 100.0
T = 1.0
r = 0.05
sigma = 0.2
S_max = 300.0

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, S, tau):
        x = torch.cat([S / S_max, tau / T], dim=1)
        raw = self.net(x)
        payoff = torch.clamp(K - S, min=0.0)
        return payoff + torch.nn.functional.softplus(raw)

def payoff(S):
    return torch.clamp(K - S, min=0.0)

def train_pinn(epochs=8000, n_interior=6000, lr=1e-3):
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        S = torch.rand(n_interior, 1, device=device) * S_max
        tau = torch.rand(n_interior, 1, device=device) * T
        S.requires_grad_(True)
        tau.requires_grad_(True)

        V = model(S, tau)

        V_tau = torch.autograd.grad(
            V, tau,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
        )[0]

        V_S = torch.autograd.grad(
            V, S,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
        )[0]

        V_SS = torch.autograd.grad(
            V_S, S,
            grad_outputs=torch.ones_like(V_S),
            create_graph=True,
        )[0]

        pde = V_tau - (
            0.5 * sigma ** 2 * S ** 2 * V_SS
            + r * S * V_S
            - r * V
        )

        pde_loss = torch.mean(pde ** 2)

        # Terminal condition at tau = 0
        S0 = torch.rand(n_interior // 4, 1, device=device) * S_max
        tau0 = torch.zeros_like(S0)
        payoff_loss = torch.mean((model(S0, tau0) - payoff(S0)) ** 2)

        # Boundary conditions
        tb = torch.rand(n_interior // 4, 1, device=device) * T

        S_left = torch.zeros_like(tb)
        left_loss = torch.mean((model(S_left, tb) - K) ** 2)

        S_right = torch.full_like(tb, S_max)
        right_loss = torch.mean(model(S_right, tb) ** 2)

        # Obstacle condition V >= payoff
        obstacle_loss = torch.mean(
            torch.relu(payoff(S) - V) ** 2
        )

        loss = (
            pde_loss
            + 10.0 * payoff_loss
            + left_loss
            + right_loss
            + 10.0 * obstacle_loss
        )

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(
                f'Epoch {epoch}, '
                f'loss={loss.item():.6e}, '
                f'pde={pde_loss.item():.6e}, '
                f'payoff={payoff_loss.item():.6e}, '
                f'obstacle={obstacle_loss.item():.6e}'
            )

    runtime = time.perf_counter() - start
    return model, runtime

def evaluate():
    grid = Grid(N=200, M=10000)

    reference_grid = Grid(N=400, M=40000)
    reference_values = rannacher_american_put(reference_grid)
    reference_on_grid = np.interp(grid.S, reference_grid.S, reference_values)

    model, runtime = train_pinn()

    S_tensor = torch.tensor(
        grid.S.reshape(-1, 1),
        dtype=torch.float32,
        device=device,
    )
    tau_tensor = torch.full_like(S_tensor, T)

    with torch.no_grad():
        pinn_values = model(S_tensor, tau_tensor).cpu().numpy().flatten()

    errors = pointwise_error(pinn_values, reference_on_grid)
    idx = np.argmin(np.abs(grid.S - grid.K))

    table = pd.DataFrame([{
        'method': 'PINN',
        'N_evaluation': grid.N,
        'training_epochs': 8000,
        'price_near_K': pinn_values[idx],
        'reference_price_near_K': reference_on_grid[idx],
        'pointwise_error_near_K': errors[idx],
        'max_absolute_error': max_absolute_error(pinn_values, reference_on_grid),
        'mean_absolute_error': mean_absolute_error(pinn_values, reference_on_grid),
        'rmse': root_mean_squared_error(pinn_values, reference_on_grid),
        'runtime_seconds': runtime,
        'stable': not (
            np.any(np.isnan(pinn_values))
            or np.any(np.isinf(pinn_values))
        ),
    }])

    table.to_csv(
        'results/tables/pinn_american_put_comparison.csv',
        index=False,
    )

    print(table)

    plt.figure(figsize=(8, 5))
    plt.plot(grid.S, reference_on_grid, label='Rannacher benchmark', linewidth=2, color='red')
    plt.plot(grid.S, pinn_values, label='PINN', linewidth=2, color='purple')
    plt.xlabel('Underlying asset price ($S$)')
    plt.ylabel('Option value')
    plt.title('American put option: PINN vs Rannacher benchmark')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        'results/figures/pinn_vs_rannacher.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(grid.S, np.abs(errors), linewidth=2, color='purple')
    plt.xlabel('Underlying asset price ($S$)')
    plt.ylabel('Absolute error')
    plt.title('PINN absolute error relative to Rannacher benchmark')
    plt.grid(alpha=0.3)
    plt.savefig(
        'results/figures/pinn_absolute_error.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

if __name__ == '__main__':
    evaluate()