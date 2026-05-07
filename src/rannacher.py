import numpy as np
from scipy.linalg import solve_banded
from src.grid import Grid
from src.payoff import put_payoff

def _build_operator_coefficients(grid: Grid):
    S = grid.S
    r = grid.r
    sigma = grid.sigma
    dS = grid.dS
    N = grid.N
    i = np.arange(1, N)
    a = 0.5 * sigma ** 2 * S[i] ** 2 / dS ** 2 - r * S[i] / (2.0 * dS)
    b = -sigma ** 2 * S[i] ** 2 / dS ** 2 - r
    c = 0.5 * sigma ** 2 * S[i] ** 2 / dS ** 2 + r * S[i] / (2.0 * dS)
    return a, b, c

def _build_banded_matrix(a, b, c, dt: float, theta: float):
    n_inner = len(b)
    lower = -theta * dt * a[1:]
    main = 1.0 - theta * dt * b
    upper = -theta * dt * c[:-1]
    ab = np.zeros((3, n_inner))
    ab[0, 1:] = upper
    ab[1, :] = main
    ab[2, :-1] = lower
    return ab

def _apply_operator_to_inner(V: np.ndarray, a, b, c) -> np.ndarray:
    return (
        a * V[:-2]
        + b * V[1:-1]
        + c * V[2:]
    )

def rannacher_american_put(grid: Grid, damping_steps: int = 2) -> np.ndarray:
    S = grid.S
    K = grid.K
    dt = grid.dt
    N = grid.N
    M = grid.M

    payoff = put_payoff(S, K)
    V = payoff.copy()
    a, b, c = _build_operator_coefficients(grid)
    # Phase 1: backward Euler
    be_steps = min(damping_steps, M)
    be_matrix = _build_banded_matrix(a, b, c, dt, theta=1.0)

    for _ in range(be_steps):
        rhs = V[1:N].copy()

        # Boundary conditions for the American put option
        rhs[0] += dt * a[0] * K
        rhs[-1] += dt * c[-1] * 0.0
        V_inner = solve_banded((1, 1), be_matrix, rhs)
        V[1:N] = np.maximum(V_inner, payoff[1:N])
        V[0] = K
        V[N] = 0.0

    # Phase 2: Crank-Nicolson
    cn_matrix = _build_banded_matrix(a, b, c, dt, theta=0.5)
    for _ in range(be_steps, M):
        rhs = V[1:N] + 0.5 * dt * _apply_operator_to_inner(V, a, b, c)
        # Boundary conditions for the American put option
        rhs[0] += 0.5 * dt * a[0] * K
        rhs[-1] += 0.5 * dt * c[-1] * 0.0
        V_inner = solve_banded((1, 1), cn_matrix, rhs)
        V[1:N] = np.maximum(V_inner, payoff[1:N])
        V[0] = K
        V[N] = 0.0
    return V

if __name__ == '__main__':
    grid = Grid(N=200, M=40000)
    values = rannacher_american_put(grid)
    print('Rannacher')
    print('With N = 200, M = 40000:')
    print('American put option price near S = K:')
    idx = np.argmin(np.abs(grid.S - grid.K))
    print(values[idx])