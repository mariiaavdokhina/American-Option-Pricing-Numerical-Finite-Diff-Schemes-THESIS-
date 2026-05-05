import numpy as np
from scipy.linalg import solve_banded
from src.grid import Grid
from src.payoff import put_payoff

def implicit_american_put(grid: Grid) -> np.ndarray:
    S = grid.S
    K = grid.K
    r = grid.r
    sigma = grid.sigma
    dS = grid.dS
    dt = grid.dt
    N = grid.N
    M = grid.M
    payoff = put_payoff(S, K)
    V = payoff.copy()
    i = np.arange(1, N)
    a = 0.5 * sigma ** 2 * S[i] ** 2 / dS ** 2 - r * S[i] / (2.0 * dS)
    b = -sigma ** 2 * S[i] ** 2 / dS ** 2 - r
    c = 0.5 * sigma ** 2 * S[i] ** 2 / dS ** 2 + r * S[i] / (2.0 * dS)
    lower = -dt * a[1:]
    main = 1.0 - dt * b
    upper = -dt * c[:-1]
    ab = np.zeros((3, N - 1))
    ab[0, 1:] = upper
    ab[1, :] = main
    ab[2, :-1] = lower
    for _ in range(M):
        rhs = V[1:N].copy()
        # boundary conditions for the put option
        rhs[0] += dt * a[0] * K
        rhs[-1] += dt * c[-1] * 0.0
        V_inner = solve_banded((1, 1), ab, rhs)
        V[1:N] = np.maximum(V_inner, payoff[1:N])
        V[0] = K
        V[N] = 0.0
    return V

if __name__ == '__main__':
    grid = Grid(N=200, M=10000)
    values = implicit_american_put(grid)
    print('American put option price near S = K:')
    idx = np.argmin(np.abs(grid.S - grid.K))
    print(values[idx])