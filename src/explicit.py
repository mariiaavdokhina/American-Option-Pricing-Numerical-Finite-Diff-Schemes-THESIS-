import numpy as np
from src.grid import Grid
from src.payoff import put_payoff

def explicit_american_put(grid: Grid) -> np.ndarray:
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

    for _ in range(M):
        V_old = V.copy()

        for i in range(1, N):
            delta = (V_old[i + 1] - V_old[i - 1]) / (2.0 * dS)
            gamma = (V_old[i + 1] - 2.0 * V_old[i] + V_old[i - 1]) / (dS ** 2)
            continuation_value = V_old[i] + dt * (
                0.5 * sigma ** 2 * S[i] ** 2 * gamma
                + r * S[i] * delta
                - r * V_old[i]
            )
            V[i] = max(continuation_value, payoff[i])

        # boundary conditions for the put option
        V[0] = K
        V[N] = 0.0
    return V

if __name__ == '__main__':
    grid = Grid(N=200, M=40000)
    values = explicit_american_put(grid)
    print('American put option price near S = K:')
    idx = np.argmin(np.abs(grid.S - grid.K))
    print(values[idx])