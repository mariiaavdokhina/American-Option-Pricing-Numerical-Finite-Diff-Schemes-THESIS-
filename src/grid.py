from dataclasses import dataclass
import numpy as np

@dataclass
# grid parameters
class Grid:
    Smax: float = 300.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2
    N: int = 200
    M: int = 1000

    def __post_init__(self) -> None:
        if self.Smax <= 0:
            raise ValueError('Smax must be > 0')
        if self.K <= 0:
            raise ValueError('K must be > 0')
        if self.T <= 0:
            raise ValueError('T must be > 0')
        if self.sigma <= 0:
            raise ValueError('sigma must be > 0')
        if self.N < 3:
            raise ValueError('N must be >= 3')
        if self.M < 1:
            raise ValueError('M must be >= 1')

        # calculating grid params
        self.dS = self.Smax / self.N
        self.dt = self.T / self.M

        # creating the actual grid
        self.S = np.linspace(0.0, self.Smax, self.N + 1)
        self.t = np.linspace(0.0, self.T, self.M + 1)