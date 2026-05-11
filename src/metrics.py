import numpy as np

def max_absolute_error(values: np.ndarray, reference_values: np.ndarray) -> float:
    return float(np.max(np.abs(values - reference_values)))

def mean_absolute_error(values: np.ndarray, reference_values: np.ndarray) -> float:
    return float(np.mean(np.abs(values - reference_values)))

def root_mean_squared_error(values: np.ndarray, reference_values: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values - reference_values) ** 2)))

def pointwise_error(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    return np.abs(values - reference_values)