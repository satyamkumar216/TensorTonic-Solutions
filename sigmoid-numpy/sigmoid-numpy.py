import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x_arr = np.asarray(x, dtype=float)
    result = 1 / (1 + np.exp(-x_arr))
    
    # Return a float if input was a scalar, otherwise return a NumPy array
    return float(result) if np.isscalar(x) else result
