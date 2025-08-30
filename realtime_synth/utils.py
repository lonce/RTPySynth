import numpy as np

def exp_map01(x: float, lo: float, hi: float) -> float:
    """Exponential map from [0,1] -> [lo,hi] (requires lo>0)."""
    x = float(np.clip(x, 0.0, 1.0))
    return lo * (hi / lo) ** x
