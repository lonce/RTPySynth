from __future__ import annotations
import numpy as np
from typing import List
from .base import BaseGenerator
from ..utils import exp_map01

class NoisyLPGenerator(BaseGenerator):
    """
    White noise -> one-pole lowpass.
      p[0] -> cutoff in [100, 8000] Hz (exp)
      p[1] -> level in [0,1]
    """
    param_labels = ["Cutoff (Hz)", "Level"]

    def __init__(self, c_lo: float = 100.0, c_hi: float = 8000.0, seed: int = 0,
                 init_norm_params: List[float] | None = None) -> None:
        super().__init__(init_norm_params=init_norm_params or [0.5, 0.2])
        self.c_lo = max(1e-3, float(c_lo))
        self.c_hi = max(self.c_lo, float(c_hi))
        self.prev = 0.0
        self.rng = np.random.default_rng(seed)
        # semantic defaults
        self.cut   = exp_map01(self.norm_params[0], self.c_lo, self.c_hi)
        self.level = self.norm_params[1]

    def set_params(self, norm_params):
        super().set_params(norm_params)
        self.cut   = exp_map01(self.norm_params[0], self.c_lo, self.c_hi)
        self.level = self.norm_params[1]

    def generate(self, frames: int, sr: int):
        if self.level <= 0.0:
            return np.zeros(frames, dtype=np.float32)
        a = 1.0 - np.exp(-2.0 * np.pi * self.cut / sr)
        x = self.rng.standard_normal(frames).astype(np.float64) * 0.5
        y = np.empty_like(x)
        y_prev = float(self.prev)
        for i in range(frames):
            y_prev = y_prev + a * (x[i] - y_prev)
            y[i] = y_prev
        self.prev = y_prev
        return (self.level * y).astype(np.float32)

    def formatted_readouts(self):
        return [f"{self.param_labels[0]}: {self.cut:7.1f} Hz",
                f"{self.param_labels[1]}: {self.level:.3f}"]
