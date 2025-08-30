from __future__ import annotations
import numpy as np
from typing import List
from .base import BaseGenerator
from ..utils import exp_map01

class SineGenerator(BaseGenerator):
    """
    Sine oscillator.
      p[0] -> frequency in [20, 2000] Hz (exp)
      p[1] -> amplitude in [0, 1] (linear)
    """
    param_labels = ["Freq (Hz)", "Amp"]

    def __init__(self, f_lo: float = 20.0, f_hi: float = 2000.0,
                 init_norm_params: List[float] | None = None) -> None:
        super().__init__(init_norm_params=init_norm_params or [0.5, 0.2])
        self.f_lo = max(1e-3, float(f_lo))
        self.f_hi = max(self.f_lo, float(f_hi))
        self.phase = 0.0
        self.twopi = 2.0 * np.pi
        # semantic defaults (safe pre-start)
        self.freq = exp_map01(self.norm_params[0], self.f_lo, self.f_hi)
        self.amp  = self.norm_params[1]

    def set_params(self, norm_params):
        super().set_params(norm_params)
        self.freq = exp_map01(self.norm_params[0], self.f_lo, self.f_hi)
        self.amp  = self.norm_params[1]

    def generate(self, frames: int, sr: int):
        if self.amp <= 0.0 or self.freq <= 0.0:
            return np.zeros(frames, dtype=np.float32)
        inc = self.twopi * self.freq / sr
        idx = self.phase + inc * np.arange(frames, dtype=np.float64)
        y = self.amp * np.sin(idx)
        self.phase = (idx[-1] + inc) % self.twopi
        return y.astype(np.float32)

    def formatted_readouts(self):
        return [f"{self.param_labels[0]}: {self.freq:7.2f} Hz",
                f"{self.param_labels[1]}: {self.amp:.3f}"]
