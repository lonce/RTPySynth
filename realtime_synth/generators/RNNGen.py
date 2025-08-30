from __future__ import annotations
import numpy as np
from typing import List
from .base import BaseGenerator
from ..utils import exp_map01


import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import deque
from ../../../audioDataLoader.audio_dataset import latents_to_audio_simple, efficient_codes_to_latents, preprocess_latents_for_RNN



class RNNGen(BaseGenerator):
    """
    Start here for new DSP ideas. Adjust labels/mapping/state as needed.
      Example mapping:
        p[0] -> something exponential in [lo1, hi1]
        p[1] -> something linear in [lo2, hi2]
      Add/remove params by editing param_labels and init_norm_params.
    """
    param_labels = ["Param 1", "Param2"]  # add more names as needed

    def __init__(self, lo2=0.0, hi2=1.0, lo1=0.1, hi1=10.0, 
                 init_norm_params: List[float] | None = None) -> None:
        super().__init__(init_norm_params=init_norm_params or [0.5, 0.5])

        # store ranges
        self.lo1, self.hi1 = float(lo1), float(hi1)
        self.lo2, self.hi2 = float(lo2), float(hi2)
        # persistent DSP state here
        self._state_var = 0.0

        # semantic initialization
        self.val1 = exp_map01(self.norm_params[0], self.lo1, self.hi1)
        self.val2 = self.lo2 + self.norm_params[1]*(self.hi2 - self.lo2)


    def set_params(self, norm_params: List[float]) -> None:
        super().set_params(norm_params)
        self.val1 = exp_map01(self.norm_params[0], self.lo1, self.hi1)
        self.val2 = self.lo2 + self.norm_params[1]*(self.hi2 - self.lo2)

    def generate(self, frames: int, sr: int):
        # Replace with real DSP; silence as a stub.
        return np.zeros(frames, dtype=np.float32)

    def formatted_readouts(self):
        return [f"{self.param_labels[0]}: {self.val1:.3f}",
                f"{self.param_labels[1]}: {self.val2:.3f}"]
