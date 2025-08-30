from __future__ import annotations
import numpy as np
from typing import List

class BaseGenerator:
    """
    Interface for realtime generators with N normalized params in [0,1].

    Subclasses should define:
      - param_labels: list[str]   (length = number of params)
      - set_params(norm_params: list[float])  -> map [0,1] to semantic values + store state
      - generate(frames:int, sr:int) -> np.ndarray float32, mono
      - formatted_readouts() -> list[str]   (one per param, human-readable)
    """
    param_labels: List[str] = []

    def __init__(self, init_norm_params: List[float] | None = None) -> None:
        n = len(self.param_labels)
        self.norm_params: List[float] = list(init_norm_params) if init_norm_params else [0.5] * n

    def num_params(self) -> int:
        return len(self.param_labels)

    def set_params(self, norm_params: List[float]) -> None:
        self.norm_params = [float(np.clip(v, 0.0, 1.0)) for v in norm_params]

    def generate(self, frames: int, sr: int):
        raise NotImplementedError

    def formatted_readouts(self) -> List[str]:
        return [f"{label}: {v:.3f}" for label, v in zip(self.param_labels, self.norm_params)]
