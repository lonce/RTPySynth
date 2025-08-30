from __future__ import annotations
import numpy as np
import sounddevice as sd
import threading
from typing import List
from .generators.base import BaseGenerator

class RealtimeSynth:
    """
    Realtime audio engine using sounddevice (PortAudio).
    Accepts a class-based generator with N normalized params in [0,1].
    """
    def __init__(self, generator: BaseGenerator, samplerate: int = 48000,
                 blocksize: int = 128, channels: int = 1) -> None:
        self.sr = int(samplerate)
        self.blocksize = int(blocksize)
        self.channels = int(channels)
        self._lock = threading.Lock()
        self.gen = generator
        self._norm: List[float] = list(self.gen.norm_params)  # normalized params

        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.channels,
            dtype='float32',
            blocksize=self.blocksize,
            latency='low',
            callback=self._callback,
        )
        self._running = False

    # -------- control --------
    def set_params(self, norm_list: List[float]) -> None:
        with self._lock:
            n = self.gen.num_params()
            vals = (list(norm_list) + [0.5]*n)[:n]
            self._norm = [float(np.clip(v, 0.0, 1.0)) for v in vals]

    def set_generator(self, generator: BaseGenerator) -> None:
        with self._lock:
            self.gen = generator
            self._norm = list(self.gen.norm_params)

    def start(self) -> None:
        if not self._running:
            self.stream.start()
            self._running = True

    def stop(self) -> None:
        if self._running:
            self.stream.stop()
            self._running = False

    def close(self) -> None:
        try:
            self.stream.close()
        except Exception:
            pass

    # -------- audio callback --------
    def _callback(self, outdata, frames, time, status):
        if status:
            # print("Audio status:", status)
            pass
        with self._lock:
            gen = self.gen
            norm = list(self._norm)

        gen.set_params(norm)
        mono = gen.generate(frames, self.sr)

        if self.channels == 1:
            outdata[:, 0] = mono
        else:
            outdata[:] = np.repeat(mono[:, None], self.channels, axis=1)
