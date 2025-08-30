# realtime-synth

Tiny real-time audio synth engine in Python with a low-latency callback backend (PortAudio via `sounddevice`), a pluggable **class-based generator** API, and an auto-building **ipywidgets UI**.  
Sliders are normalized `[0,1]`; each generator maps them to semantic values (Hz, gain, cutoff, etc.) and maintains its own DSP state.

## Features
- 🔊 **Real-time**: callback-driven `sounddevice.OutputStream` with small block sizes.
- 🧩 **Pluggable DSP**: write generators as small classes with `param_labels`, `set_params()`, and `generate()`.
- 🎚️ **Normalized controls**: UI supplies `[0,1]` params; generators decide the mapping.
- 🔄 **Hot-swap voices**: pick between included generators or bring your own in a single cell.

---

## Install

Install directly from GitHub (no PyPI needed):

```bash
# Latest from main (includes optional UI extras)
pip install "rtpysynth[ui] @ git+https://github.com/lonce/RTPySynth@main"

# Or pin a tagged release (recommended)
pip install "realtime-synth[ui] @ git+https://github.com/lonce/RTPySynth@v0.1.0"
```

**System deps**
- Linux: PortAudio runtime may be required by `sounddevice`:
  ```bash
  sudo apt-get update && sudo apt-get install -y libportaudio2
  ```
- Jupyter: ipywidgets 8+ (installed via the `[ui]` extra). In classic Notebook, if widgets don’t appear:
  ```bash
  jupyter nbextension enable --py widgetsnbextension
  ```

---

## Quickstart (Jupyter)

```python
from realtime_synth_ui import build_synth_ui

# Launch UI and audio engine. If no dict is passed, built-ins are used.
synth, ui = build_synth_ui(samplerate=48000, blocksize=128, channels=1)
```

- Press **Play** to start/stop audio.
- Use the dropdown to switch voices.
- Lower `blocksize` (64 / 32) for lower latency if your machine is stable; raise it if you hear underruns/clicks.

---

## Included Generators

- **Sine** — `["Freq (Hz)", "Amp"]`
- **Noisy LP** — white noise through a one-pole low-pass: `["Cutoff (Hz)", "Level"]`
- **Template** — a skeleton to copy for your own DSP

---

## Add Your Own Generator (no packaging required)

Just define a subclass of `BaseGenerator` anywhere (notebook cell or your own module) and pass it to the UI.  
**No `pyproject.toml` or publishing needed** for local use.

```python
import numpy as np
from realtime_synth.generators.base import BaseGenerator
from realtime_synth.utils import exp_map01
from realtime_synth_ui import build_synth_ui

class MyLocalSine(BaseGenerator):
    # 2 normalized params in [0,1]
    param_labels = ["Freq (Hz)", "Amp"]

    def __init__(self, init_norm_params=None):
        super().__init__(init_norm_params or [0.5, 0.2])  # defaults
        self.twopi = 2*np.pi
        self.phase = 0.0
        self.set_params(self.norm_params)  # initialize semantic values

    def set_params(self, norm_params):
        super().set_params(norm_params)
        # Map [0,1] → semantic values
        self.freq = float(exp_map01(self.norm_params[0], 20.0, 2000.0))  # exponential Hz
        self.amp  = float(self.norm_params[1])                           # linear gain 0..1

    def generate(self, frames, sr):
        if self.amp <= 0.0 or self.freq <= 0.0:
            return np.zeros(frames, dtype=np.float32)
        inc = self.twopi * self.freq / sr
        n = np.arange(frames, dtype=np.float64)
        y = self.amp * np.sin(self.phase + inc*n)
        self.phase = (self.phase + inc*frames) % self.twopi
        return y.astype(np.float32)

    def formatted_readouts(self):
        # Optional: pretty labels shown next to sliders
        return [f"{self.param_labels[0]}: {self.freq:7.2f} Hz",
                f"{self.param_labels[1]}: {self.amp:.3f}"]

# Run the UI with your custom voice (you can include built-ins too)
GENS = {
    "MyLocalSine": MyLocalSine,
    # "Sine": SineGenerator,            # you can import these if you like
    # "Noisy LP": NoisyLPGenerator,
}

synth, ui = build_synth_ui(GENS, samplerate=48000, blocksize=128, channels=1)
```

**What the interface expects from a generator:**
- `param_labels`: list of human-readable names (length = number of parameters).
- `__init__(...)`: set any state; call `self.set_params(self.norm_params)` to initialize semantic values.
- `set_params(norm_params: list[float])`: receive normalized `[0,1]` values, clip as needed, map to semantic.
- `generate(frames: int, sr: int) -> np.ndarray[float32]`: return **mono** audio for the next block.
- `formatted_readouts() -> list[str]` (optional): strings to display next to sliders.

> Tip: keep any persistent DSP state (phases, filter memory, envelopes, RNG) as instance attributes.  
> The engine calls `set_params()` once per audio block before `generate()`.

---

## Programmatic Control (without UI)

```python
from realtime_synth.engine import RealtimeSynth
from realtime_synth.generators.sine import SineGenerator

gen = SineGenerator()
synth = RealtimeSynth(generator=gen, samplerate=48000, blocksize=128, channels=1)

# Set normalized params [0,1]
synth.set_params([0.6, 0.25])
synth.start()
# ... do stuff ...
synth.stop()
synth.close()
```

---

## Development

Editable install for hacking:

```bash
git clone https://github.com/lonce/RTPySynth.git
cd RTPySynth
pip install -e ".[ui]"
```

---

## Troubleshooting

- **Clicks/underruns**: increase `blocksize` (e.g., 256 or 512), close CPU-heavy apps.
- **No audio / device errors**: ensure a working default output device; on Linux install `libportaudio2`.
- **Widgets not showing**: confirm ipywidgets 8+, enable notebook extension in classic Jupyter.

---

## License

MIT © lonce
