from __future__ import annotations
from typing import Dict, Callable, Optional, Union, List
import inspect

from ipywidgets import FloatSlider, ToggleButton, VBox, HBox, Label, Layout, Dropdown, HTML
from IPython.display import display

from importlib.metadata import entry_points

from realtime_synth.engine import RealtimeSynth
from realtime_synth.generators.base import BaseGenerator

# A generator provider can be either a class (subclass of BaseGenerator)
# or a zero-argument factory function that returns an instance.
GeneratorProvider = Union[Callable[[], BaseGenerator], type]


def _discover_generators() -> Dict[str, GeneratorProvider]:
    """
    Discover installed generator plugins via entry points.
    Accepts both classes and zero-arg factories.
    """
    gens: Dict[str, GeneratorProvider] = {}
    try:
        eps = entry_points(group="realtime_synth.generators")
    except TypeError:
        # Older Python fallback
        eps = entry_points().get("realtime_synth.generators", [])  # type: ignore[attr-defined]
    for ep in eps:
        obj = ep.load()
        # Either a class or a zero-arg callable is fine
        if inspect.isclass(obj):
            gens[ep.name] = obj
        else:
            # Assume zero-arg factory
            gens[ep.name] = obj  # type: ignore[assignment]
    return gens


def _instantiate(provider: GeneratorProvider) -> BaseGenerator:
    """Create a generator instance from a class or zero-arg factory."""
    if inspect.isclass(provider):
        return provider()  # type: ignore[call-arg]
    return provider()      # type: ignore[call-arg]


def build_synth_ui(
    generators: Optional[Dict[str, GeneratorProvider]] = None,
    samplerate: int = 48000,
    blocksize: int = 128,
    channels: int = 1,
):
    """
    Create and display a UI for a realtime synth with swappable generators.
    If `generators` is None, auto-discovers installed plugins via entry points,
    falling back to the built-in voices shipped with rtpysynth.

    Returns (synth_instance, ui_container).
    """
    # Auto-discover if not supplied
    if generators is None:
        generators = _discover_generators()
        if not generators:
            # Fallback to built-ins
            from realtime_synth.generators import SineGenerator, NoisyLPGenerator, MyGeneratorTemplate
            generators = {
                "Sine": SineGenerator,
                "NoisyLP": NoisyLPGenerator,  # keep key without space (matches entry point name)
                "Template": MyGeneratorTemplate,
            }

    # Instantiate the first available generator
    first_key = next(iter(generators))
    current_gen = _instantiate(generators[first_key])
    synth = RealtimeSynth(generator=current_gen, samplerate=samplerate, blocksize=blocksize, channels=channels)

    # --- UI widgets ---
    sliders_box = VBox()
    readouts_box = HBox()
    play_toggle = ToggleButton(value=False, description='▶️ Play', layout=Layout(width='120px'))
    gen_picker = Dropdown(options=list(generators.keys()), value=first_key, description="Generator:")
    status_line = HTML(value="")

    def refresh_readouts():
        ro = synth.gen.formatted_readouts()
        labels: List[Label] = list(readouts_box.children)  # type: ignore[assignment]
        for i, lbl in enumerate(labels):
            lbl.value = ro[i] if i < len(ro) else ""

    def build_param_ui():
        # Build sliders and readouts dynamically from current generator
        sliders = []
        readout_labels = []
        n = synth.gen.num_params()
        for i in range(n):
            lab = synth.gen.param_labels[i] if i < len(synth.gen.param_labels) else f"Param {i+1}"
            init = synth.gen.norm_params[i] if i < len(synth.gen.norm_params) else 0.5
            s = FloatSlider(
                value=float(init), min=0.0, max=1.0, step=0.001,
                description=lab, layout=Layout(width='420px')
            )
            def on_change(change):
                if change['name'] == 'value':
                    values = [w.value for w in sliders_box.children]  # read all current slider values
                    synth.set_params(values)
                    refresh_readouts()
            s.observe(on_change, names='value')
            sliders.append(s)
            readout_labels.append(Label("", layout=Layout(width='220px')))

        sliders_box.children = sliders
        readouts_box.children = readout_labels
        synth.set_params([w.value for w in sliders])  # push initial values
        refresh_readouts()

    def on_play(change):
        if change['name'] == 'value':
            if change['new']:
                play_toggle.description = '⏹ Stop'
                synth.start()
                status_line.value = "<i>Audio running…</i>"
            else:
                play_toggle.description = '▶️ Play'
                synth.stop()
                status_line.value = "<i>Audio stopped.</i>"

    def on_pick(change):
        if change['name'] == 'value':
            key = change['new']                    # <-- use 'new', not 'value'
            provider = generators[key]
            synth.set_generator(_instantiate(provider))
            build_param_ui()                       # rebuild sliders/readouts for the new gen

    play_toggle.observe(on_play, names='value')
    gen_picker.observe(on_pick, names='value')

    controls = VBox([
        gen_picker,
        sliders_box,
        HBox([play_toggle]),
        HBox([Label("Mapped:"), readouts_box]),
        HTML(
            "<small>Sliders are normalized [0,1]; each generator defines its own mapping &amp; state.<br>"
            "Tip: decrease blocksize for lower latency (higher xrun risk); increase if you hear clicks.</small>"
        )
    ])

    build_param_ui()
    display(controls, status_line)
    return synth, controls
