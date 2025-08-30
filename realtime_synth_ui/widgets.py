from __future__ import annotations
from typing import Dict, Callable
from ipywidgets import FloatSlider, ToggleButton, VBox, HBox, Label, Layout, Dropdown, HTML
from IPython.display import display
from realtime_synth.engine import RealtimeSynth
from realtime_synth.generators.base import BaseGenerator

def build_synth_ui(generators: Dict[str, Callable[[], BaseGenerator]],
                   samplerate: int = 48000, blocksize: int = 128, channels: int = 1):
    """
    Create and display a UI for a realtime synth with swappable generators.
    Returns (synth_instance, ui_container).
    """
    # initial synth
    first_key = next(iter(generators))
    current_gen = generators[first_key]()
    synth = RealtimeSynth(generator=current_gen, samplerate=samplerate, blocksize=blocksize, channels=channels)

    sliders_box = VBox()
    readouts_box = HBox()
    play_toggle = ToggleButton(value=False, description='▶️ Play', layout=Layout(width='120px'))
    gen_picker = Dropdown(options=list(generators.keys()), value=first_key, description="Generator:")
    status_line = HTML(value="")

    def refresh_readouts():
        ro = synth.gen.formatted_readouts()
        labels = list(readouts_box.children)
        for i, lbl in enumerate(labels):
            lbl.value = ro[i] if i < len(ro) else ""

    def build_param_ui():
        sliders = []
        readout_labels = []
        n = synth.gen.num_params()
        for i in range(n):
            lab = synth.gen.param_labels[i] if i < len(synth.gen.param_labels) else f"Param {i+1}"
            init = synth.gen.norm_params[i] if i < len(synth.gen.norm_params) else 0.5
            s = FloatSlider(value=float(init), min=0.0, max=1.0, step=0.001,
                            description=lab, layout=Layout(width='420px'))
            def on_change(change):
                if change['name'] == 'value':
                    values = [w.value for w in sliders_box.children]
                    synth.set_params(values)
                    refresh_readouts()
            s.observe(on_change, names='value')
            sliders.append(s)
            readout_labels.append(Label("", layout=Layout(width='220px')))
        sliders_box.children = sliders
        readouts_box.children = readout_labels
        synth.set_params([w.value for w in sliders])
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
            synth.set_generator(generators[change['new']]())
            build_param_ui()

    play_toggle.observe(on_play, names='value')
    gen_picker.observe(on_pick, names='value')

    controls = VBox([
        gen_picker,
        sliders_box,
        HBox([play_toggle]),
        HBox([Label("Mapped:"), readouts_box]),
        HTML("<small>Sliders are normalized [0,1]; each generator defines its own mapping & state.<br>"
             "Tip: reduce blocksize for lower latency (risking underruns); increase if you hear clicks.</small>")
    ])

    build_param_ui()
    display(controls, status_line)
    return synth, controls
