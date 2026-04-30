# Demo videos

The project page expects three short clips here. Each plays autoplay-muted-loop in a card, so they should be:

- **No audio** (it's muted anyway, just don't waste bytes on a track)
- **Short** (15-30s loops)
- **16:9** (cards have `aspect-ratio: 16/9`)
- **MP4 / H.264** (broadest browser support)

| Filename | What it should show |
|---|---|
| `midi_demo.mp4` | Hardware MIDI controller driving DEMON live: knob movement intercut with the realtime motion-graph demo UI / waveform. |
| `web_demo.mp4` | Screen capture of the browser web app: HUD, knobs, audio-reactive video background. |
| `vst_demo.mp4` | The VST plugin running inside a DAW. Until the VST is ready, this card has a "Coming soon" link &mdash; can either drop a teaser cap of an early build, or remove the third card from `index.html`. |

Compress aggressively (e.g., `ffmpeg -crf 28`) &mdash; project page videos shouldn't be 100MB each.
