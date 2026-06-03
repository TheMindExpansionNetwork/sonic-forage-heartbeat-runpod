# Sonic Forage · Jam Lab

Experimentation folder for running **DEMON** (realtime motion → music) and **AVTR-1** (realtime talking-head video) side by side on RunPod.

## Ports

| Port | Service | URL pattern (RunPod) |
|------|---------|----------------------|
| **6660** | DEMON motion-graph web UI | `https://<POD_ID>-6660.proxy.runpod.net/` |
| **1318** | DEMON engine (HTTP + WebSocket) | `wss://<POD_ID>-1318.proxy.runpod.net/` ← **TouchDesigner** |
| **7860** | AVTR local stream (WebRTC UI) | `https://<POD_ID>-7860.proxy.runpod.net/` |
| **8000** | AVTR renderer API (internal) | Used by streamer only |

**TouchDesigner:** see [TOUCHDESIGNER.md](./TOUCHDESIGNER.md) · `demonTD.tox` in this folder · fork [TheMindExpansionNetwork/demonTD](https://github.com/TheMindExpansionNetwork/demonTD)

Expose **6660**, **1318**, and **7860** in your RunPod template.

## Quick start

```bash
# From DEMON repo root
./demos/sonic_forage_jam/start_demon.sh      # music / motion graph
./demos/sonic_forage_jam/start_avtr.sh       # avatar video (7860)
./demos/sonic_forage_jam/stop_jam.sh         # stop both
```

**First time on a new GPU** (TRT engines are arch-specific), rebuild AVTR engines:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
cd /workspace/avtr-1
export AVTR1_LOCAL_STORAGE=/workspace/avtr-1/artifacts
pixi run -e renderer python scripts/build_avtr1_engines.py   # ~10–30 min
pixi run -e renderer python scripts/build_renderer_engines.py
pixi run -e renderer python scripts/build_hubert_engine.py
# or: pixi run -e renderer python scripts/build_engines.py
```

AVTR lives in **`/workspace/avtr-1`** (sibling of DEMON). Engines/weights: `$AVTR1_LOCAL_STORAGE` (default `artifacts/` inside that repo).

## Audio: full track vs vocals

**DEMON** can drive generation from:

- **Full mix** — default fixture `source.wav`
- **Vocals only** — stem mode `vocals` (MelBand-RoFormer, cached under models dir)
- **Instruments only** — stem mode `instruments`

In the DEMON HUD (Advanced / source controls), switch timbre or structure refs and stem-backed fixtures after upload or library pick.

**AVTR-1** expects **16 kHz mono PCM** on two logical streams:

- **speech** — what the avatar lip-syncs
- **listen** — peer audio for “active listening” (can be silence)

The stock **7860** UI uses your **microphone** + OpenAI Realtime or Cartesia (API keys in the browser). To lip-sync **DEMON output**:

1. **Jam mode (simple):** Play DEMON in the browser and let the mic pick it up (imperfect but fast).
2. **Offline:** `pixi run generate_offline --speech your_vocals.wav` in `avtr-1` (not live).
3. **Future:** `jam_audio_bridge.py` (see below) — resample DEMON stream to 16 kHz and POST to renderer `/stream` (experimental).

## Vocals export helper

Resample a WAV/OGG to AVTR-friendly 16 kHz mono:

```bash
cd /workspace/DEMON
uv run python demos/sonic_forage_jam/prepare_avtr_audio.py \
  --input path/to/track.wav \
  --output /tmp/vocals_16k.wav \
  --stem vocals   # or omit for full mix
```

## Logs

| Log | Contents |
|-----|----------|
| `/tmp/avtr_jam.log` | AVTR orchestrator (renderer + 7860) |
| `/tmp/demon_jam.log` | DEMON launcher |
| `/tmp/xl_demo_compile.log` | DEMON XL experiments |

## GPU note

Both stacks want the GPU. For a stable jam session:

- Run **one** heavy workload at a time, or
- DEMON with `--accel compile` / `eager` while AVTR holds the GPU for video.

## Links

- [DEMON runpod lesson](../realtime_motion_graph_web/runpod/LESSON.md)
- [AVTR-1 README](/workspace/avtr-1/README.md)
- [SONIC_FORAGE lore](../realtime_motion_graph_web/runpod/SONIC_FORAGE.md)