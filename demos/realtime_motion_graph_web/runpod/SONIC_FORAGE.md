# Sonic Forage · Heartbeat · RunPod Edition

> *Real-time motion becomes music. The GPU is the amplifier. The browser is the stage.*

This fork exists because getting **DEMON** (Daydream’s realtime motion-graph web demo) alive on **RunPod** is a rite of passage — WebSockets, proxy URLs, ten gigabytes of checkpoints, and a headless box that has never heard of PortAudio. We documented every trap so the next forager doesn’t bleed the same way.

## What we’re building (the lore, briefly)

**Sonic Forage** is the hunt for live sound: you don’t “render a track and export.” You **forage** in the signal — nudging prompts, LoRAs, and motion while the engine keeps generating. **Heartbeat** is the pulse of that loop: slices arriving over the wire, the HUD’s playhead, the kick driving the shader. **RunPod** is the portable forge: spin up an A100, open two TCP ports, and perform.

Under the hood it’s still ACE-Step + TensorRT/eager + the DEMON web UI. The edition you’re holding adds the **RunPod survival kit** in this folder.

## Who this is for

- Artists and devs deploying on **RunPod** (or any `*.proxy.runpod.net` setup)
- Anyone who saw “WebSocket failed” and questioned their life choices
- TheMindExpansionNetwork / collaborators shipping **live** AI music experiences

## Start here

| Doc | You’ll learn |
|-----|----------------|
| [LESSON.md](./LESSON.md) | Step-by-step tutorial (lesson format) |
| [README.md](./README.md) | Quick reference + troubleshooting |
| [FIXES.md](./FIXES.md) | Every patch we made and why |
| [GITHUB.md](./GITHUB.md) | Auth + fork workflow |

## One-command vibe check

```bash
./demos/realtime_motion_graph_web/runpod/download_models.sh   # once per volume
./demos/realtime_motion_graph_web/runpod/start.sh
# open https://<RUNPOD_POD_ID>-6660.proxy.runpod.net/
```

When **Play** breathes and the HUD moves — that’s the heartbeat. Welcome to the forage.