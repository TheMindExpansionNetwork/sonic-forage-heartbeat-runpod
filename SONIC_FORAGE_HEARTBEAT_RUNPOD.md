# sonic-forage-heartbeat-runpod

A **RunPod-ready** edition of [daydreamlive/DEMON](https://github.com/daydreamlive/DEMON) — realtime motion-to-music in the browser, with the fixes and lessons from actually getting it working on GPU pods.

**Fork:** https://github.com/TheMindExpansionNetwork/sonic-forage-heartbeat-runpod

## Why this exists

Deploying the realtime motion-graph web demo on RunPod hits problems you don’t see on localhost:

- WebSockets must use the **pod proxy** (`wss://<pod>-1318.proxy.runpod.net`), not `127.0.0.1`
- **~10 GB** ACE-Step checkpoints must be downloaded once per volume
- Headless pods have **no PortAudio** — server audio output isn’t needed (the browser plays audio)

This fork bundles code fixes + scripts + a full tutorial.

## Quick start

```bash
uv sync
./demos/realtime_motion_graph_web/runpod/download_models.sh
./demos/realtime_motion_graph_web/runpod/start.sh
```

Open `https://<RUNPOD_POD_ID>-6660.proxy.runpod.net/` · expose TCP **6660** and **1318**.

## Documentation

| Path | Description |
|------|-------------|
| [demos/realtime_motion_graph_web/runpod/SONIC_FORAGE.md](demos/realtime_motion_graph_web/runpod/SONIC_FORAGE.md) | Lore + what we’re building |
| [demos/realtime_motion_graph_web/runpod/LESSON.md](demos/realtime_motion_graph_web/runpod/LESSON.md) | Step-by-step lesson |
| [demos/realtime_motion_graph_web/runpod/README.md](demos/realtime_motion_graph_web/runpod/README.md) | Ops reference |
| [demos/realtime_motion_graph_web/runpod/FIXES.md](demos/realtime_motion_graph_web/runpod/FIXES.md) | Change log |

## Upstream

Based on **daydreamlive/DEMON**. Pull upstream for core engine changes; this fork focuses on RunPod deployment ergonomics.

---

*Sonic Forage · Heartbeat · RunPod — forage the signal, feel the pulse.*