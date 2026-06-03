# RunPod setup · Sonic Forage · Heartbeat edition

**Fork:** [sonic-forage-heartbeat-runpod](https://github.com/TheMindExpansionNetwork/sonic-forage-heartbeat-runpod)

| Start here | |
|------------|--|
| New to this? | [SONIC_FORAGE.md](./SONIC_FORAGE.md) — lore + vibe |
| Mythos | [LORE.md](./LORE.md) — short story |
| Learning path | [LESSON.md](./LESSON.md) — step-by-step tutorial |
| Ops / fixes | This file + [FIXES.md](./FIXES.md) |
| **Reboot runbook** | [agent-world/docs/RUNBOOK.md](./agent-world/docs/RUNBOOK.md) |

This folder documents RunPod-specific fixes, scripts, and templates for the realtime motion-graph web demo on a GPU pod.

### Agent World (network volume)

On the persistent volume at `/workspace/agent-world-workspace` — same content is in git under **`runpod/agent-world/`**. After a pod reboot:

```bash
/workspace/agent-world-workspace/POD_BOOTSTRAP.sh
/workspace/agent-world-workspace/scripts/start_demon_web_xl.sh
```

XL + `compile`, logs in `agent-world-workspace/logs/`. See **RUNBOOK** for meta-tensor and Qwen repair.

## Quick start

1. In the RunPod template, expose **TCP ports `6660` and `1318`**.
2. **First time on a fresh pod/volume** — download models (~10 GB, one-time):

```bash
./demos/realtime_motion_graph_web/runpod/download_models.sh
```

3. From the DEMON repo root:

```bash
./demos/realtime_motion_graph_web/runpod/start.sh
```

4. Open the URL printed in the log (usually `https://<RUNPOD_POD_ID>-6660.proxy.runpod.net/`).
5. Click **Play** and wait for the first cold start (~15s+ while the GPU pipeline warms up).

Stop:

```bash
./demos/realtime_motion_graph_web/runpod/stop.sh
```

## URLs on this pod

RunPod sets `RUNPOD_POD_ID` automatically. Proxy URLs follow:

| Service | Port | URL pattern |
|---------|------|-------------|
| Web UI | 6660 | `https://<POD_ID>-6660.proxy.runpod.net/` |
| Engine (HTTP + WebSocket) | 1318 | `https://<POD_ID>-1318.proxy.runpod.net/` |

Example for pod `h9ay8scybro9x3`:

- UI: https://h9ay8scybro9x3-6660.proxy.runpod.net/
- Engine: https://h9ay8scybro9x3-1318.proxy.runpod.net/

## What was broken (socket / WebSocket errors)

When you open the UI through the RunPod proxy, the browser runs JavaScript on **your machine**, not inside the pod. The app was trying to connect to:

- `ws://127.0.0.1:1318/`

That points at **your laptop’s** localhost, not the GPU server — so Play failed with WebSocket / socket errors.

HTTP `/api/*` calls worked when proxied through Next.js, but **WebSocket bypasses Next.js** and must use the public engine URL.

## What we fixed

See [FIXES.md](./FIXES.md) for file-by-file detail. Summary:

1. **Auto WebSocket URL on RunPod** (`web/engine/podUrl.ts`) — derives `wss://<pod>-1318.proxy.runpod.net/` from the page hostname.
2. **Launcher RunPod mode** (`run.py`) — binds `0.0.0.0`, sets `NEXT_PUBLIC_POD_BASE_URL` to the `1318` proxy, starts Next on `0.0.0.0`.
3. **Next.js dev origins** (`web/next.config.ts`) — allows `*.proxy.runpod.net` and your pod’s `6660` host for HMR.
4. **Env template** (`env.example`) — documents `NEXT_PUBLIC_POD_BASE_URL` when you run `npm run dev` without the launcher.

## Files in this folder

| File | Purpose |
|------|---------|
| [README.md](./README.md) | This guide |
| [FIXES.md](./FIXES.md) | Every code change, why, and how to edit |
| [env.example](./env.example) | Copy/adapt for `web/.env.development` |
| [download_models.sh](./download_models.sh) | **Required once** — fetch ACE-Step checkpoints |
| [start.sh](./start.sh) | Start backend + web (recommended) |
| [stop.sh](./stop.sh) | Stop both processes |
| [setup_github.sh](./setup_github.sh) | One-time `gh` + git auth (paste PAT) |
| [GITHUB.md](./GITHUB.md) | MCP vs CLI, fork/push examples |

## Customization

### New RunPod pod (different `RUNPOD_POD_ID`)

Usually **no code edits** — `RUNPOD_POD_ID` is set by RunPod and `run.py` + `podUrl.ts` pick it up.

If you run **only** `npm run dev` (no launcher), update `web/.env.development` from [env.example](./env.example).

### Accelerator

Default in `start.sh` is `eager` (faster to start, no TensorRT). For TRT:

```bash
ACCEL=tensorrt ./demos/realtime_motion_graph_web/runpod/start.sh
```

### Manual WebSocket override

Add to the UI URL:

```
?ws=wss://<POD_ID>-1318.proxy.runpod.net/
```

### Ports

Edit `start.sh` variables `BACKEND_PORT` / `WEB_PORT` if your template uses different exposed ports (update RunPod template too).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| Page loads but **Play spins / never starts** | Checkpoints not downloaded | Run [download_models.sh](./download_models.sh); wait for `OK`, then refresh and Play |
| Play loads ~1 min then disconnects | `PortAudio library not found` in backend log | Fixed in `audio_engine.py` — restart demo after pull |
| WebSocket / socket error on Play | Engine URL still `127.0.0.1` | Use RunPod proxy URL; see [FIXES.md](./FIXES.md); restart via `start.sh` |
| `Address already in use` on 1318 | Second backend already running | Run `stop.sh`, then `start.sh` once |
| UI loads but API 502 / empty | Port 1318 not exposed on pod | Add `1318` to RunPod TCP ports |
| Next “Blocked cross-origin” in dev | HMR from proxy host | Ensure `allowedDevOrigins` in `next.config.ts` includes your pod host |
| `--help` hangs ~2+ minutes | Python import loads GPU stack | Use `run.py` / `start.sh`; avoid `python -m demos.realtime_motion_graph_web --help` |

## Related docs

- Main demo README: [../README.md](../README.md)
- Local-only env example: [../web/.env.local.example](../web/.env.local.example)