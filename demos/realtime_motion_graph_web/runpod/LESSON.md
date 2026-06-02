# Lesson: DEMON on RunPod (Sonic Forage · Heartbeat edition)

**Time:** ~45 minutes first time (mostly model download) · **Level:** intermediate  
**You’ll need:** RunPod GPU pod, Node 20+, Python/`uv`, ports **6660** + **1318** exposed

---

## Chapter 0 — The two doors

DEMON on RunPod is really **two servers**:

| Door | Port | What it does |
|------|------|----------------|
| **Web UI** | 6660 | Next.js — what you see in the browser |
| **Engine** | 1318 | Python — GPU inference + **WebSocket** stream |

HTTP calls like `/api/loras` can ride through Next.js rewrites. **Audio streaming does not.** The browser must open `wss://<your-pod>-1318.proxy.runpod.net/` — not `127.0.0.1`.

**Lesson 0 takeaway:** If the page loads but Play dies, suspect the WebSocket URL first.

---

## Chapter 1 — Expose the ports

In your RunPod template (or pod TCP settings), publish:

- `6660` → UI  
- `1318` → engine  

Without **1318**, the proxy cannot reach the WebSocket and you’ll get socket / connection errors.

---

## Chapter 2 — GitHub & repo (optional)

```bash
export GITHUB_TOKEN=ghp_your_token
./demos/realtime_motion_graph_web/runpod/setup_github.sh
```

See [GITHUB.md](./GITHUB.md). This fork lives at:

**https://github.com/TheMindExpansionNetwork/sonic-forage-heartbeat-runpod**

---

## Chapter 3 — Install Python deps

From repo root:

```bash
uv sync
```

---

## Chapter 4 — Download the models (the big one)

Fresh volumes have **no checkpoints**. Play will fail until this finishes (~10 GB).

```bash
./demos/realtime_motion_graph_web/runpod/download_models.sh
```

Wait for `OK`. Verify:

```bash
ls ~/.daydream-scope/models/demon/checkpoints/
# expect: acestep-v15-turbo  vae  Qwen3-Embedding-0.6B  acestep-5Hz-lm-1.7B
```

**Lesson 4 takeaway:** “Page works, Play doesn’t” after 0–60s → check checkpoints. After ~85s then disconnect → check PortAudio fix (included in this fork).

---

## Chapter 5 — Start the stack

```bash
./demos/realtime_motion_graph_web/runpod/start.sh
```

Logs print something like:

```text
>>> Open https://<RUNPOD_POD_ID>-6660.proxy.runpod.net/
```

Open that URL (not `localhost` on your laptop unless you’re port-forwarding).

**Stop:**

```bash
./demos/realtime_motion_graph_web/runpod/stop.sh
```

Run `stop.sh` and `start.sh` as **separate** commands (not `stop.sh && start.sh` in one line on older scripts).

---

## Chapter 6 — First Play (the heartbeat)

1. Hard-refresh the UI (`Ctrl+Shift+R`).  
2. Click **Play**.  
3. Wait **60–90 seconds** on first connect (GPU loads DiT + VAE).  
4. You should land in the live HUD — waveform, knobs, optional motion.

If it fails, open browser **DevTools → Console** and backend logs. Match symptoms in [README.md](./README.md#troubleshooting).

---

## Chapter 7 — What we patched (study guide)

| Bug | Symptom | Fix (this fork) |
|-----|---------|-----------------|
| WS points at localhost | Socket error on Play | `podUrl.ts` + `run.py` RunPod URLs |
| Missing checkpoints | FileNotFoundError / spin | `download_models.sh` + `model_context.py` |
| PortAudio on headless pod | Disconnect after model load | `audio_engine.py` lazy sounddevice |
| Next cross-origin dev | Blocked `/_next/*` | `allowedDevOrigins` in `next.config.ts` |
| stop.sh kills start | Launcher exits instantly | Narrower `pkill` patterns |

Details: [FIXES.md](./FIXES.md).

---

## Chapter 8 — Accelerator modes

```bash
# Default in start.sh — boots fast, good for lessons
ACCEL=eager ./demos/realtime_motion_graph_web/runpod/start.sh

# Production-ish — needs TRT engines built
ACCEL=tensorrt ./demos/realtime_motion_graph_web/runpod/start.sh
```

---

## Chapter 9 — Manual overrides

**WebSocket override** (if auto-detect fails):

```text
https://<POD_ID>-6660.proxy.runpod.net/?ws=wss://<POD_ID>-1318.proxy.runpod.net/
```

**Env-only Next dev** (no Python launcher): copy [env.example](./env.example) → `web/.env.development`.

---

## Chapter 10 — What to try next (Sonic Forage path)

- Swap fixtures in the Advanced drawer — hear different seeds of the forage  
- Enable LoRAs — timbre is another layer of motion  
- Drop a video in `demos/realtime_motion_graph_web/videos/` — kick-reactive shaders  
- Read the upstream README for MCP / Claude driving the demo  

---

## Cheat sheet

```bash
# lifecycle
./demos/realtime_motion_graph_web/runpod/download_models.sh  # once
./demos/realtime_motion_graph_web/runpod/start.sh
./demos/realtime_motion_graph_web/runpod/stop.sh

# health
curl -s http://127.0.0.1:1318/api/server-info | head -c 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:6660/
```

When the stream locks in and the graph breathes with the track — that’s **Heartbeat**. You foraged the setup. Now forage the sound.