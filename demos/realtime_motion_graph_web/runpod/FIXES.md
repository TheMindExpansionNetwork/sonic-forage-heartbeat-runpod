# RunPod fixes — change log for future edits

Date applied: 2026-06-02  
Environment: RunPod A100 pod (`RUNPOD_POD_ID=h9ay8scybro9x3`)

## Problem

| Layer | Behavior before fix |
|-------|---------------------|
| Browser on RunPod proxy | Page loaded from `https://<pod>-6660.proxy.runpod.net` |
| HTTP API | Worked via Next.js rewrites to backend |
| WebSocket | Used `ws://127.0.0.1:1318/` → **wrong host** (user’s machine) |
| Next.js HMR | Could be blocked for cross-origin dev requests |

Symptom: errors about WebSocket / socket / connection failed when clicking **Play**.

---

## File changes (edit these when customizing)

### 1. `web/engine/podUrl.ts`

**What:** `defaultWsUrl()` detects RunPod hostnames and builds the engine WebSocket URL.

**Logic:**

```ts
// Host: h9ay8scybro9x3-6660.proxy.runpod.net
// WS:   wss://h9ay8scybro9x3-1318.proxy.runpod.net/
```

**To change engine port** (if not 1318): change the `-1318` suffix in the template string in `defaultWsUrl()`.

**Override without code:** `?ws=wss://<pod>-1318.proxy.runpod.net/` on the UI URL.

---

### 2. `run.py` (launcher)

**What:** When `RUNPOD_POD_ID` is set:

| Setting | Value |
|---------|--------|
| Default `--host` | `0.0.0.0` (reachable from RunPod proxy) |
| Next dev bind | `-H 0.0.0.0` |
| `NEXT_PUBLIC_POD_BASE_URL` | `https://<RUNPOD_POD_ID>-1318.proxy.runpod.net` |
| Startup banner | Prints `https://<RUNPOD_POD_ID>-6660.proxy.runpod.net/` |

**New helpers:**

- `_runpod_backend_url(port)` — builds proxy URL from env
- `_public_pod_base_url(host, port)` — RunPod proxy or local `http://127.0.0.1:port`

**Note:** Health-check still probes `127.0.0.1:1318` locally (correct for “is backend up?”).

**To change ports:** `--port` / `--web-port` on launcher, or edit `start.sh`.

---

### 3. `web/next.config.ts`

**What:**

```ts
allowedDevOrigins: [
  "*.proxy.runpod.net",
  ...(process.env.RUNPOD_POD_ID
    ? [`${process.env.RUNPOD_POD_ID}-6660.proxy.runpod.net`]
    : []),
],
```

**Why:** Next.js 16 blocks cross-origin access to `/_next/*` dev assets unless the browser’s `Origin` is allowlisted. RunPod serves the UI from `*.proxy.runpod.net`.

**To change web port:** Replace `6660` in the `${RUNPOD_POD_ID}-6660...` line if you use another exposed port.

`rewrites()` still use `NEXT_PUBLIC_POD_BASE_URL` for `/api/*`, `/fixtures/*`, etc.

---

### 4. `web/.env.development`

**What was set for this pod:**

```env
NEXT_PUBLIC_POD_BASE_URL=https://h9ay8scybro9x3-1318.proxy.runpod.net
NEXT_PUBLIC_LOCAL_MODE=1
```

**When it matters:** Running `npm run dev` **without** `run.py` (launcher sets env itself when using `start.sh`).

**On a new pod:** Copy [env.example](./env.example) and replace `h9ay8scybro9x3` with your `RUNPOD_POD_ID`, or rely on `podUrl.ts` auto-detect for WS only.

---

## What we did *not* change

- Python `server.py` protocol / ports (still default `1318`)
- WebSocket path (still `/` on engine port)
- Production build flow — this doc is **dev / RunPod** focused

---

## Reverting to local-only dev

1. `web/.env.development`:

   ```env
   NEXT_PUBLIC_POD_BASE_URL=http://127.0.0.1:1318
   ```

2. Run without `RUNPOD_POD_ID` in the environment:

   ```bash
   uv run python -u -m demos.realtime_motion_graph_web.run
   ```

3. Open `http://localhost:6660/`

---

## 6. `acestep/streaming/audio_engine.py` (2026-06-02)

**Symptom:** Models load (~85s) then WebSocket dies:

```text
OSError: PortAudio library not found
```

**Cause:** `AudioEngine.__init__` imported `sounddevice` on every session. RunPod has no PortAudio; the **browser** plays audio for the web demo — server playback is never started.

**Fix:** Lazy-import `sounddevice` only in `start()` (buffer-only `__init__`).

---

## 5. `acestep/engine/model_context.py` (2026-06-02)

**Symptom:** UI loads (HTTP 200) but **Play** never starts; backend log shows:

```text
FileNotFoundError: Cannot locate a populated checkpoints directory ...
```

**Cause:** Fresh RunPod volume had fixtures but no `~/.daydream-scope/models/demon/checkpoints/` weights (~10 GB). `_resolve_checkpoint_dir` refused an empty canonical path before auto-download could run.

**Fix:** Allow the canonical `checkpoints_dir()` path even when empty; `_ensure_downloaded` then fetches from HuggingFace.

**One-time setup on a new pod:**

```bash
./demos/realtime_motion_graph_web/runpod/download_models.sh
```

---

## 2026-06-03 — Storage, XL, meta tensor, Qwen corrupt file

### Network volume layout

| Path | Role |
|------|------|
| `/workspace/.daydream-scope` | All checkpoints (persistent) |
| `/workspace/agent-world-workspace` | Scripts, logs, runbook |
| `~/.daydream-scope` | Symlink → `/workspace/.daydream-scope` |

Overlay `/` must stay empty of models (~30 GB limit).

### `cannot copy out of meta tensor`

**File:** `acestep/engine/model_context.py`

**Fix:** `from_pretrained(..., torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)` before `.to(cuda)`.

### `header too small` while deserializing

**Cause:** Truncated/0-byte `.safetensors` from interrupted download.

**Example:** `checkpoints/Qwen3-Embedding-0.6B/model.safetensors` was 0 bytes.

**Fix:**

```bash
/workspace/agent-world-workspace/scripts/repair_qwen_encoder.sh
# or from repo copy:
./demos/realtime_motion_graph_web/runpod/agent-world/scripts/repair_qwen_encoder.sh
```

### XL + compile launcher (nohup)

Use `agent-world/scripts/start_demon_web_xl.sh` — survives shell exit; pid in `logs/demon_web_xl.pid`.

Full reboot steps: `runpod/agent-world/docs/RUNBOOK.md`

---

## Checklist after you edit anything

```bash
./demos/realtime_motion_graph_web/runpod/stop.sh
./demos/realtime_motion_graph_web/runpod/start.sh
curl -s http://127.0.0.1:1318/api/server-info | head -c 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:6660/
```

Then open the `6660` proxy URL and test **Play** in the browser devtools → Network → WS should show `wss://<pod>-1318.proxy.runpod.net`.