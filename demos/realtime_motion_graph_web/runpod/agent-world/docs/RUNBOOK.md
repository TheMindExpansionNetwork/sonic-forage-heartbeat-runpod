# RunPod reboot runbook (Agent World + DEMON XL)

**Volume path:** `/workspace/agent-world-workspace` (alias: `/workspace/agent-world-workshop`)  
**Git copy:** `DEMON/demos/realtime_motion_graph_web/runpod/agent-world/`

## After attaching the network volume to a new pod

```bash
source /workspace/agent-world-workspace/env.sh
/workspace/agent-world-workspace/POD_BOOTSTRAP.sh
```

`POD_BOOTSTRAP` verifies XL, repairs corrupt Qwen weights if needed, and hooks `env.sh` into `~/.bashrc`.

## Start DEMON XL (survives shell exit)

```bash
/workspace/agent-world-workspace/scripts/start_demon_web_xl.sh
```

- **Web:** `https://${RUNPOD_POD_ID}-6660.proxy.runpod.net/`
- **Engine WS:** `wss://${RUNPOD_POD_ID}-1318.proxy.runpod.net/`
- **Log:** `/workspace/agent-world-workspace/logs/demon_web_xl.log`
- **PID file:** `logs/demon_web_xl.pid`

Stop:

```bash
/workspace/agent-world-workspace/scripts/stop_all.sh
```

## Expose ports in RunPod template

| Port | Service |
|------|---------|
| 6660 | DEMON web UI |
| 1318 | DEMON engine / TouchDesigner |
| 7860 | AVTR (later) |

## Models (network volume only)

| Asset | Path |
|-------|------|
| All demon models | `/workspace/.daydream-scope/models/demon` |
| XL DiT (4 shards) | `.../checkpoints/acestep-v15-xl-turbo/` (~19 GB) |
| VAE | `.../checkpoints/vae/diffusion_pytorch_model.safetensors` (~322 MB) |
| Text encoder | `.../checkpoints/Qwen3-Embedding-0.6B/model.safetensors` (~1.2 GB) |
| Symlink | `~/.daydream-scope` → `/workspace/.daydream-scope` |

**Do not** put checkpoints on overlay `/` (30 GB). Set:

```bash
export ACESTEP_MODELS_DIR=/workspace/.daydream-scope/models/demon
```

## Known errors and fixes

### 1. `cannot copy out of meta tensor`

**Cause:** Transformers 4.5x loads weights on meta device; `.to(cuda)` fails.

**Fix (in repo):** `acestep/engine/model_context.py` — `from_pretrained(..., torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)`.

**Action:** Pull latest `sonic-forage-heartbeat-runpod` / DEMON with that commit.

### 2. `header too small` while deserializing

**Cause:** **0-byte or truncated** `.safetensors` (usually interrupted download).

**Checked 2026-06-03:** `Qwen3-Embedding-0.6B/model.safetensors` was **0 bytes**.

**Fix:**

```bash
/workspace/agent-world-workspace/scripts/repair_qwen_encoder.sh
```

Then restart DEMON.

**Verify any shard:**

```bash
/workspace/DEMON/.venv/bin/python -c "
from safetensors import safe_open
p='/workspace/.daydream-scope/models/demon/checkpoints/Qwen3-Embedding-0.6B/model.safetensors'
with safe_open(p, framework='pt') as h: print('OK', len(list(h.keys())))
"
```

### 3. `session_create_failed` after Play

- Use **one browser tab** (parallel sessions load XL multiple times).
- Wait **2–4 min** first Play (shard load + `torch.compile`).
- Tail log for `model_loaded` (success) vs `session_create_failed` (failure).
- Run `repair_qwen_encoder.sh` if header error.

### 4. Web works but engine dead (exit 247 / background shell died)

Old launcher used `exec | tee` in a short-lived shell. Use **`start_demon_web_xl.sh`** (nohup + pid file).

### 5. Disk full on `/`

Move models to `/workspace/.daydream-scope`, symlink `~/.daydream-scope`. Prune `uv` cache: `uv cache prune`.

## Live logs

```bash
tail -f /workspace/agent-world-workspace/logs/demon_web_xl.log | grep '\[backend\]'
```

Success line: `model_loaded duration_s=...`

## Default XL test command

```bash
source /workspace/agent-world-workspace/env.sh
cd /workspace/DEMON
uv run python -u -m demos.realtime_motion_graph_web.run \
  -- --host 0.0.0.0 --port 1318 --accel compile --checkpoint xl
```

## Deferred (jam later)

- AVTR renderer TRT + port 7860
- DEMON XL TensorRT (CUDA 35 on some pods)
- Vocal / endless-sing jam pipeline

## Links

- Fork: https://github.com/TheMindExpansionNetwork/sonic-forage-heartbeat-runpod
- Upstream: https://github.com/daydreamlive/DEMON
- demonTD: https://github.com/TheMindExpansionNetwork/demonTD