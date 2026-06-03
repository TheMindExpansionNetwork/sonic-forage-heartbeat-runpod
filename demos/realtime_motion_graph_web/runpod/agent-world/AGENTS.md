# Agent instructions (Agent World Workspace)

When this volume is attached at `/workspace`, use these paths. **Do not** store large files under `/root` (30 GB overlay).

## Canonical paths

| What | Path |
|------|------|
| **This workspace** | `/workspace/agent-world-workspace` |
| **Shell env** | `source /workspace/agent-world-workspace/env.sh` |
| **DEMON repo** | `/workspace/DEMON` or `repos/DEMON` |
| **Models** | `/workspace/.daydream-scope/models/demon` |
| **XL checkpoint** | `models/demon/checkpoints/acestep-v15-xl-turbo` |
| **demonTD** | `/workspace/demonTD` |
| **AVTR (later)** | `/workspace/avtr-1` |

## XL test (ready now)

```bash
source /workspace/agent-world-workspace/env.sh
/workspace/agent-world-workspace/scripts/verify_xl.sh
/workspace/agent-world-workspace/scripts/start_demon_web_xl.sh
```

Logs: `/workspace/agent-world-workspace/logs/`

## Reboot / errors

Read **`docs/RUNBOOK.md`**. Quick fixes:

- **header too small** → `./scripts/repair_qwen_encoder.sh` then restart
- **meta tensor** → need DEMON with `model_context.py` fix (`low_cpu_mem_usage=False`)
- **session_create_failed** → one browser tab; wait for `model_loaded` in log

Git copy of scripts/docs: `DEMON/demos/realtime_motion_graph_web/runpod/agent-world/`

## TouchDesigner (user laptop)

- Backend on pod: `scripts/start_demon_td_backend.sh` → `wss://<POD>-1318.proxy.runpod.net/`
- `.tox`: `releases/demonTD.tox`
- Doc: `docs/TOUCHDESIGNER.md`

## Deferred (jam together later)

- AVTR TRT renderer engines + port **7860**
- Vocal stem jam + endless singing experiments
- DEMON TensorRT XL (needs working TRT builder on pod)

## RunPod ports

| Port | Service |
|------|---------|
| 6660 | DEMON web |
| 1318 | DEMON WS / TD |
| 7860 | AVTR (later) |