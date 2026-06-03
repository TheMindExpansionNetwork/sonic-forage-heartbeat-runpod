# Agent World Workspace

**Attach this network volume to any RunPod.** Everything heavy lives here — not on the 30 GB overlay.

## Layout

```
agent-world-workspace/
  README.md           ← you are here
  env.sh              ← source this in every new shell
  AGENTS.md           ← paths + commands for AI agents
  repos/
    DEMON/            → sonic-forage / daydream engine
    demonTD/          → TouchDesigner operator
    avtr-1/           → talking-head video (build later)
  models/             → checkpoints, TRT, loras (persistent)
  scripts/
    start_demon_web_xl.sh   # nohup daemon (6660 + 1318)
    start_demon_td_backend.sh
    stop_all.sh
    verify_xl.sh
    repair_qwen_encoder.sh  # fix "header too small" (0-byte Qwen)
  docs/
    RUNBOOK.md              # reboot + errors (read this)
    RUNPOD_PORTS.md
    TOUCHDESIGNER.md
  releases/
    demonTD.tox
  logs/               ← runtime logs land here
```

## First boot on a fresh pod

```bash
./POD_BOOTSTRAP.sh                # verify XL + repair Qwen + env
./scripts/start_demon_web_xl.sh   # browser HUD :6660
# or
./scripts/start_demon_td_backend.sh   # TouchDesigner :1318
```

## XL checkpoint

Expected at:

`models/checkpoints/acestep-v15-xl-turbo/` (~19 GB)

Download if missing:

```bash
source env.sh
cd repos/DEMON && uv run acestep-download --model acestep-v15-xl-turbo --skip-main
```

## Default accel

**`compile` + `xl`** — TensorRT XL build is separate (pod GPU stack dependent). Jam / AVTR builds run later together.

## Links

- DEMON fork: https://github.com/TheMindExpansionNetwork/sonic-forage-heartbeat-runpod
- demonTD fork: https://github.com/TheMindExpansionNetwork/demonTD