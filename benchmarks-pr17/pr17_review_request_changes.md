Quick follow-up on the XL batched path now that 1b77dc4 has b4/b8 building cleanly via the ONNX dynbatch walk. Re-tested on the same fixture as my prior comment (xl-turbo, 5090, TRT 10.16.1.11, steps=8 depth=4 vae_window=3.0s, --no-fast-vae, 24 measured ticks) and there are two small builder knobs worth landing in this PR.

**1. Surface `--batch-opt` and `--builder-optimization-level` on the build CLI.**

`TRTBuildConfig` already carries `batch_opt: int = 1` and `builder_optimization_level: int = 3`, but neither `_run_single` nor `_run_all` exposes them, so every engine ships with the legacy `Bopt=1, opt=3` tactic set. TRT picks GEMM tactics around the OPT shape, so the XL b4/b8 engines are choosing kernels tuned for batch=1 and then running them at batch=4 in steady state.

Diff is small — two `parser.add_argument` calls in `main()` plus threading them through `_build_decoder_engine` → `TRTBuildConfig`. Defaulting both to `None` preserves current behavior for anyone not passing the flag.

**2. Build XL with `--batch-opt 4 --builder-optimization-level 5` and make b4 the canonical XL profile.**

Same machine, same fixture, mean of 24 measured ticks:

| engine | tick mean | Δ vs orig b4 |
|---|---:|---:|
| b4 (orig: bopt=1, opt=3) | 150.3 ms | — |
| b8 (orig: bopt=1, opt=3) | 148.4 ms | -1.3% |
| **b4, bopt=4, opt_level=5** | **141.7 ms** | **-5.7%** |

Build cost was 89 s vs 97 s for the original opt=3 — opt_level=5 was effectively free on this graph. The properly-tuned b4 also beats the original b8 outright, which makes sense: depth=4 streaming never exercises B>4, so the b8 engine's extra capacity is dead weight (≈8 GB on disk, larger TRT workspace at session-creation, no perf return).

Concrete asks for this PR:

- Add the two CLI flags.
- Update `_trt_build_command` to emit `--batch-max 4 --batch-opt 4 --builder-optimization-level 5` for `acestep-v15-xl-turbo` (instead of `--batch-max 8`).
- Repoint `_XL_TURBO_TRT_ENGINE_PROFILES[60.0]["decoder"]` (and 120s if it ships in this PR) at `decoder_xl-turbo_mixed_refit_b4_60s`.

Happy to send these on top of your branch if it's easier.
