"""Validate FP8 XL decoder vs bf16 baseline: numeric accuracy + speed.

Both engines have the same I/O contract; feed identical inputs (pulled
from the calibration .npz so they're real distribution samples) and
compare the velocity output. Then benchmark steady-state latency at
batch=4 for both.

Run with:
    python benchmarks-pr17/fp8_vs_bf16_validate.py
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
torch.set_grad_enabled(False)

import tensorrt as trt

ENGINES = {
    "bf16": Path(os.path.expanduser(
        "~/.daydream-scope/models/demon/trt_engines/"
        "decoder_xl-turbo_mixed_refit_b4_60s/"
        "decoder_xl-turbo_mixed_refit_b4_60s.engine"
    )),
    "fp8": Path(os.path.expanduser(
        "~/.daydream-scope/models/demon/trt_engines/"
        "decoder_xl-turbo_fp8_refit_b4_60s/"
        "decoder_xl-turbo_fp8_refit_b4_60s.engine"
    )),
}
CAL = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/calibration/decoder_xl_fp8/calibration.npz"
))
INPUT_NAMES = ("hidden_states", "timestep", "encoder_hidden_states", "context_latents")
OUTPUT_NAME = "velocity"

_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}
if hasattr(trt, "bfloat16"):
    _TRT_TO_TORCH[trt.bfloat16] = torch.bfloat16


class EngineRunner:
    def __init__(self, label, path):
        self.label = label
        self.path = path
        self.size_mb = path.stat().st_size / 1e6
        logger = trt.Logger(trt.Logger.WARNING)
        rt = trt.Runtime(logger)
        with open(path, "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load {path}")
        self.ctx = self.engine.create_execution_context()

        # I/O dtypes per tensor
        self.in_dtypes = {}
        for n in INPUT_NAMES:
            self.in_dtypes[n] = _TRT_TO_TORCH.get(
                self.engine.get_tensor_dtype(n), torch.float32,
            )
        self.out_dtype = _TRT_TO_TORCH.get(
            self.engine.get_tensor_dtype(OUTPUT_NAME), torch.float32,
        )
        self.stream = torch.cuda.Stream()

    def __repr__(self):
        dts = "/".join(str(self.in_dtypes[n]).replace("torch.", "") for n in INPUT_NAMES)
        return (
            f"{self.label} ({self.size_mb:.0f} MB) "
            f"inputs={dts} out={str(self.out_dtype).replace('torch.','')}"
        )

    def run(self, inputs: dict, sync: bool = True) -> torch.Tensor:
        """One forward pass. ``inputs`` is a dict of torch.Tensor on cuda;
        we recast each to the engine's required dtype.
        """
        dev = torch.device("cuda")
        bufs = {}
        for n in INPUT_NAMES:
            t = inputs[n].to(device=dev, dtype=self.in_dtypes[n]).contiguous()
            bufs[n] = t
            if not self.ctx.set_input_shape(n, tuple(t.shape)):
                raise RuntimeError(f"{self.label}: rejected shape for {n}: {t.shape}")
            if not self.ctx.set_tensor_address(n, t.data_ptr()):
                raise RuntimeError(f"{self.label}: rejected address for {n}")
        miss = self.ctx.infer_shapes()
        if miss:
            raise RuntimeError(f"{self.label}: shapes underspecified: {miss}")
        out_shape = tuple(self.ctx.get_tensor_shape(OUTPUT_NAME))
        out = torch.empty(out_shape, dtype=self.out_dtype, device=dev)
        if not self.ctx.set_tensor_address(OUTPUT_NAME, out.data_ptr()):
            raise RuntimeError(f"{self.label}: rejected output address")
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(f"{self.label}: execute_async_v3 failed")
        if sync:
            self.stream.synchronize()
        return out, bufs  # keep bufs alive until caller is done


def main():
    print("=" * 70)
    print("FP8 vs bf16 XL decoder validation")
    print("=" * 70)
    print()

    for label, p in ENGINES.items():
        if not p.exists():
            print(f"MISSING: {label} engine at {p}")
            return
    if not CAL.exists():
        print(f"MISSING: calibration {CAL}")
        return

    # Load engines
    runners = {label: EngineRunner(label, path) for label, path in ENGINES.items()}
    for r in runners.values():
        print(" ", r)
    print()

    # Calibration samples as inputs (real distribution)
    npz = np.load(str(CAL))
    n_total = npz["hidden_states"].shape[0]
    print(f"Calibration tensors: {n_total} samples")
    print()

    # ---- Numeric accuracy on a handful of batch=4 forwards ----
    print("-" * 70)
    print("NUMERIC: FP8 velocity vs bf16 velocity (same inputs)")
    print("-" * 70)
    B = 4
    n_eval = min(8, n_total // B)
    rows = []
    for i in range(n_eval):
        s = slice(i * B, (i + 1) * B)
        inputs = {
            "hidden_states": torch.from_numpy(npz["hidden_states"][s]),
            "timestep": torch.from_numpy(npz["timestep"][s]),
            "encoder_hidden_states": torch.from_numpy(npz["encoder_hidden_states"][s]),
            "context_latents": torch.from_numpy(npz["context_latents"][s]),
        }
        out_bf16, _b = runners["bf16"].run(inputs)
        out_fp8, _f = runners["fp8"].run(inputs)
        bf32 = out_bf16.float()
        ff32 = out_fp8.float()
        diff = (bf32 - ff32).abs()
        rel = diff / (bf32.abs() + 1e-8)
        cos = torch.nn.functional.cosine_similarity(
            bf32.flatten().unsqueeze(0), ff32.flatten().unsqueeze(0),
        ).item()
        rows.append({
            "batch": i,
            "ts": inputs["timestep"].tolist(),
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
            "max_rel": rel.max().item(),
            "mean_rel": rel.mean().item(),
            "cosine": cos,
            "bf16_std": bf32.std().item(),
            "fp8_std": ff32.std().item(),
        })
        print(
            f"  batch {i:>2}  ts={[f'{t:.2f}' for t in inputs['timestep'].tolist()]}  "
            f"cos={cos:.5f}  max_abs={diff.max().item():.4f}  "
            f"mean_abs={diff.mean().item():.5f}  "
            f"bf16_std={bf32.std().item():.4f}  fp8_std={ff32.std().item():.4f}"
        )

    cos_avg = sum(r["cosine"] for r in rows) / len(rows)
    max_max = max(r["max_abs"] for r in rows)
    mean_mean = sum(r["mean_abs"] for r in rows) / len(rows)
    print()
    print(f"AGGREGATE: cosine_sim avg={cos_avg:.5f}  "
          f"max_abs_diff peak={max_max:.4f}  mean_abs_diff avg={mean_mean:.5f}")
    print()

    # ---- Speed benchmark at batch=4 ----
    print("-" * 70)
    print("SPEED: per-tick latency at batch=4, 60s seq (steady state)")
    print("-" * 70)
    # Use one steady-state sample, run it many times
    s = slice(0, B)
    inputs = {
        "hidden_states": torch.from_numpy(npz["hidden_states"][s]),
        "timestep": torch.from_numpy(npz["timestep"][s]),
        "encoder_hidden_states": torch.from_numpy(npz["encoder_hidden_states"][s]),
        "context_latents": torch.from_numpy(npz["context_latents"][s]),
    }
    bench = {}
    for label, r in runners.items():
        # Warmup
        for _ in range(5):
            r.run(inputs)
        torch.cuda.synchronize()
        # Timed
        ts = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r.run(inputs)
            t1 = time.perf_counter()
            ts.append((t1 - t0) * 1000.0)
        ts.sort()
        bench[label] = {
            "mean_ms": sum(ts) / len(ts),
            "median_ms": ts[len(ts) // 2],
            "p10_ms": ts[len(ts) // 10],
            "p90_ms": ts[(len(ts) * 9) // 10],
            "min_ms": ts[0],
            "max_ms": ts[-1],
            "size_mb": r.size_mb,
        }
        print(
            f"  {label}: mean={bench[label]['mean_ms']:.2f}ms  "
            f"median={bench[label]['median_ms']:.2f}ms  "
            f"p10/90={bench[label]['p10_ms']:.2f}/{bench[label]['p90_ms']:.2f}ms  "
            f"size={bench[label]['size_mb']:.0f}MB"
        )

    if "bf16" in bench and "fp8" in bench:
        speedup = bench["bf16"]["mean_ms"] / bench["fp8"]["mean_ms"]
        print()
        print(f"SPEEDUP: FP8 / bf16 = {speedup:.2f}x  "
              f"(mean: {bench['bf16']['mean_ms']:.2f} -> {bench['fp8']['mean_ms']:.2f} ms)")

    # ---- Persist results ----
    out = {
        "engines": {label: str(p) for label, p in ENGINES.items()},
        "calibration_npz": str(CAL),
        "numeric_rows": rows,
        "numeric_aggregate": {
            "cosine_sim_avg": cos_avg,
            "max_abs_diff_peak": max_max,
            "mean_abs_diff_avg": mean_mean,
        },
        "benchmark": bench,
    }
    out_path = Path("benchmarks-pr17") / "fp8_vs_bf16_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
