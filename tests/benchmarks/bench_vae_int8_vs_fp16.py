"""Compare VAE decode: PT eager (DreamVAE) vs TRT fp16 vs TRT int8.

Generates a single latent once, then decodes it three ways against the same
input slice (10s == opt-shape of the TRT profiles). Reports per-path:
  - peak allocated VRAM during decode
  - wall-clock latency
  - output MSE vs PT eager reference (quality proxy)

Usage:
    uv run python tests/benchmarks/bench_vae_int8_vs_fp16.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
torch.set_grad_enabled(False)

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.vae_nodes import _trt_vae_decode
from acestep.paths import trt_engines_dir


def _load_env_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def _load_dream_vae_pt(device, dtype):
    snap = snapshot_download(
        "daydreamlive/DreamVAE",
        token=_load_env_hf_token(),
        allow_patterns=["*.safetensors", "*.json", "modeling.py"],
    )
    spec = importlib.util.spec_from_file_location(
        "dreamvae_modeling", os.path.join(snap, "modeling.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    decoder = mod.FastOobleckDecoder()
    decoder.load_state_dict(load_file(os.path.join(snap, "model.safetensors")))
    return decoder.to(device=device, dtype=dtype).eval()


def _bench_one(label, fn, reference=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    mse = None
    if reference is not None:
        mse = ((out.float() - reference.float()) ** 2).mean().item()
    print(f"  [{label:>16}]  {dt:7.1f} ms   peak_alloc={peak_alloc:.3f} GB"
          f"   mse_vs_pt={mse:.2e}" if mse is not None else
          f"  [{label:>16}]  {dt:7.1f} ms   peak_alloc={peak_alloc:.3f} GB")
    return {"label": label, "ms": dt, "peak_alloc_gb": peak_alloc, "mse": mse}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8-engine", default="vae_decode_int8_15s")
    ap.add_argument("--fp16-engine", default="vae_decode_fp16_15s_dreamvae")
    ap.add_argument("--chunk-frames", type=int, default=250,
                    help="Frames to decode (must match engine profile's opt)")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda")

    int8_engine_path = trt_engines_dir() / args.int8_engine / f"{args.int8_engine}.engine"
    fp16_engine_path = trt_engines_dir() / args.fp16_engine / f"{args.fp16_engine}.engine"
    assert int8_engine_path.exists(), f"Missing INT8 engine: {int8_engine_path}"
    assert fp16_engine_path.exists(), f"Missing fp16 engine: {fp16_engine_path}"

    print("Loading session (eager) + DreamVAE reference decoder...")
    session = Session(decoder_backend="eager", vae_backend="eager", use_flash_attention=True)
    handler = session.handler
    pt_decoder = _load_dream_vae_pt(device, torch.bfloat16)

    print("Generating a latent for decode comparison...")
    cond = session.encode_text(
        tags="electronic ambient, 120 bpm", lyrics="[instrumental]",
        duration=args.duration, instruction=TASK_INSTRUCTIONS["text2music"],
    )
    latent = session.generate(
        conditioning=cond, seed=42, duration=args.duration,
        steps=args.steps, shift=3.0, denoise=1.0,
    )
    # latent.tensor is [1, T, D]. Decoder wants [1, D, T].
    lat_bdt = latent.tensor.transpose(1, 2).contiguous()
    # Take the first chunk_frames (== opt of TRT profile)
    chunk = lat_bdt[:, :, :args.chunk_frames].contiguous()
    print(f"  chunk shape: {tuple(chunk.shape)}  dtype={chunk.dtype}")
    print()

    # Warmup each path once (TRT JIT/workspace alloc, buf reuse)
    print("Warmup...")
    _ = pt_decoder(chunk.to(torch.bfloat16))
    _ = _trt_vae_decode(chunk, str(fp16_engine_path), device)
    _ = _trt_vae_decode(chunk, str(int8_engine_path), device)
    torch.cuda.synchronize()

    # PT eager reference (DreamVAE in bf16)
    print(f"\n{args.runs} runs each:")
    ref_runs = []
    for _ in range(args.runs):
        r, out = _bench_one("PT eager bf16", lambda: pt_decoder(chunk.to(torch.bfloat16)))
        ref_runs.append(r)
    reference_audio = out.detach().clone()

    fp16_runs = []
    for _ in range(args.runs):
        r, _ = _bench_one("TRT fp16", lambda: _trt_vae_decode(chunk, str(fp16_engine_path), device),
                          reference=reference_audio)
        fp16_runs.append(r)

    int8_runs = []
    for _ in range(args.runs):
        r, _ = _bench_one("TRT int8", lambda: _trt_vae_decode(chunk, str(int8_engine_path), device),
                          reference=reference_audio)
        int8_runs.append(r)

    # Summary
    def summary(runs, name):
        ms = sorted(r["ms"] for r in runs)
        peak = max(r["peak_alloc_gb"] for r in runs)
        med_ms = ms[len(ms) // 2]
        mse_values = [r["mse"] for r in runs if r["mse"] is not None]
        mse = sum(mse_values) / len(mse_values) if mse_values else None
        return name, med_ms, peak, mse

    print("\n" + "=" * 72)
    print(f"{'Path':<18}{'Median ms':>12}{'Peak alloc GB':>18}{'MSE vs PT':>24}")
    print("-" * 72)
    for name, med, peak, mse in [
        summary(ref_runs, "PT eager bf16"),
        summary(fp16_runs, "TRT fp16"),
        summary(int8_runs, "TRT int8"),
    ]:
        mse_s = f"{mse:.2e}" if mse is not None else "(reference)"
        print(f"{name:<18}{med:>12.1f}{peak:>18.3f}{mse_s:>24}")
    print("=" * 72)


if __name__ == "__main__":
    main()
