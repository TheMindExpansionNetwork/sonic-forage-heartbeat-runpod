"""Peak-VRAM benchmark for fitting ACE-Step into 8GB.

Measures end-to-end or streaming peak allocated/reserved CUDA memory under
different quantization / VAE / backend configs. Each invocation runs in a
fresh process so peaks are isolated.

Usage:
    uv run python tests/benchmarks/bench_8gb_vram.py --config baseline
    uv run python tests/benchmarks/bench_8gb_vram.py --config int8_vaewin --use-dream-vae
    uv run python tests/benchmarks/bench_8gb_vram.py --config int8_vaewin --mode stream --stream-batch 4

Modes:
    single  — session.generate() + session.decode() (default)
    stream  — StreamPipeline-driven batched ticks, matching production path

Flags:
    --use-dream-vae   Replace VAE decoder with daydreamlive/DreamVAE distillation
                       (decoder-only; leaves vae.encoder on Oobleck if present).
"""

import argparse
import gc
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.types import Curve


CONFIGS = {
    "baseline":        dict(decoder_backend="eager",   vae_backend="eager", quantization=None,               vae_window=0.0),
    "baseline_vaewin": dict(decoder_backend="eager",   vae_backend="eager", quantization=None,               vae_window=10.0),
    "compile":         dict(decoder_backend="compile", vae_backend="eager", quantization=None,               vae_window=10.0),
    "int8":            dict(decoder_backend="compile", vae_backend="eager", quantization="int8_weight_only", vae_window=0.0),
    "int8_vaewin":     dict(decoder_backend="compile", vae_backend="eager", quantization="int8_weight_only", vae_window=10.0),
    "fp8":             dict(decoder_backend="compile", vae_backend="eager", quantization="fp8_weight_only",  vae_window=10.0),
    "fp8_dynamic":     dict(decoder_backend="compile", vae_backend="eager", quantization="fp8_dynamic",      vae_window=10.0),
    "w8a8":            dict(decoder_backend="compile", vae_backend="eager", quantization="w8a8_dynamic",     vae_window=10.0),
}


DREAM_VAE_REPO = "daydreamlive/DreamVAE"


def gb(n_bytes):
    return n_bytes / (1024**3)


def snapshot(label, events):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    peak_alloc = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak_reserved = torch.cuda.max_memory_reserved()
    events.append({
        "label": label,
        "alloc_gb": round(gb(alloc), 3),
        "peak_alloc_gb": round(gb(peak_alloc), 3),
        "reserved_gb": round(gb(reserved), 3),
        "peak_reserved_gb": round(gb(peak_reserved), 3),
    })
    print(f"[{label:>30}] alloc={gb(alloc):.2f}GB  peak_alloc={gb(peak_alloc):.2f}GB  "
          f"reserved={gb(reserved):.2f}GB  peak_reserved={gb(peak_reserved):.2f}GB",
          flush=True)


def _load_dream_vae_decoder(device, dtype):
    """Import FastOobleckDecoder from the DreamVAE repo (downloads if needed)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    token = os.environ.get("HF_TOKEN")
    if not token:
        # Fall back to repo .env
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1]
                    break

    snap = snapshot_download(
        DREAM_VAE_REPO, token=token,
        allow_patterns=["*.safetensors", "*.json", "modeling.py"],
    )
    # Load modeling.py as a module
    spec = importlib.util.spec_from_file_location("dreamvae_modeling",
                                                  os.path.join(snap, "modeling.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    decoder = mod.FastOobleckDecoder()
    sd = load_file(os.path.join(snap, "model.safetensors"))
    decoder.load_state_dict(sd)
    decoder = decoder.to(device=device, dtype=dtype).eval()
    return decoder


def swap_in_dream_vae(session):
    handler = session.handler
    if handler.vae is None:
        print("  [dream-vae] skip: TRT VAE in use")
        return
    device = next(handler.vae.parameters()).device
    dtype = next(handler.vae.parameters()).dtype
    new_decoder = _load_dream_vae_decoder(device, dtype)
    handler.vae.decoder = new_decoder
    print(f"  [dream-vae] swapped in FastOobleckDecoder on {device}/{dtype}")


def run_single(session, cond, neg_cond, duration, steps, events):
    T = int(duration * 25)
    gc_curve = Curve(tensor=torch.full((T,), 5.0, dtype=torch.bfloat16))

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = session.generate(
        conditioning=cond, seed=42, duration=duration,
        steps=steps, shift=3.0, denoise=1.0,
        negative=neg_cond, guidance_curve=gc_curve,
    )
    warm_s = time.perf_counter() - t0
    snapshot("after_generate_warmup", events)
    print(f"  [generate warmup: {warm_s:.1f}s]")

    # Timed
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = session.generate(
        conditioning=cond, seed=43, duration=duration,
        steps=steps, shift=3.0, denoise=1.0,
        negative=neg_cond, guidance_curve=gc_curve,
    )
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    snapshot("after_generate_timed", events)
    print(f"  [generate: {gen_s:.2f}s]")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    audio = session.decode(out)
    torch.cuda.synchronize()
    dec_s = time.perf_counter() - t0
    snapshot("after_vae_decode", events)
    print(f"  [vae decode: {dec_s:.2f}s]")
    return {"generate_s": gen_s, "vae_decode_s": dec_s, "warmup_s": warm_s}


def run_stream(session, cond, duration, steps, batch, events):
    """Drive StreamPipeline with ``batch`` concurrent slots, measure peak."""
    from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
    from acestep.engine.stream import StreamPipeline, SlotRequest

    handler = session.handler
    device = handler.device
    dtype = handler.dtype

    handler._ensure_silence_latent_on_device()
    T = int(duration * 25)
    ctx_lat = handler.silence_latent[:, :T, :].clone().to(device=device, dtype=dtype)
    D = ctx_lat.shape[2]
    cm = torch.ones(1, T, D, device=device, dtype=dtype)
    context_latents = torch.cat([ctx_lat, cm], dim=-1)

    entry = cond.to_entries()[0]

    # Pipeline depth = infer_steps. Filling all slots means concurrent batch = depth.
    # batch is the steady-state number of active slots we want.
    config = DiffusionConfig(infer_steps=steps, shift=3.0, noise_on_cpu=True)
    engine = DiffusionEngine(handler.model, compile_loops=False)
    pipe = StreamPipeline(engine, config, pipeline_depth=batch)
    snapshot("stream_pipe_built", events)

    def mkreq(seed):
        return SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=seed,
        )

    # Submit enough requests to keep the ring buffer full for several ticks
    num_gens = batch * 3
    for i in range(num_gens):
        pipe.submit(mkreq(1000 + i))

    # Warmup ticks until pipeline is full
    for _ in range(batch + 1):
        pipe.tick()
    snapshot("stream_warmed_up", events)
    print(f"  [active_slots after warmup = {pipe.active_slots}]")

    # Timed ticks at steady state
    torch.cuda.reset_peak_memory_stats()
    t_ticks = []
    completed_latents = []
    torch.cuda.synchronize()
    for _ in range(batch * 2):
        t0 = time.perf_counter()
        result = pipe.tick()
        torch.cuda.synchronize()
        t_ticks.append((time.perf_counter() - t0) * 1000)
        if result is not None:
            completed_latents.append(result)
    snapshot("stream_steady_state", events)
    avg_tick = sum(t_ticks) / len(t_ticks)
    print(f"  [steady-state avg tick: {avg_tick:.1f}ms, n={len(t_ticks)}]")
    print(f"  [active_slots end: {pipe.active_slots}, completed: {len(completed_latents)}]")

    # Drain remaining
    while pipe.active_slots > 0:
        pipe.tick()

    # Decode one completed latent (single) to include post-stream decode peak
    dec_s = None
    if completed_latents:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            audio = handler.tiled_decode(
                completed_latents[0].transpose(1, 2),
                chunk_size=512,
                overlap=64,
                offload_wav_to_cpu=True,
            )
        torch.cuda.synchronize()
        dec_s = time.perf_counter() - t0
        snapshot("stream_after_decode", events)
        print(f"  [vae decode one: {dec_s:.2f}s]")

    return {"avg_tick_ms": avg_tick, "vae_decode_s": dec_s, "batch": batch}


def run(config_name, duration, steps, mode, batch, use_dream_vae):
    cfg = CONFIGS[config_name]
    events = []

    torch.cuda.reset_peak_memory_stats()
    snapshot("process_start", events)

    t0 = time.perf_counter()
    session = Session(
        decoder_backend=cfg["decoder_backend"],
        vae_backend=cfg["vae_backend"],
        use_flash_attention=True,
        quantization=cfg["quantization"],
        vae_window=cfg.get("vae_window", 0.0),
    )
    load_s = time.perf_counter() - t0
    print(f"  [session load: {load_s:.1f}s]", flush=True)
    snapshot("after_session_load", events)

    if use_dream_vae:
        swap_in_dream_vae(session)
        snapshot("after_dream_vae_swap", events)

    cond = session.encode_text(
        tags="jazz piano trio, brushed drums, walking bass, 140 bpm",
        lyrics="[instrumental]",
        duration=duration,
        instruction=TASK_INSTRUCTIONS["text2music"],
    )
    neg_cond = session.null_conditioning(cond)
    snapshot("after_encode_text", events)

    if mode == "single":
        phase = run_single(session, cond, neg_cond, duration, steps, events)
    else:
        phase = run_stream(session, cond, duration, steps, batch, events)

    result = {
        "config": config_name,
        "mode": mode,
        "duration_s": duration,
        "steps": steps,
        "batch": batch,
        "use_dream_vae": use_dream_vae,
        "load_s": round(load_s, 2),
        "phase": phase,
        "events": events,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(gb(torch.cuda.get_device_properties(0).total_memory), 2),
    }
    os.makedirs("benchmark_results", exist_ok=True)
    tag = config_name
    if use_dream_vae: tag += "_dvae"
    if mode == "stream": tag += f"_stream_b{batch}"
    out_path = os.path.join("benchmark_results", f"vram_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")
    peak = max(e["peak_reserved_gb"] for e in events)
    print(f"PEAK RESERVED OVERALL: {peak:.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--mode", choices=["single", "stream"], default="single")
    ap.add_argument("--stream-batch", type=int, default=4,
                    help="Stream mode: concurrent slot count (pipeline depth)")
    ap.add_argument("--use-dream-vae", action="store_true",
                    help="Swap VAE decoder for daydreamlive/DreamVAE distillation")
    args = ap.parse_args()
    run(args.config, args.duration, args.steps, args.mode, args.stream_batch,
        args.use_dream_vae)
