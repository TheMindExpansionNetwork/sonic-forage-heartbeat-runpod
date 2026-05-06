"""Benchmark live prompt-change latency and CUDA memory pressure.

This mirrors the realtime web backend prompt path without WebSocket or browser
overhead:

1. Resolve/load the same Session and optional TensorRT engines as the demo.
2. Prepare the same fixture/source latent used for cover conditioning.
3. Optionally build and prime a stream so decoder/VAE state is resident.
4. Repeatedly encode new prompts and swap ``stream.conditioning``.

Run with:

    uv run python -u -m demos.realtime_motion_graph_web.prompt_benchmark
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

from acestep.audio.key_detection import detect_key
from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.fixtures import audio_fixture
from acestep.nodes.cond_nodes import EncodeConditioning, EncodeText
from acestep.nodes.types import Conditioning, Latent
from acestep.paths import checkpoints_dir

from .benchmark import (
    DEFAULT_CONFIG,
    DEFAULT_FIXTURE,
    VALID_ACCEL,
    cuda_sync,
    describe,
    duration_cap_for,
    fmt_stat,
    load_audio,
    load_demo_config,
    resolve_trt_engines,
)
from .protocol import SAMPLE_RATE


DEFAULT_PROMPTS = (
    "heavy dubstep loop, deathstep, afxdump, growl heavy bass distortion",
    "daft punk style loop, beautiful, four to the floor, angelic",
    "industrial techno loop, metallic percussion, distorted bass, urgent",
    "cinematic ambient pulse, wide pads, granular textures, dark tension",
)


def gb(num_bytes: int | float | None) -> float | None:
    return None if num_bytes is None else float(num_bytes) / (1024**3)


def cuda_memory() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {
            "allocated": None,
            "reserved": None,
            "max_allocated": None,
            "max_reserved": None,
        }
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
    }


def cuda_memory_gb(snapshot: dict[str, int | None]) -> dict[str, float | None]:
    return {name + "_gb": gb(value) for name, value in snapshot.items()}


@contextmanager
def stage_timer(label: str, timings: dict[str, float], peaks: dict[str, dict[str, int | None]]):
    cuda_sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        cuda_sync()
        timings[label] = (time.perf_counter() - t0) * 1000
        peaks[label] = cuda_memory()


def encode_prompt_staged(
    session: Session,
    *,
    tags: str,
    refer_latent: Latent,
    bpm: int,
    duration: float,
    key: str,
    lyrics: str = "",
    time_signature: str = "4",
    language: str = "en",
) -> tuple[Conditioning, dict[str, float], dict[str, dict[str, int | None]]]:
    """Run the same nodes as Session.encode_text, with stage-level timing."""

    timings: dict[str, float] = {}
    peaks: dict[str, dict[str, int | None]] = {}

    with stage_timer("text_encoder", timings, peaks):
        text_embed = EncodeText().execute(
            clip=session.clip,
            tags=tags,
            lyrics=lyrics,
            instruction=TASK_INSTRUCTIONS["cover"],
            bpm=bpm,
            duration=duration,
            key=key,
            time_signature=time_signature,
            language=language,
        )["text_embed"]

    with stage_timer("conditioning_fusion", timings, peaks):
        conditioning = EncodeConditioning().execute(
            model=session.model,
            text_embed=text_embed,
            timbre_ref=refer_latent,
        )["conditioning"]

    return conditioning, timings, peaks


def prompt_token_count(session: Session, *, tags: str, bpm: int, duration: float, key: str) -> int | None:
    try:
        meta_cap = (
            f"- bpm: {bpm}\n"
            "- timesignature: 4\n"
            f"- keyscale: {key}\n"
            f"- duration: {duration}\n"
        )
        text_prompt = (
            f"# Instruction\n{TASK_INSTRUCTIONS['cover']}\n\n"
            f"# Caption\n{tags}\n\n"
            f"# Metas\n{meta_cap}"
            "<|endoftext|>\n"
        )
        tokens = session.handler.text_tokenizer(
            text_prompt, return_tensors="pt", add_special_tokens=False
        )
        return int(tokens["input_ids"].shape[-1])
    except Exception:
        return None


def load_prompts(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file is not None:
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                prompts.append(stripped)
    prompts.extend(args.prompt or [])

    if not prompts:
        prompts_cfg = config.get("prompts", {})
        prompts.extend(
            prompt
            for prompt in (
                prompts_cfg.get("a"),
                prompts_cfg.get("b"),
                *DEFAULT_PROMPTS,
            )
            if prompt
        )

    unique: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt not in seen:
            unique.append(prompt)
            seen.add(prompt)
    if not unique:
        raise SystemExit("No prompts available to benchmark.")
    return unique


def prompt_for_index(prompts: list[str], index: int, *, unique_prompts: bool) -> str:
    prompt = prompts[index % len(prompts)]
    if unique_prompts:
        prompt = f"{prompt}, benchmark variation {index + 1}"
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark live prompt re-encoding and apply latency."
    )
    parser.add_argument("--accel", choices=VALID_ACCEL, default="tensorrt")
    parser.add_argument("--decoder-accel", choices=VALID_ACCEL)
    parser.add_argument("--vae-accel", choices=VALID_ACCEL)
    parser.add_argument("--checkpoint", default="acestep-v15-turbo")

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Demo config.json for default prompt/engine/control values.",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore static/config.json and use script fallbacks.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audio", type=Path, help="Local WAV/FLAC/etc source audio.")
    source.add_argument(
        "--fixture",
        default=DEFAULT_FIXTURE,
        help="Fixture name from daydreamlive/demon-fixtures.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Seconds of source audio to use. Default: source length capped like the web server.",
    )

    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to include in the change sequence. Repeat for multiple prompts.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Text file with one prompt per non-empty, non-comment line.",
    )
    parser.add_argument(
        "--unique-prompts",
        action="store_true",
        help="Append an iteration suffix so every measured prompt string is unique.",
    )
    parser.add_argument("--bpm", type=int, help="Override detected BPM.")
    parser.add_argument("--key", help="Override detected key.")
    parser.add_argument(
        "--no-detect-metadata",
        action="store_true",
        help="Skip librosa/key detection and use --bpm/--key or config defaults.",
    )
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--time-signature", default="4")
    parser.add_argument("--language", default="en")

    parser.add_argument("--steps", type=int, help="Diffusion steps for optional stream priming.")
    parser.add_argument("--depth", type=int, help="Streaming pipeline depth for optional stream priming.")
    parser.add_argument("--vae-window", type=float, help="Windowed VAE decode size.")
    parser.add_argument(
        "--fast-vae",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use DreamVAE TRT decode when available.",
    )
    parser.add_argument(
        "--offload-text-encoder",
        action="store_true",
        help=(
            "Offload the text encoder between prompt changes to reduce steady "
            "VRAM. By default it stays resident for lower prompt latency."
        ),
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build a StreamHandle and measure the conditioning swap used by the web backend.",
    )
    parser.add_argument(
        "--prime-ticks",
        type=int,
        default=None,
        help="Ticks to run before measuring so stream state is resident. Default: pipeline depth.",
    )
    parser.add_argument(
        "--ticks-between-prompts",
        type=int,
        default=1,
        help="Unmeasured stream ticks after each prompt apply; models a live stream consuming updates.",
    )
    parser.add_argument("--denoise", type=float, help="Denoise value for optional stream ticks.")
    parser.add_argument(
        "--shift-raw",
        type=float,
        help="UI-style shift knob value for optional stream ticks. Effective shift is 1 + shift_raw * 5.",
    )
    parser.add_argument("--noise-share", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=557)

    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument(
        "--retain-conditionings",
        action="store_true",
        help="Keep every measured Conditioning alive to expose worst-case accumulation pressure.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one progress row per N measured prompt changes; 0 disables.",
    )
    parser.add_argument("--json", type=Path, help="Write full metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_demo_config(None if args.no_config else args.config)

    decoder_backend = args.decoder_accel or args.accel
    vae_backend = args.vae_accel or args.accel
    cap_s, trt_profile_checkpoint = duration_cap_for(
        decoder_backend=decoder_backend,
        vae_backend=vae_backend,
        checkpoint=args.checkpoint,
    )
    requested_duration = args.duration if args.duration is not None else cap_s
    duration_s = min(requested_duration, cap_s)

    engine_cfg = config.get("engine", {})
    controls_cfg = config.get("controls", {})
    steps = args.steps if args.steps is not None else int(engine_cfg.get("steps", 8))
    depth = args.depth if args.depth is not None else int(engine_cfg.get("depth", 4))
    vae_window = (
        args.vae_window
        if args.vae_window is not None
        else float(engine_cfg.get("vae_window", 3.0))
    )
    fast_vae = (
        bool(args.fast_vae)
        if args.fast_vae is not None
        else bool(engine_cfg.get("fast_vae", False))
    )
    fallback_bpm = args.bpm if args.bpm is not None else 120
    key = args.key or engine_cfg.get("key") or "C major"
    denoise = (
        args.denoise
        if args.denoise is not None
        else float(controls_cfg.get("denoise", 0.7))
    )
    shift_raw = (
        args.shift_raw
        if args.shift_raw is not None
        else float(controls_cfg.get("shift", 0.5))
    )
    shift = 1.0 + shift_raw * 5.0
    prompts = load_prompts(args, config)

    if requested_duration > cap_s:
        print(
            f"[Setup] Requested duration {requested_duration:.1f}s exceeds "
            f"{trt_profile_checkpoint} profile cap {cap_s:.0f}s; clipping."
        )

    print("=" * 72)
    print("Realtime Motion Graph Prompt Change Benchmark")
    print("=" * 72)
    print(
        f"[Setup] checkpoint={args.checkpoint} "
        f"decoder={decoder_backend} vae={vae_backend}"
    )
    print(
        f"[Setup] prompts={len(prompts)} warmup={args.warmup} measured={args.iters} "
        f"stream={args.stream} offload_text_encoder={args.offload_text_encoder}"
    )

    setup_timings: dict[str, float] = {}

    with stage_timer("load_audio", setup_timings, {}):
        if args.audio is not None:
            audio_path = args.audio
        else:
            audio_path = audio_fixture(args.fixture)
        audio = load_audio(audio_path, duration_s=duration_s)

    waveform = audio.waveform
    actual_duration_s = waveform.shape[-1] / SAMPLE_RATE
    print(
        f"[Setup] audio={audio_path} "
        f"duration={actual_duration_s:.1f}s channels={waveform.shape[0]}"
    )

    trt_engines, picked_dur, fast_vae = resolve_trt_engines(
        decoder_backend=decoder_backend,
        vae_backend=vae_backend,
        checkpoint=args.checkpoint,
        duration_s=actual_duration_s,
        fast_vae=fast_vae,
    )
    if trt_engines:
        for key_name, engine_path in sorted(trt_engines.items()):
            print(f"[Setup] {key_name}={Path(engine_path).stem}")
        if picked_dur is not None:
            print(f"[Setup] picked_trt_profile={picked_dur:.0f}s")

    with stage_timer("model_load", setup_timings, {}):
        session = Session(
            project_root=str(checkpoints_dir()),
            config_path=args.checkpoint,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            offload_text_encoder=args.offload_text_encoder,
            trt_engines=trt_engines,
            vae_window=vae_window,
        )

    if args.no_detect_metadata:
        bpm = fallback_bpm
        print(f"[Setup] metadata detection skipped; bpm={bpm} key={key}")
    else:
        with stage_timer("detect_metadata", setup_timings, {}):
            import librosa

            mono_np = waveform.mean(dim=0).numpy()
            detected_bpm, _ = librosa.beat.beat_track(y=mono_np, sr=SAMPLE_RATE)
            bpm = int(round(float(np.asarray(detected_bpm).flat[0])))
            key = args.key or detect_key(mono_np, SAMPLE_RATE)
        if args.bpm is not None:
            bpm = args.bpm
        print(f"[Setup] metadata bpm={bpm} key={key}")

    with stage_timer("prepare_source", setup_timings, {}):
        source = session.prepare_source(audio)
    print(
        f"[Setup] latent_frames={source.latent.tensor.shape[1]} "
        f"({source.latent.tensor.shape[1] / 25.0:.1f}s)"
    )

    initial_prompt = prompt_for_index(prompts, 0, unique_prompts=args.unique_prompts)
    with stage_timer("initial_prompt_encode", setup_timings, {}):
        initial_conditioning = session.encode_text(
            tags=initial_prompt,
            instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=source.latent,
            bpm=bpm,
            duration=actual_duration_s,
            key=key,
            lyrics=args.lyrics,
            time_signature=args.time_signature,
            language=args.language,
        )

    stream = None
    if args.stream:
        with stage_timer("stream_setup", setup_timings, {}):
            stream = session.stream(
                source=source,
                conditioning=initial_conditioning,
                steps=steps,
                shift=3.0,
                pipeline_depth=depth,
                noise_sharing=args.noise_share,
            )
        prime_ticks = depth if args.prime_ticks is None else args.prime_ticks
        if prime_ticks > 0:
            print(f"[Setup] Priming stream with {prime_ticks} ticks")
            with stage_timer("stream_prime", setup_timings, {}):
                for tick_idx in range(prime_ticks):
                    stream.tick(
                        denoise=denoise,
                        seed=args.seed + tick_idx,
                        shift=shift,
                        noise_sharing=args.noise_share,
                    )

    retained: list[Conditioning] = []
    samples: list[dict[str, Any]] = []
    measured_text_ms: list[float] = []
    measured_fusion_ms: list[float] = []
    measured_apply_ms: list[float] = []
    measured_total_ms: list[float] = []
    measured_peak_alloc_gb: list[float] = []
    measured_peak_reserved_gb: list[float] = []
    measured_delta_alloc_gb: list[float] = []
    measured_delta_reserved_gb: list[float] = []

    total_changes = args.warmup + args.iters
    print("[Run] Measuring prompt re-encode + apply")
    run_t0 = time.perf_counter()

    for idx in range(total_changes):
        prompt = prompt_for_index(
            prompts,
            idx + 1,
            unique_prompts=args.unique_prompts,
        )
        is_measured = idx >= args.warmup
        before = cuda_memory()
        total_t0 = time.perf_counter()
        conditioning, timings, peaks = encode_prompt_staged(
            session,
            tags=prompt,
            refer_latent=source.latent,
            bpm=bpm,
            duration=actual_duration_s,
            key=key,
            lyrics=args.lyrics,
            time_signature=args.time_signature,
            language=args.language,
        )

        cuda_sync()
        assign_t0 = time.perf_counter()
        if stream is not None:
            stream.conditioning = conditioning
        cuda_sync()
        apply_ms = (time.perf_counter() - assign_t0) * 1000
        total_ms = (time.perf_counter() - total_t0) * 1000
        after = cuda_memory()

        text_peak = peaks["text_encoder"]
        fusion_peak = peaks["conditioning_fusion"]
        peak_alloc = max(
            value
            for value in (
                text_peak["max_allocated"],
                fusion_peak["max_allocated"],
                after["allocated"],
            )
            if value is not None
        ) if torch.cuda.is_available() else None
        peak_reserved = max(
            value
            for value in (
                text_peak["max_reserved"],
                fusion_peak["max_reserved"],
                after["reserved"],
            )
            if value is not None
        ) if torch.cuda.is_available() else None

        sample = {
            "index": idx,
            "measured": is_measured,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "prompt_text_tokens": prompt_token_count(
                session,
                tags=prompt,
                bpm=bpm,
                duration=actual_duration_s,
                key=key,
            ),
            "text_encoder_ms": timings["text_encoder"],
            "conditioning_fusion_ms": timings["conditioning_fusion"],
            "apply_swap_ms": apply_ms,
            "total_apply_ms": total_ms,
            "memory_before": cuda_memory_gb(before),
            "memory_after": cuda_memory_gb(after),
            "text_encoder_peak": cuda_memory_gb(text_peak),
            "conditioning_fusion_peak": cuda_memory_gb(fusion_peak),
            "peak_allocated_gb": gb(peak_alloc),
            "peak_reserved_gb": gb(peak_reserved),
            "delta_allocated_gb": (
                gb(after["allocated"] - before["allocated"])
                if after["allocated"] is not None and before["allocated"] is not None
                else None
            ),
            "delta_reserved_gb": (
                gb(after["reserved"] - before["reserved"])
                if after["reserved"] is not None and before["reserved"] is not None
                else None
            ),
        }
        samples.append(sample)

        if args.retain_conditionings:
            retained.append(conditioning)

        if is_measured:
            measured_idx = idx - args.warmup + 1
            measured_text_ms.append(sample["text_encoder_ms"])
            measured_fusion_ms.append(sample["conditioning_fusion_ms"])
            measured_apply_ms.append(sample["apply_swap_ms"])
            measured_total_ms.append(sample["total_apply_ms"])
            if sample["peak_allocated_gb"] is not None:
                measured_peak_alloc_gb.append(sample["peak_allocated_gb"])
            if sample["peak_reserved_gb"] is not None:
                measured_peak_reserved_gb.append(sample["peak_reserved_gb"])
            if sample["delta_allocated_gb"] is not None:
                measured_delta_alloc_gb.append(sample["delta_allocated_gb"])
            if sample["delta_reserved_gb"] is not None:
                measured_delta_reserved_gb.append(sample["delta_reserved_gb"])
            if args.progress_every and (
                measured_idx == 1
                or measured_idx == args.iters
                or measured_idx % args.progress_every == 0
            ):
                peak = sample["peak_allocated_gb"]
                peak_label = "n/a" if peak is None else f"{peak:.2f}GiB"
                print(
                    f"  #{measured_idx:3d}/{args.iters} "
                    f"text={sample['text_encoder_ms']:.1f}ms "
                    f"fusion={sample['conditioning_fusion_ms']:.1f}ms "
                    f"total={sample['total_apply_ms']:.1f}ms "
                    f"peak_alloc={peak_label}"
                )

        if stream is not None and args.ticks_between_prompts > 0:
            for tick_idx in range(args.ticks_between_prompts):
                stream.tick(
                    denoise=denoise,
                    seed=args.seed + idx + tick_idx + 10_000,
                    shift=shift,
                    noise_sharing=args.noise_share,
                )

    run_wall_ms = (time.perf_counter() - run_t0) * 1000
    text_stats = describe(measured_text_ms)
    fusion_stats = describe(measured_fusion_ms)
    swap_stats = describe(measured_apply_ms)
    total_stats = describe(measured_total_ms)
    peak_alloc_stats = describe(measured_peak_alloc_gb)
    peak_reserved_stats = describe(measured_peak_reserved_gb)
    delta_alloc_stats = describe(measured_delta_alloc_gb)
    delta_reserved_stats = describe(measured_delta_reserved_gb)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Prompt changes: {total_changes} ({args.warmup} warmup, {args.iters} measured)")
    print(f"Wall time: {run_wall_ms:.1f}ms")
    print()
    print(
        f"{'metric':<22s} {'mean':>9s} {'p50':>9s} {'p90':>9s} "
        f"{'p95':>9s} {'min':>9s} {'max':>9s}"
    )
    print("-" * 82)
    for label, stats in (
        ("text_encoder_ms", text_stats),
        ("conditioning_ms", fusion_stats),
        ("apply_swap_ms", swap_stats),
        ("total_apply_ms", total_stats),
    ):
        print(
            f"{label:<22s} "
            f"{fmt_stat(stats['mean_ms']):>9s} "
            f"{fmt_stat(stats['median_ms']):>9s} "
            f"{fmt_stat(stats['p90_ms']):>9s} "
            f"{fmt_stat(stats['p95_ms']):>9s} "
            f"{fmt_stat(stats['min_ms']):>9s} "
            f"{fmt_stat(stats['max_ms']):>9s}"
        )

    if torch.cuda.is_available():
        print()
        for label, stats in (
            ("peak_allocated_gb", peak_alloc_stats),
            ("peak_reserved_gb", peak_reserved_stats),
            ("delta_allocated_gb", delta_alloc_stats),
            ("delta_reserved_gb", delta_reserved_stats),
        ):
            print(
                f"{label:<22s} "
                f"mean={fmt_stat(stats['mean_ms'])} "
                f"p95={fmt_stat(stats['p95_ms'])} "
                f"max={fmt_stat(stats['max_ms'])}"
            )
        final_mem = cuda_memory_gb(cuda_memory())
        print(
            f"\nCUDA final: allocated={final_mem['allocated_gb']:.2f} GiB "
            f"reserved={final_mem['reserved_gb']:.2f} GiB"
        )

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "decoder_backend": decoder_backend,
            "vae_backend": vae_backend,
            "trt_profile_checkpoint": trt_profile_checkpoint,
            "picked_trt_profile_s": picked_dur,
            "trt_engines": trt_engines,
            "fast_vae": fast_vae,
            "audio": str(audio_path),
            "duration_s": actual_duration_s,
            "bpm": bpm,
            "key": key,
            "steps": steps,
            "depth": depth,
            "vae_window_s": vae_window,
            "offload_text_encoder": args.offload_text_encoder,
            "stream": args.stream,
            "prime_ticks": None if not args.stream else (depth if args.prime_ticks is None else args.prime_ticks),
            "ticks_between_prompts": args.ticks_between_prompts,
            "retain_conditionings": args.retain_conditionings,
            "unique_prompts": args.unique_prompts,
            "prompt_count": len(prompts),
        },
        "setup_timings_ms": setup_timings,
        "samples": samples,
        "stats": {
            "text_encoder": text_stats,
            "conditioning_fusion": fusion_stats,
            "apply_swap": swap_stats,
            "total_apply": total_stats,
            "peak_allocated_gb": peak_alloc_stats,
            "peak_reserved_gb": peak_reserved_stats,
            "delta_allocated_gb": delta_alloc_stats,
            "delta_reserved_gb": delta_reserved_stats,
            "run_wall_ms": run_wall_ms,
        },
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON metrics: {args.json}")


if __name__ == "__main__":
    main()
