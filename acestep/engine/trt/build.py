#!/usr/bin/env python3
"""Build TensorRT engines for the ACE-Step decoder and VAE.

Single entry point for all TRT engine creation.  Supports building individual
engines (fine-grained control) or the full matrix across durations.

ONNX exports are duration-agnostic and stored in a shared trt_engines/_onnx/
directory.  Existing ONNX files are auto-detected and reused; the model is
only loaded when an ONNX export is actually needed.

Usage:
    # Build the canonical engine matrix (60s + 120s + 240s, VAE + decoder,
    # refit + non-refit). Matches acestep.paths._TRT_ENGINE_PROFILES.
    python -m acestep.engine.trt.build --all

    # Build a single duration (e.g. just 120s):
    python -m acestep.engine.trt.build --all --duration 120

    # Build a custom subset:
    python -m acestep.engine.trt.build --all --duration 60 240

    # Build only decoders (skip VAE):
    python -m acestep.engine.trt.build --all --decoder-only

    # Preview what will be built:
    python -m acestep.engine.trt.build --all --dry-run

    # Force rebuild (existing engines are skipped by default):
    python -m acestep.engine.trt.build --all --force-rebuild

    # Single engine (fine-grained control):
    python -m acestep.engine.trt.build --max-duration 60
    python -m acestep.engine.trt.build --skip-vae --decoder --decoder-mixed --decoder-refit --max-duration 240

Requirements:
    - tensorrt (uv pip install tensorrt)
    - ACE-Step model checkpoint at checkpoints/acestep-v15-turbo
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

# Suppress flash_attn import (not needed for export)
import importlib, importlib.util
_orig = importlib.util.find_spec
def _patch(name, *a, **k):
    if "flash_attn" in str(name):
        return None
    return _orig(name, *a, **k)
importlib.util.find_spec = _patch

from loguru import logger
import torch

_SUPPORTED_TRT_MIN = (10, 16)
_SUPPORTED_TRT_MAX = (10, 17)
_ENGINE_METADATA_SCHEMA = 1


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _default_trt_dir() -> str:
    """Default TRT engine directory from acestep.paths."""
    # Import lazily to avoid circular deps at module level
    from acestep.paths import trt_engines_dir
    return str(trt_engines_dir())


def _default_checkpoints_dir() -> str:
    """Default checkpoints directory from acestep.paths."""
    from acestep.paths import checkpoints_dir
    return str(checkpoints_dir())


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    """Extract a comparable numeric prefix from versions like 10.16.1.11."""
    return tuple(int(p) for p in re.findall(r"\d+", version)[:3])


def _dist_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _nvidia_smi_summary() -> dict:
    """Best-effort driver/GPU snapshot for engine metadata."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({
                "name": parts[0],
                "driver_version": parts[1],
                "compute_capability": parts[2],
            })
    return {"available": True, "gpus": gpus}


def _active_gpu_summary(device: str) -> dict:
    if not torch.cuda.is_available():
        return {"available": False}

    torch_device = torch.device(device)
    index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "index": index,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
    }


def _preflight(device: str) -> dict:
    """Validate and log the TensorRT/CUDA stack before building engines."""
    import tensorrt as trt

    trt_version = trt.__version__
    parsed = _parse_version_tuple(trt_version)
    if parsed < _SUPPORTED_TRT_MIN or parsed >= _SUPPORTED_TRT_MAX:
        raise RuntimeError(
            "DEMON TensorRT builds target TensorRT >=10.16,<10.17; "
            f"found {trt_version}. Run `uv sync --upgrade-package tensorrt`."
        )

    env = {
        "schema_version": _ENGINE_METADATA_SCHEMA,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": {
            "tensorrt": trt_version,
            "tensorrt_cu13": _dist_version("tensorrt-cu13"),
            "tensorrt_cu13_bindings": _dist_version("tensorrt-cu13-bindings"),
            "tensorrt_cu13_libs": _dist_version("tensorrt-cu13-libs"),
            "cuda_python": _dist_version("cuda-python"),
            "cuda_toolkit": _dist_version("cuda-toolkit"),
            "polygraphy": _dist_version("polygraphy"),
            "onnx": _dist_version("onnx"),
            "torch": torch.__version__,
        },
        "torch_cuda": torch.version.cuda,
        "onnx_parser_version": getattr(trt, "get_nv_onnx_parser_version", lambda: None)(),
        "active_gpu": _active_gpu_summary(device),
        "nvidia_smi": _nvidia_smi_summary(),
    }

    logger.info("=" * 60)
    logger.info("TensorRT build preflight")
    logger.info("=" * 60)
    logger.info("TensorRT: {}", env["packages"]["tensorrt"])
    logger.info("TensorRT cu13: {}", env["packages"]["tensorrt_cu13"])
    logger.info("CUDA Python: {}", env["packages"]["cuda_python"])
    logger.info("CUDA toolkit wheel: {}", env["packages"]["cuda_toolkit"])
    logger.info("Polygraphy: {}", env["packages"]["polygraphy"])
    logger.info("ONNX: {}", env["packages"]["onnx"])
    logger.info("Torch: {} (CUDA {})", env["packages"]["torch"], env["torch_cuda"])
    gpu = env["active_gpu"]
    if gpu.get("available"):
        logger.info(
            "Active GPU: cuda:{} {} (SM {})",
            gpu["index"], gpu["name"], gpu["compute_capability"],
        )
    else:
        logger.warning("No active CUDA GPU detected in torch")
    return env


def _sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_dict(config) -> dict:
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def _metadata_path(engine_path: str | os.PathLike[str]) -> Path:
    return Path(str(engine_path) + ".metadata.json")


def _expected_metadata(
    *,
    component: str,
    onnx_path: str,
    config,
    env: dict,
) -> dict:
    gpu = env.get("active_gpu", {})
    return {
        "schema_version": _ENGINE_METADATA_SCHEMA,
        "component": component,
        "tensorrt_version": env["packages"]["tensorrt"],
        "gpu_compute_capability": gpu.get("compute_capability"),
        "gpu_name": gpu.get("name"),
        "config": _config_dict(config),
        "onnx_path": str(Path(onnx_path).resolve()),
        "onnx_sha256": _sha256_file(onnx_path),
    }


def _write_metadata(
    *,
    engine_path: str,
    expected: dict,
    env: dict,
) -> None:
    payload = dict(expected)
    payload["built_at"] = datetime.now(timezone.utc).isoformat()
    payload["environment"] = env
    path = _metadata_path(engine_path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Engine metadata saved to {}", path)


def _metadata_matches(engine_path: str, expected: dict) -> tuple[bool, str]:
    path = _metadata_path(engine_path)
    if not path.exists():
        return False, "missing metadata"
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"metadata unreadable: {exc}"

    for key in (
        "schema_version",
        "component",
        "tensorrt_version",
        "gpu_compute_capability",
        "config",
        "onnx_sha256",
    ):
        if actual.get(key) != expected.get(key):
            return False, f"metadata mismatch: {key}"
    return True, "metadata match"


def _verify_engines(engine_paths: list[tuple[str, str]]):
    """Load and print I/O info for each engine."""
    import tensorrt as trt

    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    for name, path in engine_paths:
        if not os.path.exists(path):
            logger.error("  {}: MISSING ({})", name, path)
            continue
        with open(path, "rb") as f:
            engine = rt.deserialize_cuda_engine(f.read())
        if engine is None:
            logger.error("  {}: FAILED to load", name)
            continue

        io_info = []
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(tname)
            shape = engine.get_tensor_shape(tname)
            label = "IN" if mode == trt.TensorIOMode.INPUT else "OUT"
            io_info.append(f"{label}: {tname} {shape}")

        profiles = []
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tname) == trt.TensorIOMode.INPUT:
                shapes = engine.get_tensor_profile_shape(tname, 0)
                profiles.append(f"{tname}: min={shapes[0]} opt={shapes[1]} max={shapes[2]}")

        size_mb = os.path.getsize(path) / 1e6
        logger.info("  {}: OK ({:.1f} MB)", name, size_mb)
        for s in io_info:
            logger.info("    {}", s)
        for s in profiles:
            logger.info("    Profile: {}", s)


def _engine_path(output_dir: str, engine_filename: str) -> str:
    """Resolve engine path: trt_engines/<name>/<name>.engine."""
    name = engine_filename.replace(".engine", "")
    return os.path.join(output_dir, name, engine_filename)


# ------------------------------------------------------------------
# ONNX setup
# ------------------------------------------------------------------


def _ensure_onnx(
    *,
    onnx_dir: str,
    project_root: str,
    checkpoint: str,
    device: str,
    need_vae: bool,
    need_decoder_std: bool,
    need_decoder_refit: bool,
    decoder_mixed: bool,
    skip_onnx: bool,
    force_onnx: bool = False,
) -> dict[str, str]:
    """Detect existing ONNX, load model if needed, export missing files.

    Returns dict mapping component names to ONNX paths.

    VAE ONNX exports are stored in a shared ``_onnx_vae/`` directory
    (sibling to onnx_dir) since all DiT variants share the same VAE.
    Decoder ONNX exports live in onnx_dir (checkpoint-specific).
    """
    # VAE is shared across checkpoints; decoder is checkpoint-specific
    vae_onnx_dir = os.path.join(os.path.dirname(onnx_dir), "_onnx_vae")
    os.makedirs(vae_onnx_dir, exist_ok=True)

    paths = {
        "vae_encode": os.path.join(vae_onnx_dir, "vae_encode", "vae_encode.onnx"),
        "vae_decode": os.path.join(vae_onnx_dir, "vae_decode", "vae_decode.onnx"),
        "decoder": os.path.join(onnx_dir, "decoder", "decoder.onnx"),
        "decoder_refit": os.path.join(onnx_dir, "decoder_refit", "decoder_refit.onnx"),
    }

    # Also check old _onnx/ location for VAE (backward compat)
    old_onnx_dir = os.path.join(os.path.dirname(onnx_dir), "_onnx")
    for key in ("vae_encode", "vae_decode"):
        if not os.path.exists(paths[key]):
            old_path = os.path.join(old_onnx_dir, key, f"{key}.onnx")
            if os.path.exists(old_path):
                logger.info("Found VAE ONNX at old location: {}", old_path)
                paths[key] = old_path

    # Determine what actually needs exporting
    export_vae = False
    export_decoder_std = False
    export_decoder_refit = False

    if need_vae and not skip_onnx:
        if force_onnx or not os.path.exists(paths["vae_encode"]) or not os.path.exists(paths["vae_decode"]):
            export_vae = True
            if force_onnx:
                logger.info("Forcing VAE ONNX re-export")
        else:
            logger.info("Reusing existing VAE ONNX exports in {}", onnx_dir)

    if need_decoder_std and not skip_onnx:
        if force_onnx or not os.path.exists(paths["decoder"]):
            export_decoder_std = True
            if force_onnx:
                logger.info("Forcing decoder ONNX re-export: {}", paths["decoder"])
        else:
            logger.info("Reusing existing decoder ONNX: {}", paths["decoder"])

    if need_decoder_refit and not skip_onnx:
        if force_onnx or not os.path.exists(paths["decoder_refit"]):
            export_decoder_refit = True
            if force_onnx:
                logger.info("Forcing decoder ONNX re-export (refit): {}", paths["decoder_refit"])
        else:
            logger.info("Reusing existing decoder ONNX (refit): {}", paths["decoder_refit"])

    # Validate --skip-onnx
    if skip_onnx:
        missing = []
        if need_vae:
            for k in ("vae_encode", "vae_decode"):
                if not os.path.exists(paths[k]):
                    missing.append(paths[k])
        if need_decoder_std and not os.path.exists(paths["decoder"]):
            missing.append(paths["decoder"])
        if need_decoder_refit and not os.path.exists(paths["decoder_refit"]):
            missing.append(paths["decoder_refit"])
        if missing:
            for f in missing:
                logger.error("Missing ONNX file: {}", f)
            sys.exit(1)
        logger.info("Skipping ONNX export (--skip-onnx)")

    # Load model only if we need to export something
    need_model = export_vae or export_decoder_std or export_decoder_refit

    handler = None
    if need_model:
        logger.info("Loading model from checkpoints/{}...", checkpoint)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from acestep.engine.model_context import ModelContext
        handler = ModelContext(
            project_root=project_root,
            config_path=checkpoint,
            device=device,
            use_flash_attention=False,
            compile_decoder=False,
            compile_vae=False,
            skip_vae=not need_vae,
        )
        logger.info("Model loaded.")
    else:
        logger.info("All ONNX exports found, skipping model load.")

    # Export missing ONNX files
    if export_vae:
        from .vae_export import (
            export_vae_encoder_onnx, export_vae_decoder_onnx, VAEExportConfig,
        )
        logger.info("=" * 60)
        logger.info("VAE ONNX EXPORT")
        logger.info("=" * 60)
        with handler._load_model_context("vae"):
            t0 = time.time()
            export_vae_encoder_onnx(
                handler.vae, paths["vae_encode"], device=device,
                config=VAEExportConfig(trace_audio_samples=48000 * 30),
            )
            logger.info("VAE encoder exported in {:.1f}s", time.time() - t0)

            t0 = time.time()
            export_vae_decoder_onnx(
                handler.vae, paths["vae_decode"], device=device,
                config=VAEExportConfig(trace_latent_frames=750),
            )
            logger.info("VAE decoder exported in {:.1f}s", time.time() - t0)

    if export_decoder_refit or export_decoder_std:
        from .export import OnnxExportConfig, export_decoder_onnx
        with handler._load_model_context("model"):
            if export_decoder_refit:
                logger.info("=" * 60)
                logger.info("DECODER ONNX EXPORT (refit-enabled)")
                logger.info("=" * 60)
                t0 = time.time()
                export_decoder_onnx(
                    handler.model, paths["decoder_refit"], device=device,
                    config=OnnxExportConfig(mixed_precision=decoder_mixed, for_refit=True),
                )
                logger.info("Decoder ONNX (refit) exported in {:.1f}s", time.time() - t0)

            if export_decoder_std:
                logger.info("=" * 60)
                logger.info("DECODER ONNX EXPORT (standard)")
                logger.info("=" * 60)
                t0 = time.time()
                export_decoder_onnx(
                    handler.model, paths["decoder"], device=device,
                    config=OnnxExportConfig(mixed_precision=decoder_mixed, for_refit=False),
                )
                logger.info("Decoder ONNX (standard) exported in {:.1f}s", time.time() - t0)

    # Free model memory before TRT builds
    del handler
    torch.cuda.empty_cache()

    return paths


# ------------------------------------------------------------------
# Engine builders
# ------------------------------------------------------------------

def _build_vae_engines(
    *,
    output_dir: str,
    onnx_paths: dict[str, str],
    duration: int,
    workspace_gb: float,
    env: dict,
    force_rebuild: bool = False,
) -> list[tuple[str, str, float, str]]:
    """Build VAE encode + decode TRT engines for one duration.

    Returns list of (label, engine_path, elapsed_seconds, status).
    Existing engines are skipped unless force_rebuild is True.
    """
    from .vae_export import (
        build_vae_decode_engine, build_vae_encode_engine, VAETRTBuildConfig,
    )

    config = VAETRTBuildConfig(
        workspace_gb=workspace_gb,
        decode_max_frames=duration * 25,
        encode_max_samples=duration * 48000,
    )

    results = []
    for component, builder in [
        ("vae_decode", build_vae_decode_engine),
        ("vae_encode", build_vae_encode_engine),
    ]:
        name = config.engine_filename(component).replace(".engine", "")
        engine_dir = os.path.join(output_dir, name)
        engine_path = os.path.join(engine_dir, f"{name}.engine")

        label = f"VAE {component.split('_')[1]} {duration}s"
        expected_metadata = _expected_metadata(
            component=component,
            onnx_path=onnx_paths[component],
            config=config,
            env=env,
        )

        if not force_rebuild and os.path.exists(engine_path):
            matches, reason = _metadata_matches(engine_path, expected_metadata)
            if matches:
                size_mb = os.path.getsize(engine_path) / 1e6
                logger.info("SKIP {} ({:.0f} MB, {})", name, size_mb, reason)
                results.append((label, engine_path, 0.0, "SKIPPED"))
                continue
            logger.info("REBUILD {} ({})", name, reason)

        logger.info("=" * 60)
        logger.info("VAE TRT BUILD: {} (max_duration={}s)", name, duration)
        logger.info("=" * 60)

        t0 = time.time()
        builder(onnx_paths[component], engine_path, config=config)
        _write_metadata(engine_path=engine_path, expected=expected_metadata, env=env)
        elapsed = time.time() - t0
        logger.info("Built in {:.0f}s", elapsed)
        results.append((label, engine_path, elapsed, "OK"))

    return results


def _checkpoint_to_variant(checkpoint: str) -> str:
    """Extract short variant name from checkpoint path.

    'acestep-v15-turbo' -> 'turbo'
    'acestep-v15-base'  -> 'base'
    'acestep-v15-sft'   -> 'sft'
    """
    name = os.path.basename(checkpoint)
    # Strip the common 'acestep-v15-' prefix
    prefix = "acestep-v15-"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _build_decoder_engine(
    *,
    output_dir: str,
    onnx_paths: dict[str, str],
    duration: int,
    mixed: bool,
    refit: bool,
    workspace_gb: float,
    batch_max: int,
    env: dict,
    force_rebuild: bool = False,
    checkpoint: str = "acestep-v15-turbo",
) -> tuple[str, str, float, str]:
    """Build one decoder TRT engine.

    Returns (label, engine_path, elapsed_seconds, status).
    Existing engines are skipped unless force_rebuild is True.
    """
    from .export import build_trt_engine, TRTBuildConfig

    variant = _checkpoint_to_variant(checkpoint)
    config = TRTBuildConfig(
        fp16=True,
        strongly_typed=mixed,
        refit=refit,
        workspace_gb=workspace_gb,
        batch_max=batch_max,
        seq_max=duration * 25,
        variant=variant,
    )

    name = config.engine_filename().replace(".engine", "")
    engine_dir = os.path.join(output_dir, name)
    engine_path = os.path.join(engine_dir, f"{name}.engine")

    onnx_key = "decoder_refit" if refit else "decoder"
    refit_label = "refit" if refit else "no-refit"
    label = f"Decoder {variant} {duration}s, {refit_label}"
    expected_metadata = _expected_metadata(
        component=onnx_key,
        onnx_path=onnx_paths[onnx_key],
        config=config,
        env=env,
    )

    if not force_rebuild and os.path.exists(engine_path):
        matches, reason = _metadata_matches(engine_path, expected_metadata)
        if matches:
            size_mb = os.path.getsize(engine_path) / 1e6
            logger.info("SKIP {} ({:.0f} MB, {})", name, size_mb, reason)
            return (label, engine_path, 0.0, "SKIPPED")
        logger.info("REBUILD {} ({})", name, reason)

    logger.info("=" * 60)
    logger.info("DECODER TRT BUILD (refit={}, mixed={}) -> {}",
                refit, mixed, engine_path)
    logger.info("=" * 60)

    t0 = time.time()
    build_trt_engine(onnx_paths[onnx_key], engine_path, config=config)
    _write_metadata(engine_path=engine_path, expected=expected_metadata, env=env)
    elapsed = time.time() - t0
    logger.info("Built in {:.0f}s", elapsed)

    return (label, engine_path, elapsed, "OK")


# ------------------------------------------------------------------
# Batch mode (--all)
# ------------------------------------------------------------------

def _print_matrix(durations, build_vae, build_decoder, output_dir, batch_max,
                   checkpoint="acestep-v15-turbo"):
    """Print the build matrix for --all mode, showing existing vs new."""
    variant = _checkpoint_to_variant(checkpoint)
    vtag = f"_{variant}" if variant != "turbo" else ""

    # (label, engine_dir_name) pairs
    jobs = []
    for dur in durations:
        if build_vae:
            jobs.append((f"VAE decode {dur}s", f"vae_decode_fp16_{dur}s"))
            jobs.append((f"VAE encode {dur}s", f"vae_encode_fp16_{dur}s"))
        if build_decoder:
            jobs.append((f"Decoder {variant} {dur}s, refit", f"decoder{vtag}_mixed_refit_b{batch_max}_{dur}s"))

    to_build = 0
    to_skip = 0
    lines = []
    for label, dir_name in jobs:
        engine_file = os.path.join(output_dir, dir_name, f"{dir_name}.engine")
        if os.path.exists(engine_file):
            size_mb = os.path.getsize(engine_file) / 1e6
            lines.append(f"  [exists]  {label}  ({size_mb:.0f} MB)")
            to_skip += 1
        else:
            lines.append(f"  [build]   {label}")
            to_build += 1

    print(f"\nBuild matrix: {to_build} to build, {to_skip} existing (skipped)")
    for line in lines:
        print(line)
    print()
    return jobs


def _print_summary(results, output_dir):
    """Print build summary and list engines on disk."""
    print(f"\n{'=' * 60}")
    print("BUILD SUMMARY")
    print(f"{'=' * 60}")
    for label, path, elapsed, status in results:
        print(f"  {status:7s} {elapsed:6.0f}s  {label}")

    failures = sum(1 for _, _, _, s in results if s == "FAILED")
    if failures:
        print(f"\n{failures} build(s) FAILED")
    else:
        active = sum(1 for _, _, _, s in results if s != "SKIPPED")
        skipped = sum(1 for _, _, _, s in results if s == "SKIPPED")
        parts = [f"{active} built"]
        if skipped:
            parts.append(f"{skipped} skipped")
        print(f"\nAll done ({', '.join(parts)}).")

    # List engine files on disk
    from pathlib import Path
    trt_dir = Path(output_dir)
    print(f"\nEngines in {trt_dir}:")
    for d in sorted(trt_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        engine_file = d / f"{d.name}.engine"
        if engine_file.exists():
            size_mb = engine_file.stat().st_size / 1e6
            print(f"  {d.name + '/':50s} {size_mb:8.1f} MB")

    return failures


def _save_build_report(results, output_dir):
    """Append CSV build report to trt_engines/build_report.csv."""
    import csv
    from datetime import datetime

    report_path = os.path.join(output_dir, "build_report.csv")
    write_header = not os.path.exists(report_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "engine", "status", "build_time_s", "size_mb"])
        for label, path, elapsed, status in results:
            size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else -1
            writer.writerow([timestamp, label, status, f"{elapsed:.1f}", f"{size_mb:.1f}"])

    print(f"Build report appended to: {report_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build ACE-Step TRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Batch mode
    batch = parser.add_argument_group("batch mode (--all)")
    batch.add_argument("--all", action="store_true",
                       help="Build full engine matrix (VAE + decoder, "
                            "refit + non-refit, across durations)")
    batch.add_argument("--duration", nargs="*", type=int, default=None,
                       help="Duration(s) in seconds for --all mode "
                            "(default: 60 120 240 — the canonical profile set "
                            "registered in acestep.paths._TRT_ENGINE_PROFILES)")
    batch.add_argument("--force-rebuild", action="store_true",
                       help="Rebuild engines even if they already exist "
                            "(default: skip existing engines)")
    batch.add_argument("--dry-run", action="store_true",
                       help="Print build matrix without building")
    batch.add_argument("--decoder-only", action="store_true",
                       help="Only build decoder engines (skip VAE)")
    batch.add_argument("--vae-only", action="store_true",
                       help="Only build VAE engines (skip decoder)")

    # Shared / single mode
    single = parser.add_argument_group("single mode / shared options")
    single.add_argument("--output-dir",
                        default=_default_trt_dir(),
                        help="Directory for ONNX and engine files "
                             "(default: ~/.daydream-scope/models/demon/trt_engines)")
    single.add_argument("--checkpoint", default="acestep-v15-turbo",
                        help="Model checkpoint directory name")
    single.add_argument("--skip-onnx", action="store_true",
                        help="Force-skip ONNX export (error if files missing). "
                             "Normally ONNX files in _onnx/ are auto-detected "
                             "and reused without this flag.")
    single.add_argument("--force-onnx", action="store_true",
                        help="Re-export ONNX files even if matching files already exist.")
    single.add_argument("--max-duration", type=int, default=240,
                        help="Max audio duration in seconds for single mode "
                             "(default: 240 = 4min)")
    single.add_argument("--device", default="cuda")
    single.add_argument("--workspace-gb", type=float, default=16.0,
                        help="TRT builder workspace in GB (default: 16)")
    single.add_argument("--decoder", action="store_true",
                        help="Build decoder engine(s)")
    single.add_argument("--decoder-mixed", action="store_true",
                        help="Use mixed precision for decoder")
    single.add_argument("--decoder-refit",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Build refit-enabled decoder for LoRA "
                             "(default: True, use --no-decoder-refit)")
    single.add_argument("--batch-max", type=int, default=8,
                        help="Max batch size for decoder (default: 8)")
    single.add_argument("--skip-vae", action="store_true",
                        help="Skip VAE engine build")

    args = parser.parse_args()
    if args.skip_onnx and args.force_onnx:
        parser.error("--skip-onnx and --force-onnx are mutually exclusive")

    checkpoints_root = _default_checkpoints_dir()
    env = None if args.dry_run else _preflight(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    # ONNX directory is checkpoint-specific for decoder (different weights)
    # but shared for VAE (same weights across all DiT variants).
    onnx_dir = os.path.join(args.output_dir, f"_onnx_{args.checkpoint}")
    os.makedirs(onnx_dir, exist_ok=True)

    if args.all:
        _run_all(args, checkpoints_root, onnx_dir, env)
    else:
        _run_single(args, checkpoints_root, onnx_dir, env)


def _run_all(args, project_root, onnx_dir, env):
    """Build the full engine matrix."""
    durations = tuple(args.duration) if args.duration else (60, 120, 240)
    build_vae = not args.decoder_only
    build_decoder = not args.vae_only

    # Print matrix
    _print_matrix(durations, build_vae, build_decoder,
                  args.output_dir, args.batch_max, args.checkpoint)

    if args.dry_run:
        return

    # ONNX phase (once, shared across all durations)
    # Only refit-enabled decoder ONNX is needed
    onnx_paths = _ensure_onnx(
        onnx_dir=onnx_dir,
        project_root=project_root,
        checkpoint=args.checkpoint,
        device=args.device,
        need_vae=build_vae,
        need_decoder_std=False,
        need_decoder_refit=build_decoder,
        decoder_mixed=True,
        skip_onnx=args.skip_onnx,
        force_onnx=args.force_onnx,
    )

    # Engine phase
    results = []
    for dur in durations:
        if build_vae:
            results.extend(_build_vae_engines(
                output_dir=args.output_dir,
                onnx_paths=onnx_paths,
                duration=dur,
                workspace_gb=args.workspace_gb,
                env=env,
                force_rebuild=args.force_rebuild,
            ))
        if build_decoder:
            results.append(_build_decoder_engine(
                output_dir=args.output_dir,
                onnx_paths=onnx_paths,
                duration=dur,
                mixed=True,
                refit=True,
                workspace_gb=args.workspace_gb,
                batch_max=args.batch_max,
                env=env,
                force_rebuild=args.force_rebuild,
                checkpoint=args.checkpoint,
            ))

    # Summary
    failures = _print_summary(results, args.output_dir)
    _save_build_report(results, args.output_dir)

    if failures:
        sys.exit(1)


def _run_single(args, project_root, onnx_dir, env):
    """Build a single engine configuration."""
    build_vae = not args.skip_vae
    build_decoder = args.decoder

    # ONNX phase
    onnx_paths = _ensure_onnx(
        onnx_dir=onnx_dir,
        project_root=project_root,
        checkpoint=args.checkpoint,
        device=args.device,
        need_vae=build_vae,
        need_decoder_std=build_decoder and not args.decoder_refit,
        need_decoder_refit=build_decoder and args.decoder_refit,
        decoder_mixed=args.decoder_mixed,
        skip_onnx=args.skip_onnx,
        force_onnx=args.force_onnx,
    )

    # Engine phase
    built_engines = []

    if build_vae:
        results = _build_vae_engines(
            output_dir=args.output_dir,
            onnx_paths=onnx_paths,
            duration=args.max_duration,
            workspace_gb=args.workspace_gb,
            env=env,
            force_rebuild=args.force_rebuild,
        )
        for label, path, elapsed, status in results:
            if status == "OK":
                built_engines.append((label, path))

    if build_decoder:
        result = _build_decoder_engine(
            output_dir=args.output_dir,
            onnx_paths=onnx_paths,
            duration=args.max_duration,
            mixed=args.decoder_mixed,
            refit=args.decoder_refit,
            workspace_gb=args.workspace_gb,
            batch_max=args.batch_max,
            env=env,
            force_rebuild=args.force_rebuild,
            checkpoint=args.checkpoint,
        )
        label, path, elapsed, status = result
        if status == "OK":
            built_engines.append((label, path))

    # Verify
    if built_engines:
        logger.info("=" * 60)
        logger.info("VERIFICATION")
        logger.info("=" * 60)
        _verify_engines(built_engines)

    logger.info("=" * 60)
    logger.info("Built {} engine(s):", len(built_engines))
    for name, path in built_engines:
        logger.info("  {} -> {}", name, path)
    logger.info("Output directory: {}", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
