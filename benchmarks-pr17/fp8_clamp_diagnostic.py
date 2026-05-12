"""Find which initializers got clamped to ~6e-08 during the FP8 patch.

Compares the bf16 source ONNX against the post-bf16->fp16 sibling that
ModelOpt was handed for quantize. Identifies:
  - exact zeros in bf16 that became >0 in fp16
  - bf16 denormals (|x| < fp16 min subnormal) that ended up clamped
  - fp16 values exactly equal to the suspected floor (5.96e-8)

For each suspect initializer, the graph consumers are listed so we can
tell whether the value is on a hot path (MatMul weight, residual Add,
gate Mul) or a benign one (LayerNorm bias, unused tensor).

Run:
    python benchmarks-pr17/fp8_clamp_diagnostic.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
import torch


ONNX_DIR = Path(
    r"C:\Users\ryanf\.daydream-scope\models\demon\trt_engines"
    r"\_onnx_acestep-v15-xl-turbo\decoder_refit"
)
BF16_SRC = ONNX_DIR / "decoder_refit_dynbatch.onnx"
FP16_DST = ONNX_DIR / "decoder_refit_dynbatch_fp16.onnx"
OUT_JSON = Path(__file__).parent / "fp8_clamp_diagnostic.json"

# fp16 limits
FP16_MIN_SUBNORMAL = 2.0 ** -24       # ~5.96e-08
FP16_MIN_NORMAL = 2.0 ** -14          # ~6.10e-05
# The "6e-08" floor reported by ModelOpt — actually one ULP below min subnormal.
FLOOR_LOW = FP16_MIN_SUBNORMAL * 0.5
FLOOR_HIGH = FP16_MIN_SUBNORMAL * 1.5  # accept ~5.96e-08 with rounding noise


def _init_to_fp32(init) -> np.ndarray:
    """Return a fp32 numpy view of an initializer's values."""
    if init.data_type == onnx.TensorProto.BFLOAT16:
        if not init.raw_data:
            raise NotImplementedError(f"bf16 init {init.name} has no raw_data")
        t = torch.frombuffer(bytearray(init.raw_data), dtype=torch.bfloat16)
        return t.to(torch.float32).numpy().reshape(tuple(init.dims))
    if init.data_type == onnx.TensorProto.FLOAT16:
        if init.raw_data:
            return np.frombuffer(init.raw_data, dtype=np.float16).astype(np.float32).reshape(tuple(init.dims))
        return np.asarray(init.int32_data, dtype=np.int32).view(np.float16).astype(np.float32).reshape(tuple(init.dims))
    if init.data_type == onnx.TensorProto.FLOAT:
        if init.raw_data:
            return np.frombuffer(init.raw_data, dtype=np.float32).reshape(tuple(init.dims))
        return np.asarray(init.float_data, dtype=np.float32).reshape(tuple(init.dims))
    raise NotImplementedError(f"unsupported init dtype {init.data_type} ({init.name})")


def main() -> None:
    print(f"[load] bf16 source: {BF16_SRC}")
    bf_model = onnx.load(str(BF16_SRC), load_external_data=True)
    print(f"[load] fp16 sibling: {FP16_DST}")
    fp_model = onnx.load(str(FP16_DST), load_external_data=True)

    bf_inits = {i.name: i for i in bf_model.graph.initializer}
    fp_inits = {i.name: i for i in fp_model.graph.initializer}
    print(f"  bf16 initializers: {len(bf_inits)}")
    print(f"  fp16 initializers: {len(fp_inits)}")

    common = sorted(set(bf_inits) & set(fp_inits))
    print(f"  shared by name: {len(common)}")
    bf_only = sorted(set(bf_inits) - set(fp_inits))
    fp_only = sorted(set(fp_inits) - set(bf_inits))
    if bf_only:
        print(f"  bf16-only ({len(bf_only)}): first 3 = {bf_only[:3]}")
    if fp_only:
        print(f"  fp16-only ({len(fp_only)}): first 3 = {fp_only[:3]}")

    suspects = []
    summary = {
        "checked": 0,
        "bf16_skipped_dtype": Counter(),
        "any_clamp_count": 0,
        "any_zero_to_floor_count": 0,
    }

    for name in common:
        bf_init = bf_inits[name]
        fp_init = fp_inits[name]
        if bf_init.data_type != onnx.TensorProto.BFLOAT16:
            summary["bf16_skipped_dtype"][int(bf_init.data_type)] += 1
            continue
        if fp_init.data_type != onnx.TensorProto.FLOAT16:
            # Unexpected dtype mismatch — flag it but don't analyze.
            summary["bf16_skipped_dtype"][int(fp_init.data_type)] += 1
            continue

        try:
            bf_arr = _init_to_fp32(bf_init).ravel()
            fp_arr = _init_to_fp32(fp_init).ravel()
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        if bf_arr.shape != fp_arr.shape:
            print(f"  [skip] shape mismatch {name}: {bf_arr.shape} vs {fp_arr.shape}")
            continue

        summary["checked"] += 1

        abs_bf = np.abs(bf_arr)
        abs_fp = np.abs(fp_arr)

        # Boolean masks
        bf_zero = (bf_arr == 0.0)
        # bf16 denormals from fp16's perspective: nonzero but smaller than fp16 subnormal
        bf_denormal = (~bf_zero) & (abs_bf < FP16_MIN_SUBNORMAL)
        # fp16 values within [0.5 ulp, 1.5 ulp] of the min subnormal — i.e. clamped to floor
        fp_at_floor = (abs_fp >= FLOOR_LOW) & (abs_fp <= FLOOR_HIGH)
        # fp16 exact zero
        fp_zero = (fp_arr == 0.0)

        # Three things we care about:
        #  - exact bf16 zeros that became nonzero at the floor (bug)
        #  - bf16 denormals that shifted UP to the floor (potential bug)
        #  - fp16 ended up at floor regardless of bf16 source (anything at floor)
        zero_to_floor = bf_zero & fp_at_floor
        denorm_to_floor = bf_denormal & fp_at_floor
        any_at_floor = fp_at_floor

        n_zero_to_floor = int(zero_to_floor.sum())
        n_denorm_to_floor = int(denorm_to_floor.sum())
        n_any_at_floor = int(any_at_floor.sum())

        if n_any_at_floor == 0 and n_zero_to_floor == 0 and n_denorm_to_floor == 0:
            continue

        summary["any_clamp_count"] += 1
        if n_zero_to_floor:
            summary["any_zero_to_floor_count"] += 1

        # Sample some affected values for human inspection.
        if n_zero_to_floor:
            sample_idx = np.where(zero_to_floor)[0][:5].tolist()
            zero_samples = [(int(i), float(bf_arr[i]), float(fp_arr[i])) for i in sample_idx]
        else:
            zero_samples = []
        if n_denorm_to_floor:
            sample_idx = np.where(denorm_to_floor)[0][:5].tolist()
            denorm_samples = [(int(i), float(bf_arr[i]), float(fp_arr[i])) for i in sample_idx]
        else:
            denorm_samples = []

        suspects.append({
            "name": name,
            "shape": list(bf_init.dims),
            "numel": int(bf_arr.size),
            "bf16_zero_count": int(bf_zero.sum()),
            "bf16_denormal_count": int(bf_denormal.sum()),
            "fp16_at_floor_count": n_any_at_floor,
            "fp16_zero_count": int(fp_zero.sum()),
            "zero_to_floor_count": n_zero_to_floor,
            "denorm_to_floor_count": n_denorm_to_floor,
            "zero_to_floor_frac": n_zero_to_floor / bf_arr.size,
            "any_at_floor_frac": n_any_at_floor / bf_arr.size,
            "bf16_min_nonzero": float(abs_bf[abs_bf > 0].min()) if (abs_bf > 0).any() else 0.0,
            "bf16_max_abs": float(abs_bf.max()),
            "fp16_min_nonzero": float(abs_fp[abs_fp > 0].min()) if (abs_fp > 0).any() else 0.0,
            "zero_to_floor_samples": zero_samples,
            "denorm_to_floor_samples": denorm_samples,
        })

    # Rank suspects by total floor occurrences.
    suspects.sort(key=lambda s: (s["zero_to_floor_count"], s["fp16_at_floor_count"]), reverse=True)

    print()
    print("=" * 70)
    print(f"Checked {summary['checked']} shared bf16->fp16 initializers")
    print(f"  with ANY value at fp16 floor:           {summary['any_clamp_count']}")
    print(f"  with bf16-zero -> fp16-floor transitions: {summary['any_zero_to_floor_count']}")
    print(f"  skipped dtype counts: {dict(summary['bf16_skipped_dtype'])}")

    print()
    print(f"Top {min(20, len(suspects))} suspect initializers by zero-to-floor count:")
    print(f"  {'name':<60s} {'numel':>10s} {'z->floor':>10s} {'any@floor':>10s} {'frac z->floor':>14s}")
    for s in suspects[:20]:
        print(
            f"  {s['name'][:60]:<60s} {s['numel']:>10d} "
            f"{s['zero_to_floor_count']:>10d} {s['fp16_at_floor_count']:>10d} "
            f"{s['zero_to_floor_frac']:>14.4%}"
        )

    OUT_JSON.write_text(json.dumps({
        "summary": {
            **summary,
            "bf16_skipped_dtype": dict(summary["bf16_skipped_dtype"]),
        },
        "suspects": suspects,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
