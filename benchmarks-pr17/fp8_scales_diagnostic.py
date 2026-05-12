"""Inspect the FP8 ONNX for direct NaN sources.

Three things, in order:

1. For each fp16 initializer that exists in BOTH the fp16 input and the
   FP8 output, check whether ModelOpt clamped zeros (or denormals) to
   the 6e-08 floor. This is the hypothesis we couldn't confirm by
   comparing bf16->fp16 alone.

2. Walk every DequantizeLinear node. For its scale initializer, check
   for zeros, denormals, NaN, Inf, and extreme outliers (any scale
   smaller than 1e-4 is suspect: dequant divides by it, so a
   per-channel scale of 1e-7 turns 100 into 1e9).

3. Walk every QuantizeLinear / DequantizeLinear FP8 initializer. Check
   for NaN/Inf bit patterns in the FP8 weights themselves, and how
   close the maximum |weight * scale| sits to FP8 E4M3 saturation
   (448.0). Heavy saturation on critical layers is another classic
   NaN-through-residual cause.

Run:
    python benchmarks-pr17/fp8_scales_diagnostic.py
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
FP16_SRC = ONNX_DIR / "decoder_refit_dynbatch_fp16.onnx"
FP8_DST = ONNX_DIR / "decoder_refit_dynbatch_fp8.onnx"
OUT_JSON = Path(__file__).parent / "fp8_scales_diagnostic.json"

# fp16 limits
FP16_MIN_SUBNORMAL = 2.0 ** -24       # ~5.96e-08
FP16_MIN_NORMAL = 2.0 ** -14          # ~6.10e-05
FLOOR_LOW = FP16_MIN_SUBNORMAL * 0.5
FLOOR_HIGH = FP16_MIN_SUBNORMAL * 1.5
FP8_E4M3_MAX = 448.0


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


def _fp8_e4m3_to_fp32(init) -> np.ndarray:
    """Decode an FP8 E4M3 initializer to fp32 via torch."""
    if init.data_type != onnx.TensorProto.FLOAT8E4M3FN:
        raise ValueError(f"not FP8 E4M3FN: {init.name}")
    if not init.raw_data:
        raise NotImplementedError(f"FP8 init {init.name} has no raw_data")
    # PyTorch supports torch.float8_e4m3fn since 2.1+.
    t = torch.frombuffer(bytearray(init.raw_data), dtype=torch.float8_e4m3fn)
    return t.to(torch.float32).numpy().reshape(tuple(init.dims))


def main() -> None:
    print(f"[load] fp16 input:  {FP16_SRC}")
    fp_model = onnx.load(str(FP16_SRC), load_external_data=True)
    print(f"[load] fp8 output:  {FP8_DST}")
    q_model = onnx.load(str(FP8_DST), load_external_data=True)

    fp_inits = {i.name: i for i in fp_model.graph.initializer}
    q_inits = {i.name: i for i in q_model.graph.initializer}
    print(f"  fp16 inits: {len(fp_inits)}  |  fp8-graph inits: {len(q_inits)}")

    q_dtype_counts = Counter(int(i.data_type) for i in q_model.graph.initializer)
    print(f"  fp8-graph initializer dtype histogram: {dict(q_dtype_counts)}")
    print(f"    (1=FLOAT32, 10=FLOAT16, 16=BFLOAT16, 17=FLOAT8E4M3FN, 18=FLOAT8E5M2)")

    op_counts = Counter(n.op_type for n in q_model.graph.node)
    print(f"  fp8-graph QDQ counts: Q={op_counts.get('QuantizeLinear',0)} "
          f"DQ={op_counts.get('DequantizeLinear',0)}")

    # -----------------------------------------------------------------
    # 1. fp16-input vs fp8-output: did ModelOpt clamp any zeros?
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("[1/3] fp16 input -> fp8 output: zero-to-floor on retained fp16 initializers")
    print("=" * 70)
    common = sorted(set(fp_inits) & set(q_inits))
    n_checked = 0
    n_floor_diff = 0
    n_zero_lost = 0
    floor_suspects = []
    for name in common:
        a = fp_inits[name]
        b = q_inits[name]
        if a.data_type != onnx.TensorProto.FLOAT16 or b.data_type != onnx.TensorProto.FLOAT16:
            continue
        try:
            arr_a = _init_to_fp32(a).ravel()
            arr_b = _init_to_fp32(b).ravel()
        except Exception:
            continue
        if arr_a.shape != arr_b.shape:
            continue
        n_checked += 1
        zero_a = (arr_a == 0.0)
        floor_b = (np.abs(arr_b) >= FLOOR_LOW) & (np.abs(arr_b) <= FLOOR_HIGH)
        zero_to_floor = zero_a & floor_b
        n = int(zero_to_floor.sum())
        if n:
            n_floor_diff += 1
            n_zero_lost += n
            floor_suspects.append({"name": name, "shape": list(a.dims),
                                   "numel": int(arr_a.size),
                                   "zero_to_floor": n,
                                   "frac": n / arr_a.size})
    floor_suspects.sort(key=lambda d: d["zero_to_floor"], reverse=True)
    print(f"  retained fp16 initializers compared: {n_checked}")
    print(f"  initializers where fp16-zero became fp16-floor: {n_floor_diff}")
    print(f"  total elements that flipped zero -> floor:      {n_zero_lost}")
    if floor_suspects:
        for s in floor_suspects[:15]:
            print(f"    {s['name'][:60]:<60s} numel={s['numel']:>9d} flips={s['zero_to_floor']:>6d}  ({s['frac']:.4%})")

    # -----------------------------------------------------------------
    # 2. Per-channel DQ scales: zeros, denormals, NaN, extreme small
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("[2/3] DequantizeLinear scale initializers")
    print("=" * 70)
    # Each DequantizeLinear has inputs (x, x_scale, [x_zero_point]).
    # We look up the scale initializer by the node's input[1] name.
    scale_stats = {
        "dq_count": 0,
        "fp16_scale": 0,
        "fp32_scale": 0,
        "scale_has_zero": 0,
        "scale_has_denormal": 0,
        "scale_has_nan_inf": 0,
        "scale_lt_1e_4": 0,
        "scale_lt_1e_6": 0,
        "scale_examples_small": [],
    }
    smallest_scales = []
    for node in q_model.graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        scale_stats["dq_count"] += 1
        if len(node.input) < 2:
            continue
        scale_name = node.input[1]
        s_init = q_inits.get(scale_name)
        if s_init is None:
            continue
        try:
            s_arr = _init_to_fp32(s_init).ravel()
        except Exception:
            continue
        if s_init.data_type == onnx.TensorProto.FLOAT16:
            scale_stats["fp16_scale"] += 1
        elif s_init.data_type == onnx.TensorProto.FLOAT:
            scale_stats["fp32_scale"] += 1
        has_zero = bool((s_arr == 0.0).any())
        abs_s = np.abs(s_arr)
        has_denormal = bool(((abs_s > 0) & (abs_s < FP16_MIN_NORMAL)).any())
        has_bad = bool((~np.isfinite(s_arr)).any())
        if has_zero:
            scale_stats["scale_has_zero"] += 1
        if has_denormal:
            scale_stats["scale_has_denormal"] += 1
        if has_bad:
            scale_stats["scale_has_nan_inf"] += 1
        min_pos = abs_s[abs_s > 0]
        if min_pos.size:
            mn = float(min_pos.min())
            if mn < 1e-4:
                scale_stats["scale_lt_1e_4"] += 1
            if mn < 1e-6:
                scale_stats["scale_lt_1e_6"] += 1
            smallest_scales.append((mn, scale_name, node.name, list(s_init.dims)))
    smallest_scales.sort(key=lambda t: t[0])
    print(f"  DequantizeLinear nodes: {scale_stats['dq_count']}")
    print(f"  scales as fp16: {scale_stats['fp16_scale']}  fp32: {scale_stats['fp32_scale']}")
    print(f"  scales containing exact zero:     {scale_stats['scale_has_zero']}")
    print(f"  scales containing fp16-denormal:  {scale_stats['scale_has_denormal']}")
    print(f"  scales containing NaN/Inf:        {scale_stats['scale_has_nan_inf']}")
    print(f"  scales with min positive < 1e-4:  {scale_stats['scale_lt_1e_4']}")
    print(f"  scales with min positive < 1e-6:  {scale_stats['scale_lt_1e_6']}")
    print("  10 smallest per-channel scales (any DQ):")
    for mn, sn, nn, shape in smallest_scales[:10]:
        print(f"    min={mn:.3e}  shape={shape}  scale_init={sn[:50]:<50s}  node={nn[:40]}")
    scale_stats["scale_examples_small"] = [
        {"min_pos": mn, "scale_init": sn, "node": nn, "shape": shape}
        for mn, sn, nn, shape in smallest_scales[:20]
    ]

    # -----------------------------------------------------------------
    # 3. FP8 weight sanity: NaN bit patterns, saturation pressure
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("[3/3] FP8 weight initializers")
    print("=" * 70)
    fp8_inits = [i for i in q_model.graph.initializer
                 if i.data_type == onnx.TensorProto.FLOAT8E4M3FN]
    print(f"  FP8 E4M3FN initializers: {len(fp8_inits)}")
    weight_stats = {
        "count": len(fp8_inits),
        "with_nan_bits": 0,
        "with_inf_bits": 0,
        "all_zero": 0,
        "max_abs_at_saturation": 0,
        "examples_saturating": [],
    }
    # FP8 E4M3FN: 0x7F is NaN (the only NaN bit pattern in E4M3FN — there's
    # no Inf). 0x80 is -0. Max finite magnitude is 448 (0x7E or 0xFE).
    sat_examples = []
    for init in fp8_inits[:200]:  # sample 200; full pass would be too slow
        raw = init.raw_data
        if not raw:
            continue
        bytes_arr = np.frombuffer(raw, dtype=np.uint8)
        # NaN: 0x7F or 0xFF (sign bit + all-ones exponent/mantissa)
        n_nan = int(((bytes_arr & 0x7F) == 0x7F).sum())
        # 448.0 = max finite. The bit pattern is 0x7E (or 0xFE for -448).
        n_max = int(((bytes_arr & 0x7F) == 0x7E).sum())
        n_zero = int(bytes_arr.sum() == 0)
        if n_nan:
            weight_stats["with_nan_bits"] += 1
        if n_zero:
            weight_stats["all_zero"] += 1
        if n_max > 0:
            # Saturation pressure: count of values exactly at +/-448
            weight_stats["max_abs_at_saturation"] += 1
            sat_examples.append((n_max, init.name, list(init.dims), bytes_arr.size))
    sat_examples.sort(key=lambda t: t[0], reverse=True)
    print(f"  weights sampled: {min(200, len(fp8_inits))}")
    print(f"  weights with any NaN bit pattern (0x7F or 0xFF): {weight_stats['with_nan_bits']}")
    print(f"  weights all-zero:                                 {weight_stats['all_zero']}")
    print(f"  weights containing saturated +/-448 values:       {weight_stats['max_abs_at_saturation']}")
    print("  top saturating weights (count of +/-448 values):")
    for n_max, name, shape, total in sat_examples[:10]:
        print(f"    sat_count={n_max:>6d}/{total:<7d}  ({n_max/total:.2%})  shape={shape}  name={name[:60]}")
    weight_stats["examples_saturating"] = [
        {"sat_count": n, "init": name, "shape": shape, "numel": total}
        for n, name, shape, total in sat_examples[:20]
    ]

    OUT_JSON.write_text(json.dumps({
        "fp16_to_fp8_zero_to_floor": {
            "checked": n_checked,
            "initializers_with_flip": n_floor_diff,
            "total_flips": n_zero_lost,
            "suspects": floor_suspects,
        },
        "dq_scales": scale_stats,
        "fp8_weights_sampled": weight_stats,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
