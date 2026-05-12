"""Audit the XL refit ONNX node names so we can plan FP8 exclusions cleanly.

We need to keep these regions out of FP8 quantization (they're the fp32
islands that already exist in the bf16_mixed export):
  - time_embed / time_embed_r (sinusoidal -> Linear)
  - scale_shift_table operations (AdaLN per layer + output)
  - RMSNorm operations (self_attn_norm, mlp_norm, cross_attn_norm)
  - norm_out
  - proj_out ConvTranspose1d (no FP8 ConvT kernel)
  - condition_embedder (kept fp16/bf16 to match cross-attn Q dtype)

This script walks the ONNX graph and groups MatMul/Gemm/Conv nodes by
the prefix patterns we'd use to exclude them, and reports the counts so
we can confirm the right things are getting matched.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import onnx
from collections import Counter

ONNX_PATH = os.path.expanduser(
    "~/.daydream-scope/models/demon/trt_engines/"
    "_onnx_acestep-v15-xl-turbo/decoder_refit/decoder_refit_dynbatch.onnx"
)

print(f"Loading {ONNX_PATH}")
print(f"  size on disk: {os.path.getsize(ONNX_PATH)/1e6:.1f} MB (.onnx)")
data_path = ONNX_PATH.replace("_dynbatch.onnx", ".onnx.data")
if not os.path.exists(data_path):
    data_path = ONNX_PATH.replace(".onnx", ".onnx.data")
if os.path.exists(data_path):
    print(f"  external data: {os.path.getsize(data_path)/1e9:.2f} GB")

model = onnx.load(ONNX_PATH, load_external_data=False)
g = model.graph

print()
print(f"Total nodes: {len(g.node)}")
print(f"Inputs: {[(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input]}")
print(f"Outputs: {[(o.name) for o in g.output]}")

op_types = Counter(n.op_type for n in g.node)
print()
print("Op type histogram (top 20):")
for op, n in op_types.most_common(20):
    print(f"  {op:24s} {n}")

# Find all MatMul / Gemm / Conv nodes and group by the part of their name
# before the first numeric index. The dynamo exporter typically emits
# names like "/decoder/layers.0/self_attn/q_proj/MatMul".
matmul_names = [n.name for n in g.node if n.op_type in ("MatMul", "Gemm")]
print()
print(f"MatMul/Gemm count: {len(matmul_names)}")
print("Sample names (first 8):")
for n in matmul_names[:8]:
    print(f"  {n}")

print()
print("Pattern bucket counts (substring match in node name):")
patterns = [
    "scale_shift_table",
    "time_embed",
    "AdaLayerNorm",
    "norm_out",
    "self_attn_norm",
    "mlp_norm",
    "cross_attn_norm",
    "condition_embedder",
    "proj_out",
    "proj_in",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "mlp",
    "self_attn",
    "cross_attn",
    "RMSNorm",
    "rotary",
    "ConvTranspose",
]
for p in patterns:
    matches_mm = [n for n in matmul_names if p in n]
    matches_all = [n.name for n in g.node if p in n.name]
    if matches_all:
        print(f"  {p:24s} matmul={len(matches_mm):3d} all_nodes={len(matches_all):4d}  e.g. {matches_all[0]}")

# ConvTranspose nodes specifically
print()
convT = [n for n in g.node if n.op_type == "ConvTranspose"]
print(f"ConvTranspose count: {len(convT)}")
for n in convT:
    print(f"  {n.name}")

# Conv nodes
conv = [n for n in g.node if n.op_type == "Conv"]
print(f"Conv count: {len(conv)}")
for n in conv[:5]:
    print(f"  {n.name}")
