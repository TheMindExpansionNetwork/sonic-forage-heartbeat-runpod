"""Walk MatMul nodes via their weight initializers (which preserve
PyTorch names for refit). Build the exclusion plan from initializer
name patterns, then verify with concrete counts.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import onnx
from collections import Counter

ONNX_PATH = os.path.expanduser(
    "~/.daydream-scope/models/demon/trt_engines/"
    "_onnx_acestep-v15-xl-turbo/decoder_refit/decoder_refit_dynbatch.onnx"
)
model = onnx.load(ONNX_PATH, load_external_data=False)
g = model.graph

init_names = {init.name for init in g.initializer}
print(f"Initializers: {len(init_names)}")
print()

# Sample initializer names to confirm PyTorch names are preserved
print("First 20 initializer names:")
for name in sorted(init_names)[:20]:
    print(f"  {name}")
print()

# For each MatMul / Gemm, find its weight initializer (input[1]) and
# group by stripped layer-index pattern.
matmul_to_weight = {}
for n in g.node:
    if n.op_type not in ("MatMul", "Gemm"):
        continue
    for inp in n.input:
        if inp in init_names:
            matmul_to_weight[n.name] = inp
            break

print(f"MatMul/Gemm nodes with identifiable weight initializer: "
      f"{len(matmul_to_weight)}/{sum(1 for n in g.node if n.op_type in ('MatMul','Gemm'))}")
print()

# Group MatMuls by the LAST suffix of their weight name (e.g. "q_proj.weight")
suffixes = Counter()
for mm, w in matmul_to_weight.items():
    parts = w.rsplit(".", 2)
    suffix = ".".join(parts[-2:]) if len(parts) >= 2 else w
    suffixes[suffix] += 1
print("MatMul weight suffix histogram:")
for s, n in suffixes.most_common(40):
    print(f"  {n:4d}  {s}")

# Group by the segment of the weight name before .weight
print()
print("MatMul weight module-path histogram (drop layer index):")
import re
modpath = Counter()
for mm, w in matmul_to_weight.items():
    # decoder.layers.0.self_attn.q_proj.weight -> decoder.layers.N.self_attn.q_proj.weight
    g_path = re.sub(r"\.layers\.\d+\.", ".layers.N.", w)
    modpath[g_path] += 1
for p, n in sorted(modpath.items(), key=lambda kv: (-kv[1], kv[0]))[:50]:
    print(f"  {n:4d}  {p}")

# Now look at the ConvTranspose node's weight
print()
print("Conv / ConvTranspose weights:")
for n in g.node:
    if n.op_type in ("Conv", "ConvTranspose"):
        w = None
        for inp in n.input:
            if inp in init_names:
                w = inp; break
        print(f"  {n.op_type}/{n.name}  weight={w}")

# Identify what's NOT a MatMul/Gemm/Conv but reads a known weight (e.g.
# fp32-island ops like RMSNorm, AdaLN scale_shift_table which appear as
# Mul/Add chains, not MatMul). Those won't get FP8 anyway, but worth
# noting which weights have NO MatMul consumer (they're already on the
# non-quantized side of the graph).
init_consumers = {n: [] for n in init_names}
for n in g.node:
    for inp in n.input:
        if inp in init_consumers:
            init_consumers[inp].append((n.op_type, n.name))

unmatched = [(name, c) for name, c in init_consumers.items()
             if not any(opt in ("MatMul", "Gemm", "Conv", "ConvTranspose") for opt, _ in c)]
print()
print(f"Initializers NOT consumed by MatMul/Gemm/Conv/ConvT: {len(unmatched)}")
print("Sample (these are the 'fp32 island' weights already kept un-quantized):")
for name, c in unmatched[:15]:
    consumer_ops = Counter(opt for opt, _ in c)
    print(f"  {name}  consumers={dict(consumer_ops)}")
