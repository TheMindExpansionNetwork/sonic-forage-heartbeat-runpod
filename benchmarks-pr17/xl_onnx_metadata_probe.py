"""Check whether dynamo ONNX export preserves pytorch parameter names in
initializer.metadata_props (which TRT could use to restore refit names).

If yes: exclusion strategy = walk MatMul nodes, look up their weight
initializer's metadata_props['pkg.torch.onnx.original_name'], filter.

If no: we need a topology-based identification (which MatMul is which
logical Linear) using the model's known parameter shapes.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import onnx

ONNX_PATH = os.path.expanduser(
    "~/.daydream-scope/models/demon/trt_engines/"
    "_onnx_acestep-v15-xl-turbo/decoder_refit/decoder_refit_dynbatch.onnx"
)
model = onnx.load(ONNX_PATH, load_external_data=False)
g = model.graph

# Probe metadata on a val_NNN initializer
print("Initializer metadata_props probe:")
for init in g.initializer[:20]:
    if init.name.startswith("val_"):
        props = list(init.metadata_props) if init.metadata_props else []
        print(f"  {init.name}: shape={list(init.dims)} dtype={init.data_type}")
        for p in props:
            print(f"    [{p.key}] = {p.value}")
        if not props:
            print("    (no metadata_props)")
        if len(props) > 0:
            break

# Probe doc_string
print()
print("Initializer doc_string probe (first 5 val_):")
ct = 0
for init in g.initializer:
    if init.name.startswith("val_") and init.doc_string:
        print(f"  {init.name}: {init.doc_string!r}")
        ct += 1
    if ct >= 5: break
if ct == 0:
    print("  (no doc_strings on val_ initializers)")

# Top-level model metadata
print()
print("Model metadata_props:")
for p in model.metadata_props:
    if len(p.value) > 200:
        print(f"  [{p.key}] = {p.value[:200]}... ({len(p.value)} chars)")
    else:
        print(f"  [{p.key}] = {p.value}")

# Check if there's a "value_info" entry that names tensors
print()
print(f"value_info entries: {len(g.value_info)}")
named_val = [vi for vi in g.value_info if not vi.name.startswith("val_") and not vi.name.startswith("node_")]
print(f"  named (not val_*/not node_*): {len(named_val)}")
if named_val:
    for vi in named_val[:10]:
        print(f"    {vi.name}")

# Check if MatMul nodes themselves have a doc_string or attribute that
# restores PyTorch name
print()
print("MatMul node attributes / doc_string probe:")
for n in g.node:
    if n.op_type in ("MatMul", "Gemm"):
        if n.doc_string or n.attribute:
            print(f"  {n.name}  doc={n.doc_string!r}  attrs={[a.name for a in n.attribute]}")
            break
else:
    print("  (no MatMul nodes have doc_string or attributes)")

# Last shot: maybe the metadata_props on the val_ INITIALIZERS contains
# something. Look at the FIRST val_ initializer's full repr.
print()
val_inits = [init for init in g.initializer if init.name.startswith("val_")]
print(f"Total val_* initializers: {len(val_inits)}")
if val_inits:
    init = val_inits[0]
    print(f"  Sample: {init.name}")
    print(f"    raw_data length: {len(init.raw_data) if init.raw_data else 0}")
    print(f"    external_data: {[(e.key, e.value) for e in init.external_data]}")
    print(f"    metadata_props: {[(p.key, p.value) for p in init.metadata_props]}")
    print(f"    doc_string: {init.doc_string!r}")
