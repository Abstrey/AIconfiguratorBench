"""Dry-run validation for collect_attn.py (no NPU or torch required).

Validates:
1. Parameter space generation with GQA constraints
2. CSV row formatting for context and generation
3. Checkpoint save/load round-trip
4. Spec key uniqueness
5. AttnSpec immutability
"""

import csv
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# --- Mock torch and torch_npu so we can import our modules without GPU ---
mock_torch = MagicMock()
mock_torch.bfloat16 = "bfloat16"
mock_torch.float16 = "float16"
mock_torch.float32 = "float32"
mock_torch.int32 = "int32"
mock_torch.bool = "bool"
mock_torch.device = MagicMock
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch_npu"] = MagicMock()
sys.modules["vllm"] = MagicMock()
sys.modules["vllm.config"] = MagicMock()
sys.modules["vllm_ascend"] = MagicMock()
sys.modules["vllm_ascend.utils"] = MagicMock()
sys.modules["vllm_ascend.quantization"] = MagicMock()
sys.modules["vllm_ascend.quantization.methods"] = MagicMock()
sys.modules["vllm_ascend.quantization.methods.w8a8_dynamic"] = MagicMock()

# --- 1. Validate imports ---
print("1. Checking imports...")
from bench_engine import BenchResult
from attn_factory import (
    BLOCK_SIZE,
    OP_CONTEXT,
    OP_GENERATION,
    SUPPORTED_OP_TYPES,
    AttnSpec,
    _resolve_kv_heads,
)
from collect_attn import (
    CSV_COLUMNS,
    DEFAULT_BATCH_LIST,
    DEFAULT_SEQ_CONTEXT,
    DEFAULT_SEQ_GENERATION,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_KV_HEADS,
    KERNEL_TYPE_MAP,
    _build_spec_list,
    _format_context_shapes,
    _format_generation_shapes,
    _load_checkpoint,
    _make_csv_row,
    _save_checkpoint,
    _spec_key,
)
print("   OK: all imports succeeded")

# --- 2. Validate AttnSpec ---
print("2. Checking AttnSpec...")
spec = AttnSpec(
    op_type=OP_CONTEXT, batch=4, seq_len=1024,
    num_heads=32, num_kv_heads=8,
)
assert spec.batch == 4
assert spec.head_size == 128  # default
assert spec.op_type == "attention_context"
# frozen=True
try:
    spec.batch = 8  # type: ignore
    assert False, "Should be frozen"
except AttributeError:
    pass
print("   OK: AttnSpec is immutable")

# --- 3. Validate _resolve_kv_heads ---
print("3. Checking _resolve_kv_heads...")
mha_spec = AttnSpec(op_type=OP_CONTEXT, batch=1, seq_len=128, num_heads=32, num_kv_heads=0)
assert _resolve_kv_heads(mha_spec) == 32, "MHA: kv_heads=0 should resolve to num_heads"
gqa_spec = AttnSpec(op_type=OP_CONTEXT, batch=1, seq_len=128, num_heads=32, num_kv_heads=4)
assert _resolve_kv_heads(gqa_spec) == 4
print("   OK: MHA and GQA resolution correct")

# --- 4. Validate parameter space with GQA constraints ---
print("4. Checking parameter space generation...")
specs = _build_spec_list(
    op_types=[OP_CONTEXT],
    batch_list=[1, 4],
    seq_len_list=[128, 512],
    num_heads_list=[8, 32],
    num_kv_heads_list=[0, 1, 4, 8],
    head_size=128,
)
# Check GQA filtering:
# num_heads=8:  kv=0(MHA ok), kv=1(8%1=0 ok), kv=4(8%4=0 ok), kv=8(8%8=0 ok) → 4
# num_heads=32: kv=0(MHA ok), kv=1(32%1=0 ok), kv=4(32%4=0 ok), kv=8(32%8=0 ok) → 4
# total = 1 op × 2 batch × 2 seq × (4+4) heads_kv = 32
assert len(specs) == 32, f"Expected 32, got {len(specs)}"
print(f"   OK: {len(specs)} specs generated (GQA constraints applied)")

# Test GQA filtering with invalid combos
specs_filtered = _build_spec_list(
    op_types=[OP_CONTEXT],
    batch_list=[1],
    seq_len_list=[128],
    num_heads_list=[8],
    num_kv_heads_list=[0, 3, 5, 7],  # 3,5,7 don't divide 8
    head_size=128,
)
assert len(specs_filtered) == 1, f"Expected 1 (only MHA), got {len(specs_filtered)}"
assert specs_filtered[0].num_kv_heads == 0
print(f"   OK: invalid GQA combos filtered (3,5,7 rejected for num_heads=8)")

# Full default space size
full_ctx = _build_spec_list(
    [OP_CONTEXT], DEFAULT_BATCH_LIST, DEFAULT_SEQ_CONTEXT,
    DEFAULT_NUM_HEADS, DEFAULT_NUM_KV_HEADS, 128,
)
full_gen = _build_spec_list(
    [OP_GENERATION], DEFAULT_BATCH_LIST, DEFAULT_SEQ_GENERATION,
    DEFAULT_NUM_HEADS, DEFAULT_NUM_KV_HEADS, 128,
)
print(f"   Default context space: {len(full_ctx)} specs")
print(f"   Default generation space: {len(full_gen)} specs")
print(f"   Total: {len(full_ctx) + len(full_gen)} specs")

# --- 5. Validate CSV row formatting (context) ---
print("5. Checking CSV row format (context)...")
ctx_spec = AttnSpec(
    op_type=OP_CONTEXT, batch=4, seq_len=1024,
    num_heads=32, num_kv_heads=8,
)
ctx_result = BenchResult(median_us=150.0, min_us=140.0, max_us=160.0, std_us=5.0, num_iters=100)
row = _make_csv_row(ctx_spec, ctx_result)

assert row["OP State"] == "dynamic"
assert row["Accelerator Core"] == "AI_CORE"
# Q: [4096, 32, 128], K: [4096, 8, 128], V: [4096, 8, 128]
assert row["Input Shapes"] == "4096,32,128;4096,8,128;4096,8,128", f"Got: {row['Input Shapes']}"
assert row["Output Shapes"] == "4096,32,128"
assert row["Input Data Types"] == "DT_BF16;DT_BF16;DT_BF16"
assert row["Average Duration(us)"] == "150.00"
assert row["Op Type"] == "attention_context"
assert row["Batch"] == "4"
assert row["Num KV Heads"] == "8"
print(f"   OK: Input Shapes = {row['Input Shapes']}")

# --- 6. Validate CSV row formatting (generation) ---
print("6. Checking CSV row format (generation)...")
gen_spec = AttnSpec(
    op_type=OP_GENERATION, batch=8, seq_len=2048,
    num_heads=32, num_kv_heads=8,
)
gen_result = BenchResult(median_us=50.0, min_us=45.0, max_us=55.0, std_us=2.0, num_iters=100)
row_gen = _make_csv_row(gen_spec, gen_result)

# blocks_per_seq = ceil(2048/128) = 16, num_blocks = 8*16 = 128
assert row_gen["Input Shapes"] == "8,32,128;128,128,8,128;128,128,8,128", f"Got: {row_gen['Input Shapes']}"
assert row_gen["Output Shapes"] == "8,32,128"
assert row_gen["Op Type"] == "attention_generation"
print(f"   OK: Input Shapes = {row_gen['Input Shapes']}")

# --- 7. Validate CSV write/read round-trip ---
print("7. Checking CSV write/read round-trip...")
buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
writer.writeheader()
writer.writerow(row)
writer.writerow(row_gen)

buf.seek(0)
reader = csv.DictReader(buf)
rows = list(reader)
assert len(rows) == 2
assert rows[0]["Op Type"] == "attention_context"
assert rows[1]["Op Type"] == "attention_generation"
print(f"   OK: {len(rows)} rows written and read back correctly")

# --- 8. Validate checkpoint round-trip ---
print("8. Checking checkpoint save/load...")
with tempfile.TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)
    completed = {
        "attention_context_4_1024_32_8",
        "attention_generation_8_2048_32_8",
    }
    _save_checkpoint(tmppath, completed)
    loaded = _load_checkpoint(tmppath)
    assert loaded == completed
    print(f"   OK: {len(loaded)} keys round-tripped")

    empty = _load_checkpoint(Path("/nonexistent"))
    assert empty == set()
    print("   OK: missing checkpoint returns empty set")

# --- 9. Validate spec_key uniqueness ---
print("9. Checking spec_key uniqueness...")
s1 = AttnSpec(op_type=OP_CONTEXT, batch=1, seq_len=128, num_heads=32, num_kv_heads=0)
s2 = AttnSpec(op_type=OP_GENERATION, batch=1, seq_len=128, num_heads=32, num_kv_heads=0)
s3 = AttnSpec(op_type=OP_CONTEXT, batch=1, seq_len=128, num_heads=32, num_kv_heads=8)
s4 = AttnSpec(op_type=OP_CONTEXT, batch=4, seq_len=128, num_heads=32, num_kv_heads=0)
assert _spec_key(s1) != _spec_key(s2), "Different op_type should differ"
assert _spec_key(s1) != _spec_key(s3), "Different kv_heads should differ"
assert _spec_key(s1) != _spec_key(s4), "Different batch should differ"
assert _spec_key(s1) == "attention_context_1_128_32_0"
print(f"   OK: spec_key = {_spec_key(s1)}")

# Check uniqueness across full spec list
all_keys = {_spec_key(s) for s in full_ctx + full_gen}
assert len(all_keys) == len(full_ctx) + len(full_gen), "Duplicate spec keys found!"
print(f"   OK: all {len(all_keys)} spec keys are unique")

# --- 10. Validate kernel type mapping ---
print("10. Checking kernel type mapping...")
assert KERNEL_TYPE_MAP[OP_CONTEXT] == "FusedInferAttentionScore"
assert KERNEL_TYPE_MAP[OP_GENERATION] == "PagedAttention"
assert BLOCK_SIZE == 128
print("   OK: kernel types and block size correct")

print("\n=== All attention validation checks passed ===")
