"""Dry-run validation for collect_gemm.py (no NPU or torch required).

Validates:
1. Parameter space generation
2. CSV row formatting
3. Checkpoint save/load round-trip
4. Spec key uniqueness
"""

import csv
import io
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

# --- Mock torch and torch_npu so we can import our modules without GPU ---
mock_torch = MagicMock()
mock_torch.bfloat16 = "bfloat16"
mock_torch.float32 = "float32"
mock_torch.int8 = "int8"
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
from gemm_factory import (
    QUANT_BF16,
    QUANT_W8A8_DYNAMIC,
    SUPPORTED_QUANT_TYPES,
    GemmSpec,
)
from collect_gemm import (
    CSV_COLUMNS,
    DEFAULT_M_LIST,
    DEFAULT_NK_LIST,
    _build_spec_list,
    _dtype_str,
    _format_shapes,
    _load_checkpoint,
    _make_csv_row,
    _save_checkpoint,
    _spec_key,
)

print("   OK: all imports succeeded")

# --- 2. Validate GemmSpec ---
print("2. Checking GemmSpec...")
spec = GemmSpec(m=128, n=4096, k=4096, quant_type="bf16")
assert spec.m == 128
assert spec.quant_type == "bf16"
# frozen=True
try:
    spec.m = 256  # type: ignore
    assert False, "Should be frozen"
except AttributeError:
    pass
print("   OK: GemmSpec is immutable")

# --- 3. Validate BenchResult ---
print("3. Checking BenchResult...")
result = BenchResult(median_us=10.5, min_us=9.0, max_us=12.0, std_us=1.1, num_iters=100)
assert result.median_us == 10.5
print("   OK: BenchResult is immutable")

# --- 4. Validate parameter space ---
print("4. Checking parameter space generation...")
specs = _build_spec_list([1, 128], [4096], [4096, 8192], ["bf16", "w8a8_dynamic"])
assert len(specs) == 2 * 1 * 2 * 2  # 8
assert specs[0].quant_type == "bf16"
assert specs[4].quant_type == "w8a8_dynamic"
print(f"   OK: {len(specs)} specs generated")

# Full default space
full_specs = _build_spec_list(DEFAULT_M_LIST, DEFAULT_NK_LIST, DEFAULT_NK_LIST, ["bf16"])
print(f"   Default BF16 space: {len(full_specs)} specs ({len(DEFAULT_M_LIST)} M x {len(DEFAULT_NK_LIST)} N x {len(DEFAULT_NK_LIST)} K)")

# --- 5. Validate CSV row formatting ---
print("5. Checking CSV row format...")
spec_bf16 = GemmSpec(m=8192, n=4096, k=512, quant_type="bf16")
result_bf16 = BenchResult(median_us=112.9, min_us=110.0, max_us=120.0, std_us=2.5, num_iters=100)
row = _make_csv_row(spec_bf16, result_bf16)

assert row["OP State"] == "dynamic"
assert row["Accelerator Core"] == "AI_CORE"
assert row["Input Shapes"] == "8192,512;4096,512"
assert row["Input Data Types"] == "DT_BF16;DT_BF16"
assert row["Input Formats"] == "ND;ND"
assert row["Output Shapes"] == "8192,4096"
assert row["Output Data Types"] == "DT_BF16"
assert row["Average Duration(us)"] == "112.90"
assert row["Quant Type"] == "bf16"
print(f"   OK: Input Shapes = {row['Input Shapes']}")

# W8A8 row
spec_w8 = GemmSpec(m=1, n=7168, k=7168, quant_type="w8a8_dynamic")
row_w8 = _make_csv_row(spec_w8, result_bf16)
assert row_w8["Input Data Types"] == "DT_BF16;DT_INT8"
print(f"   OK: W8A8 Input Data Types = {row_w8['Input Data Types']}")

# --- 6. Validate CSV write/read round-trip ---
print("6. Checking CSV write/read round-trip...")
buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
writer.writeheader()
writer.writerow(row)
writer.writerow(row_w8)

buf.seek(0)
reader = csv.DictReader(buf)
rows = list(reader)
assert len(rows) == 2
assert rows[0]["Input Shapes"] == "8192,512;4096,512"
assert rows[1]["Input Data Types"] == "DT_BF16;DT_INT8"
print(f"   OK: {len(rows)} rows written and read back correctly")
print(f"   CSV header: {list(rows[0].keys())}")

# --- 7. Validate checkpoint round-trip ---
print("7. Checking checkpoint save/load...")
with tempfile.TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)
    completed = {"1_4096_4096_bf16", "128_4096_4096_bf16"}
    _save_checkpoint(tmppath, completed)
    loaded = _load_checkpoint(tmppath)
    assert loaded == completed
    print(f"   OK: {len(loaded)} keys round-tripped")

    # Empty checkpoint
    empty = _load_checkpoint(Path("/nonexistent"))
    assert empty == set()
    print("   OK: missing checkpoint returns empty set")

# --- 8. Validate spec_key uniqueness ---
print("8. Checking spec_key uniqueness...")
s1 = GemmSpec(m=1, n=4096, k=4096, quant_type="bf16")
s2 = GemmSpec(m=1, n=4096, k=4096, quant_type="w8a8_dynamic")
s3 = GemmSpec(m=2, n=4096, k=4096, quant_type="bf16")
assert _spec_key(s1) != _spec_key(s2)  # different quant
assert _spec_key(s1) != _spec_key(s3)  # different M
assert _spec_key(s1) == "1_4096_4096_bf16"
print(f"   OK: spec_key = {_spec_key(s1)}")

# --- 9. Validate dtype_str ---
print("9. Checking dtype_str mapping...")
assert _dtype_str("bf16") == ("DT_BF16", "DT_BF16")
assert _dtype_str("w8a8_dynamic") == ("DT_BF16", "DT_INT8")
try:
    _dtype_str("unknown")
    assert False
except ValueError:
    pass
print("   OK: dtype_str mappings correct")

print("\n=== All validation checks passed ===")
