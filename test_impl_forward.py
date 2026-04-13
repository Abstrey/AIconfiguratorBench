"""Smoke test: verify attn_factory calls AscendAttentionBackendImpl.forward()."""
import torch
import torch_npu  # noqa: F401

from attn_factory import AttnSpec, OP_CONTEXT, OP_GENERATION, create_attn_func

device = torch.device("npu")

# Test context (prefill)
print("=== Context (PrefillNoCache) ===")
spec_ctx = AttnSpec(op_type=OP_CONTEXT, batch=2, seq_len=256, num_heads=8, num_kv_heads=8)
try:
    fn = create_attn_func(spec_ctx, device)
    out = fn()
    torch.npu.synchronize()
    print(f"OK: output shape = {out.shape}")
except Exception as e:
    print(f"FAILED: {e}")

# Test generation (decode)
print("\n=== Generation (DecodeOnly) ===")
spec_gen = AttnSpec(op_type=OP_GENERATION, batch=2, seq_len=256, num_heads=8, num_kv_heads=8)
try:
    fn = create_attn_func(spec_gen, device)
    out = fn()
    torch.npu.synchronize()
    print(f"OK: output shape = {out.shape}")
except Exception as e:
    print(f"FAILED: {e}")
