"""Quick smoke test for FIA decode path (generation attention)."""
import math

import torch
import torch_npu

device = torch.device("npu")

batch, num_heads, num_kv_heads, head_size, block_size = 1, 8, 8, 128, 128
seq_len = 256
blocks_per_seq = seq_len // block_size
num_blocks = batch * blocks_per_seq
scale = 1.0 / math.sqrt(head_size)

# Query: [batch, num_heads, head_size] (one token per sequence)
q = torch.randn(batch, num_heads, head_size, dtype=torch.bfloat16, device=device)

# KV cache: [num_blocks, block_size, num_kv_heads, head_size]
kc = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)
vc = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)

# Flatten for FIA: [num_blocks, block_size, hidden]
kc_fia = kc.view(num_blocks, block_size, -1)
vc_fia = vc.view(num_blocks, block_size, -1)

# Block table: [batch, blocks_per_seq]
bt = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(batch, blocks_per_seq)

# Causal mask: fixed 2048x2048 int8
attn_mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(device)

# Sequence lengths
actual_seq_q = [1] * batch  # one token per sequence (cumulative: [1])
actual_seq_kv = [seq_len] * batch

print(f"query:     {q.shape}")
print(f"key (FIA): {kc_fia.shape}")
print(f"val (FIA): {vc_fia.shape}")
print(f"block_tbl: {bt.shape}")
print(f"attn_mask: {attn_mask.shape}")

print("\nCalling npu_fused_infer_attention_score (decode)...")
try:
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=q,
        key=kc_fia,
        value=vc_fia,
        atten_mask=attn_mask,
        block_table=bt,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=3,
    )
    torch.npu.synchronize()
    print(f"OK, output shape: {output.shape}")
except Exception as e:
    print(f"FAILED: {e}")
