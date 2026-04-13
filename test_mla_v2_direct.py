import argparse
import math
import torch
import torch_npu

def benchmark_mla_v2(batch=1, seq_len=128, num_heads=128, 
                     qk_nope_head_dim=128, qk_rope_head_dim=64, 
                     kv_lora_rank=512, v_head_dim=128, iters=100):
    device = torch.device("npu")
    dtype = torch.bfloat16
    spec_head_size = qk_nope_head_dim + qk_rope_head_dim
    scale = 1.0 / math.sqrt(spec_head_size)

    num_kv_heads = 1
    block_size = 128
    
    print(f"Initializing MLA params: Batch={batch}, Seq={seq_len}, Heads={num_heads}")
    print("Using npu_fused_infer_attention_score_v2 with BNSD_NBSD layout")

    # BNSD_NBSD layout constraints
    # q: [batch, num_heads, seq_len=1, dimension]
    q_nope = torch.randn(batch, num_heads, 1, qk_nope_head_dim, dtype=dtype, device=device).contiguous()
    q_pe = torch.randn(batch, num_heads, 1, qk_rope_head_dim, dtype=dtype, device=device)
    
    # Cache blocks
    blocks_per_seq = math.ceil(seq_len / block_size)
    num_blocks = batch * blocks_per_seq
    
    # k_nope / v_layer: [num_blocks, num_kv_heads, block_size, dimension]
    k_nope = torch.randn(num_blocks, num_kv_heads, block_size, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_blocks, num_kv_heads, block_size, qk_rope_head_dim, dtype=dtype, device=device)
    
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(batch, blocks_per_seq)
    
    # actual_seq_kvlen denotes the actual context length for each request
    seq_lens_list = [seq_len] * batch

    # Warmup
    for _ in range(5):
        out, lse = torch_npu.npu_fused_infer_attention_score_v2(
            q_nope,
            k_nope,
            k_nope, # k_nope is passed as value in v2 MLA decode
            query_rope=q_pe,
            key_rope=k_pe,
            num_query_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            input_layout="BNSD_NBSD",
            atten_mask=None,
            sparse_mode=0,
            softmax_scale=scale,
            block_table=block_table,
            block_size=block_size,
            actual_seq_kvlen=seq_lens_list,
        )
    torch.npu.synchronize()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        out, lse = torch_npu.npu_fused_infer_attention_score_v2(
            q_nope,
            k_nope,
            k_nope,
            query_rope=q_pe,
            key_rope=k_pe,
            num_query_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            input_layout="BNSD_NBSD",
            atten_mask=None,
            sparse_mode=0,
            softmax_scale=scale,
            block_table=block_table,
            block_size=block_size,
            actual_seq_kvlen=seq_lens_list,
        )
    end_event.record()
    end_event.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / iters
    print("-" * 50)
    print(f"Result: {avg_ms:.4f} ms/iter")
    print(f"Output shape: {out.shape}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directly benchmark npu_fused_infer_attention_score_v2")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    benchmark_mla_v2(batch=args.batch, seq_len=args.seq_len, iters=args.iters)
