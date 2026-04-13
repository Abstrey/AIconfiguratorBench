"""MLA operator factory for Ascend NPU benchmarking.

Constructs MLA operators via vllm-ascend's AscendAttentionBackendImpl
with use_mla=True.
"""
import logging
import math
from dataclasses import dataclass
from typing import Callable
import torch

logger = logging.getLogger(__name__)

OP_CONTEXT = "mla_context"
OP_GENERATION = "mla_generation"
SUPPORTED_OP_TYPES = (OP_CONTEXT, OP_GENERATION)

BLOCK_SIZE = 128

@dataclass(frozen=True)
class MlaSpec:
    """Immutable MLA benchmark specification."""
    op_type: str
    batch: int
    seq_len: int
    num_heads: int        # 128 for DSV3
    kv_lora_rank: int     # 512
    qk_nope_head_dim: int # 128
    qk_rope_head_dim: int # 64
    v_head_dim: int       # 128
    dtype: torch.dtype = torch.bfloat16

    @property
    def head_size(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_cache_head_size(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

def _generate_causal_mask(device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(device)

class _MockMlaAttentionLayer:
    def __init__(self) -> None:
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0

def _create_impl(spec: MlaSpec, device: torch.device):
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    # Note: Depending on AscendAttentionBackendImpl signature, we pass head_size.
    # For MLA, standard head_size is qk_nope_head_dim + qk_rope_head_dim.
    # We must also specify num_kv_heads=1 for the single latent cache
    kwargs = {
        "num_heads": spec.num_heads,
        "head_size": spec.head_size,
        "scale": 1.0 / math.sqrt(spec.head_size),
        "num_kv_heads": 1,
        "alibi_slopes": None,
        "sliding_window": None,
        "kv_cache_dtype": "auto",
        "logits_soft_cap": None,
        "attn_type": "decoder",
        "kv_sharing_target_layer_name": None
    }
    # Trying to inject use_mla if allowed, although often it's the 
    # Metadata that triggers the MLA path in the Ascend plugin.
    impl = AscendAttentionBackendImpl(**kwargs)
    return impl

def _build_prefill_metadata(spec: MlaSpec, device: torch.device):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
    num_tokens = spec.batch * spec.seq_len
    actual_seq_q = (torch.arange(1, spec.batch + 1, dtype=torch.int32) * spec.seq_len).tolist()
    seq_lens = torch.tensor([spec.seq_len] * spec.batch, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(0, num_tokens + 1, spec.seq_len, dtype=torch.int32, device=device)

    # NPU specific fields might lack `use_mla` in older versions, but we add it if supported.
    return AscendMetadata(
        attn_state=AscendAttentionState.PrefillNoCache,
        attn_mask=_generate_causal_mask(device),
        num_actual_tokens=num_tokens,
        seq_lens=seq_lens,
        seq_lens_list=[spec.seq_len] * spec.batch,
        actual_seq_lengths_q=actual_seq_q,
        query_start_loc=query_start_loc,
        max_query_len=spec.seq_len,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        causal=True,
        model_runner_type="",
        use_mla=True # Inject MLA switch
    )

def _build_decode_metadata(spec: MlaSpec, device: torch.device, blocks_per_seq: int):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
    num_blocks = spec.batch * blocks_per_seq
    seq_lens = torch.tensor([spec.seq_len] * spec.batch, dtype=torch.int32, device=device)
    actual_seq_q = list(range(1, spec.batch + 1))
    query_start_loc = torch.arange(0, spec.batch + 1, dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(spec.batch, blocks_per_seq)
    slot_mapping = torch.zeros(spec.batch, dtype=torch.int32, device=device)

    return AscendMetadata(
        attn_state=AscendAttentionState.DecodeOnly,
        attn_mask=_generate_causal_mask(device),
        num_actual_tokens=spec.batch,
        seq_lens=seq_lens,
        seq_lens_list=[spec.seq_len] * spec.batch,
        actual_seq_lengths_q=actual_seq_q,
        query_start_loc=query_start_loc,
        max_query_len=1,
        block_tables=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        model_runner_type="",
        use_mla=True # Inject MLA switch
    )

_forward_ctx_initialized = False
_ctx_refs = []

def _ensure_forward_context() -> None:
    global _forward_ctx_initialized
    if _forward_ctx_initialized:
        return
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context

    vllm_config = VllmConfig()
    cfg_ctx = set_current_vllm_config(vllm_config)
    cfg_ctx.__enter__()
    _ctx_refs.append(cfg_ctx)

    fwd_ctx = set_forward_context(attn_metadata=None, vllm_config=vllm_config)
    fwd_ctx.__enter__()
    _ctx_refs.append(fwd_ctx)

    from vllm.forward_context import get_forward_context
    ctx = get_forward_context()
    ctx.capturing = False
    _forward_ctx_initialized = True

def _dry_run(forward: Callable, spec: MlaSpec, phase: str) -> None:
    try:
        forward()
        torch.npu.synchronize()
    except RuntimeError as e:
        try:
            torch.npu.synchronize()
        except RuntimeError:
            pass
        raise RuntimeError(
            f"impl.forward() {phase} dry run failed for "
            f"batch={spec.batch} seq={spec.seq_len} "
            f"heads={spec.num_heads}: {e}"
        ) from e

def _create_context_mla(spec: MlaSpec, device: torch.device) -> Callable[[], torch.Tensor]:
    _ensure_forward_context()
    num_tokens = spec.batch * spec.seq_len

    impl = _create_impl(spec, device)
    layer = _MockMlaAttentionLayer()
    metadata = _build_prefill_metadata(spec, device)

    # MLA specific input tensors
    # query_dtype = spec.dtype. Usually Q is [T, num_heads, head_size]
    query = torch.randn(num_tokens, spec.num_heads, spec.head_size, dtype=spec.dtype, device=device)
    # For MLA, we pass kv_c and k_pe instead of generic kv
    kv_c = torch.randn(num_tokens, spec.kv_lora_rank, dtype=spec.dtype, device=device)
    k_pe = torch.randn(num_tokens, 1, spec.qk_rope_head_dim, dtype=spec.dtype, device=device)
    
    # Prefill has empty cache
    kv_cache = ()
    
    output = torch.empty(num_tokens, spec.num_heads, spec.v_head_dim, dtype=spec.dtype, device=device)

    def forward() -> torch.Tensor:
        # PagedAttention backend signature conceptually maps kv_c->key, k_pe->value during internal unpacking
        # Or Ascend uses actual named args if it's customized.
        # Following vllm standard API: forward(self, layer, query, key, value, kv_cache, attn_metadata)
        # where key=kv_c, value=k_pe for MLA
        return impl.forward(layer, query, kv_c, k_pe, kv_cache, metadata, output=output)

    _dry_run(forward, spec, "mla_prefill")
    return forward

def _create_generation_mla(spec: MlaSpec, device: torch.device) -> Callable[[], torch.Tensor]:
    _ensure_forward_context()
    blocks_per_seq = math.ceil(spec.seq_len / BLOCK_SIZE)
    num_blocks = spec.batch * blocks_per_seq

    impl = _create_impl(spec, device)
    layer = _MockMlaAttentionLayer()
    metadata = _build_decode_metadata(spec, device, blocks_per_seq)

    query = torch.randn(spec.batch, spec.num_heads, spec.head_size, dtype=spec.dtype, device=device)
    kv_c = torch.randn(spec.batch, spec.kv_lora_rank, dtype=spec.dtype, device=device)
    k_pe = torch.randn(spec.batch, 1, spec.qk_rope_head_dim, dtype=spec.dtype, device=device)

    # For MLA decode, kv_cache is typically ONE tensor containing both [blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    # But vLLM expects a tuple, we create a pseudo tuple representing the single buffer for NPU if needed.
    cache_tensor = torch.randn(num_blocks, BLOCK_SIZE, 1, spec.kv_cache_head_size, dtype=spec.dtype, device=device)
    kv_cache = (cache_tensor, cache_tensor) # NPU MLA kernels often accept a tuple and use only the first element, or identical element based on framework.

    output = torch.empty(spec.batch, spec.num_heads, spec.v_head_dim, dtype=spec.dtype, device=device)

    def forward() -> torch.Tensor:
        return impl.forward(layer, query, kv_c, k_pe, kv_cache, metadata, output=output)

    _dry_run(forward, spec, "mla_decode")
    return forward

def create_mla_func(spec: MlaSpec, device: torch.device) -> Callable[[], torch.Tensor]:
    if spec.op_type not in SUPPORTED_OP_TYPES:
        raise ValueError(f"Unsupported op_type: {spec.op_type}")

    if spec.op_type == OP_CONTEXT:
        return _create_context_mla(spec, device)
    return _create_generation_mla(spec, device)
