"""Verify that collect_attn benchmarks exercise the correct NPU attention kernel path.

Two verification methods:
  1. Call tracing  — monkey-patch torch_npu APIs, count invocations
  2. NPU profiler — capture one forward pass, print actual NPU kernel names

Usage:
    python verify_attn_kernel_path.py --op-types context generation
    python verify_attn_kernel_path.py --op-types context --method profiler
"""

import argparse
import functools
import logging
from collections import defaultdict

import torch
import torch_npu  # noqa: F401

from attn_factory import (
    OP_CONTEXT,
    OP_GENERATION,
    SUPPORTED_OP_TYPES,
    AttnSpec,
    create_attn_func,
)
from gemm_factory import _init_vllm_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method 1: Call tracing via monkey-patch
# ---------------------------------------------------------------------------

_call_counts: dict[str, int] = defaultdict(int)
_originals: dict[str, object] = {}


def _wrap(module, attr_name: str, tag: str) -> None:
    """Replace module.attr_name with a counting wrapper."""
    original = getattr(module, attr_name)
    _originals[tag] = original

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        _call_counts[tag] += 1
        return original(*args, **kwargs)

    setattr(module, attr_name, wrapper)


def _unwrap(module, attr_name: str, tag: str) -> None:
    if tag in _originals:
        setattr(module, attr_name, _originals.pop(tag))


# FIA is a torch op object, need special handling
_fia_original = None
_pa_original = None


def _install_tracers() -> None:
    """Install call-counting wrappers on attention kernel functions."""
    global _fia_original, _pa_original
    _call_counts.clear()

    # Context path: npu_fused_infer_attention_score
    _fia_original = torch_npu.npu_fused_infer_attention_score
    def fia_wrapper(*args, **kwargs):
        _call_counts["torch_npu.npu_fused_infer_attention_score"] += 1
        return _fia_original(*args, **kwargs)
    torch_npu.npu_fused_infer_attention_score = fia_wrapper

    # Generation path: _npu_paged_attention
    _pa_original = torch_npu._npu_paged_attention
    def pa_wrapper(*args, **kwargs):
        _call_counts["torch_npu._npu_paged_attention"] += 1
        return _pa_original(*args, **kwargs)
    torch_npu._npu_paged_attention = pa_wrapper


def _remove_tracers() -> None:
    global _fia_original, _pa_original
    if _fia_original is not None:
        torch_npu.npu_fused_infer_attention_score = _fia_original
        _fia_original = None
    if _pa_original is not None:
        torch_npu._npu_paged_attention = _pa_original
        _pa_original = None


def verify_call_trace(spec: AttnSpec, num_calls: int = 5) -> dict[str, int]:
    """Run the attn func N times with call tracing, return call counts."""
    device = torch.device("npu")
    attn_func = create_attn_func(spec, device)

    _install_tracers()
    _call_counts.clear()
    try:
        for _ in range(num_calls):
            attn_func()
        torch.npu.synchronize()
    finally:
        _remove_tracers()

    return dict(_call_counts)


# ---------------------------------------------------------------------------
# Method 2: NPU profiler kernel capture
# ---------------------------------------------------------------------------

def verify_profiler(spec: AttnSpec) -> list[str]:
    """Run one forward pass under NPU profiler, return kernel names."""
    device = torch.device("npu")
    attn_func = create_attn_func(spec, device)

    for _ in range(3):
        attn_func()
    torch.npu.synchronize()

    kernel_names = []
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=lambda prof: None,
    ) as prof:
        for _ in range(2):
            attn_func()
            torch.npu.synchronize()
            prof.step()

    for evt in prof.key_averages():
        if evt.device_time_total > 0:
            kernel_names.append(evt.key)

    return kernel_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify attention kernel paths")
    parser.add_argument(
        "--op-types", nargs="+", default=["context", "generation"],
        choices=["context", "generation"],
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--method", choices=["trace", "profiler", "both"], default="trace",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Initializing vLLM context...")
    _init_vllm_context()

    op_map = {"context": OP_CONTEXT, "generation": OP_GENERATION}

    for op_name in args.op_types:
        op_type = op_map[op_name]
        spec = AttnSpec(
            op_type=op_type,
            batch=args.batch,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
        )
        print(f"\n{'='*60}")
        print(f"  Verifying: {op_type}")
        print(f"  batch={spec.batch} seq={spec.seq_len} "
              f"heads={spec.num_heads} kv_heads={spec.num_kv_heads}")
        print(f"{'='*60}")

        # --- Call tracing ---
        if args.method in ("trace", "both"):
            counts = verify_call_trace(spec, num_calls=5)
            print(f"\n  [Call Trace] 5 forward() calls:")
            if not counts:
                print("    (no traced calls detected)")
            for fn_name, count in sorted(counts.items()):
                print(f"    {fn_name}: {count} calls")

            if op_type == OP_CONTEXT:
                fia_count = counts.get("torch_npu.npu_fused_infer_attention_score", 0)
                if fia_count >= 5:
                    print("    -> PASS: npu_fused_infer_attention_score called (FIA kernel)")
                else:
                    print(f"    -> FAIL: expected >=5 FIA calls, got {fia_count}")
                pa_count = counts.get("torch_npu._npu_paged_attention", 0)
                if pa_count > 0:
                    print("    -> WARN: _npu_paged_attention called in context mode (unexpected)")

            elif op_type == OP_GENERATION:
                pa_count = counts.get("torch_npu._npu_paged_attention", 0)
                if pa_count >= 5:
                    print("    -> PASS: _npu_paged_attention called (PA kernel)")
                else:
                    print(f"    -> FAIL: expected >=5 PA calls, got {pa_count}")
                fia_count = counts.get("torch_npu.npu_fused_infer_attention_score", 0)
                if fia_count > 0:
                    print("    -> WARN: npu_fused_infer_attention_score called in generation mode (unexpected)")

        # --- NPU profiler ---
        if args.method in ("profiler", "both"):
            print(f"\n  [NPU Profiler] Kernel names:")
            try:
                kernels = verify_profiler(spec)
                if kernels:
                    for kname in kernels:
                        print(f"    {kname}")
                else:
                    print("    (no kernels captured)")
            except Exception as e:
                print(f"    Profiler failed: {e}")

    print()


if __name__ == "__main__":
    main()
