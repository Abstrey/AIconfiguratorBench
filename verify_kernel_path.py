"""Verify that collect_gemm benchmarks exercise the correct vLLM-Ascend kernel path.

Two verification methods:
  1. Call tracing  — monkey-patch key torch_npu / torch APIs, count invocations
  2. NPU profiler — capture one forward pass, print actual NPU kernel names

Usage:
    python verify_kernel_path.py --quant-types bf16 w8a8_dynamic
"""

import argparse
import functools
import logging
from collections import defaultdict

import torch
import torch_npu  # noqa: F401

from gemm_factory import (
    QUANT_BF16,
    QUANT_W8A8_DYNAMIC,
    SUPPORTED_QUANT_TYPES,
    GemmSpec,
    _init_vllm_context,
    create_gemm_func,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method 1: Call tracing via monkey-patch
# ---------------------------------------------------------------------------

_call_counts: dict[str, int] = defaultdict(int)
_originals: dict[str, object] = {}


def _wrap(module, attr_name, tag: str):
    """Replace module.attr_name with a counting wrapper."""
    original = getattr(module, attr_name)
    _originals[tag] = original

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        _call_counts[tag] += 1
        return original(*args, **kwargs)

    setattr(module, attr_name, wrapper)


def _unwrap(module, attr_name, tag: str):
    """Restore original function."""
    if tag in _originals:
        setattr(module, attr_name, _originals.pop(tag))


def _install_tracers():
    """Install call-counting wrappers on key functions."""
    _call_counts.clear()
    # torch_npu quantization APIs (W8A8 path)
    _wrap(torch_npu, "npu_dynamic_quant", "torch_npu.npu_dynamic_quant")
    _wrap(torch_npu, "npu_quant_matmul", "torch_npu.npu_quant_matmul")
    # F.linear (BF16 path — final kernel call)
    _wrap(torch.nn.functional, "linear", "F.linear")
    # vllm custom op (BF16 path — vllm-ascend dispatch)
    try:
        import vllm.model_executor.layers.utils as vllm_utils
        _wrap(vllm_utils, "default_unquantized_gemm", "vllm.default_unquantized_gemm")
    except (ImportError, AttributeError):
        pass


def _remove_tracers():
    """Remove all wrappers."""
    _unwrap(torch_npu, "npu_dynamic_quant", "torch_npu.npu_dynamic_quant")
    _unwrap(torch_npu, "npu_quant_matmul", "torch_npu.npu_quant_matmul")
    _unwrap(torch.nn.functional, "linear", "F.linear")
    try:
        import vllm.model_executor.layers.utils as vllm_utils
        _unwrap(vllm_utils, "default_unquantized_gemm", "vllm.default_unquantized_gemm")
    except (ImportError, AttributeError):
        pass


def verify_call_trace(spec: GemmSpec, num_calls: int = 5) -> dict[str, int]:
    """Run the GEMM func N times with call tracing, return call counts."""
    device = torch.device("npu")
    gemm_func = create_gemm_func(spec, device)

    _install_tracers()
    _call_counts.clear()
    try:
        for _ in range(num_calls):
            gemm_func()
        torch.npu.synchronize()
    finally:
        _remove_tracers()

    return dict(_call_counts)


# ---------------------------------------------------------------------------
# Method 2: NPU profiler kernel capture
# ---------------------------------------------------------------------------

def verify_profiler(spec: GemmSpec) -> list[str]:
    """Run one forward pass under NPU profiler, return kernel names."""
    device = torch.device("npu")
    gemm_func = create_gemm_func(spec, device)

    # Warmup outside profiler
    for _ in range(3):
        gemm_func()
    torch.npu.synchronize()

    kernel_names = []
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=lambda prof: None,  # we'll read events directly
    ) as prof:
        for _ in range(2):  # warmup=1 + active=1
            gemm_func()
            torch.npu.synchronize()
            prof.step()

    # Extract kernel names from profiler events
    for evt in prof.key_averages():
        if evt.device_time_total > 0:
            kernel_names.append(evt.key)

    return kernel_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify GEMM kernel paths")
    parser.add_argument(
        "--quant-types", nargs="+", default=["bf16", "w8a8_dynamic"],
        choices=list(SUPPORTED_QUANT_TYPES),
    )
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument(
        "--method", choices=["trace", "profiler", "both"], default="trace",
        help="Verification method: trace (call counting), profiler (NPU kernel names), both",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    logger.info("Initializing vLLM context...")
    _init_vllm_context()

    for qt in args.quant_types:
        spec = GemmSpec(m=args.m, n=args.n, k=args.k, quant_type=qt)
        print(f"\n{'='*60}")
        print(f"  Verifying: M={spec.m} N={spec.n} K={spec.k} quant={qt}")
        print(f"{'='*60}")

        # --- Call tracing ---
        if args.method in ("trace", "both"):
            counts = verify_call_trace(spec, num_calls=5)
            print(f"\n  [Call Trace] 5 forward() calls:")
            for fn_name, count in sorted(counts.items()):
                print(f"    {fn_name}: {count} calls")

            # Validate expected path
            if qt == QUANT_BF16:
                if counts.get("F.linear", 0) >= 5:
                    print("    -> PASS: F.linear called (BF16 CANN AOL MatMul path)")
                else:
                    print("    -> FAIL: F.linear NOT called!")
                if counts.get("vllm.default_unquantized_gemm", 0) >= 5:
                    print("    -> PASS: vllm.default_unquantized_gemm called (full vllm-ascend dispatch)")
                elif counts.get("vllm.default_unquantized_gemm", 0) > 0:
                    print(f"    -> WARN: vllm.default_unquantized_gemm called {counts['vllm.default_unquantized_gemm']}x (expected 5)")
                else:
                    print("    -> WARN: vllm.default_unquantized_gemm NOT called (patch may not be loaded)")
                if counts.get("torch_npu.npu_quant_matmul", 0) > 0:
                    print("    -> WARN: npu_quant_matmul called in BF16 mode (unexpected)")

            elif qt == QUANT_W8A8_DYNAMIC:
                ok = True
                if counts.get("torch_npu.npu_dynamic_quant", 0) >= 5:
                    print("    -> PASS: npu_dynamic_quant called (per-token INT8 quantization)")
                else:
                    print("    -> FAIL: npu_dynamic_quant NOT called!")
                    ok = False
                if counts.get("torch_npu.npu_quant_matmul", 0) >= 5:
                    print("    -> PASS: npu_quant_matmul called (INT8 GEMM kernel)")
                else:
                    print("    -> FAIL: npu_quant_matmul NOT called!")
                    ok = False
                if counts.get("F.linear", 0) > 0:
                    print("    -> WARN: F.linear also called in W8A8 mode (unexpected)")

        # --- NPU profiler ---
        if args.method in ("profiler", "both"):
            print(f"\n  [NPU Profiler] Kernel names:")
            try:
                kernels = verify_profiler(spec)
                if kernels:
                    for kname in kernels:
                        print(f"    {kname}")
                else:
                    print("    (no kernels captured — profiler may need different config)")
            except Exception as e:
                print(f"    Profiler failed: {e}")
                print("    (try running with msprof instead)")

    print()


if __name__ == "__main__":
    main()
