"""NPU Event-based timing engine for operator benchmarking.

Aligned with AIConfigurator's benchmark_with_power() timing rules:
- Single Event pair wrapping multiple runs (no per-iteration synchronize)
- latency = total_elapsed / num_runs / repeat_n
- Only 1 synchronize at the end (vs 200 in the old per-iteration approach)
"""

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class BenchResult:
    """Immutable benchmark result."""

    avg_us: float
    num_runs: int
    repeat_n: int


def benchmark_npu(
    kernel_func: Callable[[], None],
    warmup_iters: int = 10,
    num_runs: int = 100,
    repeat_n: int = 1,
) -> BenchResult:
    """Benchmark a kernel function using NPU Event timing.

    Matches AIConfigurator benchmark_with_power() timing rules:
    1. Warmup N iterations (eager, no timing)
    2. Single Event pair wrapping num_runs × repeat_n executions
    3. latency = total_elapsed / num_runs / repeat_n

    Args:
        kernel_func: Zero-arg callable that runs the kernel once.
        warmup_iters: Number of warmup iterations before measurement.
        num_runs: Number of timed runs (outer loop).
        repeat_n: Repetitions per run (inner loop, for multi-op kernels).

    Returns:
        BenchResult with average latency in microseconds.
    """
    # Phase 1: Warmup
    for _ in range(warmup_iters):
        for _ in range(repeat_n):
            kernel_func()
    torch.npu.synchronize()

    # Phase 2: Timed execution — single Event pair, no per-iteration sync
    start_evt = torch.npu.Event(enable_timing=True)
    end_evt = torch.npu.Event(enable_timing=True)

    start_evt.record()
    for _ in range(num_runs):
        for _ in range(repeat_n):
            kernel_func()
    end_evt.record()
    torch.npu.synchronize()

    total_ms = start_evt.elapsed_time(end_evt)
    avg_us = total_ms * 1000.0 / num_runs / repeat_n

    return BenchResult(
        avg_us=avg_us,
        num_runs=num_runs,
        repeat_n=repeat_n,
    )
