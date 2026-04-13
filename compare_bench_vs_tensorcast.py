"""Compare bench data vs TensorCast profiling data by operator + shape."""

import csv
import sys
from pathlib import Path

BENCH_DIR = Path("/Users/hudingyi/Downloads/仿真/benchdata")
TC_DIR = Path(
    "/Users/hudingyi/Downloads/msmodeling/tensor_cast/performance_model/"
    "profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/"
    "vllm0.18.0_torch2.9.0_cann8.5"
)


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _norm_shape(s: str) -> str:
    """Normalize shape string: strip quotes and whitespace."""
    return s.strip().strip('"').strip("'")


# ── GEMM comparison ──────────────────────────────────────────────────
# PLACEHOLDER_GEMM


# ── Attn comparison ──────────────────────────────────────────────────
# PLACEHOLDER_ATTN


# ── MoE comparison ───────────────────────────────────────────────────
# PLACEHOLDER_MOE


def compare_gemm():
    """Compare GEMM: MatMulV2 (BF16) and QuantBatchMatmulV3 (W8A8)."""
    for kernel in ("MatMulV2", "QuantBatchMatmulV3"):
        bench_path = BENCH_DIR / "gemm_data" / f"{kernel}.csv"
        tc_path = TC_DIR / f"{kernel}.csv"
        if not bench_path.exists() or not tc_path.exists():
            print(f"[GEMM] Skipping {kernel}: file not found")
            continue

        bench_rows = _read_csv(bench_path)
        tc_rows = _read_csv(tc_path)

        # Build TC lookup: Input Shapes → Average Duration or Profiling Average Duration
        tc_lookup: dict[str, float] = {}
        for r in tc_rows:
            shape = _norm_shape(r.get("Input Shapes", ""))
            dur_key = "Average Duration(us)" if "Average Duration(us)" in r else "Profiling Average Duration(us)"
            dur = r.get(dur_key, "")
            if shape and dur:
                try:
                    tc_lookup[shape] = float(dur)
                except ValueError:
                    pass

        # Match
        matched = 0
        diffs = []
        print(f"\n{'='*80}")
        print(f"[GEMM] {kernel}: bench={len(bench_rows)} rows, TC={len(tc_rows)} rows, TC lookup={len(tc_lookup)} shapes")
        print(f"{'Shape':<35} {'Bench(us)':>10} {'TC(us)':>10} {'Ratio':>8} {'Diff%':>8}")
        print("-" * 80)

        for r in bench_rows:
            shape = _norm_shape(r.get("Input Shapes", ""))
            bench_dur = r.get("Average Duration(us)", "")
            if not bench_dur:
                continue
            bench_us = float(bench_dur)

            if shape in tc_lookup:
                tc_us = tc_lookup[shape]
                ratio = bench_us / tc_us if tc_us > 0 else float("inf")
                diff_pct = (bench_us - tc_us) / tc_us * 100 if tc_us > 0 else 0
                diffs.append((shape, bench_us, tc_us, ratio, diff_pct))
                matched += 1

        # Sort by diff% descending, show top 20 + summary
        diffs.sort(key=lambda x: abs(x[4]), reverse=True)
        for shape, b, t, ratio, diff in diffs[:20]:
            print(f"{shape:<35} {b:>10.2f} {t:>10.2f} {ratio:>8.2f}x {diff:>+7.1f}%")
        if len(diffs) > 20:
            print(f"  ... ({len(diffs) - 20} more matched rows)")

        if diffs:
            ratios = [d[3] for d in diffs]
            avg_ratio = sum(ratios) / len(ratios)
            median_ratio = sorted(ratios)[len(ratios) // 2]
            print(f"\nSummary: {matched} matched / {len(bench_rows)} bench / {len(tc_rows)} TC")
            print(f"  Avg ratio (bench/TC): {avg_ratio:.2f}x, Median: {median_ratio:.2f}x")
        else:
            print(f"  No matching shapes found ({len(bench_rows)} bench, {len(tc_lookup)} TC)")


def compare_attn():
    """Compare Attention: FusedInferAttentionScore."""
    # Attn shapes are very different between bench and TC (bench uses simple T,H,D; TC uses full FIA params)
    # Match by extracting num_tokens and num_heads from both
    for suffix in ("", "_Decode"):
        kernel = f"FusedInferAttentionScore{suffix}"
        bench_path = BENCH_DIR / "attn_data" / f"{kernel}.csv"
        tc_path = TC_DIR / "FusedInferAttentionScore.csv"  # TC has one file for both
        if not bench_path.exists() or not tc_path.exists():
            print(f"[Attn] Skipping {kernel}: file not found")
            continue

        bench_rows = _read_csv(bench_path)
        tc_rows = _read_csv(tc_path)

        print(f"\n{'='*80}")
        print(f"[Attn] {kernel}: bench={len(bench_rows)} rows, TC={len(tc_rows)} rows")
        print("  Note: Shape formats differ significantly between bench and TC.")
        print("  Bench uses simple (T,H,D) shapes; TC uses full FIA runtime params.")
        print("  Direct shape matching not feasible — manual comparison needed.")


def compare_moe():
    """Compare MoE: no TC equivalent exists."""
    for kernel in ("GroupedMatmul_MoE_BF16", "GroupedMatmul_MoE_W8A8"):
        bench_path = BENCH_DIR / "moe_data" / f"{kernel}.csv"
        tc_path = TC_DIR / f"{kernel}.csv"
        if not tc_path.exists():
            bench_rows = _read_csv(bench_path) if bench_path.exists() else []
            print(f"\n{'='*80}")
            print(f"[MoE] {kernel}: bench={len(bench_rows)} rows, TC=N/A (no TC equivalent)")
            print("  MoE data is new — no TensorCast profiling baseline exists.")
            continue


if __name__ == "__main__":
    compare_gemm()
    compare_attn()
    compare_moe()
