"""Compare production profiler kernel data vs bench data by exact shape.

Extracts unique kernel shapes from DSV3 profiler kernel_details.csv,
matches against bench CSV data, and outputs:
1. Matched shapes: profiler kernel Duration vs bench e2e Duration
2. Missing shapes: CLI commands to run supplementary bench collection
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROF_BASE = Path("/Users/hudingyi/Downloads/仿真/profiling/profiler-dsv3-0326")
BENCH_DIR = Path("/Users/hudingyi/Downloads/仿真/benchdata")

# All rank0 profiler directories
PROF_DIRS = sorted(
    p for p in PROF_BASE.iterdir()
    if p.is_dir() and "rank0" in p.name
)


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _norm(s: str) -> str:
    return s.strip().strip('"')


def _read_all_profiler_kernels() -> list[dict]:
    """Read kernel_details.csv from all rank0 profiler dirs."""
    all_rows = []
    for d in PROF_DIRS:
        kd = d / "kernel_details.csv"
        if kd.exists():
            rows = _read_csv(kd)
            scenario = d.name.replace("profiler-dsv3-", "")
            for r in rows:
                r["_scenario"] = scenario
            all_rows.extend(rows)
    return all_rows


def extract_profiler_gemm_bf16(rows):
    """Extract MatMulV2 shapes."""
    shapes = defaultdict(list)
    for r in rows:
        name = r.get("Name", "")
        if "MatMulV2" not in name:
            continue
        if any(k in name for k in ("QuantBatch", "TransposeBatch", "BatchMatMul")):
            continue
        shape = _norm(r.get("Input Shapes", ""))
        dur = float(r.get("Duration(us)", "0"))
        shapes[shape].append((dur, r["_scenario"]))
    result = {}
    for shape, entries in shapes.items():
        durs = [e[0] for e in entries]
        result[shape] = {"avg_us": sum(durs)/len(durs), "count": len(durs), "scenario": entries[0][1]}
    return result


def extract_profiler_gemm_w8a8(rows):
    """Extract QuantBatchMatmulV3, decode NZ format."""
    shapes = defaultdict(list)
    for r in rows:
        if "QuantBatchMatmulV3" not in r.get("Name", ""):
            continue
        inp = _norm(r.get("Input Shapes", ""))
        parts = inp.split(";")
        if len(parts) < 2:
            continue
        x_dims = parts[0].strip().split(",")
        w_dims = parts[1].strip().split(",")
        if len(x_dims) < 2 or len(w_dims) < 4:
            continue
        try:
            M, d0, d1 = int(x_dims[0]), int(w_dims[0]), int(w_dims[1])
            N, K = d0 * 32, d1 * 16
            dur = float(r.get("Duration(us)", "0"))
            shapes[(M, N, K)].append((dur, r["_scenario"]))
        except (ValueError, IndexError):
            continue
    result = {}
    for key, entries in shapes.items():
        durs = [e[0] for e in entries]
        result[key] = {"avg_us": sum(durs)/len(durs), "count": len(durs), "scenario": entries[0][1]}
    return result


def extract_profiler_attn(rows):
    """Extract FusedInferAttentionScore shapes."""
    shapes = defaultdict(list)
    for r in rows:
        if "FusedInferAttentionScore" not in r.get("Name", ""):
            continue
        shape = _norm(r.get("Input Shapes", ""))
        dur = float(r.get("Duration(us)", "0"))
        shapes[shape].append(dur)
    result = {}
    for shape, durs in shapes.items():
        dims = shape.split(";")[0].split(",") if shape else []
        result[shape] = {"avg_us": sum(durs)/len(durs), "count": len(durs), "desc": f"T={dims[0]}" if dims else "?"}
    return result


def extract_profiler_moe(rows):
    """Extract DispatchFFNCombine shapes."""
    shapes = defaultdict(list)
    for r in rows:
        if "DispatchFFNCombine" not in r.get("Name", ""):
            continue
        shape = _norm(r.get("Input Shapes", ""))
        dur = float(r.get("Duration(us)", "0"))
        shapes[shape].append(dur)
    result = {}
    for shape, durs in shapes.items():
        result[shape] = {"avg_us": sum(durs)/len(durs), "count": len(durs)}
    return result


def compare_gemm_bf16(prof_shapes):
    bench_path = BENCH_DIR / "gemm_data" / "MatMulV2.csv"
    bench = {}
    if bench_path.exists():
        for r in _read_csv(bench_path):
            s = _norm(r.get("Input Shapes", ""))
            d = r.get("Average Duration(us)", "")
            if d:
                bench[s] = float(d)

    print("=" * 100)
    print("GEMM BF16 (MatMulV2) -- Profiler kernel vs Bench e2e")
    print(f"{'Input Shapes':<40} {'Prof(us)':>10} {'Bench(us)':>10} {'Ratio':>8} {'Count':>6}")
    print("-" * 100)

    miss_m, miss_n, miss_k = set(), set(), set()
    for shape, info in sorted(prof_shapes.items(), key=lambda x: -x[1]["count"]):
        p = info["avg_us"]
        b = bench.get(shape)
        ratio = f"{b/p:.2f}x" if b else "MISS"
        b_s = f"{b:.2f}" if b else "N/A"
        print(f"{shape:<40} {p:>10.2f} {b_s:>10} {ratio:>8} {info['count']:>6}")
        if b is None:
            parts = shape.split(";")
            if len(parts) == 2:
                mk, nk = parts[0].split(","), parts[1].split(",")
                if len(mk) >= 2:
                    miss_m.add(mk[0]); miss_k.add(mk[1])
                if len(nk) >= 1:
                    miss_n.add(nk[0])

    if miss_m:
        print(f"\n补采命令:")
        print(f"python collect_gemm.py --quant-types bf16 \\")
        print(f"    --m-list {' '.join(sorted(miss_m, key=int))} \\")
        print(f"    --n-list {' '.join(sorted(miss_n, key=int))} \\")
        print(f"    --k-list {' '.join(sorted(miss_k, key=int))}")


def compare_gemm_w8a8(prof_shapes):
    bench_path = BENCH_DIR / "gemm_data" / "QuantBatchMatmulV3.csv"
    bench = {}
    if bench_path.exists():
        for r in _read_csv(bench_path):
            s = _norm(r.get("Input Shapes", ""))
            out = _norm(r.get("Output Shapes", ""))
            d = r.get("Average Duration(us)", "")
            if not d:
                continue
            x_dims = s.split(";")[0].split(",")
            o_dims = out.split(",")
            if len(x_dims) >= 2 and len(o_dims) >= 2:
                try:
                    bench[(int(x_dims[0]), int(o_dims[1]), int(x_dims[1]))] = float(d)
                except ValueError:
                    pass

    print("\n" + "=" * 100)
    print("GEMM W8A8 (QuantBatchMatmulV3) -- Profiler kernel vs Bench e2e")
    print(f"{'(M, N, K)':<30} {'Prof(us)':>10} {'Bench(us)':>10} {'Ratio':>8} {'Count':>6}")
    print("-" * 100)

    miss_m, miss_n, miss_k = set(), set(), set()
    for key, info in sorted(prof_shapes.items(), key=lambda x: (-x[1]["count"], x[0])):
        M, N, K = key
        p = info["avg_us"]
        b = bench.get(key)
        ratio = f"{b/p:.2f}x" if b else "MISS"
        b_s = f"{b:.2f}" if b else "N/A"
        label = f"({M}, {N}, {K})"
        print(f"{label:<30} {p:>10.2f} {b_s:>10} {ratio:>8} {info['count']:>6}")
        if b is None:
            miss_m.add(str(M)); miss_n.add(str(N)); miss_k.add(str(K))

    if miss_m:
        print(f"\n补采命令:")
        print(f"python collect_gemm.py --quant-types w8a8_dynamic \\")
        print(f"    --m-list {' '.join(sorted(miss_m, key=int))} \\")
        print(f"    --n-list {' '.join(sorted(miss_n, key=int))} \\")
        print(f"    --k-list {' '.join(sorted(miss_k, key=int))}")


def compare_attn(prof_shapes):
    print("\n" + "=" * 100)
    print("Attn (FusedInferAttentionScore) -- Profiler kernel shapes")
    print("Note: DSV3 uses MLA, bench uses standard FIA -- shapes differ fundamentally")
    print(f"{'Desc':<15} {'Prof(us)':>10} {'Count':>6} {'Input (first 60)':<60}")
    print("-" * 100)
    for shape, info in sorted(prof_shapes.items(), key=lambda x: -x[1]["count"]):
        print(f"{info['desc']:<15} {info['avg_us']:>10.2f} {info['count']:>6} {shape[:60]}")


def compare_moe(prof_shapes):
    print("\n" + "=" * 100)
    print("MoE (DispatchFFNCombine) -- Profiler MC2 vs Bench AllGather")
    print("Note: MC2 = fused dispatch+FFN+combine with 8-card comm")

    bench_moe = {}
    for qt in ("BF16", "W8A8"):
        bp = BENCH_DIR / "moe_data" / f"GroupedMatmul_MoE_{qt}.csv"
        if not bp.exists():
            continue
        for r in _read_csv(bp):
            if r.get("Model", "") == "deepseek-v3":
                t = r.get("Num Tokens", "")
                d = r.get("Average Duration(us)", "")
                if d:
                    bench_moe[f"{qt}_T={t}"] = float(d)

    print(f"\nProfiler DispatchFFNCombine (MC2):")
    for shape, info in sorted(prof_shapes.items(), key=lambda x: -x[1]["count"]):
        print(f"  avg={info['avg_us']:.2f}us  count={info['count']}")

    if bench_moe:
        print(f"\nBench MoE DSV3 (AllGather, single card):")
        for k, v in sorted(bench_moe.items()):
            print(f"  {k}: {v:.2f}us")


def main():
    print("Loading profiler data from all rank0 directories...")
    all_rows = _read_all_profiler_kernels()
    print(f"Total kernels: {len(all_rows)} from {len(PROF_DIRS)} directories\n")

    compare_gemm_bf16(extract_profiler_gemm_bf16(all_rows))
    compare_gemm_w8a8(extract_profiler_gemm_w8a8(all_rows))
    compare_attn(extract_profiler_attn(all_rows))
    compare_moe(extract_profiler_moe(all_rows))


if __name__ == "__main__":
    main()
