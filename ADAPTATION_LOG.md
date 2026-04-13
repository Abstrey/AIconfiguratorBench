# AIConfigurator Bench — NPU 算子采集适配记录

> 项目路径: `/Users/hudingyi/Downloads/AIconfiguratorBench/`
> 基于 AIConfigurator 的 benchmark 设计，适配 CANN + vLLM Ascend 平台
> 创建日期: 2026-04-09

---

## 一、已完成：GEMM 算子采集

### 1.1 实现概要

| 文件 | 行数 | 职责 |
|------|------|------|
| `bench_engine.py` | 63 | NPU Event 计时引擎 (warmup → per-iteration Event → median) |
| `gemm_factory.py` | ~180 | GEMM 算子工厂 (BF16 / W8A8_DYNAMIC 两条路径) |
| `collect_gemm.py` | ~360 | GEMM 主脚本 (参数空间 × 量化类型 笛卡尔积, CSV 输出, checkpoint 断点续传) |
| `attn_factory.py` | ~265 | Attention 算子工厂 — 通过 `AscendAttentionBackendImpl.forward()` 调用 |
| `collect_attn.py` | ~400 | Attention 主脚本 (参数空间 × op_type 笛卡尔积, CSV 输出, checkpoint) |
| `verify_kernel_path.py` | ~220 | GEMM 验证脚本 (call trace + NPU profiler) |
| `verify_attn_kernel_path.py` | ~230 | Attention 验证脚本 (call trace FIA/PA) |

### 1.2 vLLM 接口调用链对齐验证

对照 AIConfigurator 源码 `collector/vllm/collect_gemm.py`，确认 NPU 移植版走完整的 vllm-ascend 框架层调用链：

**AIConfigurator 原版 (GPU):**
```
RowParallelLinear(quant_config=None) → UnquantizedLinearMethod.apply() → F.linear() → cuBLAS
RowParallelLinear(quant_config=Fp8Config) → Fp8LinearMethod.apply() → cutlass_scaled_mm() → CUTLASS
```

**NPU 移植版 (已对齐):**
```
AscendRowParallelLinear(quant_config=None)
  → AscendUnquantizedLinearMethod.apply()
    → default_unquantized_gemm() (vllm-ascend patch)
      → torch.ops.vllm.unquantized_gemm() (PrivateUse1 dispatch)
        → F.linear() → CANN AOL MatMul

AscendRowParallelLinear(quant_config=_BenchW8A8Config)
  → AscendLinearMethod.apply()
    → AscendW8A8DynamicLinearMethod.apply()
      → torch_npu.npu_dynamic_quant() + torch_npu.npu_quant_matmul()
```

关键对齐点:
- 分布式初始化: `init_distributed_environment()` + `ensure_model_parallel_initialized(1, 1)` (和 AIConfigurator `setup_distributed()` 一致)
- BF16: 通过 `AscendRowParallelLinear.forward()` 走完整 dispatch，不直接调 `F.linear()`
- W8A8: 通过 `_BenchW8A8Config` 轻量 QuantizationConfig 注入 scheme，走 `create_weights()` + `process_weights_after_loading()` 完整流程
- 权重后处理: `process_weights_after_loading()` 触发 NZ format 转换 (`maybe_trans_nz`)

call trace 验证结果 (M=128, N=4096, K=4096):
- BF16: `F.linear` 5/5 ✓, `vllm.default_unquantized_gemm` 5/5 ✓
- W8A8: `npu_dynamic_quant` 5/5 ✓, `npu_quant_matmul` 5/5 ✓

### 1.2.1 实现细节

**bench_engine.py — 计时引擎**

复用 TensorCast `generate_comm_microbench.py` 的 NPU Event 计时模式：
```python
# warmup N 次 → per-iteration torch.npu.Event 计时 → 取 median
# 返回 BenchResult(median_us, min_us, max_us, std_us, num_iters)
```

**gemm_factory.py — 算子工厂**

BF16 路径:
```python
import vllm_ascend.patch.worker  # 加载 unquantized_gemm custom op patch
gemm = AscendRowParallelLinear(input_size=K, output_size=N, quant_config=None, disable_tp=True)
gemm.quant_method.process_weights_after_loading(gemm)  # NZ format 转换
output = gemm.forward(x)  # 走完整 vllm-ascend dispatch
```

W8A8_DYNAMIC 路径:
```python
# 轻量 QuantizationConfig，绕过 AscendModelSlimConfig 对 quant_model_description.json 的依赖
class _BenchW8A8Config(QuantizationConfig):
    def get_quant_method(self, layer, prefix=""):
        return AscendLinearMethod(AscendW8A8DynamicLinearMethod())

gemm = AscendRowParallelLinear(input_size=K, output_size=N, quant_config=_BenchW8A8Config(), disable_tp=True)
# 填充随机 INT8 权重 + scale + offset
gemm.quant_method.process_weights_after_loading(gemm)  # transpose + NZ format + flatten scales
output = gemm.forward(x)  # npu_dynamic_quant + npu_quant_matmul
```

初始化 (和 AIConfigurator `setup_distributed()` 对齐):
```python
init_distributed_environment()  # RANK=0, WORLD_SIZE=1, 自动找可用端口
ensure_model_parallel_initialized(1, 1)  # 单卡 TP=1, PP=1
set_current_vllm_config(VllmConfig())
```

**collect_gemm.py — 主脚本**

- 参数空间: M=[1..16384] × N=[256..16384] × K=[256..16384] × quant=[bf16, w8a8_dynamic]
- 输出: `MatMulV2.csv` (BF16) / `QuantBatchMatmulV3.csv` (W8A8)，文件名匹配 op_mapping.yaml kernel_type
- checkpoint: 每 10 个 shape 写一次 `checkpoint.json`，支持 `--resume` 断点续传
- 内存管理: 每个 shape 完成后 `del gemm_func` + `torch.npu.empty_cache()` 防 OOM

### 1.3 与 AIConfigurator 原版的已知差异

| 维度 | AIConfigurator 原版 | NPU 移植版 | 影响 |
|------|-------------------|-----------|------|
| L2 Cache 冲刷 | 6 个独立 GEMM 实例轮转 | 单个实例 | latency 可能偏乐观 |
| CUDA Graph | Graph capture → replay | 无 Graph，Event 计时 | 包含 launch overhead |
| 计时方式 | 总时间 / (runs × repeat × 6ops) 取平均 | per-iteration Event 取 median | median 更抗干扰 |
| 参数空间 | ~200 M × 18 NK ≈ 97K 组合 | 15 M × 9 NK ≈ 1215 组合 | 可通过 CLI 扩展 |
| 量化类型 | FP16/FP8/FP8_Block/NVFP4 | BF16/W8A8_DYNAMIC | 不同硬件量化路线 |

### 1.4 与 TensorCast op_replay 的对比

TensorCast 已有 `tools/perf_data_collection/op_replay/` 下的 microbench 采集体系（`MatMulV2_run.py` / `QuantBatchMatmulV3_run.py`），由 `start_microbench.py` 编排，通过 msprof 外部挂载采集。两者的调用接口差异如下：

#### BF16 GEMM 调用链对比

```
collect_gemm (本脚本):
  AscendRowParallelLinear.forward(x)
    → AscendUnquantizedLinearMethod.apply(layer, x)
      → default_unquantized_gemm(layer, x, weight, bias)     ← vllm-ascend patch
        → torch.ops.vllm.unquantized_gemm(x, weight, bias)   ← PrivateUse1 custom op
          → torch.nn.functional.linear(x, weight, bias)
            → CANN aclnnMm → MatMulV2/V3 kernel

op_replay/MatMulV2_run.py (TensorCast):
  torch.mm(input_a, input_b.T)
    → CANN aclnnMm → MatMulV2/V3 kernel
```

#### W8A8 GEMM 调用链对比

```
collect_gemm (本脚本):
  AscendRowParallelLinear.forward(x)
    → AscendLinearMethod.apply(layer, x, bias)
      → AscendW8A8DynamicLinearMethod.apply(layer, x, bias, tp_rank)
        → torch_npu.npu_dynamic_quant(x)                     ← per-token 动态量化 (BF16→INT8)
        → torch_npu.npu_quant_matmul(                         ← 量化矩阵乘
            quantized_x, layer.weight, layer.weight_scale,
            pertoken_scale=pertoken_scale, output_dtype=x.dtype)
          → CANN aclnnWeightQuantBatchMatmulV2/V3 kernel

op_replay/QuantBatchMatmulV3_run.py (TensorCast):
  torch_npu.npu_quant_matmul(                                 ← 直接调，无动态量化
      x_tensor,          ← 已经是 INT8 (从 CSV 重建)
      weight_tensor,     ← INT8 FRACTAL_NZ
      scale_tensor,
      bias=..., offset=..., pertoken_scale=..., output_dtype=...)
    → CANN aclnnWeightQuantBatchMatmulV2/V3 kernel
```

#### 差异总结

| 维度 | collect_gemm (本脚本) | op_replay (TensorCast) |
|------|----------------------|------------------------|
| 算子调用层级 | vllm-ascend 框架层 (`AscendRowParallelLinear`) | PyTorch/torch_npu 原生 API (`torch.mm` / `npu_quant_matmul`) |
| BF16 dispatch | vllm custom op → F.linear | torch.mm 直接调 |
| W8A8 量化步骤 | 包含 `npu_dynamic_quant` (BF16→INT8) | 跳过（输入已是 INT8） |
| W8A8 权重格式 | NZ format (通过 `process_weights_after_loading`) | FRACTAL_NZ (从 CSV 重建 + graph mode) |
| 最终 kernel | 相同: aclnnMm / aclnnWeightQuantBatchMatmulV2/V3 | 相同 |
| 测量范围 | 完整 forward (含 dispatch + 量化开销) | 纯 kernel (不含量化) |
| shape 来源 | 穷举 M×N×K 笛卡尔积，和模型无关 | 从已有 CSV 读 shape (只 replay 已知 shape) |
| 计时方式 | `torch.npu.Event` 自行计时 | `msprof` 外部挂载，从 `op_summary` 取 `Task Duration(us)` min |
| 依赖 | 需要 vllm + vllm-ascend | 只需要 torch + torch_npu |
| 用途 | 建稠密 lookup table，覆盖未 profiling 的 shape | 对已有 profiling shape 做独立验证/补充 microbench 数据 |

核心区别：collect_gemm 测的是"从 vllm-ascend 调一次 forward 的端到端 latency"（包含 dispatch 和量化开销），op_replay 测的是"单个 CANN kernel 的纯执行时间"。两者互补——collect_gemm 更接近真实推理场景，op_replay 更接近 kernel 级理论性能。

### 1.5 NPU 实测结果 (M=1/128/4096, N=K=4096)

| M | BF16 (us) | W8A8_DYNAMIC (us) | 加速比 |
|---|-----------|-------------------|--------|
| 1 | 62.07 | 85.12 | 0.73x (量化开销 > 计算收益) |
| 128 | 65.71 | 97.80 | 0.67x (memory-bound 区间) |
| 4096 | 474.91 | 398.51 | 1.19x (compute-bound, INT8 算力优势) |

小 M 时 W8A8 更慢 — `npu_dynamic_quant` 的 per-token 量化有固定开销，在 memory-bound 区间吃不回来。大 M 时 W8A8 快 ~19%，符合 INT8 权重读取减半 + Cube Core INT8 算力更高的预期。

### 1.6 CSV 格式对齐 TensorCast

输出 CSV 直接兼容 TensorCast `ProfilingDataSource` 查询：

```
# MatMulV2.csv (BF16)
OP State,Accelerator Core,Input Shapes,Input Data Types,...,Average Duration(us),...
dynamic,AI_CORE,"128,4096;4096,4096",DT_BF16;DT_BF16,...,65.71,...

# QuantBatchMatmulV3.csv (W8A8)
dynamic,MIX_AIC,"128,4096;4096,4096",INT8;INT8,...,97.80,...
```

关键对齐点:
- 文件名 = op_mapping.yaml 的 `kernel_type` (MatMulV2 / QuantBatchMatmulV3)
- W8A8 的 `Accelerator Core` = `MIX_AIC` (AI Core + Vector Core)
- W8A8 的 `Input Data Types` = `INT8;INT8` (npu_dynamic_quant 后输入已是 INT8)
- `Average Duration(us)` 列是 ProfilingDataSource 的最高优先级 latency 来源
- op_mapping 的 `tc_input_count: 2` 使 shape matching 只比较前 2 个输入

使用方式:
```bash
cp gemm_data/MatMulV2.csv  <profiling_database>/data/{device}/{backend}/{version}/
cp gemm_data/QuantBatchMatmulV3.csv  <profiling_database>/data/{device}/{backend}/{version}/
```

### 1.7 已知限制

- **未做 L2 cache 冲刷**: AIConfigurator 用 6-op 轮转模拟 cache 竞争，NPU 版已实现 6-op 轮转（`OUTSIDE_LOOP_COUNT=6`）
- **未做 NPU Graph**: Da Vinci 架构没有 CUDA Graph 等价机制，每次 forward 包含完整 Python dispatch 开销
- **NZ format 未生效**: 环境 `VLLM_ASCEND_ENABLE_NZ` 未开启时 `maybe_trans_nz()` 跳过，权重走 ND format
- **首个 shape 耗时长**: kernel JIT 编译 + warmup 导致首个 shape ~13s，后续 ~0.2s/shape

### 1.8 Bench vs 生产 Profiler 精确对比 (DSV3)

使用 DSV3 生产 profiler (`profiler-dsv3-0326`) 的 `kernel_details.csv` 提取真实 shape，与 bench 补采数据做 kernel 级对比。

#### GEMM BF16 (MatMulV2)

| M | N | K | Profiler kernel(us) | Bench e2e(us) | Ratio | 说明 |
|---|---|---|:---:|:---:|:---:|------|
| 1 | 256 | 7168 | 10.88 | 42.20 | 3.88x | decode gate_proj，框架开销 ~31us |
| 1 | 16160 | 7168 | 220.46 | 191.49 | 0.87x | LM head，bench 6-op 轮转有 cache 效应 |
| 4 | 256 | 7168 | 10.55 | 40.29 | 3.82x | decode，同上 |
| 4 | 16160 | 7168 | 221.30 | 190.57 | 0.86x | 大 shape 对齐 |
| 4096 | 4096 | 512 | 67.05 | 58.71 | 0.88x | prefill，基本对齐 |
| 8192 | 256 | 7168 | 208.88 | 127.30 | 0.61x | prefill 大 M |

#### GEMM W8A8 (QuantBatchMatmulV3)

| M | N | K | Profiler kernel(us) | Bench e2e(us) | Ratio | 说明 |
|---|---|---|:---:|:---:|:---:|------|
| 1 | 7168 | 256 | 11.79 | 48.72 | 4.13x | decode 最小 shape，框架开销占比最大 |
| 1 | 3072 | 1536 | 15.51 | 49.08 | 3.16x | shared expert gate_up |
| 1 | 2112 | 7168 | 28.44 | 56.72 | 1.99x | shared expert down_proj |
| 1 | 4608 | 7168 | 34.61 | 62.72 | 1.81x | MLA QKV proj |
| 8192 | 7168 | 2048 | 458.52 | 605.68 | 1.32x | prefill，趋于对齐 |
| 8192 | 3072 | 1536 | 164.47 | 262.78 | 1.60x | prefill |
| 8192 | 4608 | 7168 | 916.96 | 1270.06 | 1.39x | prefill 大 shape |

#### 偏差规律

| M 范围 | BF16 Ratio | W8A8 Ratio | 根因 |
|--------|:---:|:---:|------|
| M=1 (decode) | 3.8-3.9x | 1.8-4.1x | 固定框架开销 ~30-40us 占主导 |
| M=4096-8192 (prefill) | 0.6-0.9x | 1.3-1.6x | kernel 计算占主导，两者趋于一致 |

**根因分析**：

Bench 测的是 `AscendRowParallelLinear.forward()` 端到端（含 Python dispatch + vllm-ascend 框架层 + CANN kernel），Profiler 测的是纯 CANN kernel `Duration(us)`。差值即为框架 dispatch 开销：

- BF16 dispatch overhead ≈ 30-35us（`forward()` → `apply()` → `default_unquantized_gemm()` → `F.linear()` → CANN）
- W8A8 dispatch overhead ≈ 35-40us（`forward()` → `apply()` → `npu_dynamic_quant()` + `npu_quant_matmul()` → CANN）

小 M 时 kernel 执行 ~10us，dispatch ~35us，总计 ~45us，dispatch 占 78%。大 M 时 kernel 执行 ~500us，dispatch ~35us，dispatch 占 7%。

**AIConfigurator 的对比 — CUDA Graph capture + replay 机制详解**：

AIConfigurator (GPU) 也存在同样的 dispatch overhead 问题，但通过 **CUDA Graph capture + replay** 缓解。

**核心问题：CPU Launch Overhead**

GPU 上每次执行一个 kernel，CPU 都要经历完整的调度流程：
```
Python forward() → PyTorch dispatch → CUDA driver → kernel launch → GPU 执行
```
每次 launch 约 ~30us 开销。对于 decode 阶段（M=1~4），kernel 本身只跑 ~10us，但 launch 开销 ~30us，CPU 侧反而成了瓶颈。CUDA Graph 的核心思路是：把这条调度链录制一次，之后直接在 GPU 端回放，跳过所有 CPU 侧开销。

**Capture 阶段（录制）**

AIConfigurator 的 collector（`collector/helper.py:244-260`）实现：
```python
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        for _ in range(repeat_n):
            kernel_func()       # 不真正执行，录制操作序列
    torch.cuda.synchronize()
except Exception as e:
    if allow_graph_fail:
        torch.cuda.empty_cache()  # 清理 capture 失败的残留分配
        use_graph = False          # 回退到逐 kernel 执行
```

`torch.cuda.graph(g)` 上下文管理器把 stream 切换到 capture 模式：
- 所有 CUDA 操作（kernel launch、memcpy 等）**不会真正执行**
- 而是被录制为一个 DAG（有向无环图），节点是操作，边是依赖关系
- 录制完成后，graph 被固化到 GPU 端，包含完整的 kernel 参数、内存地址、执行顺序

**Capture 的约束**：
- 录制期间不能有 CPU-GPU 同步（`synchronize()`、`.item()`、`.cpu()`）
- 不能有动态控制流（`if tensor.sum() > 0` 这种依赖 GPU 数据的分支）
- 不能有动态 shape（每次执行的 tensor shape 必须和录制时一致）
- 不能分配新的 GPU 内存（所有 tensor 必须在 capture 前预分配）

**Replay 阶段（回放）**

```python
# Warmup replay path (helper.py:267-276)
for _ in range(num_warmups):
    g.replay()

# Measurement (helper.py:298-312)
start_event.record()
for _ in range(actual_num_runs):
    g.replay()          # 一次 CPU 调用 → GPU 端整个 graph 一次性提交
end_event.record()
torch.cuda.synchronize()
```

replay 时：
- CPU 只发出**一次** launch 命令（而不是 N 个 kernel 各一次）
- GPU 按照录制好的 DAG 顺序执行所有操作
- 跳过了 Python dispatch、PyTorch 调度、CUDA driver 逐个提交的**全部开销**
- 输入数据通过**原地写入**录制时的 tensor 地址来更新（地址固定，内容可变）

**Fallback 机制**

某些复杂操作（如 MoE 的 `fused_experts()`）可能 capture 失败。AIConfigurator 通过 `allow_graph_fail=True` 优雅降级（`helper.py:253-260`）：
- capture 失败 → `torch.cuda.empty_cache()` 清理残留 → 回退到逐 kernel eager 执行
- 返回 `used_cuda_graph` 标志，让调用方知道走了哪条路径
- 两条路径都用相同的 CUDA Event 计时，测量结果可比

**量化收益**

| 场景 | 无 Graph | 有 Graph | 收益 |
|------|---------|---------|------|
| Decode (M=1) | kernel 10us + launch 30us = 40us | kernel 10us + launch ~3us = 13us | ~3x |
| Prefill (M=4096) | kernel 500us + launch 30us = 530us | kernel 500us + launch ~3us = 503us | ~5% |

小 batch 收益巨大，大 batch 收益边际递减。这也解释了为什么 GPU bench 在小 M 时和 profiler 差距很小（graph 消除了 launch overhead），而 NPU bench 在小 M 时偏差 2-4x（无 graph，launch overhead 全部包含在测量中）。

**Generator 层的 CUDA Graph 配置**

AIConfigurator 不仅在 collector 中使用 CUDA Graph 做微基准测量，还在 generator 层为部署配置生成 CUDA Graph 参数（`rule_plugin/*.rule` + `backend_config_mapping.yaml`）：

| 参数 | 生产环境 | 基准测试 | 说明 |
|------|---------|---------|------|
| `cuda_graph_batch_sizes` | 粗粒度（1-16步长1, 16-32步长4, ...） | `range(1, max_batch_size+1)` 全量 | 生产减少 capture 时间，基准对齐仿真 |
| `cuda_graph_enable_padding` | `true` | `true` | 相近 batch size 复用同一个 graph |
| `disable_cuda_graph` | `false` | `false` | 映射到 vLLM `enforce-eager` / SGLang `disable-cuda-graph` |

**NPU 的架构性差异**

NPU (Da Vinci) 没有 CUDA Graph 等价机制，每次 `forward()` 都包含完整 Python dispatch。这是 NPU bench 在小 M 时偏差较大的根本原因，也是 GPU bench 和 NPU bench 的架构性差异。

| 维度 | GPU (AIConfigurator) | NPU (本脚本) |
|------|---------------------|-------------|
| Graph 支持 | CUDA Graph capture → replay | 无等价机制 |
| Launch overhead | ~3us (graph replay) | ~30-40us (每次完整 dispatch) |
| 小 M bench 精度 | 接近 profiler（graph 消除 overhead） | 偏高 2-4x（overhead 占主导） |
| 大 M bench 精度 | 接近 profiler | 接近 profiler（overhead 占比低） |
| 动态 shape | 需要为每个 batch size 预 capture | 天然支持（无 graph 约束） |

**对 TensorCast 寻优的影响**：

| 场景 | 影响 | 建议 |
|------|------|------|
| Decode (M=1-4) | bench 数据偏高 2-4x，lookup table 预测偏悲观 | 可减去固定 dispatch overhead 校准 |
| Prefill (M>1024) | bench 和 profiler 差距 <40%，可接受 | 直接使用 |
| MoE | bench 用 AllGather 路径，profiler 用 MC2，不可直接对比 | 仅用于相对比较 |

### 1.9 运行命令

```bash
# 快速验证
python collect_gemm.py --m-list 1 128 4096 --n-list 4096 --k-list 4096 --quant-types bf16 w8a8_dynamic

# 完整采集 (15 M × 9 N × 9 K × 2 quant = 2430 specs)
python collect_gemm.py --quant-types bf16 w8a8_dynamic --output-dir ./gemm_data

# 断点续传
python collect_gemm.py --quant-types bf16 w8a8_dynamic --output-dir ./gemm_data --resume

# 验证 kernel 路径
python verify_kernel_path.py --quant-types bf16 w8a8_dynamic --method trace
```

---

## 二、待适配：其他算子采集

### 2.1 算子优先级与难度评估

基于 AIConfigurator 的六类算子 + TensorCast 的 op_mapping 覆盖需求:

| 优先级 | 算子 | AIConfigurator 脚本 | TensorCast kernel_type | 难度 | 说明 |
|--------|------|--------------------|-----------------------|------|------|
| P0 | Attention | collect_attn.py | FusedInferAttentionScore / PagedAttention | 高 | **已完成** — Context (FIA) + Generation (PA) |
| P0 | MoE | collect_moe.py | GroupedMatmul_MoE_BF16 / W8A8 | 高 | **已完成** — BF16 + W8A8_DYNAMIC, 7 个模型配置 |
| P1 | Communication | collect_nccl.py | hcom_allReduce_ 等 | 低 | TensorCast 已有 generate_comm_microbench.py |
| P1 | MLA (Kernel) | collect_mla.py | FusedInferAttentionScore (MLA) | 高 | 双组件 KV Cache (kv_c + k_pe) |
| P2 | MLA (Module) | collect_mla_module.py | 多个 kernel 组合 | 最高 | 完整 DeepseekV2MLAAttention 模块 |
| P2 | GDN | collect_gdn.py | 未定义 | 中 | AIConfigurator 也未实现 |

### 2.2 各算子适配要点 (待展开)

#### Attention (collect_attn) — 已完成

| 文件 | 行数 | 职责 |
|------|------|------|
| `attn_factory.py` | ~265 | Attention 算子工厂 — 通过 `AscendAttentionBackendImpl.forward()` 调用 |
| `collect_attn.py` | ~400 | 主脚本 (参数空间 × op_type 笛卡尔积, CSV 输出, checkpoint 断点续传) |
| `verify_attn_kernel_path.py` | ~230 | 验证脚本 (call trace 确认 FIA kernel 调用) |
| `test_impl_forward.py` | ~30 | Smoke test (验证 impl.forward() 两条路径可用) |

**vLLM 接口调用链对齐验证:**

对照 AIConfigurator 源码 `collector/vllm/collect_attn.py`，确认 NPU 移植版走 vllm-ascend 的 `AscendAttentionBackendImpl.forward()` 完整链路：

**AIConfigurator 原版 (GPU):**
```
backend = get_attn_backend_cls(use_mla=False) → FlashInfer/FlashAttn
builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
attn_metadata = builder.build(common_attn_metadata)
impl = impl_cls(num_heads, head_size, scale, num_kv_heads, ...)
impl.forward(mock_layer, Q, K, V, kv_cache, attn_metadata, output=output)
  → FlashInfer/FlashAttn CUDA kernel
```

**NPU 移植版 (已对齐):**
```
impl = AscendAttentionBackendImpl(num_heads, head_size, scale, num_kv_heads, ...)
metadata = AscendMetadata(attn_state=PrefillNoCache/DecodeOnly, attn_mask, seq_lens, ...)
impl.forward(mock_layer, Q, K, V, kv_cache, metadata, output=output)
  → reshape_and_cache() (decode only)
    → DeviceOperator.reshape_and_cache() → torch_npu._npu_reshape_and_cache()
  → forward_impl() → forward_fused_infer_attention()
    → _get_fia_params() (状态机 dispatch)
    → torch_npu.npu_fused_infer_attention_score() (CANN FIA kernel)
```

关键对齐点:
- 全局配置: `VllmConfig()` + `set_current_vllm_config()` + `set_forward_context()` (和 AIConfigurator 一致)
- 后端选择: AIConfigurator 用 `get_attn_backend_cls()` 自动选后端（GPU 有多个），NPU 直接 import `AscendAttentionBackendImpl`（Ascend 只有一个后端，等价）
- 执行入口: 都是 `impl.forward(layer, Q, K, V, kv_cache, metadata, output)` — 完全对齐
- 状态机: PrefillNoCache / DecodeOnly 由 `AscendMetadata.attn_state` 驱动，impl 内部自动 dispatch
- KV cache 写入: decode 路径 impl 内部调 `reshape_and_cache()` → `DeviceOperator.reshape_and_cache()` → `_npu_reshape_and_cache()`
- metadata 构造: AIConfigurator 用 `builder.build()`，本脚本手动填 `AscendMetadata` 字段（效果等价，跳过 builder 的 mask singleton 和 batch reorder 逻辑）

**两条路径:**

| Op Type | attn_state | impl 内部路径 | 最终 kernel | 输出 CSV |
|---------|-----------|--------------|------------|---------|
| attention_context | PrefillNoCache | `forward_fused_infer_attention(Q, K, V)` — 无 block_table | `npu_fused_infer_attention_score` | `FusedInferAttentionScore.csv` |
| attention_generation | DecodeOnly | `reshape_and_cache()` → `forward_fused_infer_attention(cache)` — 有 block_table | `_npu_reshape_and_cache` + `npu_fused_infer_attention_score` | `FusedInferAttentionScore_Decode.csv` |

**Context (Prefill) 实现:**
- PrefillNoCache 状态: impl 内部跳过 `reshape_and_cache`，直接传 Q/K/V 到 FIA
- Causal mask: 固定 `[2048, 2048]` int8 上三角（FIA kernel sparse_mode=3 硬性要求，匹配 `AttentionMaskBuilder.get_splitfuse_attn_mask()`）
- actual_seq_lengths: cumsum `[seq_len, 2*seq_len, ..., batch*seq_len]`
- Q/K/V shape: TND layout `[num_tokens, num_heads, head_size]`

**Generation (Decode) 实现:**
- DecodeOnly 状态: impl 内部先调 `reshape_and_cache()` 写 KV cache，再从 cache 读取做 FIA
- KV cache shape: `[num_blocks, block_size=128, num_kv_heads, head_size]`
- impl 内部 `_get_fia_params()` 自动 view 为 `[num_blocks, block_size, hidden]` 传给 FIA
- block_table: 连续块映射 `[batch, blocks_per_seq]`
- 注意: `_npu_paged_attention` 只在 graph capture + `pa_shape_list` 匹配时启用，非通用 decode 路径

**已解决的适配问题:**
- TND layout: FIA 要求 3D `[T, N, D]`，初版误传 2D `[T, N*D]` → `query's dim should be consistent with Layout`
- Causal mask 尺寸: FIA sparse_mode=3 硬性要求 `[2048, 2048]`，初版按 seq_len 动态生成 → `maskDim 1 shall be 2048`
- `_npu_paged_attention` 不可用: ATB `PagedAttentionOperation setup failed`，改为 FIA + paged KV cache
- NPU stream 错误级联: kernel 失败后 Event.elapsed_time 报 `event recorder null`，加 `synchronize()` 恢复
- Forward context 未设置: `AscendAttentionBackendImpl.__init__()` 内部调 `get_current_vllm_config()`，需要先 `set_current_vllm_config().__enter__()`
- `set_forward_context` context manager: 需要 `__enter__()` 并保持引用防 GC

**参数空间:**
- batch: [1, 2, 4, 8, 16, 32, 64, 128]
- seq_len (context): [128, 256, 512, 1024, 2048, 4096, 8192]
- seq_len (generation): [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
- num_heads: [8, 16, 32, 40, 64]
- num_kv_heads: [0(MHA), 1, 4, 8]
- GQA 约束: num_heads % num_kv_heads == 0

**与 AIconfigurator / TensorCast 调用链对比:**

```
collect_attn (本脚本) — 走 vllm-ascend impl.forward():
  _ensure_forward_context()
    → VllmConfig()                                         ← vllm.config
    → set_current_vllm_config(vllm_config).__enter__()     ← vllm.config
    → set_forward_context(vllm_config=...).__enter__()     ← vllm.forward_context
  _create_impl(spec, device)
    → AscendAttentionBackendImpl(                          ← vllm_ascend.attention.attention_v1
        num_heads, head_size, scale, num_kv_heads,
        kv_cache_dtype="auto", attn_type="decoder")
      → self.vllm_config = get_current_vllm_config()      ← 读全局 config
  _build_prefill_metadata / _build_decode_metadata
    → AscendMetadata(                                      ← vllm_ascend.attention.attention_v1
        attn_state=PrefillNoCache/DecodeOnly,
        attn_mask=[2048,2048] int8 causal,
        seq_lens, block_tables, slot_mapping, ...)
  impl.forward(layer, Q, K, V, kv_cache, metadata, output)  ← vllm_ascend 入口
    → reshape_and_cache()                                  ← (decode only)
    │  → DeviceOperator.reshape_and_cache()                ← vllm_ascend.device.device_op
    │    → torch_npu._npu_reshape_and_cache()              ← CANN kernel
    → forward_impl()
      → forward_fused_infer_attention()
        → _get_fia_params()                                ← 状态机: PrefillNoCache/DecodeOnly
        → torch_npu.npu_fused_infer_attention_score()      ← CANN FIA kernel

AIconfigurator (GPU) — 走 vLLM impl.forward():
  get_attn_backend_cls()                                   ← vLLM 平台检测
    → FlashInfer / FlashAttn / FlexAttn
  builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
  attn_metadata = builder.build(common_attn_metadata)      ← vLLM builder
  impl = impl_cls(num_heads, head_size, scale, ...)        ← vLLM impl
  impl.forward(layer, Q, K, V, kv_cache, metadata, output) ← vLLM 入口
    → FlashInfer/FlashAttn CUDA kernel

op_replay/FusedInferAttentionScore_run.py (TensorCast) — 直调 torch_npu:
  从 FusedInferAttentionScore.csv 读 shape
  build_row_case(row)
    → build_input_tensor(query/key/value)                  ← 从 CSV 重建张量
    → build_causal_mask() / build_block_table_tensor()     ← 重建 mask/block_table
    → infer_case_args()                                    ← 推断 num_heads/scale 等
  torch_npu.npu_fused_infer_attention_score(               ← 直接调 CANN kernel
      query, key, value, atten_mask, block_table,
      actual_seq_lengths, actual_seq_lengths_kv,
      num_heads, num_key_value_heads, scale,
      input_layout, sparse_mode, block_size)
```

| 维度 | collect_attn (本脚本) | AIconfigurator (GPU) | op_replay (TensorCast) |
|------|----------------------|---------------------|------------------------|
| 调用层级 | vllm-ascend `impl.forward()` | vLLM `impl.forward()` | `torch_npu` 直调 kernel |
| 后端选择 | 直接 import AscendAttentionBackendImpl | `get_attn_backend_cls()` 自动选 | 无 (固定 FIA) |
| metadata 构造 | 手动填 `AscendMetadata` 字段 | `builder.build()` | 从 CSV 推断参数 |
| KV cache 写入 | `impl.forward()` 内部 `reshape_and_cache` | `impl.forward()` 内部 | 无 (只读 cache) |
| 状态机 dispatch | PrefillNoCache / DecodeOnly | 由 vLLM 框架管理 | 无 (直接传参) |
| 最终 kernel | `npu_fused_infer_attention_score` | FlashInfer/FlashAttn CUDA | `npu_fused_infer_attention_score` |
| shape 来源 | 穷举 batch×seq×heads 笛卡尔积 | 穷举 batch×seq×heads 笛卡尔积 | 从已有 CSV 读 (replay) |
| 计时方式 | NPU Event 自行计时 | CUDA Event + Power 采样 | msprof 外部挂载 |
| 测量范围 | impl.forward 全链路 (含 reshape_and_cache) | impl.forward 全链路 | 纯 FIA kernel |
| 用途 | 建稠密 lookup table | 建稠密 lookup table | 对已有 shape 做验证/补充 |

核心区别：collect_attn 走 vllm-ascend 的 `AscendAttentionBackendImpl.forward()` 完整链路（含状态机 dispatch + reshape_and_cache），和 AIconfigurator 走 vLLM `impl.forward()` 的架构完全对齐。op_replay 则是从 CSV 重建张量后直调 `torch_npu.npu_fused_infer_attention_score()`，跳过了 vllm-ascend 的所有中间逻辑。

**CSV 时间列语义对比:**

| 维度 | collect_attn (本脚本) | op_replay → start_microbench (TensorCast) | AIconfigurator (GPU) |
|------|----------------------|------------------------------------------|---------------------|
| 计时机制 | NPU Event per-iteration | msprof 外部挂载 → op_summary `Task Duration(us)` | CUDA Event + Graph replay |
| `Average Duration(us)` 语义 | **median** of 100 iterations | **min** of N replay samples | total / (runs × ops) |
| 其他时间列 | Median/Min/Max/Std Duration(us) | Profiling Average/Median/Std Duration(us) + aicore_time 等 | latency + energy |
| TensorCast 查询优先级 | `Average Duration(us)` (最高) | `Average Duration(us)` > `Profiling Average Duration(us)` > `Duration(us)` | N/A (cubic 插值) |
| 测量包含范围 | impl.forward() 全链路 (reshape_and_cache + FIA dispatch + kernel) | 纯 CANN kernel (msprof Task Duration) | impl.forward() 全链路 |

注意：collect_attn 的 `Average Duration(us)` 当前填的是 median（中位数），而 TensorCast 期望的是 min（最快稳定样本）。如果要直接对接 TensorCast ProfilingDataSource 查询，需要改为填 `result.min_us`，或者在 CSV 中同时提供 `Average Duration(us)` = min 和 `Median Duration(us)` = median。

**运行命令:**
```bash
# 快速验证
python collect_attn.py --op-types context generation \
    --batch-list 1 4 --seq-len-list 128 1024 \
    --num-heads-list 32 --num-kv-heads-list 0 8

# 完整采集
python collect_attn.py --op-types context generation --output-dir ./attn_data

# 断点续传
python collect_attn.py --output-dir ./attn_data --resume
```

#### MoE (collect_moe) — 已完成 (v2: 框架层对齐)

| 文件 | 行数 | 职责 |
|------|------|------|
| `moe_factory.py` | ~445 | MoE 算子工厂 (BF16 / W8A8_DYNAMIC 两条路径) |
| `collect_moe.py` | ~380 | MoE 主脚本 (token × model_config × quant 笛卡尔积, CSV 输出, checkpoint) |

**两条 kernel 路径:**

| 量化类型 | 执行流水线 | 核心 kernel | 输出 CSV |
|---------|-----------|------------|---------|
| BF16 | routing(预计算) → `unquant_apply_mlp(need_trans=False)` → pad → `npu_moe_token_unpermute` | `npu_grouped_matmul` × 2 + `npu_swiglu` | `GroupedMatmul_MoE_BF16.csv` |
| W8A8_DYNAMIC | routing(预计算) → `quant_apply_mlp()` → pad → `npu_moe_token_unpermute` | `npu_dynamic_quant` + `npu_grouped_matmul_swiglu_quant` + `npu_grouped_matmul_gmm2` | `GroupedMatmul_MoE_W8A8.csv` |

**vLLM 接口调用链对齐验证:**

对照 vllm-ascend 推理路径 `AllGatherCommImpl.fused_experts()` (moe_comm_method.py:117)，确认 benchmark 走完整框架层调用链：

**vllm-ascend 推理路径 (ALLGATHER 模式):**
```
AllGatherCommImpl.fused_experts()                    ← moe_comm_method.py:117
  ① token_dispatcher.token_dispatch()                ← token_dispatcher.py:338
     → DeviceOperator.npu_moe_init_routing()         ← CANN custom op
     → returns: sorted_hidden, expanded_row_idx, group_list (type=1)

  ② _apply_mlp() → unified_apply_mlp()              ← moe_mlp.py:370
     [BF16] → unquant_apply_mlp(need_trans=False)    ← moe_mlp.py:323
               → npu_grouped_matmul(GMM1)            ← w 已 transpose, x @ w
               → npu_swiglu()
               → npu_grouped_matmul(GMM2)
     [W8A8] → quant_apply_mlp()                      ← moe_mlp.py:83, ALLGATHER 分支 (line 241)
               → DeviceOperator.npu_dynamic_quant()   ← BF16→INT8
               → npu_grouped_matmul + swiglu_quant    ← fused GMM1+SwiGLU+quant
               → DeviceOperator.npu_grouped_matmul_gmm2() ← GMM2 with dequant

  ③ token_dispatcher.token_combine()                 ← token_dispatcher.py:394
     → torch_npu.npu_moe_token_unpermute(permuted, sorted_indices, probs)
```

**NPU benchmark (本脚本):**
```
预计算（不在计时区间）:
  ① _pytorch_token_dispatch()                        ← PyTorch argsort 替代 CANN custom op
     → expand + argsort + unique_consecutive
     → returns: sorted_hidden [T*K, H], expanded_row_idx [T*K], group_list [E]

forward()（计时区间）:
  ② [BF16] unquant_apply_mlp(need_trans=False)       ← 同一函数 moe_mlp.py:323
     [W8A8] quant_apply_mlp()                         ← 同一函数 moe_mlp.py:83

  ③ pad: zeros(T*K, H)[:local_count] = mlp_out       ← EP 模拟需要 pad
     torch_npu.npu_moe_token_unpermute()              ← 同一 CANN kernel
```

**AIConfigurator (GPU) — 走 vLLM 框架层:**
```
setup_distributed()
  → init_distributed_environment()                    ← vllm.distributed
  → set_current_vllm_config(VllmConfig())             ← vllm.config

power_law_logits_v3(num_tokens, num_experts, topk)    ← bench 自实现 Zipf 分布
  → torch.topk → F.softmax

forward() — 计时区间 (5 组 routing 取平均):
  fused_experts(hidden, w1, w2, topk_weights, topk_ids,
                quant_config, expert_map)              ← vllm.fused_moe
    [FP16/FP8 路径]:
      → routing + GEMM1(gate+up) + SiLU + GEMM2(down) + combine
      → Triton/CUTLASS grouped GEMM kernel (全部融合)
    [NVFP4 路径]:
      → flashinfer.trtllm_fp4_block_scale_routed_moe()
    [MXFP4 路径]:
      → FusedMoE.forward() → vLLM 自动选 kernel
```

**TensorCast op_replay — 直调 torch_npu:**
```
从 GroupedMatmul.csv 读 shape
build_row_case(row)
  → build_input_tensor(x, weight)                     ← 从 CSV 重建张量
torch_npu.npu_grouped_matmul(                         ← 直调 CANN kernel
    x=[input], weight=[weight],
    split_item=2, group_list_type=..., group_list=...)
msprof 外部挂载计时
```

**逐接口精确对比:**

| 接口 | vllm-ascend 推理 | benchmark 脚本 | 对齐？ |
|------|-----------------|---------------|--------|
| `unquant_apply_mlp(need_trans=False)` | moe_mlp.py:323, w=`[E, in, out]` | 同函数，同参数 | ✓ |
| `quant_apply_mlp(hidden, [w1], [w1_scale], ...)` | moe_mlp.py:83, ALLGATHER 分支 | 同函数，同参数 | ✓ |
| `npu_moe_token_unpermute(permuted, abs(idx), probs)` | token_dispatcher.py:395 | moe_factory.py:228 | ✓ |
| `_EXTRA_CTX.moe_comm_type = ALLGATHER` | ascend_forward_context.py | _ensure_forward_context line 124 | ✓ |
| `need_trans` | 推理默认 `False` (moe_runtime_args.py:132) | 显式 `False` | ✓ |
| W8A8 NZ format | `process_weights_after_loading` → `npu_format_cast(29)` | `npu_format_cast(w.data, 29)` | ✓ |
| W8A8 scale dtype | `w2_scale.dtype` → `_output_dtype` (moe_mlp.py:137) | `bfloat16` | ✓ |
| `dispose_tensor` | quant_apply_mlp 内部 dispose 输入 | `.clone()` 保护 | ✓ |
| triton swiglu_quant | 需要 `init_device_properties_triton()` | _ensure_forward_context 中初始化 | ✓ |

**三方对比总表:**

| 维度 | collect_moe (本脚本) | AIConfigurator (GPU) | TensorCast op_replay |
|------|---------------------|---------------------|---------------------|
| 调用层级 | vllm-ascend `unquant/quant_apply_mlp()` | vLLM `fused_experts()` | `torch_npu` 直调 kernel |
| 框架对齐度 | 高 — 和推理调同一个函数 | 高 — 和推理调同一个函数 | 低 — 只测单个 kernel |
| 测量范围 | MLP compute + token_unpermute | routing + MLP + combine (全融合) | 单个 npu_grouped_matmul |
| routing | 预计算 (不在计时区间) | 内置于 fused_experts (在计时区间) | 无 |
| token combine | `npu_moe_token_unpermute` (在计时区间) | 内置于 fused_experts | 无 |
| 量化路径 | BF16 / W8A8_DYNAMIC | FP16 / FP8 / FP8_Block / NVFP4 / MXFP4 | 从 CSV 推断 |
| 分布模拟 | 均匀随机 | Power Law (Zipf), 5 组取平均 | 无 |
| EP 切分 | `ep_size` 模拟 (group_list slice + pad) | `determine_expert_map` | 无 |
| 权重格式 | BF16: `[E, in, out]`; W8A8: `[E, in, out]` + NZ | `[E, out, in]` (fused_experts 内部 transpose) | 从 CSV 重建 |
| 模型相关性 | 相关 (9 个模型配置) | 相关 (14 个模型配置) | 从 CSV 读 shape |
| 计时方式 | NPU Event per-iteration median | CUDA Event + Power 采样 | msprof 外部挂载 |
| 用途 | 建 MoE lookup table | 建 MoE lookup table | 对已有 shape 做验证 |

核心区别：
1. **GPU 的 `fused_experts()` 是全融合的** — routing + MLP + combine 在一个函数内完成（Triton/CUTLASS kernel）。NPU 是分步的 — `apply_mlp()` + `token_unpermute` 分开调用。这不是 benchmark 脚本的问题，而是 NPU 架构本身的差异（Da Vinci 没有全融合 MoE kernel），vllm-ascend 推理时也是分步执行。
2. **routing 位置不同** — AIConfigurator 的 routing 在 `fused_experts` 内部（计时区间内），NPU benchmark 的 routing 预计算（计时区间外）。NPU 测的是纯 MLP+combine 时间。
3. **三者都走各自平台的框架层 API** — `fused_experts` (GPU) / `unquant/quant_apply_mlp` (NPU) / 直调 kernel (op_replay)，不是直调 raw kernel。

注意: 使用 `npu_moe_token_unpermute` 而非 `npu_moe_finalize_routing` 做 token combine，与 vllm-ascend 推理路径一致 (后者有已知精度问题，见 token_dispatcher.py:394 注释)。

**CSV 时间列语义对比:**

| 维度 | collect_moe (本脚本) | op_replay → start_microbench (TensorCast) | AIConfigurator (GPU) |
|------|---------------------|------------------------------------------|---------------------|
| 计时机制 | NPU Event per-iteration | msprof 外部挂载 → op_summary `Task Duration(us)` | CUDA Event + Graph replay |
| `Average Duration(us)` 语义 | **median** of 100 iterations | **min** of N replay samples | total / (runs × 5 routing × 6 ops) |
| 测量包含范围 | MLP compute + token_unpermute | 纯 CANN kernel (msprof Task Duration) | routing + MLP + combine (全融合) |

**模型配置 (9 个):**
- DeepSeek-V2-Lite (2048/1408, 64 experts, topk=6)
- DeepSeek-V2 (5120/1536, 160 experts, topk=6)
- DeepSeek-V3/R1 (7168/2048, 256 experts, topk=8)
- GLM-5 (7168/2048, 256 experts, topk=8 — MoE 维度与 DeepSeek-V3 相同)
- MiniMax-Text-01 (6144/9216, 32 experts, topk=2)
- Mixtral-8x7B (4096/14336, 8 experts, topk=2)
- Mixtral-8x22B (6144/16384, 8 experts, topk=2)
- Qwen2-MoE-57B (3584/2560, 64 experts, topk=8)
- Qwen3-MoE-30B (2048/1024, 128 experts, topk=8)

**参数空间:**
- tokens: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
- 13 tokens × 9 models × 2 quant = 234 total specs

**运行命令:**
```bash
# 快速验证 (单模型)
python collect_moe.py --models deepseek-v3 --token-list 1 128 1024 --quant-types bf16 w8a8_dynamic

# 完整采集
python collect_moe.py --quant-types bf16 w8a8_dynamic --output-dir ./moe_data

# 断点续传
python collect_moe.py --output-dir ./moe_data --resume
```

**已知限制:**
- **EP 模拟而非真实 EP**: 通过 `group_list` slice + pad 模拟，单卡只分配 local experts 权重，MLP 输出需 pad 回 `T*K` 行给 `npu_moe_token_unpermute`
- **Token dispatch 用 PyTorch 替代 CANN op**: `MoeInitRoutingCustom` 在部分 CANN 版本缺少编译 binary 且失败会污染 NPU stream，改用 `argsort` + `unique_consecutive`（预计算，不影响 benchmark 数据）
- **未做 Power Law 分布**: 使用均匀随机 routing，不模拟真实推理中的 expert 负载不均（AIConfigurator 用 Zipf 分布 5 组取平均）
- **未覆盖 MXFP8**: 仅 A5 设备支持，需要 `npu_grouped_matmul_swiglu_quant_v2`
- **未覆盖 W4A8/W4A16**: 需要 antiquant_scale/antiquant_offset 路径

### 2.4 MoE 移植风险与经验总结

MoE 是目前移植难度最高的算子，不是因为 kernel 本身复杂，而是 vllm-ascend 的 MoE 实现深度耦合了运行时上下文。以下是调试过程中遇到的所有问题分类。

#### 第一层：硬件约束差异（一次性，可预见）

| 问题 | 根因 | 修复 |
|------|------|------|
| OOM：256 experts 权重超单卡 HBM | GPU 版不考虑单卡内存（有 EP），NPU 单卡 64GB 不够放全量 | 加 `ep_size` 模拟 EP 切分 |
| `group_list` 需要 slice 到 local experts | EP 模拟的连带问题，`init_routing` 返回全局 expert 计数 | `group_list[:local_num_experts]` |

#### 第二层：vllm-ascend 框架隐式依赖（最耗时，不可预见）

| 问题 | 根因 | 修复 |
|------|------|------|
| `Forward context is not set` | `quant_apply_mlp` 内部访问 `_EXTRA_CTX.moe_comm_type`，底层调 `get_forward_context()` | 手动创建 ForwardContext 并设 `_forward_context` 全局变量 |
| `get_current_vllm_config()` 失败 | `set_current_vllm_config` 是 context manager，退出 scope 后 config 消失 | 手动 `__enter__()` 保持 alive |
| `set_forward_context` 签名不匹配 | 不同 vllm 版本参数不同，内部又调 `get_current_vllm_config()` 形成循环依赖 | 绕过 API 直接设 `vllm.forward_context._forward_context` |
| `num_tokens` 未设置 | NPU 版 `DeviceOperator.npu_moe_init_routing` 新增了 `forward_context.num_tokens` 检查 | 每次调用前更新 `ctx.num_tokens` |
| `_C_ascend` custom ops 未加载 | C++ 扩展需要 `enable_custom_op()` 触发加载 | 在 `_ensure_forward_context` 中调用 |

GEMM 和 Attention 没遇到这些问题，因为它们的 kernel 不依赖 `_EXTRA_CTX`。MoE 是第一个需要完整 forward context 的算子。

#### 第三层：NPU 机器与本地源码版本不一致（持续性问题）

| 问题 | 根因 | 影响 |
|------|------|------|
| `unquant_apply_mlp` 调 `torch.ops._C_ascend.moe_grouped_matmul` | NPU 上 vllm-ascend 版本更新 | 本地调试无法复现，只能在 NPU 上试错 |
| `npu_grouped_matmul` 期望 `List[Tensor]` 而非 `Tensor` | torch_npu 版本 API 签名不同 | 报错信息不直观 |
| `group_list` 返回 2D tensor | `npu_moe_init_routing` 在不同 CANN 版本返回不同 shape | 需要 squeeze 兜底 |
| weight layout 约定差异 | NZ format 转换行为依赖 CANN 版本 | 需要在 NPU 上逐个验证 |

#### 第四层：GPU vs NPU 架构差异

| GPU (AIConfigurator) | NPU | 差异原因 |
|---------------------|-----|---------|
| `fused_experts()` 一个函数搞定 | init_routing → apply_mlp → token_unpermute 三步 | NPU 没有全融合 MoE kernel |
| routing 内置于 fused_experts | `npu_moe_init_routing` 是独立算子 | Da Vinci 架构 routing 有专用硬件加速 |
| combine 内置于 fused_experts | `npu_moe_token_unpermute` | `npu_moe_finalize_routing` 有精度问题 |
| weight 直接传 3D tensor | 需要 `List[Tensor]` 包装 + NZ format | CANN GroupedMatmul API 约定不同 |

#### 未来 vllm-ascend 版本升级适配工作量评估

| 风险等级 | 场景 | 预估工作量 | 应对策略 |
|---------|------|----------|---------|
| **高** | `quant_apply_mlp` 内部逻辑重构 | 1-2 天 | BF16 和 W8A8 都依赖框架函数，需跟踪 |
| **高** | forward context 机制变更 (vllm core) | 0.5-1 天 | 当前直接设全局变量绕过 API，如果变量名或结构变了需要改 |
| **中** | `unquant_apply_mlp` 签名/行为变更 | 0.5 天 | `need_trans` 默认值、`dispose_tensor` 行为等 |
| **中** | triton swiglu_quant 依赖变更 | 0.5 天 | `init_device_properties_triton()` 接口可能变化 |
| **低** | 新增量化路径 (MXFP8, W4A8) | 1 天/路径 | 参考现有 W8A8 模式扩展 |
| **低** | 新增模型配置 | 几分钟 | 在 `MODEL_CONFIGS` 加一行 |

#### 经验教训

1. **MoE 的移植难度远高于 GEMM/Attention**：不是 kernel 复杂，而是运行时上下文依赖链条深（forward context → _EXTRA_CTX → DeviceOperator → custom C++ ops → triton device properties）
2. **本地源码和 NPU 环境版本差异是最大的效率杀手**：每次修改都需要同步到 NPU 试错，建议保持本地源码与 NPU 安装版本严格一致
3. **框架层对齐优先于直调 kernel**：v2 版本 BF16 路径改回调 `unquant_apply_mlp(need_trans=False)`，和推理路径完全一致。`need_trans` 在推理中默认 `False`（moe_runtime_args.py:132），权重在 `process_weights_after_loading` 阶段已完成 transpose
4. **CANN custom op 可靠性不稳定**：`MoeInitRoutingCustom` 在部分 CANN 版本缺少 binary 且失败会不可逆地污染 NPU stream，改用 PyTorch fallback（预计算，不影响 benchmark）
5. **vllm-ascend 框架函数有隐式副作用**：`quant_apply_mlp` 内部调 `dispose_tensor()` 会销毁输入 tensor 的 storage，需要 `.clone()` 保护；`w2_scale.dtype` 决定 `_output_dtype`（moe_mlp.py:137），必须用 bfloat16 而非 float32
6. **每次 vllm-ascend 升级，最大风险不是 kernel API 变了，而是隐式依赖链条上的某个环节变了**

#### Communication
- **已有实现**: TensorCast 的 `generate_comm_microbench.py` 已覆盖 HCCL 通信采集
- **待确认**: 是否需要额外的 Custom AllReduce 采集 (vllm-ascend 的 MC2 路径)

### 2.3 可复用组件

以下组件已在 GEMM 采集中验证，后续算子可直接复用:

| 组件 | 文件 | 复用方式 |
|------|------|---------|
| NPU Event 计时 | `bench_engine.py` | `benchmark_npu(kernel_func)` |
| vLLM 上下文初始化 | `gemm_factory.py:_init_vllm_context()` | 所有算子共用 |
| CSV 输出 + checkpoint | `collect_gemm.py` | 提取为通用 `collect_base.py` |
| kernel 路径验证 | `verify_kernel_path.py` | 扩展 monkey-patch 目标函数 |

---

## 三、适配工作量总览

### 3.1 全算子适配矩阵

| 适配场景 | GEMM | Attention | MoE | MLA (Kernel) | MLA (Module) | 通信 |
|---------|------|-----------|-----|-------------|-------------|------|
| **当前状态** | 已完成 | 已完成 | 已完成 | 未开始 | 未开始 | 复用 TensorCast |
| **vllm-ascend 版本升级** | < 0.5 天 | 0.5-1 天 | 1-3 天 | — | — | 0 |
| 版本风险点 | NZ format、quant method 签名 | AscendMetadata 字段、FIA 状态机、set_forward_context | forward context 链条、quant_apply_mlp 重构、DeviceOperator 接口 | 双组件 KV Cache API | DeepseekV2MLAAttention 内部重构 | 不依赖 vllm-ascend |
| **新 Dense 模型** | 0 | 0 | — | — | — | 0 |
| **新 MoE 模型** | 0 | 0 | 几分钟（加一行配置） | — | — | 0 |
| **新 MLA 模型** | 0 | 0 | — | 2-3 天（首次开发） | 3-5 天（首次开发） | 0 |
| **新量化路径** | ~1 天/路径 | 不涉及 | ~1 天/路径 | 未评估 | 未评估 | 不涉及 |
| 已支持量化 | BF16, W8A8_DYNAMIC | — | BF16, W8A8_DYNAMIC | — | — | — |
| 待支持量化 | MXFP8, W4A8, W4A16 | — | MXFP8, W4A8, W4A16 | — | — | — |
| **模型相关性** | 无关（穷举 M×N×K） | 无关（穷举 batch×seq×heads） | 相关（9 个模型配置） | 相关（DeepSeek 维度参数） | 相关（需真实 HF config） | 无关 |
| **框架耦合度** | 低 | 中 | 高 | 高 | 最高 | 无 |
| 依赖 vllm-ascend 接口数 | ~5 | ~8 | ~10+ | ~12 | ~15 | 0 |

**每次 vllm-ascend 版本升级总工作量：2-4.5 天**（MoE 占大头，隐式依赖链条深是主因）

### 3.2 与 TensorCast op_replay 适配成本对比

| 维度 | AIConfigurator Bench (NPU) | TensorCast op_replay |
|------|--------------------------|---------------------|
| vLLM 版本升级 | 2-4.5 天（MoE 隐式依赖链深） | 1-2 天（op_mapping.yaml + kernel 名变化） |
| 新 Dense 模型 | 0 | 0（shape 命中）/ 需重新 profiling（miss） |
| 新 MoE 模型 | 几分钟（加配置行） | 需重新 profiling + op_mapping 更新 |
| 新量化路径 | ~1 天/路径 | ~0.5 天（加 *_run.py + op_mapping） |
| 框架耦合度 | 高（依赖 vllm-ascend 内部 API） | 低（只依赖 torch_npu） |
| 核心优势 | 穷举覆盖，新模型不用重采 | 框架耦合浅，版本升级痛感低 |
| 核心劣势 | 框架耦合深，版本升级痛 | 每个新模型都要重新 profiling |

---

## 四、参考资料

- AIConfigurator 源码: `~/Downloads/aiconfigurator/collector/vllm/collect_gemm.py` (GPU 原版)
- AIConfigurator 计时引擎: `~/Downloads/aiconfigurator/collector/helper.py` (`benchmark_with_power`)
- AIConfigurator 分布式初始化: `~/Downloads/aiconfigurator/collector/vllm/utils.py` (`setup_distributed`)
- AIConfigurator 分析文档: `~/aiconfigurator_vllm_benchmark_analysis.md`
- AIConfigurator Attention 分析: `~/aiconfigurator_vllm_benchmark_analysis.md` (§3.2 Attention)
- TensorCast op_replay: `~/Downloads/msmodeling/tools/perf_data_collection/op_replay/MatMulV2_run.py`
- TensorCast FIA op_replay: `~/Downloads/msmodeling/tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py`
- TensorCast FIA 查询引擎: `~/Downloads/msmodeling/tensor_cast/performance_model/profiling_database/profiling_data_source.py` (`_query_by_attn_params`, L1271-1402)
- TensorCast microbench 编排: `~/Downloads/msmodeling/tools/perf_data_collection/start_microbench.py`
- TensorCast op_mapping: `tensor_cast/performance_model/profiling_database/data/.../op_mapping.yaml`
- vllm-ascend Attention Backend: `~/Downloads/vllm-ascend/vllm_ascend/attention/attention_v1.py` (`AscendAttentionBackendImpl`, `AscendMetadata`)
- vllm-ascend Attention Mask: `~/Downloads/vllm-ascend/vllm_ascend/attention/attention_mask.py` (`AttentionMaskBuilder`)
- vllm-ascend Attention Utils: `~/Downloads/vllm-ascend/vllm_ascend/attention/utils.py` (`using_paged_attention`, `AscendCommonAttentionMetadata`)
- vllm-ascend Forward Context: `~/Downloads/vllm-ascend/vllm_ascend/ascend_forward_context.py` (`_EXTRA_CTX`, `set_ascend_forward_context`)
- vllm-ascend 量化方法: `~/Downloads/vllm-ascend/vllm_ascend/quantization/methods/`
- vllm-ascend 线性层: `~/Downloads/vllm-ascend/vllm_ascend/ops/linear.py`
- vllm-ascend BF16 patch: `~/Downloads/vllm-ascend/vllm_ascend/patch/worker/patch_unquantized_gemm.py`
- vllm-ascend MoE MLP: `~/Downloads/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py` (`unquant_apply_mlp` / `quant_apply_mlp`)
- vllm-ascend MoE 层: `~/Downloads/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py` (`AscendFusedMoE`)
- vllm-ascend MoE 通信: `~/Downloads/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py` (`AllGatherCommImpl`)
- vllm-ascend Token Dispatch: `~/Downloads/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py` (`TokenDispatcherWithAllGather`)
- vllm-ascend Device Op: `~/Downloads/vllm-ascend/vllm_ascend/device/device_op.py` (`DeviceOperator`)
- vllm-ascend W8A8 MoE: `~/Downloads/vllm-ascend/vllm_ascend/quantization/methods/w8a8_dynamic.py` (`AscendW8A8DynamicFusedMoEMethod`)
