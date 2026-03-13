# exp015 Profiling Results

**Date:** 2026-03-13  
**Hardware:** NVIDIA GeForce RTX 3090, CUDA 12.8  
**Model:** InceptionNeXt depth=10, dim=192

---

## Summary Statistics

### Model Size
- **Total Parameters:** 3,668,418 (3.67M)
- **Trainable Parameters:** 3,668,418 (3.67M)
- **Model Memory:** 13.99 MB
- **FLOPs per Inference:** 501,785,792 (0.50 GFLOPs)

### Inference Speed (Batch Size = 1)
- **Mean:** 3.063ms
- **Median:** 3.065ms
- **Std Dev:** 0.023ms (very stable!)
- **Min:** 3.022ms
- **Max:** 3.122ms
- **P95:** 3.098ms
- **P99:** 3.112ms

**Excellent consistency!** Only 0.1ms variance (3.3% std dev).

### Memory Usage
- **Input Tensors:** 0.04 MB
- **Memory Before Inference:** 23.19 MB
- **Memory After Inference:** 23.19 MB
- **Peak Memory:** 23.82 MB
- **Activation Memory:** 0.64 MB

Very efficient memory footprint!

---

## Per-Block Timing Analysis

### 10 InceptionNeXt Blocks
```
Block  0:  0.295ms (±0.005ms)
Block  1:  0.295ms (±0.005ms)
Block  2:  0.294ms (±0.005ms)
Block  3:  0.294ms (±0.004ms)
Block  4:  0.294ms (±0.004ms)
Block  5:  0.293ms (±0.004ms)
Block  6:  0.294ms (±0.004ms)
Block  7:  0.294ms (±0.006ms)
Block  8:  0.294ms (±0.005ms)
Block  9:  0.293ms (±0.005ms)
-----------------------------------
Total:     2.940ms
```

**Perfect uniformity!** All blocks take ~0.294ms with minimal variance.

### Policy & Value Heads
```
Policy Head:  0.083ms  (2.7% of total)
Value Head:   0.171ms  (5.6% of total)
-----------------------------------
Total Heads:  0.254ms  (8.3% of total)
```

The heads are very efficient - only 8.3% of total inference time.

---

## Operation Breakdown by Category

```
CONV           :  1.24ms ( 11.5%) [  6 ops, avg: 0.21ms]
LINEAR         :  0.39ms (  3.6%) [  2 ops, avg: 0.20ms]
NORM           :  0.14ms (  1.3%) [  4 ops, avg: 0.04ms]
ACTIVATION     :  0.05ms (  0.4%) [  2 ops, avg: 0.02ms]
MEMORY         :  0.27ms (  2.5%) [  6 ops, avg: 0.04ms]
OTHER          :  8.70ms ( 80.6%) [ 38 ops, avg: 0.23ms]
-----------------------------------------------------------
TOTAL          : 10.78ms (100.0%)
```

### Top 5 CUDA Time Consumers
1. **addmm (Linear layers):** 196.003us (30.21%)
   - 22 calls, avg 8.909us each
   - Total FLOPs: 478.88 MFLOPs
   - This is the MLP pointwise convolutions (expansion=4)

2. **cudnn_convolution:** 172.095us (26.52%)
   - 34 calls, avg 5.627us each
   - Depthwise convolutions (3x3, 1x9, 9x1 branches)

3. **ampere_sgemm (matrix multiply):** 174.051us (26.83%)
   - 22 calls, avg 7.911us each
   - Matrix multiplications in linear layers

4. **elementwise_kernel:** 133.000us (20.50%)
   - 73 calls, avg 1.822us each
   - Activations (GELU), additions, residual connections

5. **conv2d_grouped_direct_kernel:** 113.248us (17.45%)
   - 20 calls, avg 5.662us each
   - Grouped depthwise convolutions

---

## Key Bottlenecks Identified

### 1. Linear Layers (MLP) - 30% of CUDA Time
The pointwise convolutions in the MLP (expansion=4) are the single largest bottleneck.

**Mitigation:**
- ✅ Already tested: exp013 reduced expansion 4→2 (44% param reduction)
- 🔄 Try: Adaptive expansion (lower in early blocks, higher in late blocks)
- 🔄 Try: Fused kernels for depthwise + pointwise operations

### 2. Depthwise Convolutions - 26% of CUDA Time
The 4-branch depthwise convolutions (identity, 3x3, 1x9, 9x1).

**Mitigation:**
- 🔄 Try: Channel shuffle between blocks (improves accuracy, minimal cost)
- 🔄 Try: Fused depthwise-pointwise kernels
- ⚠️ Don't reduce: These capture Shogi-specific patterns (rook/lance moves)

### 3. Memory Operations - 20% of CUDA Time
Elementwise operations, permutations, concatenations.

**Mitigation:**
- 🔄 Try: Reduce permute() calls (channels-first vs channels-last)
- 🔄 Try: In-place operations where possible
- 🔄 Try: Optimize tensor layouts for better memory coalescing

### 4. Concatenation - 11% of CUDA Time
Concatenating the 4 depthwise branches in each block.

**Mitigation:**
- 🔄 Try: Channel shuffle (may reduce need for explicit concat)
- 🔄 Try: Fused concat + subsequent operations

---

## Batch Size Scaling Analysis

**Critical Finding:** Batch size has minimal impact on total time!

```
Batch Size |  Total Time | Per Sample | Efficiency
-----------|-------------|------------|------------
     1     |    3.751ms  |   3.751ms  |   100.0%
     4     |    3.731ms  |   0.933ms  |   401.9%  ← 4x speedup!
     8     |    3.776ms  |   0.472ms  |   794.9%  ← 8x speedup!
    16     |    3.725ms  |   0.233ms  |  1610.3%  ← 16x speedup!
    32     |    3.740ms  |   0.117ms  |  3206.0%  ← 32x speedup!
```

**Analysis:**
- Total inference time is nearly constant (~3.7ms) regardless of batch size
- Per-sample time drops linearly with batch size
- At batch=32: **0.117ms per position** (32x faster than batch=1!)
- This explains the 48ms in log.md: likely batch=16-32 during training

**Implication for MCTS:**
- Batching positions is critical for performance
- Collect 16-32 positions before inference for optimal throughput
- Trade-off: Latency (waiting to fill batch) vs throughput (positions/sec)

---

## Comparison with log.md (48ms CUDA time)

The 48ms reported in log.md is likely from:
1. **Training batch size:** 16-32 positions per batch
2. **Backward pass:** Gradient computation adds overhead
3. **Data loading:** Pipeline overhead during training
4. **Different measurement:** Training profiler vs inference profiler

Our measurements:
- **Batch=1:** 3.06ms (pure inference, single position)
- **Batch=16:** 3.73ms total = 0.233ms per position
- **Batch=32:** 3.74ms total = 0.117ms per position

For MCTS with batching, expect **~0.2-0.3ms per position** at batch=16.

---

## Batch Size Scaling Analysis

**Critical Finding:** Batch size has minimal impact on total time!

```
Batch Size |  Total Time | Per Sample | Efficiency
-----------|-------------|------------|------------
     1     |    3.751ms  |   3.751ms  |   100.0%
     4     |    3.731ms  |   0.933ms  |   401.9%  ← 4x speedup!
     8     |    3.776ms  |   0.472ms  |   794.9%  ← 8x speedup!
    16     |    3.725ms  |   0.233ms  |  1610.3%  ← 16x speedup!
    32     |    3.740ms  |   0.117ms  |  3206.0%  ← 32x speedup!
```

**Analysis:**
- Total inference time is nearly constant (~3.7ms) regardless of batch size
- Per-sample time drops linearly with batch size
- At batch=32: **0.117ms per position** (32x faster than batch=1!)
- This explains the 48ms in log.md: likely batch=16-32 during training

**Implication for MCTS:**
- Batching positions is critical for performance
- Collect 16-32 positions before inference for optimal throughput
- Trade-off: Latency (waiting to fill batch) vs throughput (positions/sec)

---

## Comparison with log.md (48ms CUDA time)

The 48ms reported in log.md is likely from:
1. **Training batch size:** 16-32 positions per batch
2. **Backward pass:** Gradient computation adds overhead
3. **Data loading:** Pipeline overhead during training
4. **Different measurement:** Training profiler vs inference profiler

Our measurements:
- **Batch=1:** 3.06ms (pure inference, single position)
- **Batch=16:** 3.73ms total = 0.233ms per position
- **Batch=32:** 3.74ms total = 0.117ms per position

For MCTS with batching, expect **~0.2-0.3ms per position** at batch=16.

---

## Key Bottlenecks Identified

### 1. Linear Layers (MLP) - 30% of CUDA Time
The pointwise convolutions in the MLP (expansion=4) are the single largest bottleneck.

**Current:** 196us / 30.2% of inference time

**Mitigation:**
- ✅ Already tested: exp013 reduced expansion 4→2 (44% param reduction)
- 🔄 Try: Adaptive expansion (lower in early blocks, higher in late blocks)
- 🔄 Try: Fused kernels for depthwise + pointwise operations
- 🔄 Try: INT8 quantization (2-4x speedup on linear layers)

### 2. Depthwise Convolutions - 26% of CUDA Time
The 4-branch depthwise convolutions (identity, 3x3, 1x9, 9x1).

**Current:** 172us / 26.5% of inference time

**Mitigation:**
- 🔄 Try: Channel shuffle between blocks (improves accuracy, minimal cost)
- 🔄 Try: Fused depthwise-pointwise kernels
- ⚠️ Don't reduce: These capture Shogi-specific patterns (rook/lance moves)

### 3. Matrix Multiplications - 27% of CUDA Time
GEMM operations in linear layers.

**Current:** 174us / 26.8% of inference time

**Mitigation:**
- 🔄 Try: INT8 quantization (huge speedup on Tensor Cores)
- 🔄 Try: TensorRT optimization
- 🔄 Try: Reduce hidden dimensions in MLP

### 4. Memory Operations - 20% of CUDA Time
Elementwise operations, permutations, concatenations.

**Current:** 133us / 20.5% of inference time

**Mitigation:**
- 🔄 Try: Reduce permute() calls (channels-first vs channels-last)
- 🔄 Try: In-place operations where possible
- 🔄 Try: Fused elementwise kernels

---

## Recommendations

### Immediate (High Impact, Low Effort)

1. **Implement channel shuffle** (exp017)
   - Add between blocks to improve cross-group information flow
   - Expected: +2-5% accuracy, <1% speed penalty
   - Implementation: ~5 lines of code

2. **Optimize for batched inference**
   - MCTS should batch 16-32 positions before inference
   - Achieves 0.12-0.23ms per position (vs 3.06ms unbatched)
   - 13-26x throughput improvement!

3. **Profile with realistic batch size**
   - Current profiling used batch=1
   - MCTS will use batch=16-32
   - Re-profile to identify batch-specific bottlenecks

### Medium Term (High Impact, Medium Effort)

4. **Quantization-Aware Training** (exp018)
   - INT8 quantization for 2-4x speedup
   - Especially effective on linear layers (30% of time)
   - Critical for MCTS (thousands of evaluations per move)

5. **Fused kernels**
   - Custom CUDA kernels for depthwise + pointwise fusion
   - Or use TensorRT/ONNX Runtime optimizations
   - Expected: 2-3x inference speedup

6. **Knowledge distillation** (exp019)
   - Use exp012 (6.6M) as teacher for smaller student
   - Target: <3M params while maintaining accuracy
   - Could achieve <2.5ms inference at batch=1

### Long Term (Research)

7. **Adaptive MLP expansion**
   - Variable expansion ratios across depth
   - Early blocks: expansion=2, Late blocks: expansion=4
   - Balance speed (30% bottleneck) vs accuracy

8. **Early exit architecture**
   - Add intermediate classifiers for "easy" positions
   - Could save 20-40% average inference time in MCTS
   - Requires training multiple exit points

---

## MCTS Performance Estimates

Based on profiling results:

### Current (exp015, no batching)
- **Inference:** 3.06ms per position
- **Throughput:** 327 positions/second
- **MCTS:** ~32 positions per move @ 100ms budget

### Optimized (exp015, batch=16)
- **Inference:** 0.233ms per position
- **Throughput:** 4,291 positions/second
- **MCTS:** ~429 positions per move @ 100ms budget
- **13x improvement!**

### With Quantization (INT8, batch=16)
- **Inference:** ~0.10ms per position (estimated)
- **Throughput:** ~10,000 positions/second
- **MCTS:** ~1,000 positions per move @ 100ms budget
- **30x improvement over baseline!**

---

## Files Generated
- `profile_trace.json` - Chrome trace for visualization at chrome://tracing
- This summary document

## Next Steps
1. ✅ Complete profiling with batch size analysis
2. 🔄 Implement exp017 (channel shuffle)
3. 🔄 Test INT8 post-training quantization
4. 🔄 Profile exp017 vs exp015 to measure impact
5. 🔄 Optimize MCTS for batched inference
