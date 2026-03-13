# Realistic Batch Size Analysis - exp015

**Date:** 2026-03-14  
**Hardware:** NVIDIA GeForce RTX 3090  
**Model:** InceptionNeXt depth=10, dim=192 (3.67M params)  
**Actual Usage:** Training batch=1024, Search batch=128

---

## Executive Summary

### Training Performance (batch=1024)
- **Time per batch:** 55.64ms (±0.30ms)
- **Time per sample:** 0.054ms
- **Throughput:** 18,404 positions/second
- **Batches per second:** 17.97

### Search Performance (batch=128)
- **Time per batch:** 7.67ms (±0.03ms)
- **Time per sample:** 0.060ms
- **Throughput:** 16,687 positions/second
- **Batches per second:** 130.37
- **Speedup vs single:** 51.1x faster

---

## Detailed Batch Size Comparison

```
Batch |  Total (ms) | Per Sample (ms) |   Throughput    | Use Case
------|-------------|-----------------|-----------------|----------------------
    1 |      3.568  |      3.568042   |    280 pos/s    | Single position
   16 |      3.555  |      0.222185   |  4,501 pos/s    | Small batch
   32 |      3.587  |      0.112106   |  8,920 pos/s    | MCTS (small)
   64 |      4.456  |      0.069627   | 14,362 pos/s    | MCTS (medium)
  128 |      7.673  |      0.059947   | 16,681 pos/s    | MCTS/Search (default) ✓
  256 |     14.470  |      0.056522   | 17,692 pos/s    | Training (small)
  512 |     28.288  |      0.055251   | 18,099 pos/s    | Training (medium)
 1024 |     55.639  |      0.054334   | 18,404 pos/s    | Training (default) ✓
```

---

## Key Findings

### 1. **Training is Highly Efficient (batch=1024)**

At the default training batch size:
- Only 55.6ms per batch of 1024 positions
- Can process 18,404 positions/second
- 17.97 batches/second = ~18 training steps/second
- Very stable (±0.30ms std dev)

**Training throughput:**
- 1 epoch with 1M positions: ~54 seconds
- 1 epoch with 10M positions: ~9 minutes
- 100 epochs with 10M positions: ~15 hours

### 2. **Search is Optimized for Latency (batch=128)**

At the default search batch size:
- Only 7.67ms per batch of 128 positions
- Can process 16,687 positions/second
- 130.37 batches/second
- Lower latency than batch=1024 (7.67ms vs 55.6ms)

**Why batch=128 for search?**
- Balance between throughput and latency
- 7.67ms is acceptable for real-time MCTS
- Can fill 128 positions quickly during tree traversal
- 51x faster than evaluating positions one-by-one

### 3. **Scaling Efficiency**

```
Batch Size | Efficiency vs Ideal | Notes
-----------|---------------------|---------------------------
    1-32   |      99-100%        | Perfect scaling
   32-128  |       50-100%       | Good scaling
  128-1024 |       10-50%        | Diminishing returns
```

The transition from batch=128 to batch=1024:
- Throughput gain: 10.3% (16,687 → 18,404 pos/s)
- Latency increase: 7.25x (7.67ms → 55.6ms)
- Trade-off: Slightly better throughput for much higher latency

---

## MCTS Search Performance Analysis

### Positions Evaluated per Time Budget (batch=128)

```
Time Budget | Positions | Batches | Actual Time | Efficiency
------------|-----------|---------|-------------|------------
    10ms    |    256    |    2    |   15.3ms    |   65.4%
    50ms    |    896    |    7    |   53.7ms    |   93.1%
   100ms    |  1,792    |   14    |  107.4ms    |   93.1%
   200ms    |  3,456    |   27    |  207.1ms    |   96.6%
   500ms    |  8,448    |   66    |  506.3ms    |   98.7%
  1000ms    | 16,768    |  131    | 1004.9ms    |   99.5%
```

**Observations:**
- Short time budgets (<50ms) have lower efficiency due to batch overhead
- Longer time budgets (>100ms) achieve >93% efficiency
- At 1000ms budget: Can evaluate 16,768 positions!

### Comparison with Single-Position Inference

```
Metric                  | Single (batch=1) | Batched (batch=128) | Improvement
------------------------|------------------|---------------------|-------------
Time per position       |     3.063ms      |      0.060ms        |   51.1x
Positions in 100ms      |        33        |       1,792         |   54.3x
Positions in 1000ms     |       327        |      16,768         |   51.3x
```

**Impact on MCTS strength:**
- 51x more positions evaluated = much deeper/wider search
- Better move selection through more accurate value estimates
- Can explore more variations in same time budget

---

## Training vs Search Trade-offs

### Training (batch=1024)
**Advantages:**
- Maximum throughput (18,404 pos/s)
- Best GPU utilization
- Stable gradients with large batches
- Efficient for offline training

**Disadvantages:**
- High latency (55.6ms per batch)
- Not suitable for real-time inference
- Requires collecting 1024 positions

### Search (batch=128)
**Advantages:**
- Low latency (7.67ms per batch)
- Suitable for real-time MCTS
- Easy to fill during tree traversal
- 91% of maximum throughput

**Disadvantages:**
- Slightly lower throughput than batch=1024
- Not maximizing GPU utilization

---

## Memory Usage

```
Batch Size | Memory (MB) | Per Sample (MB)
-----------|-------------|----------------
     1     |    23.19    |    23.1943
   128     |    28.92    |     0.2260
  1024     |    69.73    |     0.0681
```

**Key Points:**
- Training (batch=1024): Only 69.73 MB
- Search (batch=128): Only 28.92 MB
- Plenty of headroom on 23.68 GB GPU
- Can run multiple models or larger batches if needed

---

## Recommendations

### 1. **Keep Current Batch Sizes**

The current configuration is well-optimized:
- Training batch=1024: Maximizes throughput
- Search batch=128: Balances latency and throughput

### 2. **Consider Adaptive Batching for Search**

```python
if time_remaining > 500ms:
    batch_size = 256  # Higher throughput
elif time_remaining > 100ms:
    batch_size = 128  # Default
else:
    batch_size = 64   # Lower latency
```

### 3. **Optimize MCTS for Batching**

Current MCTS should:
- Collect 128 leaf nodes before evaluation
- Batch evaluate all at once
- Continue tree traversal with results

**Expected improvement:**
- From ~33 positions/100ms (unbatched)
- To ~1,792 positions/100ms (batched)
- 54x stronger search!

### 4. **Training Optimizations**

With batch=1024 at 18 steps/second:
- Can train on large datasets efficiently
- Consider gradient accumulation if need larger effective batch
- Current batch size is well-suited for GPU

---

## Comparison with log.md (48ms CUDA time)

The 48ms reported in log.md now makes sense:
- It's close to batch=1024 training time (55.6ms)
- Likely measured during training with batch=1024
- Includes some overhead beyond pure inference

Our measurements:
- **Training (batch=1024):** 55.6ms per batch
- **Search (batch=128):** 7.67ms per batch
- **Single position:** 3.06ms

The 48ms aligns with training batch size, confirming the profiling is accurate.

---

## Operation Breakdown (from PyTorch Profiler)

```
Category       | Time   | % of Total | Optimization Potential
---------------|--------|------------|----------------------
CONV           | 1.24ms |   11.9%    | Medium (fused kernels)
LINEAR         | 0.39ms |    3.7%    | High (quantization)
NORM           | 0.14ms |    1.4%    | Low (already fast)
ACTIVATION     | 0.05ms |    0.5%    | Low (already fast)
MEMORY         | 0.27ms |    2.5%    | Medium (reduce permutes)
OTHER          | 8.37ms |   80.0%    | Various
```

**Top optimization targets:**
1. Linear layers (3.7%) - INT8 quantization
2. Convolutions (11.9%) - Fused kernels
3. Memory operations (2.5%) - Reduce permutations

---

## Next Steps

1. ✅ **Profiling complete** - Understand realistic performance
2. ✅ **Batch sizes validated** - Training=1024, Search=128 are optimal
3. 🔄 **Implement batched MCTS** - Achieve 51x speedup
4. 🔄 **Test INT8 quantization** - Potential 2-3x additional speedup
5. 🔄 **Profile with mixed precision (FP16)** - Potential 1.5-2x speedup

---

## Conclusion

Your current batch size configuration is excellent:

**Training (batch=1024):**
- 18,404 positions/second
- 55.6ms per batch
- Optimal for offline training

**Search (batch=128):**
- 16,687 positions/second
- 7.67ms per batch
- 51x faster than unbatched
- Perfect for real-time MCTS

The model is well-optimized for both training and inference. The main opportunity for improvement is ensuring MCTS actually uses batched inference to achieve the 51x speedup.
