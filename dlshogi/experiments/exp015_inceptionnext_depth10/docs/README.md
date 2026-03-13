# exp015 Documentation

Profiling and analysis documentation for exp015 (InceptionNeXt depth=10, dim=192).

## Files

### [PROFILE_RESULTS.md](PROFILE_RESULTS.md)
Initial detailed profiling results from `profile_detailed.py`.

**Key Findings:**
- 3.06ms inference time (batch=1)
- 0.294ms per block (10 blocks)
- Linear layers: 30% of CUDA time
- Memory efficient: 23.82 MB peak

**When to read:** Understanding basic performance characteristics.

---

### [REALISTIC_BATCH_ANALYSIS.md](REALISTIC_BATCH_ANALYSIS.md)
Analysis of actual training (batch=1024) and search (batch=128) configurations.

**Key Findings:**
- Training: 55.6ms per batch, 18,404 pos/s
- Search: 7.67ms per batch, 16,687 pos/s
- 51x speedup with batching vs single position
- MCTS can evaluate 1,792 positions in 100ms

**When to read:** Understanding real-world performance for training and MCTS.

---

### [LARGE_BATCH_ANALYSIS.md](LARGE_BATCH_ANALYSIS.md)
Throughput testing with extreme batch sizes (1-4096).

**Key Findings:**
- Perfect scaling up to batch=32 (100% efficiency)
- Maximum throughput: 18,865 pos/s at batch=4096
- Diminishing returns after batch=64
- Memory usage: only 208 MB at batch=4096

**When to read:** Optimizing for maximum throughput or batch inference.

---

## Quick Reference

### Performance Summary

| Configuration | Batch Size | Time per Batch | Per Position | Throughput |
|---------------|------------|----------------|--------------|------------|
| Single        | 1          | 3.06ms         | 3.06ms       | 327 pos/s  |
| MCTS (small)  | 32         | 3.59ms         | 0.11ms       | 8,920 pos/s |
| MCTS (default)| 128        | 7.67ms         | 0.06ms       | 16,687 pos/s |
| Training      | 1024       | 55.6ms         | 0.05ms       | 18,404 pos/s |
| Maximum       | 4096       | 217ms          | 0.05ms       | 18,865 pos/s |

### Bottlenecks

1. **Linear layers (MLP)** - 30.2% of time
   - Mitigation: Quantization, reduce expansion ratio
   
2. **Depthwise convolutions** - 26.5% of time
   - Mitigation: Fused kernels, channel shuffle
   
3. **Matrix multiplications** - 26.8% of time
   - Mitigation: INT8 quantization, TensorRT

### Recommendations

1. **Use batch=128 for MCTS** - Optimal balance of latency and throughput
2. **Use batch=1024 for training** - Maximum GPU utilization
3. **Implement batched inference** - 51x speedup over single positions
4. **Consider INT8 quantization** - Potential 2-4x additional speedup

---

## Related Documents

- [../../docs/optimization_ideas.md](../../docs/optimization_ideas.md) - General optimization strategies
- [../../docs/PROFILING_GUIDE.md](../../docs/PROFILING_GUIDE.md) - Profiling guide
- [../../log.md](../../log.md) - Experiment log entry for exp015

---

## Profiling Scripts

Located in parent directory:
- `profile.py` - Quick profiling (default batch=128)
- `profile_detailed.py` - Comprehensive analysis
- `profile_large_batch.py` - Throughput testing

Run with:
```bash
bash profile.sh                    # Quick profile
python profile_detailed.py         # Detailed analysis
python profile_large_batch.py      # Throughput testing
```
