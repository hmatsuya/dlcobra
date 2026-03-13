# Profiling Guide for Experiments

This guide explains which profiling script to use for different purposes.

---

## Available Profiling Scripts

### 1. `profile.py` - Quick Standard Profiling

**Purpose:** Fast, standardized profiling for comparing experiments

**What it does:**
- Basic PyTorch profiler with CUDA timing
- Exports Chrome trace for visualization
- Configurable batch size and runs
- Works with any experiment (auto-detects model)

**Output:**
- `trace.json` - Chrome trace file (view at chrome://tracing)
- Console output with top operations

**Usage:**
```bash
bash profile.sh
# or with custom settings:
python profile.py --batch-size 128 --runs 50
```

**When to use:**
- ✅ Quick profiling during development
- ✅ Comparing multiple experiments
- ✅ Identifying major bottlenecks
- ✅ Standard workflow (called by profile.sh)

**Pros:**
- Fast (~10 seconds)
- Simple output
- Consistent across experiments
- Easy to compare results

**Cons:**
- Limited analysis
- No batch size comparison
- No detailed breakdowns
- No MCTS-specific metrics

---

### 2. `profile_detailed.py` - Comprehensive Analysis

**Purpose:** Deep dive profiling for optimization work

**What it does:**
- Everything from `profile.py` PLUS:
- Per-block timing analysis
- Memory usage breakdown
- Operation category analysis (conv, linear, norm, etc.)
- Batch size comparison (1, 16, 32, 64, 128, 256, 512, 1024)
- Training vs Search configuration analysis
- MCTS performance estimates
- Parameter counting and FLOPs estimation

**Output:**
- `profile_trace.json` - Chrome trace
- Detailed console output with multiple sections
- Analysis of bottlenecks and recommendations

**Usage:**
```bash
python profile_detailed.py
```

**When to use:**
- ✅ Optimizing a specific experiment
- ✅ Understanding performance characteristics
- ✅ Preparing for production deployment
- ✅ Writing optimization documentation
- ✅ Investigating performance issues

**Pros:**
- Comprehensive analysis
- Batch size scaling insights
- MCTS-specific metrics
- Detailed bottleneck identification
- Memory profiling

**Cons:**
- Slower (~2-3 minutes)
- More complex output
- Experiment-specific (needs model.py)
- Overkill for quick checks

---

### 3. `profile_large_batch.py` - Throughput Analysis

**Purpose:** Test extreme batch sizes for throughput limits

**What it does:**
- Tests batch sizes: 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
- Measures scaling efficiency
- Identifies optimal batch size
- Memory usage per batch size
- MCTS implications for different time budgets

**Output:**
- `large_batch_results.txt` - Raw data
- Console output with analysis

**Usage:**
```bash
python profile_large_batch.py
```

**When to use:**
- ✅ Finding maximum throughput
- ✅ Optimizing for batch inference
- ✅ Understanding GPU saturation point
- ✅ Planning production deployment

**Pros:**
- Tests extreme batch sizes
- Identifies optimal configuration
- Useful for batch inference systems

**Cons:**
- Very slow (~5 minutes)
- Not needed for most experiments
- Requires significant GPU memory

---

## Recommendation for New Experiments

### Standard Workflow

**Step 1: Quick Check (profile.py)**
```bash
bash profile.sh
```
Use this for:
- Initial profiling after creating experiment
- Quick comparison with baseline
- Verifying model runs correctly
- Checking for obvious issues

**Step 2: Detailed Analysis (profile_detailed.py)** - Optional
```bash
python profile_detailed.py
```
Use this when:
- Experiment shows promise (good accuracy)
- Planning to use in production
- Need to understand bottlenecks
- Writing documentation/paper

**Step 3: Throughput Testing (profile_large_batch.py)** - Rare
```bash
python profile_large_batch.py
```
Use this when:
- Deploying to production
- Optimizing batch inference system
- Need maximum throughput data

---

## Quick Comparison Table

| Feature                    | profile.py | profile_detailed.py | profile_large_batch.py |
|----------------------------|------------|---------------------|------------------------|
| **Time to run**            | ~10s       | ~2-3min             | ~5min                  |
| **Basic timing**           | ✅         | ✅                  | ✅                     |
| **Chrome trace**           | ✅         | ✅                  | ❌                     |
| **Per-block timing**       | ❌         | ✅                  | ❌                     |
| **Memory analysis**        | ❌         | ✅                  | ✅                     |
| **Operation breakdown**    | ❌         | ✅                  | ❌                     |
| **Batch size comparison**  | ❌         | ✅ (1-1024)         | ✅ (1-4096)            |
| **MCTS metrics**           | ❌         | ✅                  | ✅                     |
| **Scaling efficiency**     | ❌         | ❌                  | ✅                     |
| **Works with any model**   | ✅         | ⚠️ (needs model.py) | ⚠️ (needs model.py)    |
| **Called by profile.sh**   | ✅         | ❌                  | ❌                     |

---

## Recommended Usage Pattern

### For Most Experiments
```bash
# 1. Create experiment
cp -r dlshogi/experiments/_template dlshogi/experiments/exp017_my_experiment

# 2. Implement model
vim dlshogi/experiments/exp017_my_experiment/model.py

# 3. Quick profile check
bash dlshogi/experiments/exp017_my_experiment/profile.sh

# 4. If results look good, train
bash dlshogi/experiments/exp017_my_experiment/run.sh
```

### For Production-Ready Experiments
```bash
# 1. After training shows good results
bash dlshogi/experiments/exp017_my_experiment/profile.sh

# 2. Detailed analysis
python dlshogi/experiments/exp017_my_experiment/profile_detailed.py

# 3. Document findings in experiment README
# 4. If deploying, test throughput
python dlshogi/experiments/exp017_my_experiment/profile_large_batch.py
```

---

## Creating Profiling Scripts for New Experiments

### Option 1: Copy from exp015 (Recommended)
```bash
# Copy all profiling scripts
cp dlshogi/experiments/exp015_inceptionnext_depth10/profile*.py \
   dlshogi/experiments/exp017_my_experiment/

# Update imports if needed (usually automatic)
```

### Option 2: Use Template
The `_template` directory should include `profile.py` which works for any experiment.

---

## Tips

### For Quick Iteration
- Use `profile.py` with `--batch-size 128` (matches search default)
- Compare trace files side-by-side in chrome://tracing
- Focus on total time and top 5 operations

### For Optimization
- Use `profile_detailed.py` to identify bottlenecks
- Check operation breakdown percentages
- Look at per-block timing for anomalies
- Compare batch size scaling

### For Production
- Use `profile_large_batch.py` to find optimal batch size
- Test with realistic batch sizes (128 for search, 1024 for training)
- Measure memory usage to ensure headroom
- Document throughput numbers

---

## Example: Comparing Two Experiments

```bash
# Profile exp015 (baseline)
cd dlshogi/experiments/exp015_inceptionnext_depth10
bash profile.sh > profile_output.txt

# Profile exp017 (new experiment)
cd dlshogi/experiments/exp017_channel_shuffle
bash profile.sh > profile_output.txt

# Compare
diff exp015_inceptionnext_depth10/profile_output.txt \
     exp017_channel_shuffle/profile_output.txt

# Or compare trace files visually
# Open both trace.json files in chrome://tracing
```

---

## Summary

**Use `profile.py` (via `bash profile.sh`) for:**
- ✅ Standard workflow
- ✅ Quick checks
- ✅ Comparing experiments
- ✅ 90% of profiling needs

**Use `profile_detailed.py` for:**
- ✅ Deep optimization work
- ✅ Understanding bottlenecks
- ✅ Production preparation
- ✅ Documentation

**Use `profile_large_batch.py` for:**
- ✅ Throughput optimization
- ✅ Production deployment
- ✅ Batch inference systems
- ✅ Rare, specialized needs

**Default recommendation:** Start with `profile.py`, upgrade to `profile_detailed.py` only when needed.
