# exp018: Lighter Value Head (2 channels)

## Overview
Based on exp015 (InceptionNeXt depths=[10], dims=[192]) with a critical value head optimization.

## Problem
The original value head uses `MAX_MOVE_LABEL_NUM` (27 for Shogi) as the channel dimension before flattening:
```python
value_conv: dim -> 27 channels
value_fc1: 9*9*27 = 2187 -> fcl
```

This creates an unnecessarily large linear layer that:
- Wastes ~500k parameters (~14% of exp015's 3.7M)
- Slows down `addmm` CUDA operations
- Provides no clear benefit (value head only needs to predict win probability)

## Solution
Reduce value head to 2 channels before flattening (AlphaZero/KataGo pattern):
```python
value_conv: dim -> 2 channels
value_fc1: 9*9*2 = 162 -> fcl
```

## Expected Impact
- **Parameters**: ~500k reduction (3.7M → ~3.2M)
- **Speed**: Noticeable reduction in linear layer CUDA time
- **Accuracy**: Neutral or slight gain (reduced overfitting)

## Implementation Details
Changed in `model.py`:
```python
# OLD (exp015)
self.value_conv = nn.Conv2d(dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)

# NEW (exp018)
value_channels = 2
self.value_conv = nn.Conv2d(dim, value_channels, kernel_size=1, bias=False)
self.value_fc1 = nn.Linear(9 * 9 * value_channels, fcl)
```

## Usage

### Debug run (verify model)
```bash
bash dlshogi/experiments/exp018_lighter_value_head/run.sh --debug
```

### Profile (batch=128)
```bash
bash dlshogi/experiments/exp018_lighter_value_head/profile.sh
```

### Detailed profiling
```bash
python dlshogi/experiments/exp018_lighter_value_head/profile_detailed.py
```

### Full training
```bash
bash dlshogi/experiments/exp018_lighter_value_head/run.sh
```

## Comparison with exp015

| Metric | exp015 | exp018 | Change |
|--------|--------|--------|--------|
| Total params | 3.67M | 3.14M | -523k (-14.3%) |
| Value fc1 input | 2187 | 162 | -13x |
| Inference time (batch=128) | 7.61ms | 7.55ms | 1.01x faster (0.8%) |
| Throughput | 16,821 samples/s | 16,963 samples/s | +142 samples/s |

### Analysis
- **Parameter reduction achieved**: 523k parameters removed (14.3%), matching expectations
- **Speed improvement**: Modest 0.8% speedup at batch=128
  - The speedup is smaller than expected because the value head linear layer is only a small portion of total compute
  - Most time is spent in the backbone (InceptionNeXt blocks with depthwise convolutions)
  - Larger speedups may be visible at smaller batch sizes where linear layers are more dominant
- **Memory efficiency**: Reduced model size improves memory footprint for deployment
- **Accuracy**: Expected to be neutral or slightly better (reduced overfitting risk)

## References
- AlphaZero paper: uses 1-2 channels for value head
- KataGo: uses 2 channels for value head
- Original optimization idea: `dlshogi/docs/optimization_ideas-google_ai_studio.md`
