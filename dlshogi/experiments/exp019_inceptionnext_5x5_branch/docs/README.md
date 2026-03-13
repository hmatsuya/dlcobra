# exp019: InceptionNeXt with 5x5 Depthwise Branch

## Motivation

The standard InceptionNeXt block (exp015) uses 4 parallel depthwise branches:
- Identity (no conv)
- 3x3 depthwise (local patterns)
- 1x9 depthwise (horizontal, rook/lance)
- 9x1 depthwise (vertical, rook/lance)

However, this design requires multiple layers to propagate information about:
- Knight (桂馬) L-shaped jumps
- Bishop (角) diagonal moves
- Extended tactical sequences

## Solution: 5x5 Depthwise Branch

Add a 5th parallel branch with 5x5 depthwise convolution:
- Captures knight jumps and bishop diagonals in a single layer
- Negligible FLOP increase (depthwise convolution)
- Dramatically improves receptive field for Shogi-specific piece movements

## Implementation Details

### Channel Split
- Requires `dim` divisible by 5
- Changed from `dims=[192]` to `dims=[190]`
- Each branch gets `dim // 5 = 38` channels

### Architecture
```python
branch_dim = dim // 5  # 190 // 5 = 38

# 5 parallel branches
x0: identity (38 channels)
x1: 3x3 depthwise (38 channels)
x2: 1x9 depthwise (38 channels)
x3: 9x1 depthwise (38 channels)
x4: 5x5 depthwise (38 channels)  # NEW

# Concatenate and apply MLP
x = concat([x0, x1, x2, x3, x4])  # 190 channels
```

## Expected Impact

- **Accuracy**: High (improved tactical sequence evaluation)
- **Speed**: Negligible overhead (~7.6ms, same as exp015)
- **Parameters**: 3.6M (similar to exp015's 3.7M)

## Profiling Results (batch=128)

- CUDA time: ~7.6ms per forward pass
- Comparable to exp015 (7.6ms)
- Depthwise convolution overhead is minimal

## Next Steps

1. Train on full dataset to evaluate accuracy improvement
2. Compare tactical position evaluation vs exp015
3. Consider combining with lighter value head (exp018) if successful
