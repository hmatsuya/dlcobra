# exp018 Checklist - Completed ✓

## 1. Debug Mode Run ✓
```bash
bash dlshogi/experiments/exp018_lighter_value_head/run.sh --debug
```

**Result**: Success
- Model created and trained for 2 epochs
- Trainable params: 3.1M (confirmed)
- No errors during forward/backward pass

## 2. Update log.md with Parameter Count ✓
Updated `dlshogi/experiments/log.md` with:
- **パラメータ数**: 3.1M
- Parameter reduction: 523k (14.3% reduction from exp015's 3.67M)
- Inference time: 7.55ms (batch=128)

## 3. Profile Experiment ✓
```bash
bash dlshogi/experiments/exp018_lighter_value_head/profile.sh
```

**Results**:
- Total params: 3,145,168
- Batch size: 128
- Average time: 7.55 ms
- Throughput: 16,963 samples/sec

## 4. Compare with exp015 ✓
Created and ran `compare_with_exp015.py`

**Comparison Results**:

| Metric | exp015 | exp018 | Change |
|--------|--------|--------|--------|
| Total params | 3,668,418 | 3,145,168 | -523,250 (-14.3%) |
| Inference time | 7.61 ms | 7.55 ms | 1.01x faster |
| Throughput | 16,821/s | 16,963/s | +142 samples/s |

### Key Findings:
1. **Parameter reduction achieved**: 523k parameters removed (14.3%), matching expectations
2. **Speed improvement**: Modest 0.8% speedup
   - Smaller than expected because value head is small portion of total compute
   - Most time spent in backbone (InceptionNeXt blocks)
   - Larger speedups may appear at smaller batch sizes
3. **Memory efficiency**: Reduced model size benefits deployment
4. **Accuracy**: Expected neutral or slightly better (less overfitting)

## 5. Detailed Profiling ✓
Ran `profile_detailed.py` for both exp018 and exp015

**Key Findings**:
- Value head timing: exp015 ~0.172ms → exp018 ~0.15ms (~13% faster)
- Value head is only ~6% of total compute (backbone is ~95%)
- Linear layer CUDA time reduced as expected
- Created detailed comparison document: `docs/PROFILING_COMPARISON.md`

## Next Steps (Optional)
- [ ] Run full training to compare accuracy metrics
- [ ] Test with smaller batch sizes (1, 8, 32) to see if speedup is more pronounced (expected 2-5%)
- [ ] Profile with torch.compile to see if optimization benefits are amplified
- [ ] Consider applying same optimization to other experiments

## Files Created
- `__init__.py`, `config.yaml`, `model.py`, `run.sh`
- `profile.py`, `profile.sh`, `profile_detailed.py`
- `docs/README.md`
- `compare_with_exp015.py`
- `CHECKLIST_COMPLETED.md` (this file)
