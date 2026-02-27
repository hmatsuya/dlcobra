# Cherry-Pick Guide: Migrating Improvements to Upstream Features

## Overview

This guide helps you migrate the training improvements from your `wcsc35` branch while keeping upstream's full input features (attacks, attack counts, check detection).

## What Gets Preserved

✅ **From Upstream (kept)**:
- Full input features (62 + 57 planes)
- Piece attacks (利き) - 14 planes
- Attack counts (利き数) - 3 planes  
- In-check detection (王手) - 1 plane
- Packed features support
- Original network architectures

✅ **From wcsc35 (applied)**:
- Training improvements (SWA, EMA, metrics)
- Data pipeline (HDF5, wildcard patterns)
- ONNX export tools
- Infrastructure (Pipfile, setup scripts)
- Engine customization (10 GPU support)
- Docker/cloud setup

❌ **From wcsc35 (skipped)**:
- Log-count hand piece encoding
- Reduced feature dimensions
- Custom network architecture (designed for reduced features)
- Packed features removal

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
# Run the automated cherry-pick script
./cherry-pick-improvements.sh
```

The script will:
1. Create a new branch `wcsc35-with-upstream-features`
2. Cherry-pick 40+ commits in logical order
3. Handle conflicts interactively
4. Provide a summary at the end

### Option 2: Manual Cherry-Pick

If you prefer manual control:

```bash
# Create new branch from upstream
git checkout upstream/master
git checkout -b wcsc35-with-upstream-features

# Cherry-pick commits one by one (see detailed list below)
git cherry-pick <commit-hash>
```

## Detailed Commit List

### Phase 1: Infrastructure (7 commits)
```bash
git cherry-pick ea4e343  # Update .gitignore
git cherry-pick e333ba6  # Add Pipfile
git cherry-pick 6ff6b0c  # Add setup script
git cherry-pick d733022  # Update setup for pipenv
git cherry-pick 9231021  # Add requirements.txt
git cherry-pick 669954b  # Add icecream/asttokens
git cherry-pick d9a2215  # Add pandas
```

### Phase 2: Configuration & Logging (4 commits)
```bash
git cherry-pick 8f09591  # WandbLogger config
git cherry-pick ef85fac  # WandbLogger trainer config
git cherry-pick dac7394  # Redirect to fit.log
git cherry-pick a6c9025  # Training script with pipenv
```

### Phase 3: Data Pipeline (11 commits)
```bash
git cherry-pick 3f7b20f  # Wildcard patterns in config
git cherry-pick 421c4b1  # Wildcard DataLoader
git cherry-pick 736e929  # plaintext_to_hcpe
git cherry-pick e5929d6  # hcpe_to_hdf5
git cherry-pick 3326607  # hcpe_to_hdf5 with shuffle
git cherry-pick 094ce14  # Make script executable
git cherry-pick 02a5395  # HDF5 data loading
git cherry-pick c2dc527  # plaintext_to_hdf5
git cherry-pick 54c0366  # psv_to_hcpe validation
git cherry-pick 722a128  # Explicit imports
git cherry-pick 0403ee3  # File existence check
```

### Phase 4: Testing (3 commits)
```bash
git cherry-pick 903e693  # flip_sfen tests
git cherry-pick b442787  # SFEN flipping
git cherry-pick a2050cd  # flip_sfen_square tests
```

### Phase 5: Training Enhancements (5 commits)
```bash
git cherry-pick 11e3da1  # Accuracy metrics
git cherry-pick d372e9a  # Early stopping
git cherry-pick 572f9d5  # SWA implementation
git cherry-pick 7835ad3  # SWA logger
git cherry-pick 4e4d384  # Refactor ema_avg
```

### Phase 6: Model Export (2 commits)
```bash
git cherry-pick 86789b0  # wandb to ONNX
git cherry-pick 471a9c2  # wandb_to_onnx script
```

### Phase 7: Engine Updates (4 commits)
```bash
git cherry-pick a040934  # Engine name
git cherry-pick a5cc101  # Author info
git cherry-pick 3b379ec  # 10 GPU support
git cherry-pick 54a6a9c  # Optimized flags
```

### Phase 8: Docker (1 commit)
```bash
git cherry-pick 13f072a  # TensorRT setup
```

### Phase 9: Config Adjustments (6 commits)
```bash
git cherry-pick 6f8c701  # Training data config
git cherry-pick 249dea2  # Disable model save
git cherry-pick aa9a21d  # Extend patience/LR
git cherry-pick 48c8ed9  # Adjust LR/disable SWA
git cherry-pick bc5a88d  # Reduce LR
git cherry-pick e24fdca  # Adjust patience
```

## Handling Conflicts

### Common Conflicts

1. **config.yaml conflicts**
   - Usually safe to take the new version
   - Manually merge if both versions have important changes

2. **ptl.py conflicts**
   - Take the new version (has SWA/EMA improvements)
   - Verify FEATURES1_NUM/FEATURES2_NUM imports are correct

3. **data_loader.py conflicts**
   - Take the new version (has HDF5 support)

### Conflict Resolution Steps

```bash
# When conflict occurs:
git status  # See conflicted files

# Edit files to resolve conflicts
# Look for <<<<<<< HEAD markers

# After resolving:
git add <resolved-files>
git cherry-pick --continue

# Or skip if commit is problematic:
git cherry-pick --skip

# Or abort entire process:
git cherry-pick --abort
```

## Post-Migration Steps

### 1. Verify Feature Dimensions

```bash
# Check that upstream features are preserved
grep "MAX_FEATURES1_NUM\|MAX_FEATURES2_NUM" cppshogi/cppshogi.h
```

Expected output:
```cpp
constexpr u32 MAX_FEATURES1_NUM = PIECETYPE_NUM + PIECETYPE_NUM + MAX_ATTACK_NUM;  // 31
constexpr u32 MAX_FEATURES2_NUM = MAX_FEATURES2_HAND_NUM + 1/*王手*/ ...;  // 57+
```

### 2. Rebuild C++ Extensions

```bash
python setup.py build_ext --inplace
```

### 3. Test Data Loading

```python
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
print(f"FEATURES1_NUM: {FEATURES1_NUM}")  # Should be 62 (2 colors * 31)
print(f"FEATURES2_NUM: {FEATURES2_NUM}")  # Should be 57+
```

### 4. Choose Network Architecture

You have two options:

#### Option A: Use Upstream's Network (Easiest)

```yaml
# In config.yaml
model:
  network: resnet10_swish  # or resnet10_relu
```

#### Option B: Adapt Custom Network

Modify `dlshogi/network/policy_value_network_cr.py`:

```python
# Update InitialBlock to handle larger feature dimensions
class InitialBlock(torch.nn.Module):
    def __init__(self, out_channels):
        super(InitialBlock, self).__init__()
        # FEATURES1_NUM will be 62 (not 28)
        # FEATURES2_NUM will be 57+ (not 14)
        self.conv1a = torch.nn.Conv2d(FEATURES1_NUM, out_channels, 3, ...)
        self.conv1b = torch.nn.Conv2d(FEATURES1_NUM, out_channels, 1, ...)
        self.conb2  = torch.nn.Conv2d(FEATURES2_NUM, out_channels, 1, ...)
        # ... rest stays the same
```

### 5. Test Training

```bash
# Small test run
python dlshogi/ptl.py fit \
  --config dlshogi/config.yaml \
  --trainer.max_epochs 1 \
  --trainer.limit_train_batches 10
```

### 6. Verify ONNX Export

```bash
# Test ONNX export with new feature dimensions
python dlshogi/wandb_checkpoint_to_model.py \
  --checkpoint path/to/checkpoint.ckpt \
  --output test_model.onnx
```

## Troubleshooting

### Issue: "FEATURES1_NUM mismatch"

**Cause**: Some cherry-picked commit modified feature dimensions

**Solution**:
```bash
# Revert the problematic commit
git revert <commit-hash>

# Or manually fix cppshogi/cppshogi.h
```

### Issue: "Network input dimension mismatch"

**Cause**: Using custom network with upstream features

**Solution**: Either use upstream network or adapt custom network (see step 4 above)

### Issue: "HDF5 files incompatible"

**Cause**: HDF5 files created with old feature dimensions

**Solution**: Regenerate HDF5 files from HCPE:
```bash
python dlshogi/utils/hcpe_to_hdf5.py \
  --input data/*.hcpe \
  --output data/train.hdf5
```

### Issue: "ONNX export fails"

**Cause**: Model was trained with old feature dimensions

**Solution**: Retrain model with new feature dimensions

## Validation Checklist

- [ ] Feature dimensions match upstream (62 + 57+)
- [ ] C++ extensions build successfully
- [ ] Data loading works with new dimensions
- [ ] Training runs without errors
- [ ] Validation metrics are logged
- [ ] SWA/EMA features work
- [ ] ONNX export succeeds
- [ ] Engine compiles and runs

## Rollback Plan

If something goes wrong:

```bash
# Delete the new branch
git checkout wcsc35
git branch -D wcsc35-with-upstream-features

# Start over
./cherry-pick-improvements.sh
```

## Summary

**What you gain:**
- All training improvements (SWA, EMA, metrics, early stopping)
- Better data pipeline (HDF5, wildcards)
- ONNX export tools
- Infrastructure improvements
- Full upstream input features (more information for the model)

**What you lose:**
- Compact log-count encoding (but gain more detailed features)
- Custom 50-block network (but can adapt it or use upstream networks)

**Net result:** Better training infrastructure + richer input features = potentially better model performance!
