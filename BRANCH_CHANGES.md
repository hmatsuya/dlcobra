# Branch Changes Summary: wcsc35 vs upstream/master

## Overview
This document summarizes all changes made in the `wcsc35` branch compared to `upstream/master`. The branch was developed for WCSC35 (World Computer Shogi Championship 2035) competition.

---

## 1. Input Feature Changes ⚠️

### Features Removed
- ❌ **Piece attack features (利き)** - 14 planes removed
- ❌ **Attack count features (利き数)** - 3 planes removed  
- ❌ **In-check feature (王手)** - 1 plane removed
- ❌ **Packed features support** - Entire packed_features1_t and packed_features2_t removed

### Features Modified
- 🔄 **Hand piece representation**: Changed from one-hot encoding to log-count encoding
  - Old: 56 planes (8+4+4+4+4+2+2 per color)
  - New: 14 planes (1 per piece type per color)
  - Encoding: `log(count + 1)` instead of binary planes

### Net Result
- **Features1**: 62 planes → 28 planes (55% reduction)
- **Features2**: 57 planes → 14 planes (75% reduction)
- **Total**: ~65% fewer input features

### Key Commits
- `f4d4ce1` - feat(shogi): implement log-count feature representation
- `3e3dd61` - refactor(shogi): remove in-check feature from input generation
- `9ae6acb` - refactor(shogi): remove attack and attack_num features from input
- `728228d` - refactor: Remove packed_features1_t and related functionality

### Files Modified
- `cppshogi/cppshogi.h`
- `cppshogi/cppshogi.cpp`
- `cppshogi/python_module.cpp`
- `dlshogi/cppshogi.pyx`

---

## 2. Neural Network Architecture

### New Custom Network
**File**: `dlshogi/network/policy_value_network_cr.py`

**Architecture**:
```
InitialBlock (dual-path):
  - Conv2d 3x3 (FEATURES1 → 192 channels)
  - Conv2d 1x1 (FEATURES1 → 192 channels)  
  - Conv2d 1x1 (FEATURES2 → 192 channels)
  - Sum + BatchNorm + ReLU

Middle Blocks (50x ResNet blocks):
  - Conv2d 3x3 → BatchNorm → ReLU
  - Conv2d 3x3 → BatchNorm
  - Residual connection + ReLU

Policy Head:
  - Conv2d 1x1 (192 → MAX_MOVE_LABEL_NUM)
  - Flatten

Value Head:
  - Conv2d 1x1 (192 → 32)
  - BatchNorm + ReLU + Flatten
  - Linear(32*9*9 → 256) + Dropout(0.05)
  - Linear(256 → 1)
```

**Key Features**:
- 50 ResNet blocks (configurable)
- 192 channels throughout
- Dropout in value head (0.05)
- Dual-path initial convolution

### Key Commits
- `8702bd1` - feat(network): replace resnet10_relu with custom PolicyValueNetwork
- `4ade2ef` - feat(network): increase number of middle blocks to improve capacity
- `514946e` - feat(network): update PolicyValueNetwork architecture and ONNX export

---

## 3. Training Enhancements

### Stochastic Weight Averaging (SWA)
**Implementation**: `dlshogi/ptl.py`

```python
use_swa=True
swa_start_epoch=10
swa_lr=1e-4
```

- Averages model weights during training
- Updates batch normalization statistics at end
- Improves generalization

### Exponential Moving Average (EMA)
```python
use_ema=True
ema_start_epoch=1
ema_freq=250
ema_decay=0.9
update_bn=True
```

- Maintains exponentially weighted average of model parameters
- Updates every 250 steps
- Optional batch norm update at end of training

### Enhanced Metrics
- Policy accuracy logging
- Value accuracy logging
- Policy entropy monitoring
- Value entropy monitoring
- Configurable val_lambda decay

### Key Commits
- `572f9d5` - feat(swa): implement Stochastic Weight Averaging (SWA)
- `d372e9a` - feat(training): add early stopping callback
- `11e3da1` - feat(logging): add policy and value accuracy metrics

---

## 4. Data Pipeline

### HDF5 Support
**New Files**:
- `dlshogi/utils/hcpe_to_hdf5.py`
- `dlshogi/utils/hcpe_to_hdf5.sh`

**Features**:
- Efficient loading of large datasets
- Optional shuffling during conversion
- Dask-based lazy loading
- Reduced memory footprint

**Usage**:
```python
from dlshogi.data_loader import Hdf5DataLoader
dataset = Hdf5DataLoader.load_files(files)
```

### Data Conversion Scripts
**New Files**:
- `dlshogi/utils/plaintext_to_hcpe.py` - Convert plaintext games to HCPE
- `dlshogi/utils/plaintext_to_hdf5.sh` - Pipeline script
- `dlshogi/utils/test_plaintext_to_hcpe.py` - Unit tests

**Features**:
- SFEN position flipping
- SFEN move flipping
- Validation checks
- Wildcard file pattern support

### Key Commits
- `02a5395` - feat(data-loading): replace individual file loading with concatenated HDF5
- `e5929d6` - feat(utils): add script to convert hcpe files to hdf5
- `736e929` - feat(utils): add plaintext_to_hcpe conversion script
- `421c4b1` - Enhance DataLoader to support wildcard file patterns

---

## 5. Model Export & Deployment

### ONNX Export
**New Files**:
- `dlshogi/wandb_checkpoint_to_model.py` - W&B checkpoint → NPZ
- `dlshogi/wandb_to_onnx.sh` - W&B checkpoint → ONNX
- `dlshogi/model.onnx` - Pre-built ONNX model (43MB)

**Features**:
- Export PyTorch Lightning checkpoints to ONNX
- Support for W&B artifact loading
- Automatic feature dimension logging

### ONNX Runtime Integration
**New Files**:
- `usi_onnxruntime/include/onnxruntime_c_api.h`
- `usi_onnxruntime/include/onnxruntime_cxx_api.h`
- `usi_onnxruntime/include/cpu_provider_factory.h`

**Features**:
- CPU inference support
- Cross-platform compatibility

### Key Commits
- `86789b0` - feat(onnx-export): add script to convert wandb checkpoints to ONNX
- `471a9c2` - feat(script): add wandb_to_onnx conversion script
- `840824e` - Add ONNX Runtime library

---

## 6. Infrastructure & DevOps

### Dependency Management
**New Files**:
- `Pipfile` - Pipenv dependency specification
- `Pipfile.lock` - Locked dependency versions
- `requirements.txt` - Pip requirements with pinned versions

**Key Dependencies Added**:
- pandas
- icecream (debugging)
- asttokens
- pytorch-lightning
- wandb

### Setup Scripts
**New Files**:
- `setup.sh` - Project installation script
- `dlshogi/ptl.sh` - Training launch script
- `docker/setup-vast.sh` - Cloud GPU setup (Vast.ai)

**Features**:
- Automated environment setup
- Pipenv integration
- Docker support for cloud training

### Key Commits
- `e333ba6` - Add Pipfile to manage project dependencies
- `6ff6b0c` - Add setup script for project installation
- `13f072a` - feat(docker): add setup script for dlcobra with TensorRT

---

## 7. Engine Customization

### USI Engine Updates
**Modified Files**:
- `usi/main.cpp`
- `usi/UctSearch.cpp`
- `usi/Makefile`

**Changes**:
- Engine name: `"dlcobra wcsc35"`
- Author: `"Hiroaki Matsuyama"`
- GPU support: Expanded to 10 GPUs (from previous limit)
- Compilation: Switched to optimized flags (release mode)

### Key Commits
- `a040934` - feat(usi): update engine name to 'dlcobra wcsc35'
- `a5cc101` - feat(usi): update author information
- `3b379ec` - feat(search): expand UCT thread and DNN model support to 10 GPUs
- `54a6a9c` - perf(usi/Makefile): switch from debug to optimized compilation flags

---

## 8. Configuration Changes

### Training Config
**File**: `dlshogi/config.yaml`

**Major Changes**:
- WandbLogger integration
- Wildcard file pattern support
- Early stopping configuration
- SWA/EMA parameters
- Learning rate schedules
- Model checkpoint settings

### Key Commits
- `8f09591` - Update config.yaml for WandbLogger integration
- `ef85fac` - Add WandbLogger configuration to the trainer
- `3f7b20f` - Update config.yaml to support wildcard patterns

---

## 9. Removed/Cleaned Up

### Deleted Files
- `dlshogi/utils/spsa_usi_tuner.py` - SPSA tuning script
- `dlshogi/utils/stat_hcpe.py` - Statistics script
- `dlshogi/utils/csa_to_important_position.py`
- `dlshogi/utils/csa_to_important_sfen.py`
- `dlshogi/utils/find_position_in_csa_dir.py`

### Removed Features
- Packed features support (entire subsystem)
- Old ResNet10 with ReLU (replaced with custom network)

---

## Recommendation: Should You Clean Start?

### ✅ YES - Clean Start Recommended IF:

1. **You want upstream's input features back**
   - The input feature changes are deeply integrated
   - Reverting just features while keeping other changes is complex
   - Risk of incompatibility between old features and new network

2. **You want to cherry-pick specific improvements**
   - Start fresh from upstream/master
   - Selectively apply commits you want to keep
   - Easier to maintain and understand

3. **You want to avoid technical debt**
   - Current branch has 70+ commits ahead
   - Some experimental changes that were reverted
   - Cleaner to rebuild with intention

### ❌ NO - Keep Current Branch IF:

1. **You want ALL the improvements together**
   - Training infrastructure (SWA, EMA)
   - Data pipeline (HDF5)
   - ONNX export
   - Custom network architecture

2. **You're okay with the reduced input features**
   - The log-count encoding is intentional
   - Network was designed for these features

---

## Recommended Approach: Selective Cherry-Pick

### Step 1: Create New Branch from Upstream
```bash
git checkout upstream/master
git checkout -b wcsc35-v2
```

### Step 2: Cherry-Pick Non-Feature Commits

**Infrastructure (safe to apply):**
```bash
git cherry-pick e333ba6  # Pipfile
git cherry-pick 6ff6b0c  # setup.sh
git cherry-pick 13f072a  # docker setup
git cherry-pick 9231021  # requirements.txt
```

**Training Enhancements (safe to apply):**
```bash
git cherry-pick 572f9d5  # SWA implementation
git cherry-pick d372e9a  # Early stopping
git cherry-pick 11e3da1  # Accuracy metrics
git cherry-pick 8f09591  # WandbLogger config
```

**Data Pipeline (safe to apply):**
```bash
git cherry-pick 02a5395  # HDF5 data loading
git cherry-pick e5929d6  # hcpe_to_hdf5
git cherry-pick 736e929  # plaintext_to_hcpe
git cherry-pick 421c4b1  # Wildcard patterns
```

**Model Export (safe to apply):**
```bash
git cherry-pick 86789b0  # ONNX export script
git cherry-pick 471a9c2  # wandb_to_onnx
```

**Engine Updates (safe to apply):**
```bash
git cherry-pick a040934  # Engine name
git cherry-pick a5cc101  # Author info
git cherry-pick 3b379ec  # 10 GPU support
```

### Step 3: Skip Feature-Related Commits

**DO NOT cherry-pick:**
- `f4d4ce1` - Log-count features
- `3e3dd61` - Remove in-check
- `9ae6acb` - Remove attack features
- `728228d` - Remove packed features
- `8702bd1` - Custom network (designed for reduced features)

### Step 4: Adapt Network Architecture

You'll need to either:
1. Use upstream's existing network (resnet10_relu)
2. Modify the custom network to work with upstream's feature dimensions

---

## Summary

**Total Changes**: 70+ commits
**Files Modified**: ~60 files
**New Files**: ~20 files
**Lines Changed**: ~3000+ lines

**Core Innovation**: Reduced input features + custom 50-block ResNet + advanced training techniques

**Recommendation**: **Clean start with selective cherry-picking** to get the training improvements without the input feature changes.
