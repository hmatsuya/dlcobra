#!/bin/bash
# Cherry-pick script to apply non-feature improvements from wcsc35 branch
# This preserves upstream's input features while adding training/infrastructure improvements

set -e  # Exit on error

echo "=========================================="
echo "Cherry-Pick Script for wcsc35 Improvements"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Create a new branch from upstream/master"
echo "2. Cherry-pick non-feature commits from wcsc35"
echo "3. Preserve upstream's input features (attacks, attack counts, check)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Ensure we're in a git repository
if [ ! -d .git ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Fetch latest upstream
echo ""
echo "Fetching latest upstream..."
git fetch upstream

# Create new branch
BRANCH_NAME="wcsc35-with-upstream-features"
echo ""
echo "Creating new branch: $BRANCH_NAME"
git checkout upstream/master
git checkout -b "$BRANCH_NAME"

echo ""
echo "=========================================="
echo "Starting cherry-pick process..."
echo "=========================================="

# Function to cherry-pick with error handling
cherry_pick_safe() {
    local commit=$1
    local description=$2
    
    echo ""
    echo ">>> Cherry-picking: $description"
    echo "    Commit: $commit"
    
    if git cherry-pick "$commit"; then
        echo "    ✓ Success"
        return 0
    else
        echo "    ✗ Conflict detected"
        echo ""
        echo "    Options:"
        echo "    1. Resolve conflicts manually, then run: git cherry-pick --continue"
        echo "    2. Skip this commit: git cherry-pick --skip"
        echo "    3. Abort entire process: git cherry-pick --abort"
        echo ""
        read -p "    Choose (1=resolve, 2=skip, 3=abort): " -n 1 -r
        echo
        
        case $REPLY in
            1)
                echo "    Please resolve conflicts, then press Enter to continue..."
                read
                git cherry-pick --continue || {
                    echo "    Still have conflicts. Please resolve and run script again."
                    exit 1
                }
                ;;
            2)
                git cherry-pick --skip
                echo "    Skipped"
                ;;
            3)
                git cherry-pick --abort
                echo "    Aborted"
                exit 1
                ;;
            *)
                echo "    Invalid choice. Aborting."
                git cherry-pick --abort
                exit 1
                ;;
        esac
    fi
}

# ==========================================
# PHASE 1: Infrastructure & Dependencies
# ==========================================
echo ""
echo "=== PHASE 1: Infrastructure & Dependencies ==="

cherry_pick_safe ea4e343 "Update .gitignore"
cherry_pick_safe e333ba6 "Add Pipfile for dependency management"
cherry_pick_safe 6ff6b0c "Add setup script"
cherry_pick_safe d733022 "Update setup script to use pipenv"
cherry_pick_safe 9231021 "Add requirements.txt"
cherry_pick_safe 669954b "Add icecream and asttokens packages"
cherry_pick_safe d9a2215 "Add pandas to dependencies"

# ==========================================
# PHASE 2: Configuration & Logging
# ==========================================
echo ""
echo "=== PHASE 2: Configuration & Logging ==="

cherry_pick_safe 8f09591 "Update config.yaml for WandbLogger"
cherry_pick_safe ef85fac "Add WandbLogger configuration"
cherry_pick_safe dac7394 "Redirect training output to fit.log"
cherry_pick_safe a6c9025 "Add script to run training with pipenv"

# ==========================================
# PHASE 3: Data Pipeline Improvements
# ==========================================
echo ""
echo "=== PHASE 3: Data Pipeline Improvements ==="

cherry_pick_safe 3f7b20f "Support wildcard patterns in config"
cherry_pick_safe 421c4b1 "Enhance DataLoader for wildcard patterns"
cherry_pick_safe 736e929 "Add plaintext_to_hcpe conversion script"
cherry_pick_safe e5929d6 "Add hcpe_to_hdf5 conversion script"
cherry_pick_safe 3326607 "Add hcpe_to_hdf5 with shuffling option"
cherry_pick_safe 094ce14 "Make hcpe_to_hdf5.sh executable"
cherry_pick_safe 02a5395 "Replace individual file loading with HDF5"
cherry_pick_safe c2dc527 "Add plaintext_to_hdf5 script"
cherry_pick_safe 54c0366 "Enhance psv_to_hcpe with validation"
cherry_pick_safe 722a128 "Switch to explicit imports in psv_to_hcp"
cherry_pick_safe 0403ee3 "Add file existence check before conversion"

# ==========================================
# PHASE 4: Testing & Utilities
# ==========================================
echo ""
echo "=== PHASE 4: Testing & Utilities ==="

cherry_pick_safe 903e693 "Add unit tests for flip_sfen"
cherry_pick_safe b442787 "Add SFEN position and move flipping"
cherry_pick_safe a2050cd "Add unit tests for flip_sfen_square and flip_sfen_move"

# ==========================================
# PHASE 5: Training Enhancements
# ==========================================
echo ""
echo "=== PHASE 5: Training Enhancements ==="

cherry_pick_safe 11e3da1 "Add policy and value accuracy metrics"
cherry_pick_safe d372e9a "Add early stopping callback"
cherry_pick_safe 572f9d5 "Implement Stochastic Weight Averaging (SWA)"
cherry_pick_safe 7835ad3 "Add logger and handle None case in SWA"
cherry_pick_safe 4e4d384 "Refactor ema_avg function"

# ==========================================
# PHASE 6: Model Export & ONNX
# ==========================================
echo ""
echo "=== PHASE 6: Model Export & ONNX ==="

cherry_pick_safe 86789b0 "Add script to convert wandb checkpoints to ONNX"
cherry_pick_safe 471a9c2 "Add wandb_to_onnx conversion script"

# ==========================================
# PHASE 7: Engine Customization
# ==========================================
echo ""
echo "=== PHASE 7: Engine Customization ==="

cherry_pick_safe a040934 "Update engine name to 'dlcobra wcsc35'"
cherry_pick_safe a5cc101 "Update author information"
cherry_pick_safe 3b379ec "Expand UCT thread and DNN model support to 10 GPUs"
cherry_pick_safe 54a6a9c "Switch to optimized compilation flags"

# ==========================================
# PHASE 8: Docker & Cloud Setup
# ==========================================
echo ""
echo "=== PHASE 8: Docker & Cloud Setup ==="

cherry_pick_safe 13f072a "Add setup script for dlcobra with TensorRT"

# ==========================================
# PHASE 9: Configuration Adjustments
# ==========================================
echo ""
echo "=== PHASE 9: Configuration Adjustments ==="

cherry_pick_safe 6f8c701 "Add new training data file to config"
cherry_pick_safe 249dea2 "Disable model save on train epoch end"
cherry_pick_safe aa9a21d "Extend patience and initial learning rate schedule"
cherry_pick_safe 48c8ed9 "Adjust learning rate settings and disable SWA"
cherry_pick_safe bc5a88d "Reduce learning rate from 0.003 to 0.001"
cherry_pick_safe e24fdca "Adjust training patience and learning rates"

echo ""
echo "=========================================="
echo "Cherry-pick process completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "- New branch: $BRANCH_NAME"
echo "- Based on: upstream/master"
echo "- Applied: Training improvements, data pipeline, ONNX export, infrastructure"
echo "- Preserved: Upstream's input features (attacks, attack counts, check)"
echo ""
echo "Next steps:"
echo "1. Review the changes: git log --oneline upstream/master..$BRANCH_NAME"
echo "2. Test the build: python setup.py build_ext --inplace"
echo "3. Verify features: Check cppshogi/cppshogi.h for FEATURES1_NUM and FEATURES2_NUM"
echo "4. Update network if needed: The custom network may need adjustment for upstream features"
echo ""
echo "Note: You may need to use upstream's network architecture or modify"
echo "      policy_value_network_cr.py to handle the larger feature dimensions."
echo ""
