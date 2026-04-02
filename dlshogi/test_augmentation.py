"""Test script for horizontal flip augmentation.

Run: python dlshogi/test_augmentation.py
"""

import sys
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, ".")

from dlshogi.augmentation import (
    _FLIP_SQUARE, _FLIP_MOVE_LABEL, _FLIP_DIRECTION,
    flip_features1, flip_features2, flip_move_label, flip_probability,
    apply_horizontal_flip,
)
from dlshogi.common import (
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT,
    DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE,
    LEFT_PROMOTE, RIGHT_PROMOTE,
    DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE,
    FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM,
)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")

# ============================================================
# 1. Square flip table
# ============================================================
print("=== Square flip table ===")

# sq = file*9 + rank. Flip: file -> 8-file, rank unchanged.
# SQ11 (file=0, rank=0) -> SQ91 (file=8, rank=0) = 72
check("SQ11 -> SQ91", _FLIP_SQUARE[0] == 72)
# SQ91 (file=8, rank=0) -> SQ11 (file=0, rank=0) = 0
check("SQ91 -> SQ11", _FLIP_SQUARE[72] == 0)
# SQ55 (file=4, rank=4) = 40 -> stays at center
check("SQ55 center invariant", _FLIP_SQUARE[40] == 40)
# SQ19 (file=0, rank=8) = 8 -> SQ99 (file=8, rank=8) = 80
check("SQ19 -> SQ99", _FLIP_SQUARE[8] == 80)
# Double flip = identity
check("double flip = identity", np.all(_FLIP_SQUARE[_FLIP_SQUARE] == np.arange(81)))

# ============================================================
# 2. Direction flip mapping
# ============================================================
print("\n=== Direction flip mapping ===")

check("UP invariant", _FLIP_DIRECTION[UP] == UP)
check("DOWN invariant", _FLIP_DIRECTION[DOWN] == DOWN)
check("LEFT <-> RIGHT", _FLIP_DIRECTION[LEFT] == RIGHT and _FLIP_DIRECTION[RIGHT] == LEFT)
check("UP_LEFT <-> UP_RIGHT", _FLIP_DIRECTION[UP_LEFT] == UP_RIGHT and _FLIP_DIRECTION[UP_RIGHT] == UP_LEFT)
check("DOWN_LEFT <-> DOWN_RIGHT", _FLIP_DIRECTION[DOWN_LEFT] == DOWN_RIGHT and _FLIP_DIRECTION[DOWN_RIGHT] == DOWN_LEFT)
check("UP2_LEFT <-> UP2_RIGHT", _FLIP_DIRECTION[UP2_LEFT] == UP2_RIGHT and _FLIP_DIRECTION[UP2_RIGHT] == UP2_LEFT)
# Promote variants
check("UP_PROMOTE invariant", _FLIP_DIRECTION[UP_PROMOTE] == UP_PROMOTE)
check("DOWN_PROMOTE invariant", _FLIP_DIRECTION[DOWN_PROMOTE] == DOWN_PROMOTE)
check("LEFT_PROMOTE <-> RIGHT_PROMOTE", _FLIP_DIRECTION[LEFT_PROMOTE] == RIGHT_PROMOTE)
check("UP2_LEFT_PROMOTE <-> UP2_RIGHT_PROMOTE", _FLIP_DIRECTION[UP2_LEFT_PROMOTE] == UP2_RIGHT_PROMOTE)
# Hand piece directions (20-26) should be unchanged
for d in range(20, MAX_MOVE_LABEL_NUM):
    check(f"hand piece dir {d} invariant", _FLIP_DIRECTION[d] == d)

# ============================================================
# 3. Move label flip table
# ============================================================
print("\n=== Move label flip table ===")

# Double flip = identity
check("move label double flip = identity",
      np.all(_FLIP_MOVE_LABEL[_FLIP_MOVE_LABEL] == np.arange(81 * MAX_MOVE_LABEL_NUM)))

# UP move to SQ11 (label = 81*UP + 0) -> UP move to SQ91 (label = 81*UP + 72)
check("UP to SQ11 -> UP to SQ91", _FLIP_MOVE_LABEL[81 * UP + 0] == 81 * UP + 72)

# LEFT move to SQ55 (label = 81*LEFT + 40) -> RIGHT move to SQ55 (label = 81*RIGHT + 40)
check("LEFT to SQ55 -> RIGHT to SQ55", _FLIP_MOVE_LABEL[81 * LEFT + 40] == 81 * RIGHT + 40)

# Hand pawn drop to SQ11 -> hand pawn drop to SQ91
hand_pawn_dir = 20  # MOVE_DIRECTION_NUM + 0 (pawn)
check("pawn drop SQ11 -> SQ91",
      _FLIP_MOVE_LABEL[81 * hand_pawn_dir + 0] == 81 * hand_pawn_dir + 72)

# ============================================================
# 4. flip_features1
# ============================================================
print("\n=== flip_features1 ===")

batch = 2
f1 = torch.zeros(batch, FEATURES1_NUM, 9, 9)
# Place a piece at file=0, rank=0 in channel 0
f1[0, 0, 0, 0] = 1.0
flipped = flip_features1(f1)
check("piece at (0,0) -> (8,0)", flipped[0, 0, 8, 0].item() == 1.0)
check("original position cleared", flipped[0, 0, 0, 0].item() == 0.0)
# Double flip = identity
check("features1 double flip = identity", torch.allclose(flip_features1(flipped), f1))

# ============================================================
# 5. flip_features2
# ============================================================
print("\n=== flip_features2 ===")

f2 = torch.zeros(batch, FEATURES2_NUM, 9, 9)
f2[0, 0, 2, 3] = 1.0
flipped2 = flip_features2(f2)
check("f2 piece at (2,3) -> (6,3)", flipped2[0, 0, 6, 3].item() == 1.0)
check("features2 double flip = identity", torch.allclose(flip_features2(flipped2), f2))

# ============================================================
# 6. flip_move_label
# ============================================================
print("\n=== flip_move_label ===")

moves = torch.tensor([81 * UP + 0, 81 * LEFT + 40], dtype=torch.int64)
flipped_moves = flip_move_label(moves)
check("UP to SQ11 flipped", flipped_moves[0].item() == 81 * UP + 72)
check("LEFT to SQ55 flipped", flipped_moves[1].item() == 81 * RIGHT + 40)

# ============================================================
# 7. flip_probability
# ============================================================
print("\n=== flip_probability ===")

total_labels = 81 * MAX_MOVE_LABEL_NUM
prob = torch.zeros(1, total_labels)
prob[0, 81 * UP_LEFT + 10] = 0.7  # UP_LEFT to sq10 (file=1, rank=1)
prob[0, 81 * RIGHT + 40] = 0.3   # RIGHT to sq40 (center)
flipped_prob = flip_probability(prob)
# UP_LEFT to sq10 -> UP_RIGHT to flip(10). sq10: file=1,rank=1 -> file=7,rank=1 = 64
check("prob UP_LEFT sq10 -> UP_RIGHT sq64",
      abs(flipped_prob[0, 81 * UP_RIGHT + 64].item() - 0.7) < 1e-6)
# RIGHT to sq40 -> LEFT to sq40 (center stays)
check("prob RIGHT sq40 -> LEFT sq40",
      abs(flipped_prob[0, 81 * LEFT + 40].item() - 0.3) < 1e-6)
# Sum preserved
check("probability sum preserved",
      abs(flipped_prob.sum().item() - prob.sum().item()) < 1e-6)

# ============================================================
# 8. apply_horizontal_flip (ratio=1.0 -> all flipped)
# ============================================================
print("\n=== apply_horizontal_flip ===")

f1 = torch.randn(4, FEATURES1_NUM, 9, 9)
f2 = torch.randn(4, FEATURES2_NUM, 9, 9)
mv = torch.randint(0, total_labels, (4,), dtype=torch.int64)
res = torch.randn(4, 1)
val = torch.randn(4, 1)

f1f, f2f, mvf, resf, valf = apply_horizontal_flip(f1, f2, mv, res, val, flip_ratio=1.0)
check("all flipped: features1", torch.allclose(f1f, flip_features1(f1)))
check("all flipped: features2", torch.allclose(f2f, flip_features2(f2)))
check("all flipped: move", torch.equal(mvf, flip_move_label(mv)))
check("result unchanged", torch.equal(resf, res))
check("value unchanged", torch.equal(valf, val))

# ratio=0.0 -> nothing flipped
f1n, f2n, mvn, resn, valn = apply_horizontal_flip(f1, f2, mv, res, val, flip_ratio=0.0)
check("none flipped: features1", torch.equal(f1n, f1))
check("none flipped: move", torch.equal(mvn, mv))

# probability variant
prob = torch.randn(4, total_labels).abs()
f1f, f2f, pf, resf, valf = apply_horizontal_flip(f1, f2, prob, res, val, flip_ratio=1.0)
check("all flipped: probability", torch.allclose(pf, flip_probability(prob)))

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("All tests passed.")
