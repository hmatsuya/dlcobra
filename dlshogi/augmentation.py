"""Horizontal flip data augmentation for Shogi training.

Shogi boards are left-right symmetric, so flipping the board horizontally
(mirroring files) is a valid augmentation. This module provides functions
to flip features and move labels in-place on already-decoded tensors.

Board coordinate system:
  Square = File * 9 + Rank  (File: 0-8, Rank: 0-8)
  Horizontal flip: file -> 8 - file, rank unchanged
  flipped_sq = (8 - file) * 9 + rank

features1 shape: (batch, FEATURES1_NUM, 9, 9) — last two dims are (file, rank)
  -> flip along the file axis (dim=-2)

features2 shape: (batch, FEATURES2_NUM, 9, 9)
  -> flip along the file axis (dim=-2)

Move label: 81 * direction + to_sq
  -> flip to_sq's file, swap left/right directions
"""

import numpy as np
import torch

from dlshogi.common import (
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT,
    DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE,
    LEFT_PROMOTE, RIGHT_PROMOTE,
    DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE,
    MAX_MOVE_LABEL_NUM,
)

# Square horizontal flip table: for sq = file*9 + rank, flipped = (8-file)*9 + rank
_FLIP_SQUARE = np.array(
    [(8 - (sq // 9)) * 9 + (sq % 9) for sq in range(81)],
    dtype=np.int64,
)

# Direction horizontal flip mapping (left <-> right)
_FLIP_DIRECTION = list(range(MAX_MOVE_LABEL_NUM))
_FLIP_DIRECTION[UP_LEFT] = UP_RIGHT
_FLIP_DIRECTION[UP_RIGHT] = UP_LEFT
_FLIP_DIRECTION[LEFT] = RIGHT
_FLIP_DIRECTION[RIGHT] = LEFT
_FLIP_DIRECTION[DOWN_LEFT] = DOWN_RIGHT
_FLIP_DIRECTION[DOWN_RIGHT] = DOWN_LEFT
_FLIP_DIRECTION[UP2_LEFT] = UP2_RIGHT
_FLIP_DIRECTION[UP2_RIGHT] = UP2_LEFT
_FLIP_DIRECTION[UP_LEFT_PROMOTE] = UP_RIGHT_PROMOTE
_FLIP_DIRECTION[UP_RIGHT_PROMOTE] = UP_LEFT_PROMOTE
_FLIP_DIRECTION[LEFT_PROMOTE] = RIGHT_PROMOTE
_FLIP_DIRECTION[RIGHT_PROMOTE] = LEFT_PROMOTE
_FLIP_DIRECTION[DOWN_LEFT_PROMOTE] = DOWN_RIGHT_PROMOTE
_FLIP_DIRECTION[DOWN_RIGHT_PROMOTE] = DOWN_LEFT_PROMOTE
_FLIP_DIRECTION[UP2_LEFT_PROMOTE] = UP2_RIGHT_PROMOTE
_FLIP_DIRECTION[UP2_RIGHT_PROMOTE] = UP2_LEFT_PROMOTE

# Build full move label flip table: label = 81 * direction + to_sq
# flipped_label = 81 * flip_direction + flip_to_sq
_FLIP_MOVE_LABEL = np.empty(81 * MAX_MOVE_LABEL_NUM, dtype=np.int64)
for _d in range(MAX_MOVE_LABEL_NUM):
    for _sq in range(81):
        _FLIP_MOVE_LABEL[81 * _d + _sq] = 81 * _FLIP_DIRECTION[_d] + _FLIP_SQUARE[_sq]


def flip_features1(features1: torch.Tensor) -> torch.Tensor:
    """Flip features1 horizontally. Shape: (batch, C, 9, 9) where dim -2 is file."""
    return features1.flip(-2)


def flip_features2(features2: torch.Tensor) -> torch.Tensor:
    """Flip features2 horizontally. Shape: (batch, C, 9, 9) where dim -2 is file."""
    return features2.flip(-2)


def flip_move_label(move: torch.Tensor) -> torch.Tensor:
    """Flip move label indices. Shape: (batch,) of int64 label indices."""
    table = torch.from_numpy(_FLIP_MOVE_LABEL).to(move.device)
    return table[move]


def flip_probability(probability: torch.Tensor) -> torch.Tensor:
    """Flip move probability vector. Shape: (batch, 81*MAX_MOVE_LABEL_NUM).

    Reorders the probability entries so that each move label maps to its
    horizontally flipped counterpart.
    """
    table = torch.from_numpy(_FLIP_MOVE_LABEL).to(probability.device)
    return probability[:, table]


def apply_horizontal_flip(features1, features2, move_or_prob, result, value,
                          flip_ratio=0.5):
    """Apply horizontal flip augmentation to a random subset of the batch.

    Args:
        features1: (batch, C1, 9, 9) board features
        features2: (batch, C2, 9, 9) hand/check features
        move_or_prob: either (batch,) int64 move labels or (batch, 2187) float probabilities
        result: (batch, 1) game result — unchanged by flip
        value: (batch, 1) position value — unchanged by flip
        flip_ratio: probability of flipping each sample (default 0.5)

    Returns:
        Tuple of (features1, features2, move_or_prob, result, value) with
        augmentation applied in-place where possible.
    """
    batch_size = features1.shape[0]
    mask = torch.rand(batch_size, device=features1.device) < flip_ratio
    if not mask.any():
        return features1, features2, move_or_prob, result, value

    idx = mask.nonzero(as_tuple=True)[0]

    features1 = features1.clone()
    features2 = features2.clone()
    move_or_prob = move_or_prob.clone()

    features1[idx] = flip_features1(features1[idx])
    features2[idx] = flip_features2(features2[idx])

    if move_or_prob.dim() == 1:
        # Single move label per sample
        move_or_prob[idx] = flip_move_label(move_or_prob[idx])
    else:
        # Probability distribution over all moves
        move_or_prob[idx] = flip_probability(move_or_prob[idx])

    return features1, features2, move_or_prob, result, value
