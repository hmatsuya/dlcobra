"""Isotropic InceptionNeXt with adaptive MLP expansion ratios.

depths=[12], dims=[192], expansions=[2,2,2,3,3,3,3,3,3,4,4,4]

Based on exp015 (depths=[10], dims=[192], uniform expansion=4).
exp013 showed uniform expansion=2 reduces params 44% but may lose capacity.
This experiment uses variable expansion: small in early blocks (feature extraction),
large in late blocks (tactical reasoning), inspired by EfficientNet compound scaling.
Block distribution 3:6:3.

Early blocks (1-3):   expansion=2 → 192→384→192   (feature extraction)
Core blocks (4-9):    expansion=3 → 192→576→192   (pattern recognition)
Late blocks (10-12):  expansion=4 → 192→768→192   (tactical reasoning)
"""
import torch
import torch.nn as nn

from dlshogi.common import *

# Expansion schedule: 3:6:3 block distribution
DEFAULT_EXPANSIONS = (2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4)


class InceptionNeXtBlock(nn.Module):
    """InceptionNeXt block with 4 parallel depthwise branches + MLP.

    Channel split: dim // 4 each for identity, 3x3, 1x9, 9x1.
    Requires dim divisible by 4.
    """

    def __init__(self, dim, expansion=4):
        super().__init__()
        assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
        branch_dim = dim // 4

        self.dw3x3 = nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1, groups=branch_dim)
        self.dw1x9 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(1, 9), padding=(0, 4), groups=branch_dim)
        self.dw9x1 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(9, 1), padding=(4, 0), groups=branch_dim)

        self.norm = nn.LayerNorm(dim)
        hidden_dim = dim * expansion if expansion > 0 else dim
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        residual = x
        b = x.shape[1] // 4

        x0, x1, x2, x3 = x[:, :b], x[:, b:2*b], x[:, 2*b:3*b], x[:, 3*b:]
        x1 = self.dw3x3(x1)
        x2 = self.dw1x9(x2)
        x3 = self.dw9x1(x3)

        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        return x + residual


class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        depths=(12,),
        dims=(192,),
        expansions=DEFAULT_EXPANSIONS,
        fcl=256,
    ):
        super().__init__()
        dim = dims[0]
        depth = depths[0]
        assert len(expansions) == depth, (
            f"expansions length ({len(expansions)}) must match depth ({depth})"
        )

        self.stem_1 = nn.Conv2d(FEATURES1_NUM, dim, kernel_size=3, padding=1, bias=False)
        self.stem_2 = nn.Conv2d(FEATURES2_NUM, dim, kernel_size=1, bias=False)
        self.stem_norm = nn.LayerNorm(dim)

        self.blocks = nn.Sequential(*[
            InceptionNeXtBlock(dim, expansion=exp) for exp in expansions
        ])

        self.policy = nn.Conv2d(dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = nn.Parameter(torch.zeros(9 * 9 * MAX_MOVE_LABEL_NUM))

        self.value_conv = nn.Conv2d(dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm = nn.LayerNorm(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)
        self.act = nn.GELU()

    def forward(self, x1, x2):
        x = self.stem_1(x1) + self.stem_2(x2)
        x = x.permute(0, 2, 3, 1)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.blocks(x)

        h_policy = self.policy(x)
        h_policy = torch.flatten(h_policy, 1) + self.policy_bias

        h_value = self.value_conv(x)
        h_value = h_value.permute(0, 2, 3, 1)
        h_value = self.act(self.value_norm(h_value))
        h_value = h_value.permute(0, 3, 1, 2)
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value
