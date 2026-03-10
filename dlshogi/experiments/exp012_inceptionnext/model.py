"""Isotropic InceptionNeXt architecture for Shogi.

depths=[20], dims=[192]

Single-stage isotropic design (no downsampling, 9x9 throughout).
Each block splits channels into 4 depthwise branches:
  - Identity (pass-through)
  - 3x3 depthwise conv (local tactics)
  - 1x9 depthwise conv (horizontal / rank-based moves)
  - 9x1 depthwise conv (vertical / file-based moves)
Followed by LayerNorm + inverted bottleneck MLP (expansion=4) + GELU.
"""
import torch
import torch.nn as nn

from dlshogi.common import *


class InceptionNeXtBlock(nn.Module):
    """InceptionNeXt block with 4 parallel depthwise branches + MLP.

    Channel split: dim // 4 each for identity, 3x3, 1x9, 9x1.
    Requires dim divisible by 4.
    """

    def __init__(self, dim, expansion=4):
        super().__init__()
        assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
        branch_dim = dim // 4

        # Depthwise branches (identity needs no layer)
        self.dw3x3 = nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1, groups=branch_dim)
        self.dw1x9 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(1, 9), padding=(0, 4), groups=branch_dim)
        self.dw9x1 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(9, 1), padding=(4, 0), groups=branch_dim)

        # MLP (channels-last)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        residual = x
        b = x.shape[1] // 4  # branch_dim

        # Split into 4 equal chunks along channel axis
        x0, x1, x2, x3 = x[:, :b], x[:, b:2*b], x[:, 2*b:3*b], x[:, 3*b:]

        # Apply depthwise branches
        x1 = self.dw3x3(x1)
        x2 = self.dw1x9(x2)
        x3 = self.dw9x1(x3)

        # Concatenate and apply MLP
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)   # (B, C, H, W)

        return x + residual


class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        depths=(20,),
        dims=(192,),
        fcl=256,
    ):
        super().__init__()
        dim = dims[0]

        # Stem: project board features into stage channels
        self.stem_1 = nn.Conv2d(FEATURES1_NUM, dim, kernel_size=3, padding=1, bias=False)
        self.stem_2 = nn.Conv2d(FEATURES2_NUM, dim, kernel_size=1, bias=False)
        self.stem_norm = nn.LayerNorm(dim)

        # Single isotropic stage
        self.blocks = nn.Sequential(*[InceptionNeXtBlock(dim) for _ in range(depths[0])])

        # policy head
        self.policy = nn.Conv2d(dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = nn.Parameter(torch.zeros(9 * 9 * MAX_MOVE_LABEL_NUM))

        # value head
        self.value_conv = nn.Conv2d(dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm = nn.LayerNorm(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)
        self.act = nn.GELU()

    def forward(self, x1, x2):
        # Stem
        x = self.stem_1(x1) + self.stem_2(x2)
        x = x.permute(0, 2, 3, 1)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.blocks(x)

        # policy head
        h_policy = self.policy(x)
        h_policy = torch.flatten(h_policy, 1) + self.policy_bias

        # value head
        h_value = self.value_conv(x)
        h_value = h_value.permute(0, 2, 3, 1)
        h_value = self.act(self.value_norm(h_value))
        h_value = h_value.permute(0, 3, 1, 2)
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value
