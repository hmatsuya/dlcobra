"""Hierarchical InceptionNeXt architecture for Shogi.

depths=[3, 3, 9, 3], dims=[64, 128, 192, 256], expansion=2

Multi-stage design with channel projection between stages.
Spatial resolution stays 9x9 throughout (no downsampling).
Each block splits channels into 4 depthwise branches:
  - Identity (pass-through)
  - 3x3 depthwise conv (local tactics)
  - 1x9 depthwise conv (horizontal / rank-based moves)
  - 9x1 depthwise conv (vertical / file-based moves)
Followed by LayerNorm + MLP (expansion=2) + GELU.
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

        self.dw3x3 = nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1, groups=branch_dim)
        self.dw1x9 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(1, 9), padding=(0, 4), groups=branch_dim)
        self.dw9x1 = nn.Conv2d(branch_dim, branch_dim, kernel_size=(9, 1), padding=(4, 0), groups=branch_dim)

        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        residual = x
        b = x.shape[1] // 4

        x0, x1, x2, x3 = x[:, :b], x[:, b:2*b], x[:, 2*b:3*b], x[:, 3*b:]
        x1 = self.dw3x3(x1)
        x2 = self.dw1x9(x2)
        x3 = self.dw9x1(x3)

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
        depths=(3, 3, 9, 3),
        dims=(64, 128, 192, 256),
        fcl=256,
    ):
        super().__init__()
        assert len(depths) == len(dims), "depths and dims must have the same length"
        assert all(d % 4 == 0 for d in dims), "all dims must be divisible by 4"

        # Stem: project board features into first stage channels
        self.stem_1 = nn.Conv2d(FEATURES1_NUM, dims[0], kernel_size=3, padding=1, bias=False)
        self.stem_2 = nn.Conv2d(FEATURES2_NUM, dims[0], kernel_size=1, bias=False)
        self.stem_norm = nn.LayerNorm(dims[0])

        # Build stages with channel-projection transitions
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = nn.Sequential(*[InceptionNeXtBlock(dim, expansion=2) for _ in range(depth)])
            self.stages.append(stage)
            # Add transition (channel projection) after all but the last stage
            if i < len(dims) - 1:
                self.transitions.append(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dims[i + 1]),
                ))

        final_dim = dims[-1]

        # Policy head
        self.policy = nn.Conv2d(final_dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = nn.Parameter(torch.zeros(9 * 9 * MAX_MOVE_LABEL_NUM))

        # Value head
        self.value_conv = nn.Conv2d(final_dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
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

        # Stages with transitions
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.transitions):
                x = x.permute(0, 2, 3, 1)          # (B, H, W, C)
                x = self.transitions[i](x)
                x = x.permute(0, 3, 1, 2)          # (B, C, H, W)

        # Policy head
        h_policy = self.policy(x)
        h_policy = torch.flatten(h_policy, 1) + self.policy_bias

        # Value head
        h_value = self.value_conv(x)
        h_value = h_value.permute(0, 2, 3, 1)
        h_value = self.act(self.value_norm(h_value))
        h_value = h_value.permute(0, 3, 1, 2)
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value
