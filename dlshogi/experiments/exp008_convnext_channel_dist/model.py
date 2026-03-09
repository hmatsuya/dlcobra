"""ConvNeXt-style channel distribution with staged architecture.

depths=[1, 1, 3, 1], dims=[96, 192, 384, 768]

4 stages with increasing channel widths and downsampling between stages.
Each stage uses ConvNeXt-style blocks (depthwise conv, inverted bottleneck, GELU).
Downsampling via LayerNorm + 1x1 conv to match spatial dims on 9x9 board.
"""
import torch
import torch.nn as nn

from dlshogi.common import *


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise conv -> LN -> pointwise expand -> GELU -> pointwise contract."""

    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        # (B, C, H, W) -> (B, H, W, C) for LayerNorm/Linear
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return x + residual


class DownsampleLayer(nn.Module):
    """Channel projection between stages (no spatial downsampling on 9x9 board)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.proj(x)


class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        depths=(1, 1, 3, 1),
        dims=(96, 192, 384, 768),
        fcl=256,
    ):
        super().__init__()

        # Stem: project board features into first stage channels
        self.stem_1 = nn.Conv2d(FEATURES1_NUM, dims[0], kernel_size=3, padding=1, bias=False)
        self.stem_2 = nn.Conv2d(FEATURES2_NUM, dims[0], kernel_size=1, bias=False)
        self.stem_norm = nn.LayerNorm(dims[0])

        # Build stages
        stages = []
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stages.append(nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)]))
        self.stages = nn.ModuleList(stages)

        # Downsampling layers between stages
        self.downsamples = nn.ModuleList([
            DownsampleLayer(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])

        final_dim = dims[-1]

        # policy head
        self.policy = nn.Conv2d(final_dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = nn.Parameter(torch.zeros(9 * 9 * MAX_MOVE_LABEL_NUM))

        # value head
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

        # Stages with downsampling between them
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

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
