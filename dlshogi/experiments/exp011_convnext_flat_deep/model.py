"""ConvNeXt flat (single-stage) architecture.

depths=[20], dims=[192]

Single stage with uniform channel width throughout.
Uses ConvNeXt-style blocks (depthwise conv, inverted bottleneck, GELU, LayerNorm).
Comparable to resnet20 (256ch) but with ConvNeXt blocks at 192ch.
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

        # Single stage
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depths[0])])

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
