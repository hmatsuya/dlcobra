"""ResNet-style channel distribution with staged architecture.

depths=[1, 1, 3, 1], dims=[96, 192, 384, 768]

4 stages with increasing channel widths between stages.
Each stage uses standard ResNet blocks (BN -> ReLU -> Conv -> BN -> ReLU -> Conv).
Channel projection between stages via 1x1 conv.
"""
import torch
import torch.nn as nn

from dlshogi.common import *


class ResNetBlock(nn.Module):
    """Standard ResNet block: BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3 + residual."""

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x + residual


class ChannelProjection(nn.Module):
    """Channel projection between stages via BN + 1x1 conv."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_dim)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(self.norm(x))


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
        self.stem_norm = nn.BatchNorm2d(dims[0])
        self.stem_act = nn.ReLU(inplace=True)

        # Build stages
        stages = []
        for depth, dim in zip(depths, dims):
            stages.append(nn.Sequential(*[ResNetBlock(dim) for _ in range(depth)]))
        self.stages = nn.ModuleList(stages)

        # Channel projection layers between stages
        self.projections = nn.ModuleList([
            ChannelProjection(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])

        final_dim = dims[-1]

        # policy head
        self.policy = nn.Conv2d(final_dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = nn.Parameter(torch.zeros(9 * 9 * MAX_MOVE_LABEL_NUM))

        # value head
        self.value_conv = nn.Conv2d(final_dim, MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        # Stem
        x = self.stem_act(self.stem_norm(self.stem_1(x1) + self.stem_2(x2)))

        # Stages with channel projection between them
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.projections):
                x = self.projections[i](x)

        # policy head
        h_policy = self.policy(x)
        h_policy = torch.flatten(h_policy, 1) + self.policy_bias

        # value head
        h_value = self.value_conv(x)
        h_value = self.act(self.value_norm(h_value))
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value
