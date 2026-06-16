"""Hybrid backbone: fast dense 3x3 ResNet blocks + sparse axial-NeXt blocks.

Rationale (from exp037 benchmark): dense 3x3 convs are math-bound and saturate
tensor cores (fast); depthwise axial convs (1x9 / 9x1) are memory-bound (slow)
but give global row/column receptive field in one layer (the InceptionNeXt bias).

Hybrid idea: use mostly dense ResNet blocks for speed, and insert an axial block
every `axial_period` blocks so global axial information still propagates. Tune
`blocks` so total params land near the ~86M equal-param budget.

Block types:
  - DenseResNetBlock: 2x (conv3x3 + BN + act), residual. Same as resnet backbone.
  - AxialNeXtBlock:   parallel depthwise 1x9 + 9x1 (+ identity) -> LN -> MLP, residual.
"""
import torch
import torch.nn as nn

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM


class Bias(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


class DenseResNetBlock(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = activation

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class AxialNeXtBlock(nn.Module):
    """Channels split in 3: identity / depthwise 1x9 / depthwise 9x1, then MLP."""

    def __init__(self, dim, expansion=4):
        super().__init__()
        assert dim % 3 == 0
        b = dim // 3
        self.b = b
        self.dw1x9 = nn.Conv2d(b, b, (1, 9), padding=(0, 4), groups=b)
        self.dw9x1 = nn.Conv2d(b, b, (9, 1), padding=(4, 0), groups=b)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        residual = x
        b = self.b
        x0, x1, x2 = x[:, :b], x[:, b:2 * b], x[:, 2 * b:3 * b]
        x1 = self.dw1x9(x1)
        x2 = self.dw9x1(x2)
        x = torch.cat([x0, x1, x2], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)
        return residual + x


class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks=30, channels=384, axial_period=4, expansion=4,
                 activation=nn.ReLU(), fcl=256):
        super().__init__()
        self.l1_1_1 = nn.Conv2d(FEATURES1_NUM, channels, 3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(FEATURES1_NUM, channels, 1, bias=False)
        self.l1_2 = nn.Conv2d(FEATURES2_NUM, channels, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.act = activation

        layers = []
        n_axial = 0
        for i in range(blocks):
            # insert an axial block every `axial_period` blocks (1-indexed positions)
            if axial_period > 0 and (i + 1) % axial_period == 0:
                layers.append(AxialNeXtBlock(channels, expansion))
                n_axial += 1
            else:
                layers.append(DenseResNetBlock(channels, activation))
        self.n_axial = n_axial
        self.n_dense = blocks - n_axial
        self.blocks = nn.Sequential(*layers)

        self.policy = nn.Conv2d(channels, MAX_MOVE_LABEL_NUM, 1, bias=False)
        self.policy_bias = Bias(9 * 9 * MAX_MOVE_LABEL_NUM)

        self.value_conv1 = nn.Conv2d(channels, MAX_MOVE_LABEL_NUM, 1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        u1 = self.act(self.norm1(self.l1_1_1(x1) + self.l1_1_2(x1) + self.l1_2(x2)))
        h = self.blocks(u1)

        h_policy = self.policy(h)
        h_policy = self.policy_bias(torch.flatten(h_policy, 1))

        h_value = self.act(self.value_norm1(self.value_conv1(h)))
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)
        return h_policy, h_value
