"""ResNet with fewer activations (ConvNeXt-style).

Only one activation per residual block, after the residual addition:
  Conv -> BN -> Conv -> BN -> (+residual) -> Act

Instead of the standard two-activation pattern:
  Conv -> BN -> Act -> Conv -> BN -> (+residual) -> Act
"""
import torch
import torch.nn as nn

from dlshogi.common import *


class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResNetBlock(nn.Module):
    def __init__(self, channels, activation):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = activation

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # No activation here (fewer activations)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.act(out + x)


class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks=10, channels=192, activation=nn.SiLU(), fcl=256):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.act = activation

        self.blocks = nn.Sequential(*[ResNetBlock(channels, activation) for _ in range(blocks)])

        # policy network
        self.policy = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(9 * 9 * MAX_MOVE_LABEL_NUM)

        # value network
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = self.act(self.norm1(u1_1_1 + u1_1_2 + u1_2))

        h = self.blocks(u1)

        # policy network
        h_policy = self.policy(h)
        h_policy = self.policy_bias(torch.flatten(h_policy, 1))

        # value network
        h_value = self.act(self.value_norm1(self.value_conv1(h)))
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value

    def set_swish(self, memory_efficient=True):
        activation = nn.SiLU() if memory_efficient else Swish()
        for n, m in self.named_modules():
            if isinstance(m, PolicyValueNetwork) or isinstance(m, ResNetBlock):
                m.act = activation
