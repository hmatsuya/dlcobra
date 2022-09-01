import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

k = 192 # 192
fcl = 256 # fully connected layers

swish = nn.SiLU()

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = swish(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return swish(out + x)

class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks=15, channels=192, fcl=256):
        super(PolicyValueNetwork, self).__init__()

        self.norm1 = nn.BatchNorm2d(k)
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand

        # resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=True)
        # self.policy_bias = Bias(MAX_MOVE_LABEL_NUM)

        # value head
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = swish(self.norm1(u1_1_1 + u1_1_2 + u1_2))
        
        # resnet blocks
        x = self.blocks(u1)

        # policy head
        policy = self.policy_conv(x)
        policy = torch.flatten(policy, 1)
        # policy = self.policy_bias(torch.flatten(policy, 1))

        # value head
        value = swish(self.value_norm1(self.value_conv1(x)))
        value = swish(self.value_fc1(torch.flatten(value, 1)))
        value = self.value_fc2(value)

        return policy, value