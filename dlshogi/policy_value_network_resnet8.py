import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *
from .resnet import BasicBlock, ResNet

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


fcl = 256 # fully connected layers
class PolicyValueNetwork(ResNet):
    def __init__(self, use_aux=False):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=64, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=64, kernel_size=1, bias=False) # pieces_in_hand
        self.inplanes = 64
        self.l2 = self._make_layer(BasicBlock,  64, 2, activation=nn.LeakyReLU(inplace=True))
        self.l3 = self._make_layer(BasicBlock, 128, 2, activation=nn.LeakyReLU(inplace=True))
        self.l4 = self._make_layer(BasicBlock, 256, 2, activation=nn.LeakyReLU(inplace=True))
        self.l5 = self._make_layer(BasicBlock, 512, 2, activation=nn.LeakyReLU(inplace=True))
        # policy network
        k = 512
        self.l22 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l22_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)
        # value network
        self.l22_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l23_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l24_v = nn.Linear(fcl, 1)
        # sennichite, nyugyoku
        if use_aux:
            self.l24_aux = nn.Linear(fcl, 2)

        self.norm1 = nn.BatchNorm2d(64)
        self.norm22_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)

        self.activation = nn.LeakyReLU(inplace=True)
        self.use_aux = use_aux

    def __call__(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = self.activation(self.norm1(u1_1_1 + u1_1_2 + u1_2))
        # Residual block
        u2 = self.l2(u1)
        u3 = self.l3(u2)
        u4 = self.l4(u3)
        u5 = self.l5(u4)
        # policy network
        h22 = self.l22(u5)
        h22_1 = self.l22_2(h22.view(-1, 9*9*MAX_MOVE_LABEL_NUM))
        # value network
        h22_v = self.activation(self.norm22_v(self.l22_v(u5)))
        h23_v = self.activation(self.l23_v(h22_v.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))
        if self.use_aux:
            return h22_1, self.l24_v(h23_v), self.l24_aux(h23_v)
        else:
            return h22_1, self.l24_v(h23_v)


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self.activation = nn.SiLU() if memory_efficient else Swish()
        self.l2.activation = self.activation
