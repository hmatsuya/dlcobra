# https://medium.com/@bentou.pub/alphazero-from-scratch-in-pytorch-for-the-game-of-chain-reaction-part-3-c3fbf0d6f986
# https://gist.github.com/BentouAI/6b61e0f01a913e30101bd96a802b1716

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import logging
import sys
from icecream import ic

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM


# Configure logging to output to stdout
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)
if sys.platform == 'win32':
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


MOVE_CHANNELS = 27
POSITION_CHANNELS = 42

class SmallBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallBlock, self).__init__()
        self.model = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channels)
        )
    def forward(self, X):
        return self.model(X)

class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(ResnetBlock, self).__init__()
        self.model = torch.nn.Sequential(
            SmallBlock(in_channels, mid_channels),
            torch.nn.ReLU(),
            SmallBlock(mid_channels, in_channels)
        )
    def forward(self, X):
        Y = self.model(X)
        Y = Y + X
        Y = torch.nn.ReLU()(Y)
        return Y

class DropoutBlock(torch.nn.Module):
    def __init__(self, in_units, out_units, dropout):
        super(DropoutBlock, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_units, out_units),
            torch.nn.BatchNorm1d(out_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout)
        )
    def forward(self, X):
        return self.model(X)

class InitialBlock(torch.nn.Module):
    def __init__(self, out_channels):
        super(InitialBlock, self).__init__()
        self.conv1a = torch.nn.Conv2d(FEATURES1_NUM, out_channels, 3, stride=1, padding=1, bias=False)
        self.conv1b = torch.nn.Conv2d(FEATURES1_NUM, out_channels, 1, stride=1, padding=0, bias=False)
        self.conb2  = torch.nn.Conv2d(FEATURES2_NUM, out_channels, 1, stride=1, padding=0, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x1, x2):
        u1_1_1 = self.conv1a(x1)
        u1_1_2 = self.conv1b(x1)
        u1_2 = self.conb2(x2)
        u1 = self.relu(self.norm(u1_1_1 + u1_1_2 + u1_2))
        return u1


class PolicyValueNetwork(torch.nn. Module):
    def __init__(
        self,
        H=[32, 256],
        num_channels=192,
        args=None,
        dropout=0.05,
        num_middle_blocks=50,
    ):
        super(PolicyValueNetwork, self).__init__()

        if args is None:
            args = {
                "M": 9,
                "N": 9,
            }
        self.args = args

        self.initial_block = InitialBlock(num_channels)

        self.middle_blocks = torch.nn.Sequential(
            *[ResnetBlock(num_channels,num_channels) for _ in range(num_middle_blocks)]
        )

        self.model = torch.nn.Sequential(
            self.middle_blocks,
        )

        self.my_policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, MAX_MOVE_LABEL_NUM, kernel_size=1, stride=1, padding=0),
            torch.nn.Flatten(start_dim=1),
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, H[0], kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(H[0]),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1),
            DropoutBlock(H[0] * args['M'] * args['N'], H[1], dropout=dropout),
            torch.nn.Linear(H[1], 1),
        )


    def forward(self, x1, x2):
        X = self.initial_block(x1, x2)
        Y = self.model(X)
        my_p = self.my_policy_head(Y)
        v = self.value_head(Y)

        return my_p, v
