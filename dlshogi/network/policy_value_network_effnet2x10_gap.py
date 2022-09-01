import copy
import math
import warnings
from typing import Any, Callable, Dict, Optional, List, Sequence, Tuple, Union
import torch
from torch import nn, Tensor
from torchvision.models.efficientnet import MBConv, MBConvConfig 
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation, Conv2dNormActivation
from torchvision.utils import _log_api_usage_once
from dlshogi.common import *

k = 192 # 192
fcl = 256 # fully connected layers

class PolicyValueNetworkV1(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig]  = [
            MBConvConfig(1, 3, 1, k, k, 1, 1, 1),
            MBConvConfig(4, 3, 1, k, k, 2, 1, 1),
            MBConvConfig(4, 3, 1, k, k, 2, 1, 1),
            MBConvConfig(4, 3, 1, k, k, 3, 1, 1),
            MBConvConfig(6, 3, 1, k, k, 3, 1, 1),
            MBConvConfig(6, 3, 1, k, k*2, 1, 1, 1),
        ],
        dropout: float = 0.2,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = MAX_MOVE_LABEL_NUM * 9 * 9, # 27 * 9 * 9 = 2187
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        # last_channel: Optional[int] = k,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class
        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.swish = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(k)
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                # 3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
                k, k, kernel_size=3, stride=1, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 2 * lastconv_input_channels # 4 * lastconv_input_channels

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=lastconv_input_channels,
                out_channels=MAX_MOVE_LABEL_NUM,
                kernel_size=1,
                bias=True,
            ),
            nn.Flatten(1),
        )

        self.regressor = nn.Sequential(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            ),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x1: Tensor, x2: Tensor) -> tuple([Tensor, Tensor]):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = self.swish(self.norm1(u1_1_1 + u1_1_2 + u1_2))

        x = self.features(u1)

        # policy head
        policy = self.classifier(x)

        # value head
        value = self.regressor(x)

        return (policy, value)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple([Tensor, Tensor]):
        return self._forward_impl(x1, x2)

class GlobalAvgPool2d(nn.Module):
    """
    Reduce mean over last two dimensions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.mean(dim=-1, keepdim=True)
        return x.mean(dim=-2, keepdim=True)

class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig]  = [
            MBConvConfig(2, 3, 1, k, k, 1, 1, 10),
        ],
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = MAX_MOVE_LABEL_NUM * 9 * 9,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class
        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if "block" in kwargs:
            warnings.warn(
                "The parameter 'block' is deprecated since 0.13 and will be removed 0.15. "
                "Please pass this information on 'MBConvConfig.block' instead."
            )
            if kwargs["block"] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs["block"]

        self.swish = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(k)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                k, firstconv_output_channels, kernel_size=3, stride=1, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 2 * lastconv_input_channels

        self.features = nn.Sequential(*layers)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=lastconv_input_channels,
                out_channels=MAX_MOVE_LABEL_NUM,
                kernel_size=1,
                bias=True,
            ),
            nn.Flatten(1),
        )

        self.regressor = nn.Sequential(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            ),
            GlobalAvgPool2d(),
            nn.Flatten(1),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x1: Tensor, x2: Tensor) -> tuple([Tensor, Tensor]):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = self.swish(self.norm1(u1_1_1 + u1_1_2 + u1_2))

        x = self.features(u1)

        # policy head
        policy = self.classifier(x)

        # value head
        value = self.regressor(x)

        return (policy, value)

    def forward(self, x1: Tensor, x2: Tensor) -> tuple([Tensor, Tensor]):
        return self._forward_impl(x1, x2)