import warnings

import torch
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn as nn
from torch.nn import functional as F

from modules.tools.types import *

warnings.filterwarnings("ignore")


class SReLU(nn.ReLU):
    """
    ReLU shifted by 0.5 as proposed in fast.ai
    https://forums.fast.ai/t/shifted-relu-0-5/41467
    (likely no visible effect)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) - 0.5


class Shortcut(nn.Module):
    def __init__(self, downsample: bool = False) -> None:
        """
        ResNet shortcut layer
        See the code to adjust pooling properties (concatenate avg pooling by default)
        :param downsample: whether to downsample with concatenation
        """
        super().__init__()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample:
            # mp = F.max_pool2d(x, kernel_size=2, stride=2)
            ap = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = torch.cat([ap, ap], dim=1)
        return x


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        use_srelu: bool = False,
    ) -> None:
        """
        Residual unit from ResNet v2
        https://arxiv.org/abs/1603.05027
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param downsample: whether to downsample in this unit
        :param use_srelu: whether to use shifted ReLU
        """
        super().__init__()
        assert (
            in_channels == out_channels
            if not downsample
            else in_channels == out_channels // 2
        ), "With downsampling out_channels = in_channels * 2"

        self.use_srelu = use_srelu
        activation = SReLU if use_srelu else nn.ReLU
        self.shortcut = Shortcut(downsample)
        self.stacks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation(),
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=2 if downsample else 1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stacks(x) + self.shortcut(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_units: int,
        downsample: bool = False,
    ) -> None:
        """
        Block of `num_units` residual units
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param num_units: number of residual units in the block
        :param downsample: whether to downsample in this unit
        """
        super().__init__()
        self.units = nn.Sequential(
            *[
                ResidualUnit(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    downsample=(downsample and i == 0),
                )
                for i in range(num_units)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.units(x)


class ResidualNetworkHeadless(nn.Module):
    def __init__(
        self,
        num_units: int = 2,
        in_channels: int = 1,
        base_channels: int = 16,
        dropout: float = 0.0,
    ):
        """
        ResNet v2
        https://arxiv.org/abs/1603.05027
        (creates ResNet18 by default)
        :param num_units: number of residual units in the block
        :param in_channels: number of input channels
        """
        super().__init__()
        self.dropout = dropout
        self.base_channels = base_channels
        self.out_channels = base_channels * 8
        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResidualBlock(
                in_channels=base_channels,
                out_channels=base_channels,
                num_units=num_units,
                downsample=False,
            ),
            ResidualBlock(
                in_channels=base_channels,
                out_channels=base_channels * 2,
                num_units=num_units,
                downsample=True,
            ),
            ResidualBlock(
                in_channels=base_channels * 2,
                out_channels=base_channels * 4,
                num_units=num_units,
                downsample=True,
            ),
            ResidualBlock(
                in_channels=base_channels * 4,
                out_channels=base_channels * 8,
                num_units=num_units,
                downsample=True,
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


class ResidualNetwork(ResidualNetworkHeadless):
    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_classes = n_classes
        self.fc = nn.Sequential(
            nn.Linear(self.base_channels * 8, self.n_classes, bias=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return self.fc(x)


class EncoderBase(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.out_channels = encoder.out_channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        # return last level features only
        return self.encoder(x)[-1]


class EncoderHeadless(nn.Module):
    def __init__(self, encoder_name: str, in_channels: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = nn.Sequential(
            EncoderBase(get_encoder(encoder_name, in_channels=in_channels)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.out_channels = self.backbone[0].out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def load_state_dict_from_segmentation(
        self,
        state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
        strict: bool = True,
    ):
        encoder_weights = {
            k.replace("network.", ""): v
            for k, v in state_dict.items()
            if "encoder" in k
        }
        self.backbone[0].load_state_dict(encoder_weights, strict=strict)


class Encoder(EncoderHeadless):
    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.fc = nn.Linear(self.backbone[0].out_channels, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return self.fc(x)


def resnet10(n_classes: int, base_channels: int = 16, **kwargs: Any) -> nn.Module:
    """
    ResNet-10 v2
    :param n_classes: number of classes for the last dense layer
    :param base_channels: number of channels in the first layer
    :param kwargs: keyword arguments for ResidualNetwork
    :return: ResidualNetwork
    """
    return ResidualNetwork(
        num_units=1, base_channels=base_channels, n_classes=n_classes, **kwargs
    )


def resnet18(n_classes: int, base_channels: int = 16, **kwargs: Any) -> nn.Module:
    """
    ResNet-18 v2
    :param n_classes: number of classes for the last dense layer
    :param base_channels: number of channels in the first layer
    :param kwargs: keyword arguments for ResidualNetwork
    :return: ResidualNetwork
    """
    return ResidualNetwork(
        num_units=2, base_channels=base_channels, n_classes=n_classes, **kwargs
    )
