"""
ResNet-18 implemented from scratch in PyTorch.

Key differences from the canonical ImageNet ResNet-18:
  - The stem conv/maxpool is adapted for small images (MNIST/CIFAR/Pokemon):
      * 32-px images → 3×3 stem, stride=1, no maxpool
      * 64-px images → 7×7 stem, stride=2, maxpool
      * 224-px images (ImageNet) → standard 7×7, stride=2, maxpool
  - Supports arbitrary CHANNELS (e.g. 1 for greyscale).
  - QuantStub / DeQuantStub placeholders are included so the same model
    can be used in PTQ experiments just like your existing CNN.
"""

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.nn.quantized import FloatFunctional
from src.utility.config import CHANNELS, NUM_CLASSES, IMAGE_SIZE


class BasicBlock(nn.Module):
    """Standard ResNet basic block (two 3×3 convs + residual shortcut).

    FloatFunctional is used for the residual addition instead of the plain `+=`
    operator. PyTorch's PTQ observer cannot instrument a bare tensor `+`, so
    without this the quantization scale/zero-point for the add node is never
    calibrated and accuracy after INT8 conversion degrades silently.
    Ref: https://pytorch.org/docs/stable/quantization.html#torch.ao.quantization.FloatFunctional
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut projection when spatial size or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # Quantization-safe addition — replaces the bare `+=` operator
        self.add_func = FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add_func.add(out, identity)   # PTQ-observable add
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 with adaptive stem for small / medium / large input sizes.

    Args:
        num_classes: number of output classes (default: from config)
        channels:    input channels (default: from config)
        image_size:  spatial size of input (default: from config)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        channels: int = CHANNELS,
        image_size: int = IMAGE_SIZE,
    ):
        super().__init__()
        self._in_channels = 64

        self.quant = QuantStub()

        # --- Adaptive stem ---------------------------------------------------
        if image_size <= 32:
            # MNIST / CIFAR: small images — keep spatial resolution
            self.stem = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        elif image_size <= 64:
            # Pokemon 64 px: moderate downsampling
            self.stem = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            # ImageNet 224 px: standard stem
            self.stem = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        # --- Residual stages -------------------------------------------------
        self.layer1 = self._make_layer(64,  num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self.dequant = DeQuantStub()

        self._init_weights()

    # ------------------------------------------------------------------
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock(self._in_channels, out_channels, stride=s))
            self._in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)
        return x