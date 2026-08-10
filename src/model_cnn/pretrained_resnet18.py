"""
Pretrained ResNet-18 and ResNet-50 wrappers.

Uses torchvision.models with ImageNet-pretrained weights. The final
fully-connected layer is replaced to match NUM_CLASSES from config so
the models can be fine-tuned on any of the six datasets.

Fine-tuning strategy (configurable via FINETUNE_MODE):
  "full"     — unfreeze all layers from the start (good for small datasets)
  "head"     — freeze backbone, train only the new FC head
  "gradual"  — freeze backbone first; call unfreeze_backbone() after warm-up
"""

import torch
import torch.nn as nn
from torchvision import models
from src.utility.config import NUM_CLASSES, CHANNELS

# "full" | "head" | "gradual"
FINETUNE_MODE = "full"


def _adapt_first_conv(model: nn.Module, channels: int, image_size: int) -> None:
    """Replace the first conv and pool layers based on input resolution."""
    old = model.conv1
    
    if image_size == 32 or image_size == 28:
        # CIFAR/MNIST Architecture: 3x3 conv, stride 1, padding 1, NO maxpool
        model.conv1 = nn.Conv2d(
            channels, old.out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # Disable the maxpool layer by replacing it with an Identity pass-through
        model.maxpool = nn.Identity()
        
    else:
        # Standard Architecture: 7x7 conv, stride 2, keep maxpool
        model.conv1 = nn.Conv2d(
            channels, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=False,
        )
        # Average pretrained RGB weights for greyscale (only if weights exist)
        if channels != 3 and getattr(old, 'weight', None) is not None:
            with torch.no_grad():
                model.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))


def get_resnet18(
    num_classes: int = NUM_CLASSES,
    channels: int    = CHANNELS,
    image_size: int = 32,
    finetune_mode: str = FINETUNE_MODE,
    pretrained: bool = False,
) -> nn.Module:
    """Return a torchvision ResNet-18 with ImageNet weights, adapted for the task."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)

    _adapt_first_conv(model, channels, image_size)

    # Replace head
    in_features  = model.fc.in_features
    model.fc     = nn.Linear(in_features, num_classes)

    if finetune_mode == "head":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    return model


def get_resnet50(
    num_classes: int = NUM_CLASSES,
    channels: int    = CHANNELS,
    image_size: int = 32,
    finetune_mode: str = FINETUNE_MODE,
    pretrained: bool = False,
) -> nn.Module:
    """Return a torchvision ResNet-50 with ImageNet weights, adapted for the task."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model   = models.resnet50(weights=weights)

    _adapt_first_conv(model, channels, image_size)

    in_features = model.fc.in_features
    model.fc    = nn.Linear(in_features, num_classes)

    if finetune_mode == "head":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters (call after warm-up when using 'gradual' mode)."""
    for param in model.parameters():
        param.requires_grad = True