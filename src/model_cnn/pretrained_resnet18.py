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
from src.utility.config import NUM_CLASSES, CHANNELS, IMAGE_SIZE

# "full" | "head" | "gradual"
FINETUNE_MODE = "full"


def _adapt_first_conv(model: nn.Module, channels: int) -> None:
    """Replace the first conv to accept CHANNELS != 3 (e.g. greyscale)."""
    if channels == 3:
        return
    old = model.conv1
    model.conv1 = nn.Conv2d(
        channels, old.out_channels,
        kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=False,
    )
    # average pretrained RGB weights across channel dim for warm initialisation
    with torch.no_grad():
        model.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))


def _adapt_stem_for_cifar(model: nn.Module, channels: int) -> None:
    """
    CIFAR-sized (<=32px) images are far smaller than ImageNet's native
    224px, so torchvision's default 7x7/stride-2 stem + maxpool would
    throw away most of the image's resolution before the first residual
    block even runs. Swap in a 3x3/stride-1 stem and drop the maxpool --
    the same adaptation src/model_cnn/resnet18.py's own from-scratch
    ResNet18 documents for its 32px case ("32-px images -> 3x3 stem,
    stride=1, no maxpool"), applied here to the torchvision architecture.
    """
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()


def get_pretrained_resnet18(
    num_classes: int = NUM_CLASSES,
    channels: int    = CHANNELS,
    finetune_mode: str = FINETUNE_MODE,
) -> nn.Module:
    """Return a torchvision ResNet-18 with ImageNet weights, adapted for the task."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model   = models.resnet18(weights=weights)

    _adapt_first_conv(model, channels)

    # Replace head
    in_features  = model.fc.in_features
    model.fc     = nn.Linear(in_features, num_classes)

    if finetune_mode == "head":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    return model


def get_pretrained_resnet50(
    num_classes: int = NUM_CLASSES,
    channels: int    = CHANNELS,
    finetune_mode: str = FINETUNE_MODE,
) -> nn.Module:
    """Return a torchvision ResNet-50 with ImageNet weights, adapted for the task."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model   = models.resnet50(weights=weights)

    _adapt_first_conv(model, channels)

    in_features = model.fc.in_features
    model.fc    = nn.Linear(in_features, num_classes)

    if finetune_mode == "head":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    return model


def get_resnet18_no_weights(
    num_classes: int = NUM_CLASSES,
    channels: int    = CHANNELS,
    image_size: int  = IMAGE_SIZE,
) -> nn.Module:
    """Torchvision ResNet-18 architecture with randomly initialised weights (no ImageNet pretraining)."""
    model = models.resnet18(weights=None)
    if image_size <= 32:
        _adapt_stem_for_cifar(model, channels)
    else:
        _adapt_first_conv(model, channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet50_no_weights(
    num_classes: int = NUM_CLASSES,
    channels: int    = CHANNELS,
    image_size: int  = IMAGE_SIZE,
) -> nn.Module:
    """Torchvision ResNet-50 architecture with randomly initialised weights (no ImageNet pretraining)."""
    model = models.resnet50(weights=None)
    if image_size <= 32:
        _adapt_stem_for_cifar(model, channels)
    else:
        _adapt_first_conv(model, channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters (call after warm-up when using 'gradual' mode)."""
    for param in model.parameters():
        param.requires_grad = True