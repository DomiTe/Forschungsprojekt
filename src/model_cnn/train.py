"""
train.py — universal training loop for CNN, ResNet-18 (scratch),
           pretrained ResNet-18, and pretrained ResNet-50.

Tracks per-epoch wall-clock time for both the training and validation phases
using TimingTracker from utils.py.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from src.utility.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MOMENTUM,
    USE_COSINE_LR,
    MODELS_DIR,
    CSV_DIR,
    LOG_DIR,
    DATASET_SPECS,
    MODEL_NAME,
    DATASET_NAME,
)
from src.utility.utils import TimingTracker, plot_training_curves, get_model_size

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(num_classes: int, model_name: str, channels: int, image_size: int) -> nn.Module:
    """Instantiate the model specified by model_name."""
    if model_name == "cnn":
        from src.model_cnn.model import CNN
        return CNN(num_classes=num_classes, channels=channels)

    elif model_name == "resnet18_scratch":
        from src.model_cnn.resnet18 import ResNet18
        return ResNet18(num_classes=num_classes, channels=channels, image_size=image_size)

    elif model_name == "resnet18_pretrained":
        from src.model_cnn.pretrained_resnet18 import get_pretrained_resnet18
        return get_pretrained_resnet18(num_classes=num_classes, channels=channels)

    elif model_name == "resnet50_pretrained":
        from src.model_cnn.pretrained_resnet18 import get_pretrained_resnet50
        return get_pretrained_resnet50(num_classes=num_classes, channels=channels)

    else:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         "Choose: cnn | resnet18_scratch | resnet18_pretrained | resnet50_pretrained")


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    num_classes:  int,
    model_name:   str = MODEL_NAME,
    dataset_name: str = DATASET_NAME,
) -> tuple[nn.Module, dict, TimingTracker]:
    """
    Full training run.

    Returns:
        model       — trained model (best validation accuracy)
        history     — dict with lists: train_loss, val_loss, train_acc, val_acc
        timer       — TimingTracker instance (use .save_csv() / .summary())
    """
    specs      = DATASET_SPECS[dataset_name]
    channels   = specs["channels"]
    image_size = specs["image_size"]

    baseline_path = os.path.join(MODELS_DIR, f"baseline_{model_name}_{dataset_name}_float32.pt")
    timing_path   = os.path.join(CSV_DIR,    f"timing_{model_name}_{dataset_name}.csv")
    curves_path   = os.path.join(LOG_DIR,    f"training_curves_{model_name}_{dataset_name}.png")

    model = build_model(num_classes, model_name, channels, image_size).to(DEVICE)
    logger.info(f"Model: {model_name} | Dataset: {dataset_name} | "
                f"Size: {get_model_size(model):.2f} MB | "
                f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )

    if USE_COSINE_LR:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    timer   = TimingTracker()

    best_acc   = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        timer.start_epoch()

        # --- Training phase -------------------------------------------------
        timer.start_split("train")
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted  = outputs.max(1)
            total        += targets.size(0)
            correct      += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc  = 100.0 * correct / total
        timer.end_split()

        # --- Validation phase -----------------------------------------------
        timer.start_split("val")
        val_loss, val_acc = _evaluate(model, val_loader, criterion)
        timer.end_split()

        record = timer.end_epoch(epoch)
        scheduler.step()

        # --- Logging --------------------------------------------------------
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"LR {lr_now:.5f} | "
            f"Train L {train_loss:.4f} A {train_acc:.2f}% | "
            f"Val L {val_loss:.4f} A {val_acc:.2f}% | "
            f"Train {record['train_s']:.1f}s  Val {record['val_s']:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- Finalise -----------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Best validation accuracy: {best_acc:.2f}%")

    torch.save(model.state_dict(), baseline_path)
    logger.info(f"Model saved → {baseline_path}")

    timer.summary()
    timer.save_csv(timing_path)
    plot_training_curves(history, save_path=curves_path)

    return model, history, timer


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return loss_sum / total, 100.0 * correct / total
