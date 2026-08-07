import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score


def compute_classification_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    average: str = "macro",
) -> dict:
    """
    Runs one full pass over loader and returns aggregate classification metrics.

    Returns:
        dict with keys: accuracy, precision, recall, f1 (all in percent)
    """
    model.eval()

    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average=average).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average=average).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average=average).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average=average).to(device)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            accuracy_metric.update(preds, targets)
            precision_metric.update(preds, targets)
            recall_metric.update(preds, targets)
            f1_metric.update(preds, targets)

    return {
        "accuracy": accuracy_metric.compute().item() * 100.0,
        "precision": precision_metric.compute().item() * 100.0,
        "recall": recall_metric.compute().item() * 100.0,
        "f1": f1_metric.compute().item() * 100.0,
    }