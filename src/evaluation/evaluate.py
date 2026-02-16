import torch
import torch.nn.functional as F
import time
import logging
from typing import Dict
from sklearn.metrics import precision_recall_fscore_support
from src.utility.config import DEVICE

logger = logging.getLogger(__name__)

def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, desc: str
) -> Dict[str, float]:
    """
    Evaluates the model and returns a dictionary of performance metrics.
    """
    model.eval()
    model.to(DEVICE)

    correct = 0
    test_loss = 0
    num_samples = len(loader.dataset)
    all_preds = []
    all_targets = []

    # --- 1. WARM-UP ---
    if DEVICE.type == "cuda":
        warmup_batches = 5
        with torch.no_grad():
            for i, (data, target) in enumerate(loader):
                if i >= warmup_batches: break
                model(data.to(DEVICE))
        torch.cuda.synchronize()

    logger.info(f"Starting evaluation: {desc}")
    
    # --- 2. MEASUREMENT ---
    start_time = time.time()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # Standard Metrics
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            
            # Collect for Scikit-Learn
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.time() - start_time

    # --- 3. CALCULATE ADVANCED METRICS ---
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    
    accuracy = 100.0 * correct / num_samples
    
    logger.info(
        f"Result [{desc}] - Acc: {accuracy:.2f}% | F1: {f1:.4f} | "
        f"Prec: {precision:.4f} | Rec: {recall:.4f} | Time: {inference_time:.2f}s"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "inference_time": inference_time
    }