import torch
import torch.nn.functional as F
import time
import logging
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, Tuple, Any
from src.utility.config import DEVICE

logger = logging.getLogger(__name__)

def evaluate(
    model: torch.nn.Module, 
    loader: torch.utils.data.DataLoader, 
    desc: str, 
    device=DEVICE
) -> Dict[str, float]:
    """
    Evaluates the model and returns a comprehensive dictionary of metrics.
    """

    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0
    num_samples = len(loader.dataset)
    
    all_targets = []
    all_preds = []

    # --- 1. WARM-UP ---
    if device.type == "cuda":
        warmup_batches = 5
        with torch.no_grad():
            for i, (data, target) in enumerate(loader):
                if i >= warmup_batches: break
                data = data.to(device)
                model(data)
        torch.cuda.synchronize()

    logger.info(f"Starting evaluation: {desc} (Sample Size: {num_samples})")
    
    # --- 2. MEASUREMENT ---
    start_time = time.time()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            # Loss & Accuracy
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Collect for Scikit-Learn Metrics
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    inference_time = end_time - start_time

    # --- 3. METRICS ---
    test_loss /= num_samples
    accuracy = 100.0 * correct / num_samples
    
    # Calculate Precision, Recall, F1 (Macro Average handles multi-class well)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    logger.info(
        f"Evaluation [{desc}] - "
        f"Acc: {accuracy:.2f}%, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, "
        f"Time: {inference_time:.4f}s"
    )

    # Return a dictionary for easier CSV logging later
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "loss": test_loss,
        "inference_time": inference_time
    }