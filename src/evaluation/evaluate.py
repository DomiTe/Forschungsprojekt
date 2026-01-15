import torch
import torch.nn.functional as F
import time
import logging
from typing import Tuple
from src.utility.config import DEVICE

logger = logging.getLogger(__name__)

def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, desc: str
) -> Tuple[float, float]:
    """
    Evaluiert das Modell auf dem gegebenen DataLoader.
    
    Returns:
        accuracy (float): Genauigkeit in Prozent
        inference_time (float): Gesamtzeit für die Inferenz in Sekunden
    """
    model.eval()
    model.to(DEVICE)

    # --- 1. WARM-UP (Wichtig für präzise Zeitmessung) ---
    # Wir schicken ein paar Dummy-Batches durch, damit Caches gefüllt sind.
    # Das Ergebnis verwerfen wir.
    
    warmup_batches = 20
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            data = data.to(DEVICE)
            model(data)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize() # Warten bis Warm-up fertig ist

    correct = 0
    # test_loss = 0
    num_samples = len(loader.dataset)

    logger.info(f"Starting evaluation: {desc} (Sample Size: {num_samples})")
    
    # --- 2. ECHTE MESSUNG ---
    start_time = time.time()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Forward Pass
            output = model(data)

            # Metrics
            pred = output.argmax(dim=1, keepdim=True)
            # test_loss += F.nll_loss(output, target, reduction="sum").item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Synchronisieren für exakte Zeit auf GPU
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    inference_time = end_time - start_time

    # --- 3. BERECHNUNG & LOGGING ---
    # test_loss /= num_samples
    accuracy = 100.0 * correct / num_samples
    
    # Latenz pro Bild (in Millisekunden) ist oft interessant für Vergleiche
    latency_ms = (inference_time / num_samples) * 1000 

    logger.info(
        f"Evaluation Result [{desc}] - "
        f"Acc: {accuracy:.2f}% | "
        # f"Loss: {test_loss:.4f} | "
        f"Total Time: {inference_time:.4f}s | "
        f"Latency: {latency_ms:.4f} ms/img"
    )

    return accuracy, inference_time