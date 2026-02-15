import torch
import torch.nn.functional as F
import time
import logging
from typing import Tuple
from src.utility.config import DEVICE

logger = logging.getLogger(__name__)

def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, desc: str, device=DEVICE
) -> Tuple[float, float]:
    """
    Evaluiert das Modell auf dem gegebenen DataLoader.
    
    Returns:
        accuracy (float): Genauigkeit in Prozent
        inference_time (float): Gesamtzeit für die Inferenz in Sekunden
    """

    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    model.to(device)

    correct = 0
    test_loss = 0
    num_samples = len(loader.dataset)

    # --- 1. WARM-UP (Wichtig für präzise Zeitmessung) ---
    # Wir schicken ein paar Dummy-Batches durch, damit Caches gefüllt sind.
    # Das Ergebnis verwerfen wir.
    if device.type == "cuda":
        warmup_batches = 5
        with torch.no_grad():
            for i, (data, target) in enumerate(loader):
                if i >= warmup_batches:
                    break
                data = data.to(device)
                model(data)
        torch.cuda.synchronize() # Warten bis Warm-up fertig ist

    logger.info(f"Starting evaluation: {desc} (Sample Size: {num_samples})")
    
    # --- 2. ECHTE MESSUNG ---
    start_time = time.time()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            # Forward Pass
            output = model(data)

            # Metrics
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Synchronisieren für exakte Zeit auf GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    inference_time = end_time - start_time

    # --- 3. BERECHNUNG & LOGGING ---
    test_loss /= num_samples
    accuracy = 100.0 * correct / num_samples
    
    # Latenz pro Bild (in Millisekunden) ist oft interessant für Vergleiche
    latency_ms = (inference_time / num_samples) * 1000 

    logger.info(
        f"Evaluation Result [{desc}] - "
        f"Acc: {accuracy:.2f}% | "
        f"Loss: {test_loss:.4f} | "
        f"Total Time: {inference_time:.4f}s | "
        f"Latency: {latency_ms:.4f} ms/img"
    )

    return accuracy, inference_time