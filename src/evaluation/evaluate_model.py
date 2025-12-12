import torch
import torch.nn.functional as F
import time
from typing import Tuple
from src.utility.config import DEVICE
from src.utility.logging import get_logger

logger = get_logger(__name__)


def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, desc: str
) -> Tuple[float, float]:
    model.eval()
    model.to(DEVICE)

    correct = 0
    test_loss = 0

    logger.info(f"Starting evaluation: {desc}")
    start_time = time.time()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    end_time = time.time()
    inference_time = end_time - start_time

    test_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    logger.info(
        f"Evaluation Result [{desc}] - Acc: {accuracy:.2f}% | "
        f"Loss: {test_loss:.4f} | Time: {inference_time:.4f}s"
    )

    return accuracy, inference_time
