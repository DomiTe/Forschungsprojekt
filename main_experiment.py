import torch
import torch.nn.functional as F
import time
import copy
from typing import Tuple
import os
from src.model import CNN
from src.utils import get_data_loaders, get_model_size
from src.train import train_model
from src.config import DEVICE, MODEL_SAVE_PATH


def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, desc: str
) -> Tuple[float, float]:
    model.eval()
    model.to(DEVICE)

    correct = 0
    test_loss = 0

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
    duration = end_time - start_time

    print(
        f"[{desc}] Acc: {accuracy:.2f}% | Loss: {test_loss:.4f} | Zeit: {duration:.4f}s"
    )

    return accuracy, inference_time


def run_experiment():
    _, test_loader = get_data_loaders()

    if not os.path.exists(MODEL_SAVE_PATH):
        model = train_model()
    else:
        model = CNN().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    acc_base, time_base = evaluate(model, test_loader, "Baseline Float32")
    size_base = get_model_size(MODEL_SAVE_PATH)

    print("\n==========================================")
    print("         ERGEBNIS BERICHT                 ")
    print("==========================================")
    print(f"Modell Größe (Float32):   {size_base:.2f} MB")
    print(f"Theor. Größe (Int8):      {size_base / 4:.2f} MB")
    print("------------------------------------------")
    print(f"Accuracy Float32:         {acc_base:.2f}%")
    # print(f"Accuracy Int8 (Sim):      {acc_quant:.2f}%")
    # print(f"Verlust an Genauigkeit:   {acc_base - acc_quant:.2f}%")
    print("==========================================")


if __name__ == "__main__":
    run_experiment()
