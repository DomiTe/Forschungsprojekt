import torch
from torchvision import datasets, transforms
from typing import Tuple
from src.config import PIN_MEMORY, DATA_DIR, BATCH_SIZE, TEST_BATCH_SIZE
import os


def get_data_loaders() -> Tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    kwargs = {"num_workers": 1, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    train_dataset = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs
    )
    test_dataset = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs
    )
    return train_loader, test_loader


def get_model_size(model_path: str) -> float:
    size = os.path.getsize(model_path)
    return size / (1024 * 1024)
