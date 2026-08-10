"""
utils.py — data loaders, logging, timing, and plotting helpers.

New additions compared to the original:
  - ImageNet loader (_get_imagenet_loaders)
  - ImageNet100 loader (_get_imagenet100_loaders)
  - TimingTracker  — records per-epoch and total train/val wall-clock times
  - save_timing_csv — persists timing results
  - Normalisation means/stds added for CIFAR datasets
  - measure_throughput — single-batch-shape latency/throughput benchmark
"""

import os
import sys
import csv
import copy
import time
import logging

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from src.utility.config import (
    PIN_MEMORY,
    DATA_DIR,
    # IMAGENET_DIR,
    IMAGENET100_DIR,
    LOG_DIR,
    CSV_DIR,
    BATCH_SIZE,
    TEST_BATCH_SIZE,
    DATASET_NAME,
    DATASET_SPECS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------
_NORM = {
    "MNIST":         {"mean": (0.1307,),                   "std": (0.3081,)},
    "FASHION_MNIST": {"mean": (0.2860,),                   "std": (0.3530,)},
    "CIFAR10":       {"mean": (0.4914, 0.4822, 0.4465),    "std": (0.2023, 0.1994, 0.2010)},
    "CIFAR100":      {"mean": (0.5071, 0.4867, 0.4408),    "std": (0.2675, 0.2565, 0.2761)},
    "POKEMON":       {"mean": (0.5,    0.5,    0.5),       "std": (0.5,    0.5,    0.5)},
    # "IMAGENET":      {"mean": (0.485,  0.456,  0.406),     "std": (0.229,  0.224,  0.225)},
    "IMAGENET100":   {"mean": (0.485,  0.456,  0.406),     "std": (0.229,  0.224,  0.225)},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_size(model: torch.nn.Module) -> float:
    """Model parameter + buffer size in MB (theoretical, in-memory)."""
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2


def get_data_loaders(dataset_name: str = DATASET_NAME, batch_size: int | None = None):
    """
    Dispatch to the correct dataset loader.

    batch_size, when given, overrides BATCH_SIZE for the *train* loader only
    (the test/val loader always uses TEST_BATCH_SIZE) -- e.g. Hessian-trace
    computation needs a much smaller batch than normal training.
    """
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    image_size = DATASET_SPECS[dataset_name]["image_size"]
    dispatch = {
        "MNIST":         _get_mnist_loaders,
        "FASHION_MNIST": _get_fashion_loaders,
        "CIFAR10":       _get_cifar10_loaders,
        "CIFAR100":      _get_cifar100_loaders,
        "POKEMON":       _get_pokemon_loaders,
        # "IMAGENET":      _get_imagenet_loaders,
        "IMAGENET100":   _get_imagenet100_loaders,
    }
    return dispatch[dataset_name](image_size, batch_size)


def measure_throughput(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple,
    warmup: int = 20,
    iters: int = 100,
) -> dict:
    """
    Single-batch-shape latency/throughput benchmark. input_shape includes
    the batch dimension (e.g. (1, channels, H, W)).

    Synchronizes around timed CUDA work so kernel-launch queuing doesn't
    make async GPU ops look free; a no-op on CPU. Reports the median (not
    mean) of `iters` runs to stay robust against one-off stalls.

    Returns: dict with keys latency_ms (median), throughput_fps.
    """
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    latencies_ms.sort()
    mid = len(latencies_ms) // 2
    median_latency_ms = (
        latencies_ms[mid] if len(latencies_ms) % 2
        else (latencies_ms[mid - 1] + latencies_ms[mid]) / 2.0
    )
    throughput_fps = input_shape[0] * 1000.0 / median_latency_ms

    return {
        "latency_ms": median_latency_ms,
        "throughput_fps": throughput_fps,
    }


# ---------------------------------------------------------------------------
# Timing tracker
# ---------------------------------------------------------------------------

class TimingTracker:
    """
    Lightweight per-epoch wall-clock timer.

    Usage:
        tracker = TimingTracker()
        for epoch in range(epochs):
            tracker.start_epoch()
            train_one_epoch(...)
            tracker.split("train")
            validate(...)
            tracker.split("val")
            tracker.end_epoch()
        tracker.summary()        # prints table
        tracker.save_csv(path)
    """

    def __init__(self):
        self.records: list[dict] = []
        self._epoch_start: float | None = None
        self._splits: dict[str, float] = {}
        self._split_start: float | None = None
        self._current_split: str | None = None

    def start_epoch(self) -> None:
        self._epoch_start = time.perf_counter()
        self._splits = {}

    def start_split(self, name: str) -> None:
        """Optional fine-grained timing of a named phase within an epoch."""
        self._current_split = name
        self._split_start   = time.perf_counter()

    def end_split(self) -> float:
        """End current split, return elapsed seconds."""
        elapsed = time.perf_counter() - self._split_start
        self._splits[self._current_split] = elapsed
        return elapsed

    def split(self, name: str) -> None:
        """Convenience: end previous split (if any) and start a new one."""
        if self._split_start is not None:
            self.end_split()
        self.start_split(name)

    def end_epoch(self, epoch: int | None = None) -> dict:
        """Finalise the epoch; returns the record dict."""
        # Close the last open split
        if self._split_start is not None:
            self.end_split()

        total = time.perf_counter() - self._epoch_start
        record = {"epoch": epoch if epoch is not None else len(self.records) + 1,
                  "total_s": round(total, 3),
                  **{f"{k}_s": round(v, 3) for k, v in self._splits.items()}}
        self.records.append(record)
        self._split_start = None
        return record

    def summary(self) -> None:
        if not self.records:
            return
        keys = list(self.records[0].keys())
        header = " | ".join(f"{k:>10}" for k in keys)
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))
        for r in self.records:
            logger.info(" | ".join(f"{str(r[k]):>10}" for k in keys))
        total = sum(r["total_s"] for r in self.records)
        logger.info(f"\nTotal wall-clock time: {total:.1f}s  ({total/60:.2f} min)")

    def save_csv(self, path: str | None = None) -> None:
        if path is None:
            path = os.path.join(CSV_DIR, "timing.csv")
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        logger.info(f"Timing data saved → {path}")


# ---------------------------------------------------------------------------
# Dataset loaders (private)
# ---------------------------------------------------------------------------

def _norm(name: str):
    return _NORM.get(name, {"mean": (0.5,), "std": (0.5,)})


def _get_mnist_loaders(image_size: int, batch_size: int | None = None):
    n = _norm("MNIST")
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    train = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=batch_size or BATCH_SIZE, shuffle=True,  pin_memory=PIN_MEMORY),
            DataLoader(test,  batch_size=TEST_BATCH_SIZE,          shuffle=False, pin_memory=PIN_MEMORY),
            10)


def _get_fashion_loaders(image_size: int, batch_size: int | None = None):
    n = _norm("FASHION_MNIST")
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    train = datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test  = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=batch_size or BATCH_SIZE, shuffle=True,  pin_memory=PIN_MEMORY),
            DataLoader(test,  batch_size=TEST_BATCH_SIZE,          shuffle=False, pin_memory=PIN_MEMORY),
            10)


def _get_cifar10_loaders(image_size: int, batch_size: int | None = None):
    n = _norm("CIFAR10")
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    tf_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    train = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=tf_train)
    test  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_test)
    kw = {"num_workers": 8, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    logger.info(f"CIFAR-10: {len(train)} train / {len(test)} test")
    return (DataLoader(train, batch_size=batch_size or BATCH_SIZE, shuffle=True,  **kw),
            DataLoader(test,  batch_size=TEST_BATCH_SIZE,          shuffle=False, **kw),
            10)


def _get_cifar100_loaders(image_size: int, batch_size: int | None = None):
    n = _norm("CIFAR100")
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    tf_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    train = datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=tf_train)
    test  = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=tf_test)
    kw = {"num_workers": 8, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    logger.info(f"CIFAR-100: {len(train)} train / {len(test)} test")
    return (DataLoader(train, batch_size=batch_size or BATCH_SIZE, shuffle=True,  **kw),
            DataLoader(test,  batch_size=TEST_BATCH_SIZE,          shuffle=False, **kw),
            100)


def _get_pokemon_loaders(image_size: int, batch_size: int | None = None):
    n = _norm("POKEMON")
    tf_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    dataset_path = os.path.join(DATA_DIR, "PokemonData")
    full_train   = datasets.ImageFolder(dataset_path, transform=tf_train)
    full_val     = datasets.ImageFolder(dataset_path, transform=tf_val)
    total        = len(full_train)
    train_sz     = int(0.8 * total)
    val_sz       = total - train_sz
    gen          = torch.Generator().manual_seed(42)
    train_data, _ = random_split(full_train, [train_sz, val_sz], generator=gen)
    _, val_data   = random_split(full_val,   [train_sz, val_sz], generator=gen)
    kw = {"num_workers": 0, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    num_classes  = len(full_train.classes)
    logger.info(f"Pokemon: {len(train_data)} train / {len(val_data)} val — {num_classes} classes")
    return (DataLoader(train_data, batch_size=batch_size or BATCH_SIZE, shuffle=True,  **kw),
            DataLoader(val_data,   batch_size=TEST_BATCH_SIZE,          shuffle=False, **kw),
            num_classes)


# def _get_imagenet_loaders(image_size: int, batch_size: int | None = None):
#     """
#     Expects ImageNet data laid out as:
#         data/imagenet/train/<class_dir>/*.JPEG
#         data/imagenet/val/<class_dir>/*.JPEG

#     The standard ImageNet validation folder structure is produced by
#     the official torchvision download script or ILSVRC devkit.
#     """
#     n = _norm("IMAGENET")
#     tf_train = transforms.Compose([
#         transforms.RandomResizedCrop(image_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize(n["mean"], n["std"]),
#     ])
#     tf_val = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(n["mean"], n["std"]),
#     ])
#     train_dir = os.path.join(IMAGENET_DIR, "train")
#     val_dir   = os.path.join(IMAGENET_DIR, "val")
#     if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
#         raise FileNotFoundError(
#             f"ImageNet not found at {IMAGENET_DIR}.\n"
#             "Expected sub-directories: train/ and val/\n"
#             "Download from https://image-net.org/request  or use a torrent/academic mirror."
#         )
#     train = datasets.ImageFolder(train_dir, transform=tf_train)
#     val   = datasets.ImageFolder(val_dir,   transform=tf_val)
#     kw = {"num_workers": 0, "pin_memory": PIN_MEMORY, "persistent_workers": False} if PIN_MEMORY else {"num_workers": 0}
#     logger.info(f"ImageNet: {len(train)} train / {len(val)} val")
#     return (DataLoader(train, batch_size=batch_size or BATCH_SIZE, shuffle=True,  **kw),
#             DataLoader(val,   batch_size=TEST_BATCH_SIZE,          shuffle=False, **kw),
#             1000)


def _get_imagenet100_loaders(image_size: int, batch_size: int | None = None):
    """
    100-class subset of ImageNet. Expects data laid out as:
        data/imagenet100/train/<class_dir>/*.JPEG
        data/imagenet100/val/<class_dir>/*.JPEG
    """
    n = _norm("IMAGENET100")
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(n["mean"], n["std"]),
    ])
    train_dir = os.path.join(IMAGENET100_DIR, "train")
    val_dir   = os.path.join(IMAGENET100_DIR, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"IMAGENET100 not found at {IMAGENET100_DIR}.\n"
            "Expected sub-directories: train/ and val/"
        )
    train = datasets.ImageFolder(train_dir, transform=tf_train)
    val   = datasets.ImageFolder(val_dir,   transform=tf_val)
    kw = {"num_workers": 8, "pin_memory": PIN_MEMORY, "persistent_workers": False} if PIN_MEMORY else {"num_workers": 4}
    num_classes = len(train.classes)
    logger.info(f"ImageNet100: {len(train)} train / {len(val)} val across {num_classes} classes")
    return (DataLoader(train, batch_size=batch_size or BATCH_SIZE, shuffle=True,  **kw),
            DataLoader(val,   batch_size=TEST_BATCH_SIZE,          shuffle=False, **kw),
            num_classes)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path: str | None = None) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss",      marker=".")
    plt.plot(epochs, history["val_loss"],   label="Validation Loss", marker=".")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc",      color="blue",  marker=".")
    plt.plot(epochs, history["val_acc"],   label="Validation Acc", color="green", marker=".")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(LOG_DIR, "training_curves.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training curves saved → {save_path}")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_csv(results: list[dict], filename: str, fieldnames: list[str]) -> None:
    filepath = os.path.join(CSV_DIR, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"CSV saved → {filepath}")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_global_logging() -> None:
    log_filename = os.path.join(LOG_DIR, "experiment_log.txt")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
