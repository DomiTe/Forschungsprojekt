"""Data loaders, logging setup, timing, and plotting helpers."""

import os
import sys
import csv
import time
import logging

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utility.config import (
    PIN_MEMORY,
    DATA_DIR,
    IMAGENET100_DIR,
    LOG_DIR,
    CSV_DIR,
    BATCH_SIZE,
    TEST_BATCH_SIZE,
    DATASET_NAME,
    DATASET_SPECS,
)

logger = logging.getLogger(__name__)

_NORM = {
    "CIFAR10":       {"mean": (0.4914, 0.4822, 0.4465),    "std": (0.2023, 0.1994, 0.2010)},
    "IMAGENET100":   {"mean": (0.485,  0.456,  0.406),     "std": (0.229,  0.224,  0.225)},
}

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
        "CIFAR10":       _get_cifar10_loaders,
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
        tracker.summary()
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
        total = sum(r["total_s"] for r in self.records)
        logger.info(f"Total wall-clock time: {total:.1f}s ({total/60:.2f} min)")

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

def _norm(name: str):
    return _NORM.get(name, {"mean": (0.5,), "std": (0.5,)})


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


def save_csv(results: list[dict], filename: str, fieldnames: list[str]) -> None:
    filepath = os.path.join(CSV_DIR, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"CSV saved → {filepath}")

def setup_global_logging(level: str | int = "WARNING") -> None:
    """
    Configure the root logger. Default level is WARNING, so third-party
    traces (torch, matplotlib, etc.) and routine info-level messages stay
    silent unless an anomaly occurs; pass a lower level (e.g. "INFO") or
    use --log-level for more verbose output.
    """
    log_filename = os.path.join(LOG_DIR, "experiment_log.txt")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
