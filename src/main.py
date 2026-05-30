"""
main.py — train all models on all datasets.

Usage:
    uv run python -m src.main

Trains the full 4-model × 5-dataset matrix and writes a summary CSV to
results/csv/main_summary.csv.

Models:   cnn | resnet18_scratch | resnet18_pretrained | resnet50_pretrained
Datasets: MNIST | FASHION_MNIST | CIFAR10 | CIFAR100 | POKEMON
"""

import os
import sys
import csv
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utility.utils import setup_global_logging, get_data_loaders
from src.model_cnn.train import train_model
from src.utility.config import CSV_DIR

logger = logging.getLogger(__name__)

MODELS = [
    "cnn",
    "resnet18_scratch",
    "resnet18_pretrained",
    "resnet50_pretrained",
]

DATASETS = [
    "MNIST",
    "FASHION_MNIST",
    "CIFAR10",
    "CIFAR100",
    "POKEMON",
]


def main() -> None:
    setup_global_logging()

    total = len(MODELS) * len(DATASETS)
    logger.info(f"=== Pipeline start: {len(MODELS)} models - {len(DATASETS)} datasets = {total} runs ===")

    summary: list[dict] = []
    run_idx = 0

    for dataset_name in DATASETS:
        # Load data once per dataset, reuse across all models
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            train_loader, val_loader, num_classes = get_data_loaders(dataset_name)
        except Exception as exc:
            logger.error(f"Failed to load {dataset_name}: {exc}")
            for model_name in MODELS:
                summary.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "best_val_acc": "LOAD_ERROR",
                    "wall_time_min": "",
                    "status": "failed",
                })
            continue

        for model_name in MODELS:
            run_idx += 1
            logger.info(f"\n--- Run {run_idx}/{total}: {model_name} on {dataset_name} ---")
            t0 = time.perf_counter()
            try:
                _, history, _ = train_model(
                    train_loader, val_loader, num_classes,
                    model_name=model_name,
                    dataset_name=dataset_name,
                )
                best_val_acc = max(history["val_acc"])
                elapsed_min  = (time.perf_counter() - t0) / 60
                logger.info(
                    f"Finished {model_name}/{dataset_name}: "
                    f"best_val_acc={best_val_acc:.2f}%  time={elapsed_min:.1f} min"
                )
                summary.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "best_val_acc": f"{best_val_acc:.2f}",
                    "wall_time_min": f"{elapsed_min:.1f}",
                    "status": "ok",
                })
            except Exception as exc:
                elapsed_min = (time.perf_counter() - t0) / 60
                logger.error(f"FAILED {model_name}/{dataset_name}: {exc}", exc_info=True)
                summary.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "best_val_acc": "ERROR",
                    "wall_time_min": f"{elapsed_min:.1f}",
                    "status": "failed",
                })

    _save_summary(summary)
    _print_summary(summary)
    logger.info("=== Pipeline complete ===")


def _save_summary(summary: list[dict]) -> None:
    path = os.path.join(CSV_DIR, "pipeline_summary.csv")
    fieldnames = ["model", "dataset", "best_val_acc", "wall_time_min", "status"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    logger.info(f"Summary saved → {path}")


def _print_summary(summary: list[dict]) -> None:
    logger.info("\n=== PIPELINE SUMMARY ===")
    header = f"{'Model':<24} {'Dataset':<16} {'Best Val Acc':>12} {'Time (min)':>10} {'Status':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for row in summary:
        logger.info(
            f"{row['model']:<24} {row['dataset']:<16} "
            f"{row['best_val_acc']:>12} {row['wall_time_min']:>10} {row['status']:>8}"
        )


if __name__ == "__main__":
    main()
