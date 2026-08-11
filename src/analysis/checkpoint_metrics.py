"""
checkpoint_metrics.py -- per-layer Hessian trace, top eigenvalue, weight-
quantization error (MSE/SQNR), and whole-model classification metrics
(accuracy/precision/recall/f1) for FP32/PTQ/QAT, computed from SAVED
checkpoints rather than inline during training.

Motivation: these four measurements used to run INLINE inside main.py's
default (no-flag) training loop, coupling "how well does the model
quantize" analysis to "how long does training take" -- you could not get
one without paying for the other, and a code fix to the analysis meant
retraining to re-measure it. --train-only now does FP32/PTQ/QAT training
and checkpointing ONLY; this module is that same Hessian/eigenvalue/quant-
error/classification computation, reading the checkpoints --train-only (or
any other stage that writes baseline_*/ptq_po2_*/qat_po2_* in the expected
layout) saved, so it can be rerun independently and is one of the passes
--analyze-cifar10/--analyze-imagenet100 bundle.

Per model x dataset x stage (FP32 always run as the reference; PTQ/QAT
skipped with a warning if their checkpoint is missing): classification
metrics, Hessian trace (compute_layerwise_hessian_trace_pyhessian,
unchanged), top eigenvalue (compute_top_eigenvalue, unchanged), and
(PTQ/QAT only) weight-quantization error vs the FP32 reference
(compute_layerwise_quant_error, unchanged).

Reuses (does not duplicate): compute_layerwise_hessian_trace_pyhessian,
compute_top_eigenvalue, compute_layerwise_quant_error,
compute_classification_metrics (all unchanged); the checkpoint loader
(_load_quant_model, _load_fp32_reference) and _append_row CSV writer from
diagnose_activations.py; the robust checkpoint resolver from
_ablation_common.py; MODELS/DATASETS/STAGES and the quantized-checkpoint
directory resolver from deploy_fbgemm.py.

Analysis only. Runs as a single local process (`python -m src.main
--checkpoint-metrics ...`), no SLURM/torchrun required; prefers CUDA.
"""

import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.analysis.pyhessian import compute_layerwise_hessian_trace_pyhessian
from src.analysis.top_eigenvalue import compute_top_eigenvalue
from src.analysis.quant_error import compute_layerwise_quant_error
from src.analysis.classification_metrics import compute_classification_metrics
from src.analysis.diagnose_activations import (
    _resolve_fp32_models_dir,
    _load_quant_model,
    _load_fp32_reference,
    _append_row,
)
from src.analysis._ablation_common import _resolve_checkpoint_robust, WeightAblationCheckpointError
from src.quantization.deploy_fbgemm import MODELS, DATASETS, STAGES, _resolve_checkpoint_dir
from src.utility.config import CSV_DIR, DATASET_SPECS, HESSIAN_BATCH_SIZE
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

# Per-dataset trace budget for the Hessian-trace and top-eigenvalue calls.
# CIFAR10 uses compute_layerwise_hessian_trace_pyhessian's/compute_top_
# eigenvalue's own defaults (num_batches=5, max_iter=100, tol=1e-3) --
# passed explicitly so nothing is left to an upstream default drifting.
# IMAGENET100 is reduced -- 224x224 HVP cost is far higher per iteration
# than 32x32 -- matching every other trace-estimating mode in this
# codebase's own IMAGENET100 budget for consistency.
TRACE_CONFIG = {
    "CIFAR10":     {"num_batches": 5, "max_iter": 100, "tol": 1e-3},
    "IMAGENET100": {"num_batches": 3, "max_iter": 30,  "tol": 1e-3},
}

HESSIAN_FIELDNAMES = ["model", "dataset", "stage", "layer", "trace"]
EIGENVALUE_FIELDNAMES = ["model", "dataset", "stage", "layer", "eigenvalue"]
QUANT_ERROR_FIELDNAMES = ["model", "dataset", "stage", "layer", "mse", "sqnr"]
CLASSIFICATION_FIELDNAMES = ["model", "dataset", "stage", "accuracy", "precision", "recall", "f1"]


def _build_hessian_loader(dataset_name: str) -> DataLoader:
    _, val_loader, _ = get_data_loaders(dataset_name)
    return DataLoader(val_loader.dataset, batch_size=HESSIAN_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)


def _run_one_stage(
    model_name: str, dataset_name: str, stage: str, model: nn.Module, fp32_model: nn.Module,
    val_loader: DataLoader, hessian_loader: DataLoader, device: torch.device, num_classes: int, trace_cfg: dict,
    hessian_csv: str, eigenvalue_csv: str, quant_error_csv: str, classification_csv: str,
) -> None:
    label = f"{stage} {model_name}/{dataset_name}"
    criterion = nn.CrossEntropyLoss()
    model.eval()

    class_metrics = compute_classification_metrics(model, val_loader, device, num_classes=num_classes)
    _append_row(classification_csv, {
        "model": model_name, "dataset": dataset_name, "stage": stage,
        "accuracy": class_metrics["accuracy"], "precision": class_metrics["precision"],
        "recall": class_metrics["recall"], "f1": class_metrics["f1"],
    }, CLASSIFICATION_FIELDNAMES)

    traces = compute_layerwise_hessian_trace_pyhessian(model, hessian_loader, criterion, device, **trace_cfg)
    for layer, trace_val in traces.items():
        _append_row(hessian_csv, {"model": model_name, "dataset": dataset_name, "stage": stage, "layer": layer, "trace": trace_val}, HESSIAN_FIELDNAMES)

    eigenvalues = compute_top_eigenvalue(model, hessian_loader, criterion, device, **trace_cfg)
    for layer, eigenvalue in eigenvalues.items():
        _append_row(eigenvalue_csv, {"model": model_name, "dataset": dataset_name, "stage": stage, "layer": layer, "eigenvalue": eigenvalue}, EIGENVALUE_FIELDNAMES)

    if stage != "FP32":
        quant_error = compute_layerwise_quant_error(fp32_model, model)
        for layer, metrics in quant_error.items():
            _append_row(quant_error_csv, {
                "model": model_name, "dataset": dataset_name, "stage": stage, "layer": layer,
                "mse": metrics["mse"], "sqnr": metrics["sqnr"],
            }, QUANT_ERROR_FIELDNAMES)

    logger.info(f"[CheckpointMetrics] {label}: accuracy={class_metrics['accuracy']:.2f}% -- {len(traces)} layers traced")


def run_checkpoint_metrics(
    checkpoint_dir: str | None, load_run_id: str | None, datasets: list[str] | None = None,
) -> None:
    """datasets restricts the sweep to a subset of DATASETS (e.g. one dataset,
    for a parallel per-dataset analysis run) -- default None means every
    dataset in DATASETS."""
    datasets = datasets if datasets is not None else DATASETS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[CheckpointMetrics] CUDA not available -- falling back to CPU, this will be slow.")
    logger.info(f"[CheckpointMetrics] device={device} datasets={datasets}")

    fp32_models_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)
    quant_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)

    os.makedirs(CSV_DIR, exist_ok=True)
    hessian_csv = os.path.join(CSV_DIR, "layerwise_hessian_traces.csv")
    eigenvalue_csv = os.path.join(CSV_DIR, "layerwise_top_eigenvalues.csv")
    quant_error_csv = os.path.join(CSV_DIR, "layerwise_quant_error.csv")
    classification_csv = os.path.join(CSV_DIR, "classification_metrics.csv")

    for dataset_name in datasets:
        specs = DATASET_SPECS[dataset_name]
        trace_cfg = TRACE_CONFIG[dataset_name]
        try:
            _, val_loader, num_classes = get_data_loaders(dataset_name)
        except Exception as exc:
            logger.warning(f"[CheckpointMetrics] {dataset_name}: could not load dataset ({exc}) -- skipping")
            continue
        hessian_loader = _build_hessian_loader(dataset_name)

        for model_name in MODELS:
            label = f"{model_name}/{dataset_name}"
            try:
                fp32_ckpt = _resolve_checkpoint_robust(fp32_models_dir, {"model": model_name, "dataset": dataset_name})
            except (FileNotFoundError, WeightAblationCheckpointError) as exc:
                logger.warning(f"[CheckpointMetrics] {label}: FP32 baseline checkpoint unresolvable ({exc}) -- skipping model")
                continue

            fp32_model = _load_fp32_reference(model_name, fp32_ckpt, num_classes, specs["channels"], specs["image_size"]).to(device)
            try:
                _run_one_stage(
                    model_name, dataset_name, "FP32", fp32_model, fp32_model, val_loader, hessian_loader, device, num_classes, trace_cfg,
                    hessian_csv, eigenvalue_csv, quant_error_csv, classification_csv,
                )
            except Exception as exc:
                logger.error(f"[CheckpointMetrics] FAILED FP32 {label}: {exc}", exc_info=True)

            for stage in STAGES:
                try:
                    stage_ckpt = _resolve_checkpoint_robust(quant_dir, {"stage": stage, "model": model_name, "dataset": dataset_name})
                except FileNotFoundError as exc:
                    logger.warning(f"[CheckpointMetrics] {stage} {label}: checkpoint missing ({exc}) -- skipping")
                    continue
                except WeightAblationCheckpointError as exc:
                    logger.error(f"[CheckpointMetrics] {stage} {label}: checkpoint AMBIGUOUS/NEAR-MISS -- {exc} -- skipping")
                    continue

                stage_model, _, _ = _load_quant_model(model_name, stage_ckpt, num_classes, specs["channels"], specs["image_size"])
                stage_model = stage_model.to(device)
                try:
                    _run_one_stage(
                        model_name, dataset_name, stage, stage_model, fp32_model, val_loader, hessian_loader, device, num_classes, trace_cfg,
                        hessian_csv, eigenvalue_csv, quant_error_csv, classification_csv,
                    )
                except Exception as exc:
                    logger.error(f"[CheckpointMetrics] FAILED {stage} {label}: {exc}", exc_info=True)
                finally:
                    del stage_model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            del fp32_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    logger.info("[CheckpointMetrics] === Checkpoint-Metrics complete ===")
