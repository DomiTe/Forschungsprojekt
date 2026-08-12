"""
layer_ablation.py -- tests whether excluding specific layers from fbgemm
INT8 quantization recovers accuracy, to isolate curvature (Hessian trace)
as the cause of resnet50/CIFAR10/PTQ's collapse from 75.46% FP32 to 23.19%
INT8 (every other model/dataset/stage combination loses under 0.7 points).
resnet50's conv1 carries a Hessian trace of 13,161.8 at PTQ -- 84% of the
model's total trace -- versus 18.7 at FP32 (a 704x increase) and 183.2
after QAT, while the quantization perturbation on that layer (MSE, SQNR)
is essentially identical between PTQ and QAT. If curvature is responsible,
excluding conv1 from quantization should restore most of the lost accuracy.

Per model x dataset x stage:
  1. Build the baked FP32 PoT model, evaluate -> fp32_acc.
  2. Quantize every eligible layer via the existing fbgemm PTQ path,
     evaluate -> all_quantized_acc (reproduces the numbers above).
  3. For each layer in an exclusion list, repeat the conversion leaving
     that one layer in FP32 (module.qconfig = None before prepare()),
     evaluate -> ablated_acc.

Exclusion list is either explicit (--ablate-layers) or trace-guided
(--ablate-top-k): the N highest-trace layers, read from
layerwise_hessian_traces.csv. The N lowest-nonzero-trace layers are always
run alongside the top-k set as a control -- if excluding high-trace layers
recovers accuracy but excluding low-trace layers does not, the effect is
attributable to curvature rather than to merely leaving *some* layer in
FP32.

Reuses (does not duplicate) the checkpoint reconstruction and fbgemm
conversion path from src/quantization/deploy_fbgemm.py -- the same
_build_baked_model / _apply_fbgemm_ptq / _evaluate_accuracy /
_audit_quantized_modules this module's --deploy-cpu-fbgemm sibling uses,
with _apply_fbgemm_ptq's `excluded_layers` parameter (added for this mode)
as the only new hook into that path.

Runs as a single local process (`python -m src.main
--ablate-layer-quantization ...`), no torchrun/SLURM/torch.distributed.
"""

import os
import csv
import copy
import logging

import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import pandas as pd

from src.utility.config import RESULTS_DIR, RUN_ID, CSV_DIR
from src.utility.utils import get_data_loaders
from src.quantization.deploy_fbgemm import (
    MODELS,
    DATASETS,
    STAGES,
    NUM_CALIBRATION_BATCHES,
    FbgemmBuildError,
    _resolve_checkpoint_dir,
    _checkpoint_path,
    _build_baked_model,
    _apply_fbgemm_ptq,
    _audit_quantized_modules,
    _evaluate_accuracy,
    _resolve_module_name,
)
from src.utility.config import DATASET_SPECS

logger = logging.getLogger(__name__)

OUTPUT_FIELDNAMES = [
    "model", "dataset", "stage", "excluded_layer", "selection", "excluded_trace",
    "fp32_acc", "all_quantized_acc", "ablated_acc", "recovery_pts",
    "fraction_of_fp32_recovered", "excluded_layer_still_fp32", "other_layers_quantized",
]


# ---------------------------------------------------------------------------
# Hessian trace CSV
# ---------------------------------------------------------------------------

def _resolve_hessian_csv_path(checkpoint_dir: str | None, load_run_id: str | None) -> str:
    # layerwise_hessian_traces.csv lives in the "csv" directory that sits
    # next to "quantized_models" under the same run, matching how every
    # other per-run artifact (models/, csv/, logs/) is laid out under
    # results/<RUN_ID>/. When checkpoint_dir is an explicit override (e.g.
    # a flat backup directory with no sibling csv/), this correctly
    # resolves to a path that doesn't exist, and _load_hessian_traces
    # degrades gracefully (top-k/low-k selection is skipped, --ablate-layers
    # still works).
    if checkpoint_dir:
        run_root = os.path.dirname(os.path.normpath(checkpoint_dir))
        path = os.path.join(run_root, "csv", "layerwise_hessian_traces.csv")
    else:
        run_id = load_run_id or RUN_ID
        path = os.path.join(RESULTS_DIR, run_id, "csv", "layerwise_hessian_traces.csv")
    logger.info(f"[LayerAblation] Hessian trace CSV path: {path}")
    return path


def _load_hessian_traces(csv_path: str) -> pd.DataFrame | None:
    if not os.path.exists(csv_path):
        logger.warning(f"[LayerAblation] Hessian trace CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df["layer_name"] = df["layer"].apply(lambda s: s[:-len(".weight")] if s.endswith(".weight") else s)
    logger.info(f"[LayerAblation] Loaded Hessian traces from {csv_path} ({len(df)} rows)")
    return df


def _combo_subset(df: pd.DataFrame, model_name: str, dataset_name: str, stage: str) -> pd.DataFrame:
    return df[(df["model"] == model_name) & (df["dataset"] == dataset_name) & (df["stage"] == stage)]


def _traces_for_combo(df: pd.DataFrame | None, model_name: str, dataset_name: str, stage: str) -> dict[str, float]:
    if df is None:
        return {}
    subset = _combo_subset(df, model_name, dataset_name, stage)
    return dict(zip(subset["layer_name"], subset["trace"]))


def _select_layers_by_trace(
    df: pd.DataFrame, model_name: str, dataset_name: str, stage: str, top_k: int
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    subset = _combo_subset(df, model_name, dataset_name, stage)
    if subset.empty:
        return [], []

    ranked_desc = subset.sort_values("trace", ascending=False)
    top = list(zip(ranked_desc["layer_name"].head(top_k), ranked_desc["trace"].head(top_k)))

    nonzero_asc = subset[subset["trace"] > 0].sort_values("trace", ascending=True)
    low = list(zip(nonzero_asc["layer_name"].head(top_k), nonzero_asc["trace"].head(top_k)))

    return top, low


# ---------------------------------------------------------------------------
# Verification: excluded layer stayed FP32, other layers were quantized
# ---------------------------------------------------------------------------

def _verify_ablation(model: nn.Module, layer_name: str, label: str) -> tuple[bool, bool]:
    resolved_name = _resolve_module_name(model, layer_name)
    if resolved_name is None:
        raise FbgemmBuildError(
            f"{label}: excluded layer '{layer_name}' not found in the converted model -- "
            f"cannot verify ablation."
        )

    named = dict(model.named_modules())
    excluded_module = named[resolved_name]

    if isinstance(excluded_module, (nnq.Conv2d, nnq.Linear)):
        raise FbgemmBuildError(
            f"{label}: excluded layer '{resolved_name}' was quantized anyway "
            f"(qconfig=None was not honored) -- ablation result is meaningless."
        )
    if not isinstance(excluded_module, (nn.Conv2d, nn.Linear)):
        raise FbgemmBuildError(
            f"{label}: excluded layer '{resolved_name}' is neither a plain nn.Conv2d/nn.Linear "
            f"nor a quantized one after conversion (got {type(excluded_module).__name__}) -- "
            f"ablation result is meaningless."
        )

    other_conv_hit = None
    other_linear_hit = None
    for name, module in named.items():
        if name == resolved_name:
            continue
        if other_conv_hit is None and isinstance(module, nnq.Conv2d):
            other_conv_hit = name
        if other_linear_hit is None and isinstance(module, nnq.Linear):
            other_linear_hit = name

    if other_conv_hit is None:
        raise FbgemmBuildError(
            f"{label}: excluding '{resolved_name}' left no OTHER Conv2d layer quantized -- "
            f"model is silently fully-unquantized, ablation result is meaningless."
        )
    if other_linear_hit is None:
        raise FbgemmBuildError(
            f"{label}: excluding '{resolved_name}' left no OTHER Linear layer quantized -- "
            f"model is silently fully-unquantized, ablation result is meaningless."
        )

    logger.info(
        f"[LayerAblation] {label}: verified -- '{resolved_name}' still fp32, "
        f"other Conv2d '{other_conv_hit}' and Linear '{other_linear_hit}' quantized"
    )
    return True, True


# ---------------------------------------------------------------------------
# CSV (append mode -- one row written to disk immediately after computation)
# ---------------------------------------------------------------------------

def _append_row(path: str, row: dict) -> None:
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"[LayerAblation] row appended -> {path} ({row['selection']}: {row['excluded_layer'] or '<none>'})")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_layer_ablation(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    ablate_top_k: int,
    ablate_layers: str | None,
    eval_subset: int | None,
    datasets: list[str] | None = None,
) -> None:
    """datasets restricts the sweep to a subset of DATASETS (e.g. one dataset,
    for a parallel per-dataset analysis run) -- default None means every
    dataset in DATASETS. Requires layerwise_hessian_traces.csv to already
    exist for --checkpoint-dir/--load-run-id (written by the checkpoint-
    metrics pipeline) -- the trace-guided top-k/low-k selection reads it."""
    datasets = datasets if datasets is not None else DATASETS
    torch.backends.quantized.engine = "fbgemm"
    num_threads = os.cpu_count() or 1
    torch.set_num_threads(num_threads)
    logger.info(f"[LayerAblation] backend=fbgemm num_threads={num_threads}")
    logger.info(f"[LayerAblation] calibration batches = {NUM_CALIBRATION_BATCHES} (same as --deploy-cpu-fbgemm)")

    resolved_checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)
    hessian_csv_path = _resolve_hessian_csv_path(checkpoint_dir, load_run_id)
    hessian_df = _load_hessian_traces(hessian_csv_path)

    os.makedirs(CSV_DIR, exist_ok=True)
    output_csv_path = os.path.join(CSV_DIR, "layer_ablation.csv")

    explicit_layer_names = (
        [name.strip() for name in ablate_layers.split(",") if name.strip()]
        if ablate_layers else None
    )
    if explicit_layer_names:
        logger.info(f"[LayerAblation] Explicit exclusion mode: {explicit_layer_names}")
    else:
        logger.info(f"[LayerAblation] Trace-guided mode: top-{ablate_top_k} + low-{ablate_top_k} (control)")

    for dataset_name in datasets:
        specs = DATASET_SPECS[dataset_name]
        channels, image_size = specs["channels"], specs["image_size"]

        try:
            train_loader, val_loader, loaded_num_classes = get_data_loaders(dataset_name)
        except Exception as exc:
            logger.warning(f"[LayerAblation] {dataset_name}: could not load dataset ({exc}) -- skipping")
            continue

        for model_name in MODELS:
            for stage in STAGES:
                label = f"{stage} {model_name}/{dataset_name}"
                logger.info(f"[LayerAblation] --- {label} ---")

                try:
                    checkpoint_path = _checkpoint_path(resolved_checkpoint_dir, stage, model_name, dataset_name)
                except FileNotFoundError as exc:
                    logger.warning(f"[LayerAblation] {label}: missing checkpoint ({exc}) -- skipping")
                    continue

                baked_model = _build_baked_model(
                    model_name=model_name,
                    checkpoint_path=checkpoint_path,
                    num_classes=loaded_num_classes,
                    channels=channels,
                    image_size=image_size,
                )
                fp32_acc = _evaluate_accuracy(baked_model, val_loader, eval_subset)
                logger.info(f"[LayerAblation] {label}: fp32_acc={fp32_acc:.2f}%")

                int8_model_all, _ = _apply_fbgemm_ptq(
                    copy.deepcopy(baked_model), train_loader, label, model_name,
                )
                _audit_quantized_modules(int8_model_all, label)
                all_quantized_acc = _evaluate_accuracy(int8_model_all, val_loader, eval_subset)
                logger.info(f"[LayerAblation] {label}: all_quantized_acc={all_quantized_acc:.2f}%")
                del int8_model_all

                _append_row(output_csv_path, {
                    "model": model_name, "dataset": dataset_name, "stage": stage,
                    "excluded_layer": "", "selection": "none", "excluded_trace": "",
                    "fp32_acc": fp32_acc, "all_quantized_acc": all_quantized_acc,
                    "ablated_acc": all_quantized_acc, "recovery_pts": 0.0,
                    "fraction_of_fp32_recovered": 0.0,
                    "excluded_layer_still_fp32": "", "other_layers_quantized": True,
                })

                # Build this combo's exclusion plan: one entry per layer to
                # ablate individually (each becomes its own conversion run).
                exclusions: list[tuple[str, object, str]] = []
                if explicit_layer_names:
                    combo_traces = _traces_for_combo(hessian_df, model_name, dataset_name, stage)
                    for layer_name in explicit_layer_names:
                        exclusions.append((layer_name, combo_traces.get(layer_name, ""), "explicit"))
                elif hessian_df is None:
                    logger.warning(
                        f"[LayerAblation] {label}: no Hessian trace CSV -- skipping top-k/low-k selection"
                    )
                else:
                    top, low = _select_layers_by_trace(hessian_df, model_name, dataset_name, stage, ablate_top_k)
                    if not top:
                        logger.warning(f"[LayerAblation] {label}: no trace rows for this model/dataset/stage")
                    exclusions.extend((name, trace, "top_trace") for name, trace in top)
                    exclusions.extend((name, trace, "low_trace") for name, trace in low)

                if exclusions:
                    plan = ", ".join(f"{name}({sel}, trace={trace})" for name, trace, sel in exclusions)
                    logger.info(f"[LayerAblation] {label}: exclusion plan = {plan}")

                fp32_minus_all = fp32_acc - all_quantized_acc

                for layer_name, trace_val, selection in exclusions:
                    excl_label = f"{label} excl={layer_name}"
                    int8_model_excl, _ = _apply_fbgemm_ptq(
                        copy.deepcopy(baked_model), train_loader, excl_label, model_name,
                        excluded_layers=frozenset({layer_name}),
                    )

                    excluded_still_fp32, others_quantized = _verify_ablation(int8_model_excl, layer_name, excl_label)

                    ablated_acc = _evaluate_accuracy(int8_model_excl, val_loader, eval_subset)
                    del int8_model_excl

                    recovery_pts = ablated_acc - all_quantized_acc
                    fraction_recovered = (
                        recovery_pts / fp32_minus_all if fp32_minus_all != 0 else float("nan")
                    )

                    logger.info(
                        f"[LayerAblation] {excl_label} ({selection}): ablated_acc={ablated_acc:.2f}% "
                        f"recovery={recovery_pts:+.2f}pts fraction_of_fp32_recovered={fraction_recovered:.3f}"
                    )

                    _append_row(output_csv_path, {
                        "model": model_name, "dataset": dataset_name, "stage": stage,
                        "excluded_layer": layer_name, "selection": selection, "excluded_trace": trace_val,
                        "fp32_acc": fp32_acc, "all_quantized_acc": all_quantized_acc,
                        "ablated_acc": ablated_acc, "recovery_pts": recovery_pts,
                        "fraction_of_fp32_recovered": fraction_recovered,
                        "excluded_layer_still_fp32": excluded_still_fp32,
                        "other_layers_quantized": others_quantized,
                    })

                del baked_model

    logger.info("[LayerAblation] === Layer ablation complete ===")
