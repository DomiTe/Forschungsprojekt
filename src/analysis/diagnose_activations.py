"""
diagnose_activations.py -- isolates how much of PTQ's accuracy collapse comes
from activation quantization versus weight quantization, and locates which
layers' activation ranges are pathological.

Motivation: for resnet50/CIFAR10, PoT PTQ drops accuracy 80.56% -> 11.73%.
The same checkpoint, after bake_pot_into_standard_layers (which replaces
QuantizedConv2d/QuantizedLinear with plain nn.Conv2d/nn.Linear carrying
weight_fake_quant(weight), dropping act_fake_quant), evaluates at 75.46%.
Baking removes activation quantization, so the ~64-point gap between 75.46%
and 11.73% is attributable to activation quantization, not weights.
Layer-wise weight-Hessian analysis (src/analysis/layer_ablation.py) cannot
explain this collapse -- excluding the highest-trace layer (conv1, trace
13,161.8 = 84% of the model total) from weight quantization recovered only
3.07 points. This mode confirms the decomposition and locates the
responsible activation quantizers.

Four parts, run per model x dataset x stage (CIFAR10; cnn, resnet18_no_weights,
resnet50_no_weights; PTQ, QAT):

  Part 0 (gate): confirm the reloaded checkpoint's activation observers are
    actually calibrated. A collapse caused by an uncalibrated reload would
    look identical to a genuine PTQ property but mean something entirely
    different -- Parts 1-3 are skipped for any combo that fails this gate.
  Part 1: evaluate FP32 / full-quantized / weights-only / activations-only
    variants of the same checkpoint to decompose the accuracy loss by source.
  Part 2: per-layer calibrated activation range vs. actual batch activation
    distribution (percentiles, outlier factor) to find pathological ranges.
  Part 3: cumulative activation-quantization ablation over the layers with
    the highest outlier factor (causal test), with a low-outlier-factor
    control sweep required to attribute any recovery to range pathology
    specifically rather than to leaving any quantizer off.

Reuses (does not duplicate) the checkpoint reconstruction primitives already
established for the fbgemm/PoT pipeline: build_model, fuse_model_architectures,
replace_layers_for_quantization (src/quantization/quantizer.py), and the CPU
accuracy evaluator / MODELS / DATASETS / STAGES / checkpoint-path resolution
from src/quantization/deploy_fbgemm.py.

Runs as a single local process (`python -m src.main --diagnose-activation-quant
...`), no torchrun/SLURM/torch.distributed involved.
"""

import os
import csv
import copy
import math
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model_cnn.train import build_model
from src.quantization.quantizer import (
    QuantizedConv2d,
    QuantizedLinear,
    fuse_model_architectures,
    replace_layers_for_quantization,
)
from src.quantization.deploy_fbgemm import (
    MODELS,
    DATASETS,
    STAGES,
    _resolve_checkpoint_dir,
    _checkpoint_path,
    _evaluate_accuracy,
)
from src.analysis._seed_stats import derive_seeds, aggregate
from src.utility.config import RESULTS_DIR, RUN_ID, CSV_DIR, DATASET_SPECS
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

# A small slice of one validation batch is enough to characterize each
# layer's activation distribution; capturing all ~50+ quantized layers'
# pre-quantization tensors simultaneously via hooks on a full TEST_BATCH_SIZE
# (512) batch would be needlessly memory-heavy for resnet50.
RANGE_STATS_BATCH_SIZE = 64
CUMULATIVE_ABLATION_TOP_K = 5

LOAD_CHECK_FIELDNAMES = [
    "model", "dataset", "stage", "missing_observer_keys", "unexpected_keys_count",
    "observers_total", "observers_uncalibrated", "calibrated", "note",
]
DECOMPOSITION_FIELDNAMES = [
    "model", "dataset", "stage", "fp32_acc", "weights_only_acc", "activations_only_acc",
    "full_acc", "weight_loss_pts", "activation_loss_pts", "total_loss_pts", "dominant_source",
]
RANGES_FIELDNAMES = [
    "model", "dataset", "stage", "layer", "calib_min", "calib_max", "range_width",
    "scale", "zero_point", "act_p99", "act_p999", "act_max", "outlier_factor", "range_over_p99",
    "seed", "n_seeds", "metric_std",
]
ABLATION_FIELDNAMES = [
    "model", "dataset", "stage", "selection", "num_disabled", "newly_disabled_layer",
    "disabled_layers", "outlier_factor", "baseline_acc", "ablated_acc", "recovery_pts",
]


class DiagnoseActivationsError(RuntimeError):
    pass


def _append_row(path: str, row: dict, fieldnames: list[str]) -> None:
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"[DiagnoseActivations] row appended -> {path}")


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_fp32_models_dir(checkpoint_dir: str | None, load_run_id: str | None) -> str:
    # baseline_{model}_{dataset}_float32.pt lives in the "models" directory
    # that sits next to "quantized_models" under the same run -- same sibling-
    # directory convention used for layerwise_hessian_traces.csv in
    # src/analysis/layer_ablation.py.
    if checkpoint_dir:
        run_root = os.path.dirname(os.path.normpath(checkpoint_dir))
        path = os.path.join(run_root, "models")
    else:
        run_id = load_run_id or RUN_ID
        path = os.path.join(RESULTS_DIR, run_id, "models")
    logger.info(f"[DiagnoseActivations] FP32 baseline models directory: {path}")
    return path


def _fp32_checkpoint_path(models_dir: str, model_name: str, dataset_name: str) -> str:
    path = os.path.join(models_dir, f"baseline_{model_name}_{dataset_name}_float32.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing FP32 baseline checkpoint: {path}")
    return path


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def _load_quant_model(
    model_name: str, checkpoint_path: str, num_classes: int, channels: int, image_size: int,
) -> tuple[nn.Module, list[str], list[str]]:
    device = torch.device("cpu")
    model = build_model(num_classes=num_classes, model_name=model_name, channels=channels, image_size=image_size)
    fuse_model_architectures(model, model_name)
    replace_layers_for_quantization(model)
    model = model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, list(missing_keys), list(unexpected_keys)


def _load_fp32_reference(
    model_name: str, checkpoint_path: str, num_classes: int, channels: int, image_size: int,
) -> nn.Module:
    device = torch.device("cpu")
    model = build_model(num_classes=num_classes, model_name=model_name, channels=channels, image_size=image_size)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Part 0: observer calibration gate
# ---------------------------------------------------------------------------

def _check_observer_calibration(
    model: nn.Module, missing_keys: list[str], unexpected_keys: list[str], label: str,
) -> dict:
    missing_observer_keys = [k for k in missing_keys if "activation_post_process" in k]
    if missing_observer_keys:
        logger.warning(
            f"[DiagnoseActivations] {label}: load_state_dict missing_keys includes observer "
            f"buffers -- reload is uncalibrated: {missing_observer_keys}"
        )

    observers_total = 0
    observers_uncalibrated = 0
    dir_dumped = False

    for name, module in model.named_modules():
        if not isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            continue
        act_fq = module.act_fake_quant
        if not hasattr(act_fq, "activation_post_process"):
            if not dir_dumped:
                logger.error(
                    f"[DiagnoseActivations] {label}: '{name}.act_fake_quant' has no "
                    f"'activation_post_process' attribute. dir(act_fake_quant) = {dir(act_fq)}"
                )
                dir_dumped = True
            raise DiagnoseActivationsError(
                f"{label}: observer path 'act_fake_quant.activation_post_process' absent on "
                f"'{name}' -- see dir() dump above, do not guess the path."
            )

        observer = act_fq.activation_post_process
        observers_total += 1
        min_val = observer.min_val.item()
        max_val = observer.max_val.item()
        is_calibrated = math.isfinite(min_val) and math.isfinite(max_val) and min_val < max_val
        logger.info(
            f"[DiagnoseActivations] {label}: {name}.act_fake_quant.activation_post_process "
            f"min_val={min_val} max_val={max_val} calibrated={is_calibrated}"
        )
        if not is_calibrated:
            observers_uncalibrated += 1

    calibrated = (not missing_observer_keys) and (observers_uncalibrated == 0)
    return {
        "missing_observer_keys": missing_observer_keys,
        "unexpected_keys_count": len(unexpected_keys),
        "observers_total": observers_total,
        "observers_uncalibrated": observers_uncalibrated,
        "calibrated": calibrated,
    }


def _run_load_check(
    model_name: str, dataset_name: str, stage: str,
    loaded_model: nn.Module, missing_keys: list[str], unexpected_keys: list[str],
    output_csv_path: str,
) -> bool:
    label = f"{stage} {model_name}/{dataset_name}"
    logger.info(f"[DiagnoseActivations] {label}: missing_keys={missing_keys}")
    logger.info(f"[DiagnoseActivations] {label}: unexpected_keys={unexpected_keys}")

    try:
        gate_info = _check_observer_calibration(loaded_model, missing_keys, unexpected_keys, label)
    except DiagnoseActivationsError as exc:
        _append_row(output_csv_path, {
            "model": model_name, "dataset": dataset_name, "stage": stage,
            "missing_observer_keys": "", "unexpected_keys_count": len(unexpected_keys),
            "observers_total": "", "observers_uncalibrated": "",
            "calibrated": "no", "note": f"ERROR: {exc}",
        }, LOAD_CHECK_FIELDNAMES)
        logger.error(f"[DiagnoseActivations] {label}: {exc} -- GATE FAILED, skipping this combo.")
        return False

    note = "" if gate_info["calibrated"] else (
        "GATE FAILED: reload is uncalibrated (missing observer buffers and/or observers at "
        "init sentinels) -- Parts 1-3 skipped for this combo, would measure a loading bug."
    )
    _append_row(output_csv_path, {
        "model": model_name, "dataset": dataset_name, "stage": stage,
        "missing_observer_keys": ";".join(gate_info["missing_observer_keys"]),
        "unexpected_keys_count": gate_info["unexpected_keys_count"],
        "observers_total": gate_info["observers_total"],
        "observers_uncalibrated": gate_info["observers_uncalibrated"],
        "calibrated": "yes" if gate_info["calibrated"] else "no",
        "note": note,
    }, LOAD_CHECK_FIELDNAMES)

    if not gate_info["calibrated"]:
        logger.error(f"[DiagnoseActivations] {label}: {note}")
        return False

    logger.info(f"[DiagnoseActivations] {label}: GATE PASSED -- observers calibrated, proceeding to Parts 1-3")
    return True


# ---------------------------------------------------------------------------
# Identity-based quantizer disabling (uniform for both quantizer types)
# ---------------------------------------------------------------------------

def _disable_activation_quant(model: nn.Module, layer_names: set[str] | None = None) -> list[str]:
    disabled = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            if layer_names is None or name in layer_names:
                module.act_fake_quant = nn.Identity()
                disabled.append(name)
    return disabled


def _disable_weight_quant(model: nn.Module, layer_names: set[str] | None = None) -> list[str]:
    disabled = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            if layer_names is None or name in layer_names:
                module.weight_fake_quant = nn.Identity()
                disabled.append(name)
    return disabled


def _verify_identity_swap(model: nn.Module, attr_name: str, layer_names: list[str], label: str) -> None:
    named = dict(model.named_modules())
    for name in layer_names:
        target = getattr(named[name], attr_name)
        if not isinstance(target, nn.Identity):
            raise DiagnoseActivationsError(
                f"{label}: '{name}.{attr_name}' was not replaced with nn.Identity "
                f"(got {type(target).__name__}) -- modification did not take effect, "
                f"evaluating now would silently measure an unmodified model."
            )


# ---------------------------------------------------------------------------
# Part 1: damage decomposition
# ---------------------------------------------------------------------------

def _run_decomposition(
    model_name: str, dataset_name: str, stage: str,
    loaded_model: nn.Module, fp32_model: nn.Module, val_loader, eval_subset: int | None,
    output_csv_path: str,
) -> float:
    label = f"{stage} {model_name}/{dataset_name}"

    fp32_acc = _evaluate_accuracy(fp32_model, val_loader, eval_subset)

    full_model = copy.deepcopy(loaded_model)
    full_acc = _evaluate_accuracy(full_model, val_loader, eval_subset)
    del full_model

    weights_only_model = copy.deepcopy(loaded_model)
    act_layers = _disable_activation_quant(weights_only_model)
    _verify_identity_swap(weights_only_model, "act_fake_quant", act_layers, f"{label} weights-only")
    weights_only_acc = _evaluate_accuracy(weights_only_model, val_loader, eval_subset)
    del weights_only_model

    activations_only_model = copy.deepcopy(loaded_model)
    weight_layers = _disable_weight_quant(activations_only_model)
    _verify_identity_swap(activations_only_model, "weight_fake_quant", weight_layers, f"{label} activations-only")
    activations_only_acc = _evaluate_accuracy(activations_only_model, val_loader, eval_subset)
    del activations_only_model

    weight_loss_pts = fp32_acc - weights_only_acc
    activation_loss_pts = fp32_acc - activations_only_acc
    total_loss_pts = fp32_acc - full_acc
    if activation_loss_pts > weight_loss_pts:
        dominant_source = "activation"
    elif weight_loss_pts > activation_loss_pts:
        dominant_source = "weight"
    else:
        dominant_source = "tied"

    logger.info(
        f"[DiagnoseActivations] {label}: fp32={fp32_acc:.2f}% full={full_acc:.2f}% "
        f"weights_only={weights_only_acc:.2f}% activations_only={activations_only_acc:.2f}% "
        f"-- weight_loss={weight_loss_pts:.2f}pts activation_loss={activation_loss_pts:.2f}pts "
        f"dominant={dominant_source}"
    )

    _append_row(output_csv_path, {
        "model": model_name, "dataset": dataset_name, "stage": stage,
        "fp32_acc": fp32_acc, "weights_only_acc": weights_only_acc,
        "activations_only_acc": activations_only_acc, "full_acc": full_acc,
        "weight_loss_pts": weight_loss_pts, "activation_loss_pts": activation_loss_pts,
        "total_loss_pts": total_loss_pts, "dominant_source": dominant_source,
    }, DECOMPOSITION_FIELDNAMES)

    return full_acc


# ---------------------------------------------------------------------------
# Part 2: per-layer activation range statistics
# ---------------------------------------------------------------------------

def _get_observer(act_fq: nn.Module, label: str, layer_name: str):
    if not hasattr(act_fq, "activation_post_process"):
        logger.error(
            f"[DiagnoseActivations] {label}: '{layer_name}.act_fake_quant' has no "
            f"'activation_post_process' attribute. dir(act_fake_quant) = {dir(act_fq)}"
        )
        raise DiagnoseActivationsError(
            f"{label}: observer path 'act_fake_quant.activation_post_process' absent on "
            f"'{layer_name}' -- see dir() dump above, do not guess the path."
        )
    return act_fq.activation_post_process


def _collect_activation_tensors(model: nn.Module, batch_input: torch.Tensor) -> dict[str, torch.Tensor]:
    # Hooked on act_fake_quant's INPUT (a forward_pre_hook), not its output --
    # the input is the raw pre-quantization conv/linear activation, i.e. the
    # actual signal the calibrated observer range was fit to. The output
    # would already be clamped/dequantized to within the calibrated range by
    # construction, making any outlier/range comparison against that range
    # trivially bounded and meaningless.
    tensors: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_name: str):
        def hook(module, inputs):
            tensors[layer_name] = inputs[0].detach().flatten().float()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            handles.append(module.act_fake_quant.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(batch_input)

    for h in handles:
        h.remove()

    return tensors


def _draw_range_batch(val_loader, seed: int, single_seed_mode: bool) -> torch.Tensor:
    """
    single_seed_mode (n_seeds==1): the historical, unconditional draw --
    next(iter(val_loader)), the loader's own first batch, no shuffling, no
    seed dependence at all. This is what makes n_seeds=1 reproduce
    pre-n_seeds behavior bit-exact regardless of base_seed for this phase
    specifically (unlike the probe-seed phases, which still depend on
    base_seed's value at n_seeds=1).

    Multi-seed (n_seeds>1): a fresh shuffle=True loader over the same
    dataset, ambient torch RNG seeded immediately before the draw -- the
    --n-seeds variance pass's stochastic source for this phase (val-batch
    selection), per spec.
    """
    if single_seed_mode:
        inputs, _ = next(iter(val_loader))
        return inputs
    torch.manual_seed(seed)
    shuffled_loader = DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    inputs, _ = next(iter(shuffled_loader))
    return inputs


_RANGE_METRICS = ("calib_min", "calib_max", "range_width", "scale", "zero_point", "act_p99", "act_p999", "act_max", "outlier_factor", "range_over_p99")

# torch.quantile refuses tensors with more elements than this (2^24). ImageNet100's
# 224x224 inputs make early-layer activation tensors far larger than CIFAR10's
# (32x32) ever were, so this is routinely hit there. Subsample with a fixed
# generator seed for reproducibility -- the exact p99/p999 of a >16M-element
# activation tensor from a fixed subsample is stable enough for range diagnostics.
_QUANTILE_MAX_ELEMENTS = 16_000_000
_QUANTILE_SUBSAMPLE_SEED = 0


def _safe_quantile(tensor: torch.Tensor, q: float) -> float:
    if tensor.numel() > _QUANTILE_MAX_ELEMENTS:
        generator = torch.Generator().manual_seed(_QUANTILE_SUBSAMPLE_SEED)
        idx = torch.randperm(tensor.numel(), generator=generator)[:_QUANTILE_MAX_ELEMENTS]
        tensor = tensor[idx]
    return torch.quantile(tensor, q).item()


def _run_range_analysis(
    model_name: str, dataset_name: str, stage: str,
    loaded_model: nn.Module, val_loader, output_csv_path: str, seeds: list[int],
) -> list[dict]:
    label = f"{stage} {model_name}/{dataset_name}"
    single_seed_mode = (len(seeds) == 1)

    per_seed_layer_metrics: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        inputs = _draw_range_batch(val_loader, seed, single_seed_mode)
        inputs = inputs[:RANGE_STATS_BATCH_SIZE]
        activation_tensors = _collect_activation_tensors(loaded_model, inputs)

        layer_metrics: dict[str, dict] = {}
        for name, module in loaded_model.named_modules():
            if not isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                continue

            observer = _get_observer(module.act_fake_quant, label, name)
            calib_min = observer.min_val.item()
            calib_max = observer.max_val.item()
            range_width = calib_max - calib_min
            scale_t, zero_point_t = observer.calculate_qparams()
            scale = scale_t.item()
            zero_point = int(zero_point_t.item())

            tensor = activation_tensors.get(name)
            if tensor is None or tensor.numel() == 0:
                logger.warning(f"[DiagnoseActivations] {label} seed={seed}: no activation samples captured for '{name}', skipping")
                continue

            act_p99 = _safe_quantile(tensor, 0.99)
            act_p999 = _safe_quantile(tensor, 0.999)
            act_max = tensor.max().item()
            outlier_factor = (act_max / act_p99) if act_p99 != 0 else float("inf")
            range_over_p99 = (range_width / act_p99) if act_p99 != 0 else float("inf")

            metrics = {
                "calib_min": calib_min, "calib_max": calib_max, "range_width": range_width,
                "scale": scale, "zero_point": zero_point,
                "act_p99": act_p99, "act_p999": act_p999, "act_max": act_max,
                "outlier_factor": outlier_factor, "range_over_p99": range_over_p99,
            }
            layer_metrics[name] = metrics
            _append_row(output_csv_path, {
                "model": model_name, "dataset": dataset_name, "stage": stage, "layer": name,
                **metrics, "seed": seed, "n_seeds": len(seeds), "metric_std": "",
            }, RANGES_FIELDNAMES)

        per_seed_layer_metrics[seed] = layer_metrics

    # Aggregate across seeds per layer -- mean for every batch-dependent
    # metric (calib_min/max/scale/zero_point aren't actually batch-
    # dependent, since they come from the calibrated observer, not the
    # drawn batch; averaging them is a harmless no-op when they're constant
    # across seeds). metric_std reports the std of outlier_factor
    # specifically -- the one metric Part 3's cumulative ablation actually
    # ranks/selects layers by, and this schema's single std column.
    all_layers: set[str] = set()
    for lm in per_seed_layer_metrics.values():
        all_layers.update(lm.keys())

    rows = []
    for name in sorted(all_layers):
        present_seeds = [s for s in seeds if name in per_seed_layer_metrics.get(s, {})]
        if not present_seeds:
            continue
        agg = {}
        for metric in _RANGE_METRICS:
            vals = [per_seed_layer_metrics[s][name][metric] for s in present_seeds]
            agg[metric], _ = aggregate(vals)
        _, outlier_std = aggregate([per_seed_layer_metrics[s][name]["outlier_factor"] for s in present_seeds])

        row = {
            "model": model_name, "dataset": dataset_name, "stage": stage, "layer": name,
            **agg, "seed": "aggregate", "n_seeds": len(present_seeds), "metric_std": outlier_std,
        }
        _append_row(output_csv_path, row, RANGES_FIELDNAMES)
        rows.append(row)

    ranked_desc = sorted(rows, key=lambda r: r["outlier_factor"], reverse=True)
    worst10 = ranked_desc[:10]
    logger.info(
        f"[DiagnoseActivations] {label}: worst 10 layers by outlier factor (n_seeds={len(seeds)}) -- "
        + ", ".join(f"{r['layer']}({r['outlier_factor']:.2f}+/-{r['metric_std']:.2f})" for r in worst10)
    )

    return ranked_desc


# ---------------------------------------------------------------------------
# Part 3: cumulative activation-quantization ablation
# ---------------------------------------------------------------------------

def _run_cumulative_ablation(
    model_name: str, dataset_name: str, stage: str,
    loaded_model: nn.Module, val_loader, eval_subset: int | None,
    baseline_acc: float, ranked_layers: list[dict],
    output_csv_path: str,
) -> None:
    label = f"{stage} {model_name}/{dataset_name}"

    top_outlier = ranked_layers[:CUMULATIVE_ABLATION_TOP_K]
    low_outlier = list(reversed(ranked_layers))[:CUMULATIVE_ABLATION_TOP_K]

    for selection, ordered in (
        ("cumulative_top_outlier", top_outlier),
        ("cumulative_low_outlier", low_outlier),
    ):
        disabled_so_far: list[str] = []
        for k, row in enumerate(ordered, start=1):
            layer_name = row["layer"]
            outlier_factor = row["outlier_factor"]
            disabled_so_far.append(layer_name)

            model_copy = copy.deepcopy(loaded_model)
            _disable_activation_quant(model_copy, layer_names=set(disabled_so_far))
            _verify_identity_swap(
                model_copy, "act_fake_quant", disabled_so_far, f"{label} {selection} k={k}"
            )
            ablated_acc = _evaluate_accuracy(model_copy, val_loader, eval_subset)
            del model_copy

            recovery_pts = ablated_acc - baseline_acc
            logger.info(
                f"[DiagnoseActivations] {label} {selection} k={k} (+{layer_name}): "
                f"ablated_acc={ablated_acc:.2f}% recovery={recovery_pts:+.2f}pts"
            )

            _append_row(output_csv_path, {
                "model": model_name, "dataset": dataset_name, "stage": stage,
                "selection": selection, "num_disabled": k, "newly_disabled_layer": layer_name,
                "disabled_layers": ";".join(disabled_so_far), "outlier_factor": outlier_factor,
                "baseline_acc": baseline_acc, "ablated_acc": ablated_acc, "recovery_pts": recovery_pts,
            }, ABLATION_FIELDNAMES)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_diagnose_activation_quant(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    eval_subset: int | None,
    datasets: list[str] | None = None,
    n_seeds: int = 1,
    base_seed: int = 42,
) -> None:
    """datasets restricts the sweep to a subset of DATASETS (e.g. one dataset,
    for a parallel per-dataset analysis run) -- default None means every
    dataset in DATASETS.

    n_seeds/base_seed only affect Part 2 (per-layer activation range
    statistics, which depends on val-batch selection). Parts 0/1/3 are
    deterministic on a fixed model+dataset (full-val-set or fixed-subset
    evaluation, no probe/batch stochasticity) and are left untouched --
    they always run their single existing deterministic pass. At n_seeds=1,
    Part 2 also reproduces its pre-n_seeds behavior bit-exact regardless of
    base_seed (see _draw_range_batch's single_seed_mode)."""
    datasets = datasets if datasets is not None else DATASETS
    seeds = derive_seeds(base_seed, n_seeds)
    resolved_checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)
    fp32_models_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)

    os.makedirs(CSV_DIR, exist_ok=True)
    load_check_csv = os.path.join(CSV_DIR, "activation_load_check.csv")
    decomposition_csv = os.path.join(CSV_DIR, "activation_decomposition.csv")
    ranges_csv = os.path.join(CSV_DIR, "activation_ranges.csv")
    ablation_csv = os.path.join(CSV_DIR, "activation_ablation.csv")

    for dataset_name in datasets:
        specs = DATASET_SPECS[dataset_name]
        channels, image_size = specs["channels"], specs["image_size"]

        try:
            train_loader, val_loader, num_classes = get_data_loaders(dataset_name)
        except Exception as exc:
            logger.warning(f"[DiagnoseActivations] {dataset_name}: could not load dataset ({exc}) -- skipping")
            continue

        for model_name in MODELS:
            for stage in STAGES:
                label = f"{stage} {model_name}/{dataset_name}"
                logger.info(f"[DiagnoseActivations] === {label} ===")

                try:
                    quant_ckpt_path = _checkpoint_path(resolved_checkpoint_dir, stage, model_name, dataset_name)
                    fp32_ckpt_path = _fp32_checkpoint_path(fp32_models_dir, model_name, dataset_name)
                except FileNotFoundError as exc:
                    logger.warning(f"[DiagnoseActivations] {label}: missing checkpoint ({exc}) -- skipping")
                    continue

                # ---- Part 0: gate ----
                loaded_model, missing_keys, unexpected_keys = _load_quant_model(
                    model_name, quant_ckpt_path, num_classes, channels, image_size,
                )
                gate_passed = _run_load_check(
                    model_name, dataset_name, stage, loaded_model, missing_keys, unexpected_keys, load_check_csv,
                )
                if not gate_passed:
                    del loaded_model
                    continue

                # ---- Part 1: decomposition ----
                fp32_model = _load_fp32_reference(model_name, fp32_ckpt_path, num_classes, channels, image_size)
                full_acc = _run_decomposition(
                    model_name, dataset_name, stage, loaded_model, fp32_model, val_loader, eval_subset,
                    decomposition_csv,
                )
                del fp32_model

                # ---- Part 2: per-layer activation range statistics ----
                ranked_layers = _run_range_analysis(
                    model_name, dataset_name, stage, loaded_model, val_loader, ranges_csv, seeds,
                )

                # ---- Part 3: cumulative activation ablation (top vs. low outlier control) ----
                _run_cumulative_ablation(
                    model_name, dataset_name, stage, loaded_model, val_loader, eval_subset,
                    full_acc, ranked_layers, ablation_csv,
                )

                del loaded_model

    logger.info("[DiagnoseActivations] === Diagnose-Activation-Quant complete ===")
