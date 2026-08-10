"""
weight_ablation.py -- measures each layer's *weight*-quantization damage in
isolation and correlates it against that layer's precomputed weight-Hessian
trace.

Motivation: the PoT PTQ accuracy collapse is ~70pts activation-driven and
~5pts weight-driven (src/analysis/diagnose_activations.py). The weight-Hessian
story can therefore only be tested against weight-quant damage specifically.
Naively "quantizing only layer l" is invalid: a QuantizedConv2d/QuantizedLinear
carries both weight_fake_quant and act_fake_quant, so it would also turn on
that layer's activation quantizer -- for the high-outlier-factor layers the
measured damage would be activation-clipping damage the weight trace does not
predict. This mode isolates weights by disabling ALL activation quantization
first, then quantizes one layer's weights at a time.

Three parts, per model x dataset x stage (CIFAR10; resnet18_no_weights and
resnet50_no_weights required, cnn included as a bonus; PTQ, QAT):

  Part 0 (gate): compute FP32 and weights-only-all-layers accuracy up front.
    Cross-checked against KNOWN_ANCHORS (independently confirmed values from
    prior runs) when available for that combo; combos with no known anchor
    still compute and log these numbers as a sanity print, just without a
    hard pass/fail gate. A failed gate skips Parts 1-2 for that combo --
    they would otherwise be measuring a broken pipeline.
  Part 1: reconstruct the model fresh per layer, isolate that one layer's
    weight quantization (everything else FP32, all activations FP32),
    evaluate on the full validation set. Optionally also runs the
    leave-one-out complement (every layer quantized except l) for direct
    comparison against the existing conv1 exclusion result
    (src/analysis/layer_ablation.py: excluding conv1 recovered 3.07pts).
  Part 2: Spearman correlation (+ top-5 overlap, tie-immune) between
    per-layer weight_damage_pts and the precomputed weight-Hessian trace,
    reusing (not recomputing) layerwise_hessian_traces.csv via
    src/analysis/layer_ablation.py's existing loader.

Analysis only -- no torchao, no INT8 conversion, no deployment/benchmark
path. Reuses (does not duplicate) the checkpoint loader, evaluation function,
and Identity-swap helpers already established in
src/analysis/diagnose_activations.py, and the Hessian trace CSV loader in
src/analysis/layer_ablation.py.

Runs as a single local process (`python -m src.main --weight-ablation ...`),
CPU or CUDA -- prefers CUDA when available, since this sweep reconstructs
and evaluates the model fresh per layer (resnet50: ~53 layers x up to 2
evaluations x 2 stages).
"""

import os
import csv
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from src.quantization.quantizer import QuantizedConv2d, QuantizedLinear
from src.quantization.deploy_fbgemm import (
    MODELS,
    DATASETS,
    STAGES,
    _resolve_checkpoint_dir,
    _checkpoint_path,
)
from src.analysis.diagnose_activations import (
    _resolve_fp32_models_dir,
    _fp32_checkpoint_path,
    _load_quant_model,
    _load_fp32_reference,
    _disable_activation_quant,
    _disable_weight_quant,
    _append_row,
)
from src.analysis.layer_ablation import (
    _resolve_hessian_csv_path,
    _load_hessian_traces,
    _traces_for_combo,
)
from src.utility.config import CSV_DIR, DATASET_SPECS, TEST_BATCH_SIZE
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

SEED = 42

# Known-good anchors from prior independently-confirmed runs, used as a
# regression gate in Part 0: if a freshly-computed anchor for a covered
# combo drifts more than ANCHOR_TOLERANCE_PTS from these, the checkpoint
# reload / Identity-swap pipeline broke and Parts 1-2 for that combo are
# skipped rather than silently measuring a broken setup. Combos not listed
# have no known-good anchor yet -- Part 0 still computes and logs
# fp32_acc/weights_only_acc for them (e.g. for the resnet18/CIFAR10/PTQ
# structural validation run), just without a hard pass/fail check.
KNOWN_ANCHORS = {
    ("resnet50_no_weights", "CIFAR10", "PTQ"): {"fp32_acc": 80.56, "weights_only_acc": 75.46},
}
ANCHOR_TOLERANCE_PTS = 1.0

ABLATION_FIELDNAMES = [
    "model", "dataset", "stage", "layer", "hessian_trace", "fp32_acc", "isolated_acc",
    "weight_damage_pts", "leave_one_out_acc", "leave_one_out_recovery_pts",
]
CORRELATION_FIELDNAMES = [
    "model", "dataset", "stage", "n_layers_matched", "spearman_rho", "spearman_p",
    "top5_overlap", "note",
]


class WeightAblationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Eval loader: num_workers=0, pin_memory=False, shuffle=False (explicit, per
# this mode's constraints -- overrides the get_data_loaders default of
# num_workers=8 / PIN_MEMORY=True on CUDA machines).
# ---------------------------------------------------------------------------

def _build_eval_loader(dataset_name: str) -> tuple[DataLoader, int]:
    _, val_loader, num_classes = get_data_loaders(dataset_name)
    eval_loader = DataLoader(
        val_loader.dataset, batch_size=TEST_BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    return eval_loader, num_classes


# ---------------------------------------------------------------------------
# Weight-mask verification (Part 1 step 3): exactly the given set of layers
# has an active (non-Identity) weight_fake_quant, every other quantized
# layer's weight_fake_quant is Identity, and every act_fake_quant is
# Identity. One generic check covers the isolation case (one active layer),
# the leave-one-out case (all but one active), and Part 0's
# weights-only-all-layers case (all active).
# ---------------------------------------------------------------------------

def _verify_weight_mask(model: nn.Module, expected_active_layers: set[str], label: str) -> None:
    for name, module in model.named_modules():
        if not isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            continue
        if not isinstance(module.act_fake_quant, nn.Identity):
            raise WeightAblationError(
                f"{label}: '{name}.act_fake_quant' is not nn.Identity (got "
                f"{type(module.act_fake_quant).__name__}) -- activation quantization leaked "
                f"into a weight-isolation measurement, damage figure would be meaningless."
            )
        is_active = not isinstance(module.weight_fake_quant, nn.Identity)
        should_be_active = name in expected_active_layers
        if is_active != should_be_active:
            raise WeightAblationError(
                f"{label}: '{name}.weight_fake_quant' active={is_active}, expected "
                f"active={should_be_active} -- weight-mask verification failed, evaluating now "
                f"would silently measure the wrong configuration."
            )


# ---------------------------------------------------------------------------
# Part 0: anchors + regression gate
# ---------------------------------------------------------------------------

def _check_anchor_gate(
    model_name: str, dataset_name: str, stage: str, fp32_acc: float, weights_only_acc: float,
) -> tuple[bool, str | None]:
    label = f"{stage} {model_name}/{dataset_name}"
    anchor = KNOWN_ANCHORS.get((model_name, dataset_name, stage))
    if anchor is None:
        logger.info(
            f"[WeightAblation] {label}: no known-good anchor for this combo -- proceeding "
            f"without a hard regression check (fp32_acc={fp32_acc:.2f}%, "
            f"weights_only_acc={weights_only_acc:.2f}%)"
        )
        return True, None

    fp32_diff = abs(fp32_acc - anchor["fp32_acc"])
    weights_diff = abs(weights_only_acc - anchor["weights_only_acc"])
    passed = fp32_diff <= ANCHOR_TOLERANCE_PTS and weights_diff <= ANCHOR_TOLERANCE_PTS
    logger.info(
        f"[WeightAblation] {label}: anchor check -- fp32 {fp32_acc:.2f}% vs "
        f"{anchor['fp32_acc']:.2f}% (diff {fp32_diff:.2f}pt), weights_only {weights_only_acc:.2f}% "
        f"vs {anchor['weights_only_acc']:.2f}% (diff {weights_diff:.2f}pt), "
        f"tolerance={ANCHOR_TOLERANCE_PTS}pt -- {'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        return False, (
            f"anchor mismatch: fp32 diff={fp32_diff:.2f}pt, weights_only diff={weights_diff:.2f}pt "
            f"(tolerance={ANCHOR_TOLERANCE_PTS}pt)"
        )
    return True, None


# ---------------------------------------------------------------------------
# Part 1: per-layer weights-only isolation (+ optional leave-one-out)
# ---------------------------------------------------------------------------

def _run_isolation_sweep(
    model_name: str, dataset_name: str, stage: str, quant_ckpt_path: str,
    num_classes: int, channels: int, image_size: int,
    eval_loader: DataLoader, device: torch.device,
    fp32_acc: float, weights_only_all_acc: float, all_layer_names: list[str],
    hessian_traces: dict[str, float], output_csv_path: str,
    run_leave_one_out: bool = True,
) -> list[dict]:
    from src.main import evaluate

    label = f"{stage} {model_name}/{dataset_name}"
    rows: list[dict] = []

    for layer_name in all_layer_names:
        other_layers = {n for n in all_layer_names if n != layer_name}

        model, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
        model = model.to(device)
        _disable_activation_quant(model)
        _disable_weight_quant(model, layer_names=other_layers)
        _verify_weight_mask(model, {layer_name}, f"{label} isolate={layer_name}")
        isolated_acc = evaluate(model, eval_loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        weight_damage_pts = fp32_acc - isolated_acc

        row = {
            "model": model_name, "dataset": dataset_name, "stage": stage, "layer": layer_name,
            "hessian_trace": hessian_traces.get(layer_name, ""),
            "fp32_acc": fp32_acc, "isolated_acc": isolated_acc, "weight_damage_pts": weight_damage_pts,
            "leave_one_out_acc": "", "leave_one_out_recovery_pts": "",
        }

        if run_leave_one_out:
            loo_model, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
            loo_model = loo_model.to(device)
            _disable_activation_quant(loo_model)
            _disable_weight_quant(loo_model, layer_names={layer_name})
            _verify_weight_mask(loo_model, other_layers, f"{label} leave-one-out={layer_name}")
            loo_acc = evaluate(loo_model, eval_loader, device)
            del loo_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            row["leave_one_out_acc"] = loo_acc
            row["leave_one_out_recovery_pts"] = loo_acc - weights_only_all_acc

        loo_msg = (
            f" leave_one_out_recovery={row['leave_one_out_recovery_pts']:.2f}pts"
            if run_leave_one_out else ""
        )
        logger.info(
            f"[WeightAblation] {label} layer={layer_name}: isolated_acc={isolated_acc:.2f}% "
            f"weight_damage={weight_damage_pts:.2f}pts{loo_msg}"
        )

        _append_row(output_csv_path, row, ABLATION_FIELDNAMES)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Part 2: correlation with the precomputed weight-Hessian trace
# ---------------------------------------------------------------------------

def _run_correlation(
    model_name: str, dataset_name: str, stage: str,
    ablation_rows: list[dict], hessian_traces: dict[str, float], output_csv_path: str,
) -> None:
    label = f"{stage} {model_name}/{dataset_name}"

    ablation_layer_names = {r["layer"] for r in ablation_rows}
    trace_layer_names = set(hessian_traces.keys())

    only_in_trace = sorted(trace_layer_names - ablation_layer_names)
    only_in_ablation = sorted(ablation_layer_names - trace_layer_names)
    if only_in_trace:
        logger.warning(
            f"[WeightAblation] {label}: layers in Hessian trace but not in ablation set "
            f"(excluded from correlation): {only_in_trace}"
        )
    if only_in_ablation:
        logger.warning(
            f"[WeightAblation] {label}: layers in ablation set but not in Hessian trace "
            f"(excluded from correlation): {only_in_ablation}"
        )

    matched_layers = [r["layer"] for r in ablation_rows if r["layer"] in hessian_traces]
    n_matched = len(matched_layers)

    if n_matched == 0:
        note = "no matched layers between ablation set and Hessian trace set -- correlation undefined"
        logger.error(f"[WeightAblation] {label}: {note}")
        _append_row(output_csv_path, {
            "model": model_name, "dataset": dataset_name, "stage": stage,
            "n_layers_matched": 0, "spearman_rho": "", "spearman_p": "",
            "top5_overlap": "", "note": note,
        }, CORRELATION_FIELDNAMES)
        return

    damage_by_layer = {r["layer"]: r["weight_damage_pts"] for r in ablation_rows}
    traces = [hessian_traces[l] for l in matched_layers]
    damages = [damage_by_layer[l] for l in matched_layers]

    if n_matched >= 3:
        rho, p_value = spearmanr(traces, damages)
    else:
        rho, p_value = float("nan"), float("nan")
        logger.warning(
            f"[WeightAblation] {label}: only {n_matched} matched layers -- Spearman correlation "
            f"not meaningful (need >= 3), reporting NaN"
        )

    k = min(5, n_matched)
    top_k_by_trace = set(sorted(matched_layers, key=lambda l: hessian_traces[l], reverse=True)[:k])
    top_k_by_damage = set(sorted(matched_layers, key=lambda l: damage_by_layer[l], reverse=True)[:k])
    overlap = len(top_k_by_trace & top_k_by_damage)
    top5_overlap = f"{overlap}/{k}"

    logger.info(
        f"[WeightAblation] {label}: n_matched={n_matched} spearman_rho={rho:.4f} "
        f"p={p_value:.4g} top{k}_overlap={top5_overlap}"
    )

    note = ""
    if only_in_trace or only_in_ablation:
        note = f"excluded {len(only_in_trace)} trace-only + {len(only_in_ablation)} ablation-only layers"

    _append_row(output_csv_path, {
        "model": model_name, "dataset": dataset_name, "stage": stage,
        "n_layers_matched": n_matched, "spearman_rho": rho, "spearman_p": p_value,
        "top5_overlap": top5_overlap, "note": note,
    }, CORRELATION_FIELDNAMES)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_weight_ablation(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    run_leave_one_out: bool = True,
) -> None:
    from src.main import evaluate

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[WeightAblation] CUDA not available -- falling back to CPU, this will be slow.")
    logger.info(f"[WeightAblation] device={device} seed={SEED}")

    resolved_checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)
    fp32_models_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)
    hessian_csv_path = _resolve_hessian_csv_path(checkpoint_dir, load_run_id)
    hessian_df = _load_hessian_traces(hessian_csv_path)

    os.makedirs(CSV_DIR, exist_ok=True)
    ablation_csv = os.path.join(CSV_DIR, "weight_ablation.csv")
    correlation_csv = os.path.join(CSV_DIR, "weight_ablation_correlation.csv")

    for dataset_name in DATASETS:
        specs = DATASET_SPECS[dataset_name]
        channels, image_size = specs["channels"], specs["image_size"]

        try:
            eval_loader, num_classes = _build_eval_loader(dataset_name)
        except Exception as exc:
            logger.warning(f"[WeightAblation] {dataset_name}: could not load dataset ({exc}) -- skipping")
            continue

        for model_name in MODELS:
            for stage in STAGES:
                label = f"{stage} {model_name}/{dataset_name}"
                logger.info(f"[WeightAblation] === {label} ===")

                try:
                    quant_ckpt_path = _checkpoint_path(resolved_checkpoint_dir, stage, model_name, dataset_name)
                    fp32_ckpt_path = _fp32_checkpoint_path(fp32_models_dir, model_name, dataset_name)
                except FileNotFoundError as exc:
                    logger.warning(f"[WeightAblation] {label}: missing checkpoint ({exc}) -- skipping")
                    continue

                # ---- Part 0: anchors ----
                fp32_model = _load_fp32_reference(
                    model_name, fp32_ckpt_path, num_classes, channels, image_size,
                ).to(device)
                fp32_acc = evaluate(fp32_model, eval_loader, device)
                del fp32_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                wo_model, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
                wo_model = wo_model.to(device)
                all_layer_names = [
                    name for name, module in wo_model.named_modules()
                    if isinstance(module, (QuantizedConv2d, QuantizedLinear))
                ]
                _disable_activation_quant(wo_model)
                _verify_weight_mask(wo_model, set(all_layer_names), f"{label} weights-only-all")
                weights_only_all_acc = evaluate(wo_model, eval_loader, device)
                del wo_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                logger.info(
                    f"[WeightAblation] {label}: fp32_acc={fp32_acc:.2f}% "
                    f"weights_only_all_acc={weights_only_all_acc:.2f}% ({len(all_layer_names)} layers)"
                )

                gate_passed, note = _check_anchor_gate(model_name, dataset_name, stage, fp32_acc, weights_only_all_acc)
                if not gate_passed:
                    logger.error(f"[WeightAblation] {label}: GATE FAILED -- {note}. Skipping Parts 1-2.")
                    continue

                # ---- Part 1: per-layer isolation ----
                combo_traces = _traces_for_combo(hessian_df, model_name, dataset_name, stage)
                ablation_rows = _run_isolation_sweep(
                    model_name, dataset_name, stage, quant_ckpt_path, num_classes, channels, image_size,
                    eval_loader, device, fp32_acc, weights_only_all_acc, all_layer_names, combo_traces,
                    ablation_csv, run_leave_one_out=run_leave_one_out,
                )

                # ---- Part 2: correlation with the precomputed weight-Hessian trace ----
                _run_correlation(model_name, dataset_name, stage, ablation_rows, combo_traces, correlation_csv)

    logger.info("[WeightAblation] === Weight-Ablation complete ===")
