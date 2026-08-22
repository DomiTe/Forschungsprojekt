"""
weight_ablation_canonical.py -- P1, revised: tests whether each layer's
weight-only PoT quantization damage is predicted by (a) the raw canonical
weight-Hessian trace, (b) the weight-quantization perturbation ||delta W||^2
alone, or (c) the HAWQ product Tr(H)*||delta W||^2.

Motivation: relock_traces.py (src/analysis/relock_traces.py) showed the raw
weight-Hessian trace does NOT single out conv1 -- on resnet50 it is below the
model median in the unfused basis. Yet conv1 is still the dominant
weight-quant damage layer in the original (pre-relock) weight_ablation.py
sweep (~3 of the model's ~5 total weight-damage points). Raw trace therefore
does not explain the damage. HAWQ (Dong et al., "HAWQ-V2", NeurIPS 2020)
predicts per-layer quantization sensitivity by Tr(H)*||delta W||^2, not raw
Tr(H) alone -- and conv1 plausibly has both high curvature-per-parameter and
a large PoT weight-quantization error (a 7x7x3-ish stem quantizes poorly
under a narrow power-of-two grid). This mode decides, honestly and without
picking the best-looking predictor after the fact, which of the three
candidates (if any) actually predicts weight-only damage.

Three parts, per model x stage (CIFAR10; resnet18_no_weights and
resnet50_no_weights required, cnn optional -- too few layers to correlate;
PTQ required, QAT optional -- skipped with a warning if its checkpoint is
missing):

  Part 0 (gate): reuses _ablation_common.py's path-equivalence gate
    unchanged -- the act_fake_quant->Identity weights-only construction must
    match bake_pot_into_standard_layers's weights-only construction within
    PATH_EQUIVALENCE_TOLERANCE_PTS, or the isolation sweep is unsound and is
    skipped. Then a second, NEW alignment gate: the canonical fp32_fused
    trace's layer set (read from canonical_traces.csv, written by
    relock_traces.py's frozen config) must align bijectively -- by name AND
    by weight shape -- with the ablation layer set (this model's
    QuantizedConv2d/QuantizedLinear modules). Any mismatch is printed in
    full and stops that model/stage combo; no fuzzy fallback.
  Part 1: reconstructs the model fresh per layer (as in P1), isolates that
    one layer's weight quantization (every other layer's weight_fake_quant
    and every layer's act_fake_quant -> Identity), evaluates on the full,
    fixed val set, and records weight_damage_pts = fp32_acc - isolated_acc.
    Pure forward evaluation -- no Hessian-vector products here, so
    torch.no_grad() is used throughout (unlike the trace-estimation modes).
  Part 2: for each layer, computes delta_w_sq = ||W_fused - weight_fake_quant
    (W_fused)||^2 on that layer's ACTUAL fused weights (the exact PoT
    quantization error Part 1's ablation applied), plus a size-normalised
    delta_w_sq_per_param.
  Part 3: Spearman rho (+ top-5 overlap, conv1's rank under each predictor
    vs. its rank by damage) between weight_damage_pts and each of
    raw_trh (canonical fp32_fused Tr(H)), dwsq (||delta W||^2 alone), and
    trh_dwsq (their product) -- all three reported, never just the
    best-looking one.

Reuses (does not duplicate): the checkpoint loader (_load_quant_model,
_load_fp32_reference), bake_pot_into_standard_layers (src/main.py, deferred
import to avoid a circular import -- same pattern used by
src/analysis/_ablation_common.py and src/quantization/deploy_fbgemm.py), the
evaluation function (src/main.py's evaluate), and the Identity-swap helpers
(_disable_activation_quant, _disable_weight_quant, _verify_identity_swap),
all from src/analysis/diagnose_activations.py; the path-equivalence gate
(_run_part0), the weight-mask verifier (_verify_weight_mask), and the robust
checkpoint resolver (_resolve_checkpoint_robust), all from
src/analysis/_ablation_common.py.

Analysis only -- no torchao/deployment code. Runs as a single local process
(`python -m src.main --weight-ablation-canonical ...`), no SLURM/torchrun
required; prefers CUDA (A100 in production).

Reference: Dong, Yao, Gholami, Mahoney & Keutzer, "HAWQ-V2: Hessian Aware
trace-Weighted Quantization of Neural Networks" (NeurIPS 2020).
"""

import os
import csv
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import pandas as pd

from src.quantization.quantizer import QuantizedConv2d, QuantizedLinear
from src.quantization.deploy_fbgemm import _resolve_checkpoint_dir
from src.model_cnn.train import _evaluate
from src.analysis.diagnose_activations import (
    _resolve_fp32_models_dir,
    _load_quant_model,
    _load_fp32_reference,
    _disable_activation_quant,
    _disable_weight_quant,
    _verify_identity_swap,
    _append_row,
    DiagnoseActivationsError,
)
from src.analysis._ablation_common import (
    _resolve_checkpoint_robust,
    WeightAblationCheckpointError,
    _run_part0,
    _verify_weight_mask,
    _build_eval_loader,
)
from src.utility.config import CSV_DIR, DATASET_SPECS

logger = logging.getLogger(__name__)

DATASETS = ["CIFAR10", "IMAGENET100"]
REQUIRED_MODELS = ["resnet18_no_weights", "resnet50_no_weights", "cnn"]   # resnet18 first -- run/report order per spec
STAGES = ["PTQ", "QAT"]                                            # QAT optional, skipped with a warning if missing

SEED = 42

ABLATION_FIELDNAMES = [
    "model", "dataset", "stage", "layer", "hessian_trace_fused", "delta_w_sq", "delta_w_sq_per_param",
    "trh_times_dwsq", "fp32_acc", "isolated_acc", "weight_damage_pts", "abs_weight_damage_pts",
]
# Signed vs. |damage| (see module docstring addendum below): PTQ damage is
# mostly positive (quant hurts) and QAT damage is mostly negative (the QAT
# model has adapted TO the quantized weights, so isolating one layer's quant
# noise on an otherwise-FP32 forward pass reads as an IMPROVEMENT), but the
# underlying quantity in both is the same sign-agnostic one: how much does
# this layer's quantization state matter. abs_weight_damage_pts is the
# primary correlation target (--damage-mode default "both" still reports
# signed alongside it, never dropped -- the sign is what carries the
# PTQ-vs-QAT distinction).
CORRELATION_FIELDNAMES = [
    "model", "dataset", "stage", "predictor", "n_layers", "damage_mode",
    "spearman_rho_abs", "spearman_p_abs", "top5_overlap_abs", "conv1_damage_rank_abs",
    "spearman_rho_signed", "spearman_p_signed", "top5_overlap_signed", "conv1_damage_rank_signed",
    "conv1_predictor_rank",
]

PREDICTORS = ["raw_trh", "dwsq", "trh_dwsq"]
DAMAGE_MODES = ("signed", "abs", "both")
PREDICTOR_SOURCE_COLUMNS = {"raw_trh": "hessian_trace_fused", "dwsq": "delta_w_sq", "trh_dwsq": "trh_times_dwsq"}

# ---------------------------------------------------------------------------
# Loss-based damage extension (see module docstring addendum far below):
# adds fp32_loss/isolated_loss/loss_damage/abs_loss_damage alongside the
# existing accuracy-only columns, incrementally written and resumable.
# ---------------------------------------------------------------------------

LOSS_ABLATION_FIELDNAMES = [
    "model", "dataset", "stage", "layer",
    "fp32_acc", "isolated_acc", "weight_damage_pts", "abs_weight_damage_pts",
    "fp32_loss", "isolated_loss", "loss_damage", "abs_loss_damage",
]
ACCURACY_MISMATCH_FIELDNAMES = [
    "model", "dataset", "stage", "layer", "field", "existing_value", "recomputed_value", "abs_diff",
]
LOSS_CORRELATION_FIELDNAMES = [
    "model", "dataset", "stage", "predictor", "n_layers", "damage_mode", "status",
    "spearman_rho_abs", "spearman_p_abs", "top5_overlap_abs", "conv1_damage_rank_abs",
    "spearman_rho_signed", "spearman_p_signed", "top5_overlap_signed", "conv1_damage_rank_signed",
    "conv1_predictor_rank",
]

# Same tolerance philosophy as PATH_EQUIVALENCE_TOLERANCE_PTS in
# _ablation_common.py: this reuses the exact same eval path (model, loader,
# order, determinism), so any drift beyond float noise means something is
# genuinely different between this sweep and the stored accuracy-only run,
# not just re-run jitter.
ACC_MATCH_TOLERANCE_PTS = 0.01

# Cheapest-first combo order (Part 6 of the task spec): CIFAR10 is fast
# regardless of model order (~9 min total for all 3 models x both stages in
# the original accuracy-only run), so it keeps REQUIRED_MODELS' existing
# resnet18/resnet50/cnn order; IMAGENET100 is reordered smallest-network-first
# (cnn, resnet18, resnet50) so a partial overnight run leaves only the single
# most expensive combo (resnet50/IMAGENET100, 54 layers) possibly unfinished.
_IMAGENET100_MODEL_ORDER = ["cnn", "resnet18_no_weights", "resnet50_no_weights"]

LOSS_COMBO_ORDER: list[tuple[str, str, str]] = (
    [("CIFAR10", model, stage) for stage in STAGES for model in REQUIRED_MODELS]
    + [("IMAGENET100", model, stage) for model in _IMAGENET100_MODEL_ORDER for stage in STAGES]
)


class WeightAblationCanonicalError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Canonical trace source (relock_traces.py's frozen-config output)
# ---------------------------------------------------------------------------

def _resolve_trace_config_path(canonical_traces_csv: str) -> str:
    # trace_config.json is written as a sibling of the "csv" directory that
    # canonical_traces.csv lives in (results/<RUN_ID>/trace_config.json vs.
    # results/<RUN_ID>/csv/canonical_traces.csv) -- same sibling-directory
    # convention as every other per-run artifact in this project.
    run_root = os.path.dirname(os.path.dirname(os.path.normpath(canonical_traces_csv)))
    return os.path.join(run_root, "trace_config.json")


def _log_trace_config(canonical_traces_csv: str) -> None:
    path = _resolve_trace_config_path(canonical_traces_csv)
    if not os.path.exists(path):
        logger.warning(
            f"[WeightAblationCanonical] no sibling trace_config.json found at {path} for "
            f"{canonical_traces_csv} -- proceeding without confirming the frozen config's "
            f"basis/probe_seed, but the canonical_traces.csv data itself is still used as-is."
        )
        return
    import json
    with open(path) as f:
        config = json.load(f)
    basis = config.get("basis", {}).get("canonical_for_cross_stage_comparison", "?")
    seed = config.get("estimator", {}).get("probe_seed", "?")
    logger.info(f"[WeightAblationCanonical] canonical trace source config: {path} (basis={basis}, probe_seed={seed})")


def _load_canonical_traces(canonical_traces_csv: str, model_name: str, dataset_name: str, variant: str) -> dict[str, tuple[float, str]]:
    """
    Returns {canonical_layer: (trace_raw, weight_shape_str)} for the given
    (model, dataset, variant).

    Since the --n-seeds pass (relock_traces.py), canonical_traces.csv can
    hold multiple rows per layer -- one per probe seed plus one
    seed="aggregate" mean/std row. Filtering to seed=="aggregate" when that
    column is present is required, not optional: without it, the dict
    comprehension below would keep whichever row happens to appear last for
    a given layer (a single noisy per-seed draw, not the mean) --
    non-deterministic-feeling and silently wrong. Falls back to using every
    row as-is for older canonical_traces.csv files that predate the "seed"
    column (single row per layer already, so the filter is a no-op there).
    """
    if not os.path.exists(canonical_traces_csv):
        raise WeightAblationCanonicalError(f"canonical traces CSV not found: {canonical_traces_csv}")
    df = pd.read_csv(canonical_traces_csv)
    if "seed" in df.columns:
        df = df[df["seed"] == "aggregate"]
    subset = df[(df["model"] == model_name) & (df["dataset"] == dataset_name) & (df["variant"] == variant)]
    if subset.empty:
        raise WeightAblationCanonicalError(
            f"no canonical traces for model={model_name} dataset={dataset_name} variant={variant} in {canonical_traces_csv}"
        )
    return {row["canonical_layer"]: (row["trace_raw"], row["weight_shape"]) for _, row in subset.iterrows()}


# ---------------------------------------------------------------------------
# Part 0: alignment gate (canonical trace layers <-> ablation layers)
# ---------------------------------------------------------------------------

def _align_canonical_to_ablation(
    model_name: str, canonical_traces: dict[str, tuple[float, str]], all_layer_names: list[str], model: nn.Module,
) -> dict[str, float]:
    """
    Verifies the canonical trace layer set is bijective with the ablation
    layer set (this model's QuantizedConv2d/QuantizedLinear module names)
    AND that each layer's recorded canonical weight_shape matches the live
    model's actual weight shape. Prints the full table and raises on any
    mismatch -- no fuzzy fallback, mirroring quant_induced_trace.py's Part 0
    mapping gate.

    Returns {layer_name: trace_raw} on success.
    """
    named = dict(model.named_modules())
    canon_names = set(canonical_traces.keys())
    ablation_names = set(all_layer_names)

    rows = []
    ok = canon_names == ablation_names
    for name in sorted(canon_names | ablation_names):
        in_canon = name in canon_names
        in_ablation = name in ablation_names
        shape_match = None
        if in_canon and in_ablation:
            recorded_shape = canonical_traces[name][1]
            live_shape = str(tuple(named[name].weight.shape))
            shape_match = (recorded_shape == live_shape)
            ok = ok and shape_match
        rows.append((name, in_canon, in_ablation, shape_match))

    header = f"{'layer':<40} {'in_canonical':<13} {'in_ablation':<12} shape_match"
    logger.info(
        f"[WeightAblationCanonical] {model_name}: Part 0 canonical<->ablation alignment table "
        f"({len(canon_names)} canonical layers vs {len(ablation_names)} ablation layers):\n" + header + "\n"
        + "\n".join(f"{n:<40} {ic!s:<13} {ia!s:<12} {sm!s}" for n, ic, ia, sm in rows)
    )

    if not ok:
        raise WeightAblationCanonicalError(
            f"{model_name}: Part 0 canonical<->ablation alignment gate FAILED -- layer sets are "
            f"not bijective and/or a weight shape mismatched (see table above). Refusing fuzzy "
            f"name fallback."
        )

    logger.info(f"[WeightAblationCanonical] {model_name}: Part 0 alignment gate PASSED -- {len(canon_names)} layers bijective and shape-matched")
    return {name: canonical_traces[name][0] for name in canon_names}


# ---------------------------------------------------------------------------
# Part 1: weight-only isolation damage
# ---------------------------------------------------------------------------

def _run_isolation_sweep(
    model_name: str, dataset_name: str, stage: str, quant_ckpt_path: str,
    num_classes: int, channels: int, image_size: int,
    eval_loader: DataLoader, device: torch.device,
    fp32_acc: float, all_layer_names: list[str],
) -> dict[str, float]:
    """Returns {layer_name: weight_damage_pts}."""
    from src.main import evaluate

    label = f"{stage} {model_name}/{dataset_name}"
    damage: dict[str, float] = {}

    for layer_name in all_layer_names:
        other_layers = {n for n in all_layer_names if n != layer_name}

        model, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
        model = model.to(device)
        _disable_activation_quant(model)
        _disable_weight_quant(model, layer_names=other_layers)
        _verify_weight_mask(model, {layer_name}, f"{label} isolate={layer_name}")

        with torch.no_grad():
            isolated_acc = evaluate(model, eval_loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        weight_damage_pts = fp32_acc - isolated_acc
        damage[layer_name] = weight_damage_pts
        logger.info(f"[WeightAblationCanonical] {label} layer={layer_name}: isolated_acc={isolated_acc:.2f}% weight_damage={weight_damage_pts:.3f}pts")

    return damage


# ---------------------------------------------------------------------------
# Part 2: perturbation term ||delta W||^2
# ---------------------------------------------------------------------------

def _compute_delta_w_sq(
    model_name: str, quant_ckpt_path: str, num_classes: int, channels: int, image_size: int,
    device: torch.device, all_layer_names: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Returns {layer_name: (delta_w_sq, delta_w_sq_per_param)}, computed on the
    checkpoint's own fused weights (module.weight) run through that same
    module's weight_fake_quant -- the exact PoT quantization error Part 1's
    isolation ablation applies to that layer. No activation/weight Identity
    swaps needed here; this is a pure weight-space computation.
    """
    model, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
    model = model.to(device)
    named = dict(model.named_modules())

    result: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for layer_name in all_layer_names:
            module = named[layer_name]
            assert isinstance(module, (QuantizedConv2d, QuantizedLinear))
            w = module.weight.detach()
            w_q = module.weight_fake_quant(w).detach()
            dwsq = (w - w_q).pow(2).sum().item()
            numel = w.numel()
            result[layer_name] = (dwsq, dwsq / numel if numel else float("nan"))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Part 3: correlations
# ---------------------------------------------------------------------------

def _rank_desc(values: dict[str, float], layer: str) -> int | None:
    if layer not in values:
        return None
    ranked = sorted(values.keys(), key=lambda l: values[l], reverse=True)
    return ranked.index(layer) + 1


def _spearman_and_overlap(
    layers: list[str], predictor_vals: dict[str, float], target_vals: dict[str, float], top_k: int,
) -> tuple[float, float, int]:
    n = len(layers)
    if n >= 3:
        rho, p_value = spearmanr([predictor_vals[l] for l in layers], [target_vals[l] for l in layers])
    else:
        rho, p_value = float("nan"), float("nan")
    top_k_by_target = set(sorted(layers, key=lambda l: target_vals[l], reverse=True)[:top_k]) if n else set()
    top_k_by_predictor = set(sorted(layers, key=lambda l: predictor_vals[l], reverse=True)[:top_k]) if n else set()
    overlap = len(top_k_by_target & top_k_by_predictor)
    return rho, p_value, overlap


def _run_correlations(
    model_name: str, dataset_name: str, stage: str,
    damage: dict[str, float], raw_trh: dict[str, float], dwsq: dict[str, float],
    correlation_csv: str, damage_mode: str = "both",
) -> None:
    """
    damage_mode selects which target(s) the primary spearman_rho_*/top5_overlap_*/
    conv1_damage_rank_* columns are computed for -- "abs" (the paper's headline;
    see module docstring addendum), "signed" (the original P1-revised target), or
    "both" (default -- computes and writes both column sets in the same row, so
    the signed-vs-QAT-null / abs-vs-top-rank contrast is visible without
    re-running anything).
    """
    assert damage_mode in DAMAGE_MODES, f"damage_mode must be one of {DAMAGE_MODES}, got {damage_mode!r}"
    label = f"{stage} {model_name}/{dataset_name}"
    layers = sorted(set(damage) & set(raw_trh) & set(dwsq))
    n = len(layers)
    if n < 3:
        logger.warning(f"[WeightAblationCanonical] {label}: only {n} layers with all three quantities present -- correlations not meaningful (need >= 3)")

    trh_dwsq = {l: raw_trh[l] * dwsq[l] for l in layers}
    predictor_values = {"raw_trh": raw_trh, "dwsq": dwsq, "trh_dwsq": trh_dwsq}
    damage_signed = {l: damage[l] for l in layers}
    damage_abs = {l: abs(damage[l]) for l in layers}
    conv1_damage_rank_signed = _rank_desc(damage_signed, "conv1")
    conv1_damage_rank_abs = _rank_desc(damage_abs, "conv1")

    top_k = min(5, n)
    want_abs = damage_mode in ("abs", "both")
    want_signed = damage_mode in ("signed", "both")

    summary_lines = []
    for predictor in PREDICTORS:
        vals = {l: predictor_values[predictor][l] for l in layers}
        conv1_predictor_rank = _rank_desc(vals, "conv1")

        row = {
            "model": model_name, "dataset": dataset_name, "stage": stage, "predictor": predictor,
            "n_layers": n, "damage_mode": damage_mode,
            "spearman_rho_abs": "", "spearman_p_abs": "", "top5_overlap_abs": "", "conv1_damage_rank_abs": "",
            "spearman_rho_signed": "", "spearman_p_signed": "", "top5_overlap_signed": "", "conv1_damage_rank_signed": "",
            "conv1_predictor_rank": conv1_predictor_rank if conv1_predictor_rank is not None else "",
        }

        summary_bits = []
        if want_abs:
            rho_abs, p_abs, overlap_abs = _spearman_and_overlap(layers, vals, damage_abs, top_k)
            row["spearman_rho_abs"] = rho_abs
            row["spearman_p_abs"] = p_abs
            row["top5_overlap_abs"] = f"{overlap_abs}/{top_k}" if n else ""
            row["conv1_damage_rank_abs"] = conv1_damage_rank_abs if conv1_damage_rank_abs is not None else ""
            summary_bits.append(f"abs: rho={rho_abs:.4f} p={p_abs:.4g} top{top_k}_overlap={overlap_abs}/{top_k}")
        if want_signed:
            rho_signed, p_signed, overlap_signed = _spearman_and_overlap(layers, vals, damage_signed, top_k)
            row["spearman_rho_signed"] = rho_signed
            row["spearman_p_signed"] = p_signed
            row["top5_overlap_signed"] = f"{overlap_signed}/{top_k}" if n else ""
            row["conv1_damage_rank_signed"] = conv1_damage_rank_signed if conv1_damage_rank_signed is not None else ""
            summary_bits.append(f"signed: rho={rho_signed:.4f} p={p_signed:.4g} top{top_k}_overlap={overlap_signed}/{top_k}")

        _append_row(correlation_csv, row, CORRELATION_FIELDNAMES)
        summary_lines.append(f"{predictor} [{', '.join(summary_bits)}] conv1_predictor_rank={conv1_predictor_rank}/{n}")

    logger.info(
        f"[WeightAblationCanonical] {label}: n_layers={n} damage_mode={damage_mode} "
        f"conv1_damage_rank_abs={conv1_damage_rank_abs}/{n} conv1_damage_rank_signed={conv1_damage_rank_signed}/{n} -- "
        + " | ".join(summary_lines)
    )

    # Honest, non-cherry-picked verdict -- reported, not used to alter any CSV row.
    # Always computed (both targets) regardless of --damage-mode, purely as a log
    # message -- this is what motivated the |damage| reframing in the first place
    # (conv1 sits at the top by |damage| in every configuration; signed rank
    # buries this on QAT, where per-layer damage is mostly negative).
    if conv1_damage_rank_abs is not None:
        rho_raw_abs, _ = spearmanr([raw_trh[l] for l in layers], [damage_abs[l] for l in layers]) if n >= 3 else (float("nan"), float("nan"))
        rho_dwsq_abs, _ = spearmanr([dwsq[l] for l in layers], [damage_abs[l] for l in layers]) if n >= 3 else (float("nan"), float("nan"))
        rho_trh_dwsq_abs, _ = spearmanr([trh_dwsq[l] for l in layers], [damage_abs[l] for l in layers]) if n >= 3 else (float("nan"), float("nan"))
        conv1_rank_raw = _rank_desc(raw_trh, "conv1")
        conv1_rank_dwsq = _rank_desc(dwsq, "conv1")
        conv1_rank_trh_dwsq = _rank_desc(trh_dwsq, "conv1")

        logger.info(
            f"[WeightAblationCanonical] {label}: VERDICT (|damage|) -- conv1 ranks #{conv1_damage_rank_abs}/{n} "
            f"by |damage| (signed rank #{conv1_damage_rank_signed}/{n}), #{conv1_rank_raw}/{n} by raw_trh "
            f"(rho_abs={rho_raw_abs:.3f}), #{conv1_rank_dwsq}/{n} by dwsq (rho_abs={rho_dwsq_abs:.3f}), "
            f"#{conv1_rank_trh_dwsq}/{n} by trh_dwsq (rho_abs={rho_trh_dwsq_abs:.3f}). "
            f"Interpretation left to the reader per predictor's numbers above -- not selected post hoc."
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_one_combo(
    model_name: str, dataset_name: str, stage: str, specs: dict, num_classes: int, device: torch.device,
    eval_loader: DataLoader, quant_ckpt_path: str, fp32_ckpt_path: str, canonical_traces_csv: str,
    ablation_csv: str, correlation_csv: str, damage_mode: str = "both",
) -> None:
    channels, image_size = specs["channels"], specs["image_size"]
    label = f"{stage} {model_name}/{dataset_name}"

    # ---- Part 0a: path-equivalence gate (reused unchanged from P1; this
    # internally deferred-imports bake_pot_into_standard_layers from
    # src.main itself, so it is not re-imported here) ----
    gate_passed, fp32_acc, weights_only_all_acc, all_layer_names, note = _run_part0(
        model_name, dataset_name, stage, quant_ckpt_path, fp32_ckpt_path,
        num_classes, channels, image_size, eval_loader, device,
    )
    if not gate_passed:
        logger.error(f"[WeightAblationCanonical] {label}: Part 0 path-equivalence GATE FAILED -- {note}. Skipping.")
        return

    # ---- Part 0b: canonical trace <-> ablation layer alignment ----
    variant = "fp32_fused_qat" if stage == "QAT" else "fp32_fused"
    
    try:
        canonical_traces = _load_canonical_traces(canonical_traces_csv, model_name, dataset_name, variant)
    except WeightAblationCanonicalError as exc:
        logger.error(f"[WeightAblationCanonical] {label}: {exc} -- skipping.")
        return

    probe_model, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
    try:
        raw_trh = _align_canonical_to_ablation(model_name, canonical_traces, all_layer_names, probe_model)
    except WeightAblationCanonicalError as exc:
        logger.error(f"[WeightAblationCanonical] {label}: {exc} -- skipping.")
        del probe_model
        return
    finally:
        del probe_model

    # ---- Part 1: weight-only isolation damage ----
    damage = _run_isolation_sweep(
        model_name, dataset_name, stage, quant_ckpt_path, num_classes, channels, image_size,
        eval_loader, device, fp32_acc, all_layer_names,
    )

    # ---- Part 2: perturbation term ----
    dwsq_map = _compute_delta_w_sq(model_name, quant_ckpt_path, num_classes, channels, image_size, device, all_layer_names)

    for layer_name in all_layer_names:
        dwsq, dwsq_per_param = dwsq_map.get(layer_name, (float("nan"), float("nan")))
        trh = raw_trh.get(layer_name, float("nan"))
        layer_damage = damage.get(layer_name, float("nan"))
        _append_row(ablation_csv, {
            "model": model_name, "dataset": dataset_name, "stage": stage, "layer": layer_name,
            "hessian_trace_fused": trh, "delta_w_sq": dwsq, "delta_w_sq_per_param": dwsq_per_param,
            "trh_times_dwsq": trh * dwsq if trh == trh and dwsq == dwsq else float("nan"),
            "fp32_acc": fp32_acc, "isolated_acc": fp32_acc - layer_damage,
            "weight_damage_pts": layer_damage, "abs_weight_damage_pts": abs(layer_damage),
        }, ABLATION_FIELDNAMES)

    # ---- Part 3: correlations (decision) ----
    dwsq_only = {l: dwsq_map[l][0] for l in dwsq_map}
    _run_correlations(model_name, dataset_name, stage, damage, raw_trh, dwsq_only, correlation_csv, damage_mode=damage_mode)


def run_weight_ablation_canonical(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    canonical_traces_csv: str,
    damage_mode: str = "both",
    datasets: list[str] | None = None,
) -> None:
    """datasets restricts the sweep to a subset of DATASETS (e.g. one dataset,
    for a parallel per-dataset analysis run) -- default None means every
    dataset in DATASETS."""
    assert damage_mode in DAMAGE_MODES, f"damage_mode must be one of {DAMAGE_MODES}, got {damage_mode!r}"
    datasets = datasets if datasets is not None else DATASETS
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[WeightAblationCanonical] CUDA not available -- falling back to CPU, this will be slow.")
    logger.info(f"[WeightAblationCanonical] device={device} seed={SEED} damage_mode={damage_mode} datasets={datasets} canonical_traces_csv={canonical_traces_csv}")
    _log_trace_config(canonical_traces_csv)

    fp32_models_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)
    quant_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)

    os.makedirs(CSV_DIR, exist_ok=True)
    ablation_csv = os.path.join(CSV_DIR, "weight_ablation_canonical_v2.csv")
    correlation_csv = os.path.join(CSV_DIR, "weight_ablation_canonical_correlation_v2.csv")

    for dataset_name in datasets:
        specs = DATASET_SPECS[dataset_name]
        try:
            eval_loader, num_classes = _build_eval_loader(dataset_name)
        except Exception as exc:
            logger.error(f"[WeightAblationCanonical] {dataset_name}: could not load dataset ({exc}) -- skipping dataset")
            continue

        # Stage-outer, model-inner: resnet18/PTQ, resnet50/PTQ, then (if reached)
        # resnet18/QAT, resnet50/QAT -- matches the spec's explicit run/report order.
        for stage in STAGES:
            for model_name in REQUIRED_MODELS:
                label = f"{stage} {model_name}/{dataset_name}"
                logger.info(f"[WeightAblationCanonical] === {label} ===")
                try:
                    fp32_ckpt_path = _resolve_checkpoint_robust(fp32_models_dir, {"model": model_name, "dataset": dataset_name})
                    quant_ckpt_path = _resolve_checkpoint_robust(quant_dir, {"stage": stage, "model": model_name, "dataset": dataset_name})
                except FileNotFoundError as exc:
                    level = logger.warning if stage == "QAT" else logger.error
                    level(f"[WeightAblationCanonical] {label}: checkpoint missing ({exc}) -- skipping{' (QAT optional)' if stage == 'QAT' else ''}")
                    continue
                except WeightAblationCheckpointError as exc:
                    logger.error(f"[WeightAblationCanonical] {label}: checkpoint resolution AMBIGUOUS/NEAR-MISS -- {exc} -- skipping, needs human attention")
                    continue

                try:
                    _run_one_combo(
                        model_name, dataset_name, stage, specs, num_classes, device, eval_loader,
                        quant_ckpt_path, fp32_ckpt_path, canonical_traces_csv, ablation_csv, correlation_csv,
                        damage_mode=damage_mode,
                    )
                except DiagnoseActivationsError as exc:
                    logger.error(f"[WeightAblationCanonical] {label}: Identity-swap verification FAILED -- {exc}")
                except Exception as exc:
                    logger.error(f"[WeightAblationCanonical] FAILED {label}: {exc}", exc_info=True)
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    logger.info("[WeightAblationCanonical] === Weight-Ablation-Canonical complete ===")


# =============================================================================
# Loss-based damage extension
# =============================================================================
"""
Extends the accuracy-only isolation sweep above with per-layer isolated
validation LOSS, not just accuracy -- both supervisors asked whether the
damage metric should be loss-based, since the Hessian/Taylor motivation
(Sec 3.3) is about loss, not accuracy. Reuses the isolation harness above
unchanged (_run_part0's path-equivalence gate, _load_quant_model,
_disable_activation_quant/_disable_weight_quant, _verify_weight_mask,
_append_row) -- the only new logic is the eval call itself (loss+accuracy
instead of accuracy-only, via train.py's _evaluate -- same mean-reduced
CrossEntropyLoss already used everywhere else full-model loss is measured
in this codebase, deliberately not a new reduction choice; see the trace-
measurement discrepancy this project already documented in
trace_reconciliation_ledger.csv for what happens when a "same" quantity is
silently measured two different ways) and a validation gate comparing the
freshly recomputed accuracy against the already-stored accuracy-only run
(weight_ablation_canonical_v2.csv), since both should be numerically
identical (same model, same loader, same order, same determinism).

Sign convention: loss_damage(l) = loss_iso(l) - fp32_loss, positive = worse
(loss went up), matching weight_damage(l) = fp32_acc - acc_iso(l), also
positive = worse -- a reader comparing the two columns never has to mentally
flip a sign.

Designed to run unattended overnight: every row is appended and the file
flushed immediately (_append_row, reused unchanged from
diagnose_activations.py), and re-launching after an interruption skips any
(model, dataset, stage, layer) already present in the output CSV rather than
recomputing or duplicating it. The separate, fast correlation step
(run_weight_ablation_loss_correlation, below) works against whatever subset
of rows exists at the time it's run.
"""


def _existing_completed_keys(output_csv: str) -> set[tuple[str, str, str, str]]:
    if not os.path.exists(output_csv):
        return set()
    df = pd.read_csv(output_csv)
    return set(zip(df["model"], df["dataset"], df["stage"], df["layer"]))


def _load_existing_ablation_csvs(existing_ablation_csvs: dict[str, str]) -> pd.DataFrame:
    """
    existing_ablation_csvs: {dataset_name: path to that dataset's canonical
    weight_ablation_canonical_v2.csv}. Each dataset's canonical accuracy run
    lives in its own results/<RUN_ID>/csv/ directory (a separate relock-
    traces/weight-ablation-canonical invocation per dataset), so this is a
    small dict, not a single shared file -- returns the concatenation with
    every row's 'dataset' column cross-checked against the key it was loaded
    under.
    """
    frames = []
    for dataset_name, path in existing_ablation_csvs.items():
        if not os.path.exists(path):
            raise WeightAblationCanonicalError(f"existing ablation CSV not found for dataset={dataset_name}: {path}")
        df = pd.read_csv(path)
        mismatched = df[df["dataset"] != dataset_name]
        if not mismatched.empty:
            raise WeightAblationCanonicalError(
                f"--existing-ablation-csv for dataset={dataset_name} ({path}) contains "
                f"{len(mismatched)} row(s) with a different 'dataset' value -- wrong file/dataset pairing"
            )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _backup_existing_file(path: str) -> None:
    if not os.path.exists(path):
        return
    backup_path = f"{path}.bak-{time.strftime('%Y%m%dT%H%M%S')}"
    os.rename(path, backup_path)
    logger.warning(f"[WeightAblationLoss] --force-recompute: existing {path} moved to {backup_path} (not deleted)")


def _ensure_csv_with_header(path: str, fieldnames: list[str]) -> None:
    # Part 2 requires accuracy_mismatch.csv to exist (with header) even on a
    # completely clean run, so its absence is never confused with "the check
    # never ran" -- _append_row alone only materializes the file on its first
    # write, which would leave it missing entirely on a clean run.
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    logger.info(f"[WeightAblationLoss] initialized {path} (header only)")


def _check_and_log_mismatch(
    mismatch_csv: str, model_name: str, dataset_name: str, stage: str, layer: str,
    field: str, existing_value: float, recomputed_value: float,
) -> bool:
    diff = abs(existing_value - recomputed_value)
    if diff <= ACC_MATCH_TOLERANCE_PTS:
        return False
    _append_row(mismatch_csv, {
        "model": model_name, "dataset": dataset_name, "stage": stage, "layer": layer, "field": field,
        "existing_value": existing_value, "recomputed_value": recomputed_value, "abs_diff": diff,
    }, ACCURACY_MISMATCH_FIELDNAMES)
    logger.warning(
        f"[WeightAblationLoss] ACCURACY MISMATCH {stage} {model_name}/{dataset_name} layer={layer} "
        f"field={field}: existing={existing_value:.4f} recomputed={recomputed_value:.4f} "
        f"diff={diff:.4f} > tolerance {ACC_MATCH_TOLERANCE_PTS} -- logged, not silently accepted"
    )
    return True


def _run_combo_loss_sweep(
    model_name: str, dataset_name: str, stage: str, specs: dict, num_classes: int, device: torch.device,
    eval_loader: DataLoader, quant_ckpt_path: str, fp32_ckpt_path: str,
    existing_combo_df: pd.DataFrame, output_csv: str, mismatch_csv: str,
    completed_keys: set[tuple[str, str, str, str]], force_recompute: bool,
) -> tuple[int, int]:
    """Returns (n_layers_written, n_layers_skipped_already_done) for this combo."""
    channels, image_size = specs["channels"], specs["image_size"]
    label = f"{stage} {model_name}/{dataset_name}"
    t0 = time.monotonic()

    # ---- Part 0: reuse the unchanged path-equivalence gate + layer set ----
    gate_passed, fp32_acc_gate, _weights_only_acc, all_layer_names, note = _run_part0(
        model_name, dataset_name, stage, quant_ckpt_path, fp32_ckpt_path,
        num_classes, channels, image_size, eval_loader, device,
    )
    if not gate_passed:
        logger.error(f"[WeightAblationLoss] {label}: Part 0 path-equivalence GATE FAILED -- {note}. Skipping.")
        return 0, 0

    # ---- FP32 baseline: loss + accuracy together (same eval, one extra
    # forward pass over the full val set, once per combo -- _run_part0 above
    # already paid for this pass via evaluate(); this second pass adds loss
    # via _evaluate(), train.py's own mean-reduced-CrossEntropyLoss path,
    # deliberately reused rather than reimplemented) ----
    fp32_model = _load_fp32_reference(model_name, fp32_ckpt_path, num_classes, channels, image_size).to(device)
    fp32_loss, fp32_acc = _evaluate(fp32_model, eval_loader, nn.CrossEntropyLoss(reduction="mean"), device)
    del fp32_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info(
        f"[WeightAblationLoss] {label}: fp32_loss={fp32_loss:.4f} fp32_acc={fp32_acc:.2f}% "
        f"(Part0 gate's own fp32_acc={fp32_acc_gate:.2f}%, diff={abs(fp32_acc - fp32_acc_gate):.4f}pt)"
    )

    existing_fp32_acc_rows = existing_combo_df["fp32_acc"].unique()
    if len(existing_fp32_acc_rows) > 0:
        _check_and_log_mismatch(
            mismatch_csv, model_name, dataset_name, stage, "__fp32_baseline__",
            "fp32_acc", float(existing_fp32_acc_rows[0]), fp32_acc,
        )

    n_written, n_skipped = 0, 0
    for layer_name in all_layer_names:
        key = (model_name, dataset_name, stage, layer_name)
        if key in completed_keys and not force_recompute:
            n_skipped += 1
            continue

        other_layers = {n for n in all_layer_names if n != layer_name}
        model, _, _ = _load_quant_model(model_name, quant_ckpt_path, num_classes, channels, image_size)
        model = model.to(device)
        _disable_activation_quant(model)
        _disable_weight_quant(model, layer_names=other_layers)
        _verify_weight_mask(model, {layer_name}, f"{label} loss-isolate={layer_name}")

        isolated_loss, isolated_acc = _evaluate(model, eval_loader, nn.CrossEntropyLoss(reduction="mean"), device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        weight_damage_pts = fp32_acc - isolated_acc
        loss_damage = isolated_loss - fp32_loss

        existing_row = existing_combo_df[existing_combo_df["layer"] == layer_name]
        if not existing_row.empty:
            _check_and_log_mismatch(
                mismatch_csv, model_name, dataset_name, stage, layer_name,
                "isolated_acc", float(existing_row["isolated_acc"].iloc[0]), isolated_acc,
            )

        _append_row(output_csv, {
            "model": model_name, "dataset": dataset_name, "stage": stage, "layer": layer_name,
            "fp32_acc": fp32_acc, "isolated_acc": isolated_acc,
            "weight_damage_pts": weight_damage_pts, "abs_weight_damage_pts": abs(weight_damage_pts),
            "fp32_loss": fp32_loss, "isolated_loss": isolated_loss,
            "loss_damage": loss_damage, "abs_loss_damage": abs(loss_damage),
        }, LOSS_ABLATION_FIELDNAMES)
        n_written += 1
        logger.info(
            f"[WeightAblationLoss] {label} layer={layer_name}: isolated_acc={isolated_acc:.2f}% "
            f"weight_damage={weight_damage_pts:.3f}pts isolated_loss={isolated_loss:.4f} "
            f"loss_damage={loss_damage:+.4f} ({n_written} written, {n_skipped} already done this combo)"
        )

    elapsed = time.monotonic() - t0
    logger.info(
        f"[WeightAblationLoss] === {label} block complete: {n_written} layer(s) computed, "
        f"{n_skipped} already done (resumed), {len(all_layer_names)} total -- elapsed {elapsed:.1f}s ==="
    )
    return n_written, n_skipped


def run_weight_ablation_loss(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    existing_ablation_csvs: dict[str, str],
    force_recompute: bool = False,
    datasets: list[str] | None = None,
) -> None:
    """
    Part 1/2/3/4 of the loss-damage extension: sweeps all requested (model,
    dataset, stage) combos in cheapest-first order (LOSS_COMBO_ORDER),
    writing results/<RUN_ID>/csv/weight_ablation_loss_damage.csv one row at a
    time (resumable -- rows already present are skipped unless
    force_recompute) and results/<RUN_ID>/csv/accuracy_mismatch.csv (always
    created, even empty).
    """
    datasets = datasets if datasets is not None else ["CIFAR10", "IMAGENET100"]
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[WeightAblationLoss] CUDA not available -- falling back to CPU, this will be slow.")

    output_csv = os.path.join(CSV_DIR, "weight_ablation_loss_damage.csv")
    mismatch_csv = os.path.join(CSV_DIR, "accuracy_mismatch.csv")

    if force_recompute:
        _backup_existing_file(output_csv)
        _backup_existing_file(mismatch_csv)
    _ensure_csv_with_header(mismatch_csv, ACCURACY_MISMATCH_FIELDNAMES)

    existing_ablation_df = _load_existing_ablation_csvs(existing_ablation_csvs)
    completed_keys = _existing_completed_keys(output_csv)
    logger.info(
        f"[WeightAblationLoss] device={device} seed={SEED} datasets={datasets} force_recompute={force_recompute} "
        f"output_csv={output_csv} -- {len(completed_keys)} (model,dataset,stage,layer) row(s) already done"
    )

    fp32_models_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)
    quant_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)

    combos = [c for c in LOSS_COMBO_ORDER if c[0] in datasets]
    run_start = time.monotonic()
    total_written, total_skipped = 0, 0
    _loader_cache: dict[str, tuple[DataLoader, int]] = {}

    for combo_idx, (dataset_name, model_name, stage) in enumerate(combos, start=1):
        label = f"{stage} {model_name}/{dataset_name}"
        existing_combo_df = existing_ablation_df[
            (existing_ablation_df["model"] == model_name)
            & (existing_ablation_df["dataset"] == dataset_name)
            & (existing_ablation_df["stage"] == stage)
        ]
        if existing_combo_df.empty:
            logger.error(
                f"[WeightAblationLoss] {label}: no rows in the supplied --existing-ablation-csv for this "
                f"combo -- cannot validate accuracy or determine the canonical layer set. Skipping."
            )
            continue

        expected_layers = set(existing_combo_df["layer"])
        done_layers = {k[3] for k in completed_keys if k[0] == model_name and k[1] == dataset_name and k[2] == stage}
        if not force_recompute and expected_layers.issubset(done_layers):
            logger.info(
                f"[WeightAblationLoss] [{combo_idx}/{len(combos)}] {label}: all {len(expected_layers)} layers "
                f"already done -- skipping combo entirely (no model load)."
            )
            total_skipped += len(expected_layers)
            continue

        specs = DATASET_SPECS[dataset_name]
        if dataset_name not in _loader_cache:
            try:
                _loader_cache[dataset_name] = _build_eval_loader(dataset_name)
            except Exception as exc:
                logger.error(f"[WeightAblationLoss] {dataset_name}: could not load dataset ({exc}) -- skipping all its combos")
                continue
        eval_loader, num_classes = _loader_cache[dataset_name]

        try:
            fp32_ckpt_path = _resolve_checkpoint_robust(fp32_models_dir, {"model": model_name, "dataset": dataset_name})
            quant_ckpt_path = _resolve_checkpoint_robust(quant_dir, {"stage": stage, "model": model_name, "dataset": dataset_name})
        except FileNotFoundError as exc:
            level = logger.warning if stage == "QAT" else logger.error
            level(f"[WeightAblationLoss] {label}: checkpoint missing ({exc}) -- skipping{' (QAT optional)' if stage == 'QAT' else ''}")
            continue
        except WeightAblationCheckpointError as exc:
            logger.error(f"[WeightAblationLoss] {label}: checkpoint resolution AMBIGUOUS/NEAR-MISS -- {exc} -- skipping")
            continue

        logger.info(f"[WeightAblationLoss] [{combo_idx}/{len(combos)}] === {label} ===")
        try:
            n_written, n_skipped = _run_combo_loss_sweep(
                model_name, dataset_name, stage, specs, num_classes, device, eval_loader,
                quant_ckpt_path, fp32_ckpt_path, existing_combo_df, output_csv, mismatch_csv,
                completed_keys, force_recompute,
            )
            total_written += n_written
            total_skipped += n_skipped
        except DiagnoseActivationsError as exc:
            logger.error(f"[WeightAblationLoss] {label}: Identity-swap verification FAILED -- {exc}")
        except Exception as exc:
            logger.error(f"[WeightAblationLoss] FAILED {label}: {exc}", exc_info=True)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed_total = time.monotonic() - run_start
        logger.info(
            f"[WeightAblationLoss] progress: {total_written} layer(s) newly computed, {total_skipped} "
            f"already-done/skipped, {combo_idx}/{len(combos)} combos processed -- running total {elapsed_total:.1f}s"
        )

    logger.info(
        f"[WeightAblationLoss] === Weight-Ablation-Loss complete: {total_written} layer(s) newly written, "
        f"{total_skipped} already done -- {output_csv} ==="
    )


# ---------------------------------------------------------------------------
# Part 5: loss-based correlation (separate, fast, safe on partial data)
# ---------------------------------------------------------------------------

def run_weight_ablation_loss_correlation(
    loss_damage_csv: str,
    existing_ablation_csvs: dict[str, str],
    output_csv: str,
) -> None:
    """
    Independent of run_weight_ablation_loss's sweep -- reads whatever rows
    currently exist in loss_damage_csv (partial or complete) and correlates
    S_raw/S_pert/S_hawq (pulled from the canonical accuracy-only CSVs, which
    already carry hessian_trace_fused/delta_w_sq/trh_times_dwsq per layer)
    against abs_loss_damage/loss_damage, mirroring
    weight_ablation_canonical_correlation_v2.csv's schema exactly (same
    _abs/_signed suffixes, same rho/p/top5/rank columns) plus one addition:
    a status column marking any (model, dataset, stage) combo that doesn't
    yet have all its layers -- those get n_layers/status filled in but no
    correlation numbers, rather than a correlation computed on a truncated,
    non-representative subset of layers.
    """
    if not os.path.exists(loss_damage_csv):
        raise WeightAblationCanonicalError(f"loss damage CSV not found: {loss_damage_csv} -- run --weight-ablation-loss first")

    loss_df = pd.read_csv(loss_damage_csv)
    ablation_df = _load_existing_ablation_csvs(existing_ablation_csvs)
    predictor_df = ablation_df[["model", "dataset", "stage", "layer"] + list(PREDICTOR_SOURCE_COLUMNS.values())]
    merged = loss_df.merge(predictor_df, on=["model", "dataset", "stage", "layer"], how="left", validate="one_to_one")

    missing_predictors = merged[merged[list(PREDICTOR_SOURCE_COLUMNS.values())].isna().any(axis=1)]
    if not missing_predictors.empty:
        logger.warning(
            f"[WeightAblationLossCorrelation] {len(missing_predictors)} loss row(s) have no matching "
            f"predictor row in --existing-ablation-csv (layer name not found there) -- dropped from correlation"
        )
        merged = merged.dropna(subset=list(PREDICTOR_SOURCE_COLUMNS.values()))

    expected_layer_counts = ablation_df.groupby(["model", "dataset", "stage"])["layer"].nunique()

    rows = []
    for (model_name, dataset_name, stage), g in merged.groupby(["model", "dataset", "stage"]):
        label = f"{stage} {model_name}/{dataset_name}"
        n_present = len(g)
        n_total = int(expected_layer_counts.get((model_name, dataset_name, stage), n_present))
        complete = n_present >= n_total

        if not complete:
            status = f"incomplete_{n_present}_of_{n_total}"
            logger.info(f"[WeightAblationLossCorrelation] {label}: {status} -- correlations NOT computed on this partial subset")
            for predictor in PREDICTORS:
                rows.append({
                    "model": model_name, "dataset": dataset_name, "stage": stage, "predictor": predictor,
                    "n_layers": n_present, "damage_mode": "both", "status": status,
                    "spearman_rho_abs": "", "spearman_p_abs": "", "top5_overlap_abs": "", "conv1_damage_rank_abs": "",
                    "spearman_rho_signed": "", "spearman_p_signed": "", "top5_overlap_signed": "", "conv1_damage_rank_signed": "",
                    "conv1_predictor_rank": "",
                })
            continue

        status = "complete"
        layers = list(g["layer"])
        damage_abs = dict(zip(layers, g["abs_loss_damage"]))
        damage_signed = dict(zip(layers, g["loss_damage"]))
        conv1_damage_rank_abs = _rank_desc(damage_abs, "conv1")
        conv1_damage_rank_signed = _rank_desc(damage_signed, "conv1")
        top_k = min(5, n_present)

        raw_trh = dict(zip(layers, g["hessian_trace_fused"]))
        dwsq = dict(zip(layers, g["delta_w_sq"]))
        trh_dwsq = dict(zip(layers, g["trh_times_dwsq"]))
        predictor_values = {"raw_trh": raw_trh, "dwsq": dwsq, "trh_dwsq": trh_dwsq}

        summary_bits = []
        for predictor in PREDICTORS:
            vals = predictor_values[predictor]
            conv1_predictor_rank = _rank_desc(vals, "conv1")
            rho_abs, p_abs, overlap_abs = _spearman_and_overlap(layers, vals, damage_abs, top_k)
            rho_signed, p_signed, overlap_signed = _spearman_and_overlap(layers, vals, damage_signed, top_k)
            rows.append({
                "model": model_name, "dataset": dataset_name, "stage": stage, "predictor": predictor,
                "n_layers": n_present, "damage_mode": "both", "status": status,
                "spearman_rho_abs": rho_abs, "spearman_p_abs": p_abs,
                "top5_overlap_abs": f"{overlap_abs}/{top_k}", "conv1_damage_rank_abs": conv1_damage_rank_abs if conv1_damage_rank_abs is not None else "",
                "spearman_rho_signed": rho_signed, "spearman_p_signed": p_signed,
                "top5_overlap_signed": f"{overlap_signed}/{top_k}", "conv1_damage_rank_signed": conv1_damage_rank_signed if conv1_damage_rank_signed is not None else "",
                "conv1_predictor_rank": conv1_predictor_rank if conv1_predictor_rank is not None else "",
            })
            summary_bits.append(f"{predictor}: rho_abs={rho_abs:.4f}")
        logger.info(f"[WeightAblationLossCorrelation] {label}: complete ({n_present}/{n_total}) -- " + ", ".join(summary_bits))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOSS_CORRELATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"[WeightAblationLossCorrelation] === wrote {len(rows)} row(s) -> {output_csv} ===")
