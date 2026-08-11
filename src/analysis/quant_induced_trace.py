"""
quant_induced_trace.py -- decomposes the FP32 -> quantized change in each
layer's weight-Hessian trace into a fusion effect (BN folded into conv) and a
quantization-induced effect (fake-quant noise), across four model variants:
unfused FP32, fused FP32, PTQ, QAT.

Motivation: the random-init control (src/analysis/random_init_control.py, P2)
showed conv1 is unremarkable in the FP32 model -- at the median for resnet50
(trace 13.87, ~1x median), below median for resnet18 -- while the established
conv1 trace of 13,161.8 (84% of the model total) was a PTQ figure. That
~950x gap says the conv1 curvature dominance is most likely
quantization-induced, not an FP32 property. This module makes that rigorous
per-layer and reconciles the discrepancy: it confirms whether conv1 (and
which other layers) gain curvature specifically from quantization,
controlling for the reparametrisation that fusion alone introduces, and
checks the FP32 profile computed here against a banked FP32 profile (e.g.
layerwise_hessian_traces.csv) that produced the "conv1 14x" claim.

Two parts, CIFAR10 only; models resnet18_no_weights and resnet50_no_weights
required, cnn included only if it fuses cleanly (it does, per the existing
fuse_model_architectures path already used across the pipeline):

  Part 0 (gate): build the unfused FP32 skeleton and the fused/quantized
    skeleton (no checkpoint needed for this -- shapes are checkpoint-
    independent), enumerate their conv/linear weight tensors in forward
    order (named_modules() traversal order, which matches attribute
    registration order for these standard sequential model definitions), and
    pair them by position. Verified by weight shape at every position. The
    unfused FP32 name is the canonical layer id; the quantized module name is
    carried alongside. Bijective + shape-matched is asserted, not assumed --
    the full table is always printed, and a failure raises rather than
    falling back to fuzzy name matching (which could pair conv1's FP32 trace
    with another layer's quantized trace). In practice fuse_modules(inplace=
    True) and replace_layers_for_quantization's setattr both preserve module
    names, so canonical_name == quantized_name in every observed case here --
    the position+shape check is the actual gate, name equality is a bonus
    sanity signal, not the mechanism.
  Part 1: trace fp32_unfused (build_model + baseline checkpoint), fp32_fused
    (PTQ checkpoint with both weight_fake_quant and act_fake_quant ->
    nn.Identity()), ptq (PTQ checkpoint, both fake-quants active), qat (QAT
    checkpoint, both fake-quants active), and fp32_fused_qat (QAT checkpoint
    with both fake-quants -> Identity -- QAT's own weights were retrained,
    so its fake-quants-off reference is NOT fp32_unfused). One fixed probe
    seed (PROBE_SEED) reset immediately before every
    compute_layerwise_hessian_trace_pyhessian call (reused unchanged), so
    Hutchinson probe draws are identical across all variants and the
    FP32->PTQ/QAT difference is a model difference, not estimator noise. A
    stage (PTQ or QAT) whose checkpoint is missing is skipped with a warning,
    not a hard failure -- the model's other stage still runs.
  Part 2: per canonical layer, fusion_ratio = fp32_fused / fp32_unfused,
    ptq_amplification = ptq / fp32_fused, qat_amplification = qat /
    fp32_fused_qat, elev_over_median per variant, and a verdict from the
    amplification number itself (never a hidden threshold): conv1 spotlight
    (trace across all four variants, amplification, rank among all layers),
    reconciliation against an optional banked FP32 profile CSV, and Spearman
    between fp32_unfused and ptq/qat (resnets only -- cnn has too few
    layers).

Reuses (does not duplicate): build_model, fuse_model_architectures,
replace_layers_for_quantization, compute_layerwise_hessian_trace_pyhessian
(unchanged), the Identity-swap helpers (_disable_weight_quant,
_disable_activation_quant, _verify_identity_swap) and FP32 checkpoint
resolution from src/analysis/diagnose_activations.py, the robust checkpoint
resolver (_resolve_checkpoint_robust) from src/analysis/_ablation_common.py
(filenames in this project are not perfectly uniform, so exact f-string
paths are never used), the quantized-checkpoint directory resolver
from src/quantization/deploy_fbgemm.py, and _safe_div /
_enable_determinism from src/analysis/random_init_control.py.

Analysis only -- no torchao/INT8/deployment code. Runs as a single local
process (`python -m src.main --quant-induced-trace ...`), no SLURM/torchrun
required; prefers CUDA (A100).
"""

import os
import math
import logging
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import pandas as pd

from src.model_cnn.train import build_model
from src.quantization.quantizer import (
    fuse_model_architectures,
    replace_layers_for_quantization,
)
from src.analysis.pyhessian import compute_layerwise_hessian_trace_pyhessian
from src.analysis.diagnose_activations import (
    _resolve_fp32_models_dir,
    _disable_activation_quant,
    _disable_weight_quant,
    _verify_identity_swap,
    _append_row,
    DiagnoseActivationsError,
)
from src.analysis._ablation_common import (
    _resolve_checkpoint_robust,
    WeightAblationCheckpointError,
)
from src.analysis.random_init_control import _safe_div, _enable_determinism
from src.quantization.deploy_fbgemm import _resolve_checkpoint_dir
from src.utility.config import CSV_DIR, DATASET_SPECS, HESSIAN_BATCH_SIZE
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

DATASETS = ["CIFAR10"]
# resnet50 first -- it carries the headline conv1 spike this mode targets.
ORDERED_MODELS = ["resnet50_no_weights", "resnet18_no_weights", "cnn"]
REQUIRED_MODELS = {"resnet50_no_weights", "resnet18_no_weights"}

# Fixed and reset immediately before every estimator call, across every
# variant -- the config-lock this mode's cross-variant ratios depend on.
PROBE_SEED = 20260811

# compute_layerwise_hessian_trace_pyhessian's own defaults -- passed
# explicitly so the "identical estimator config across variants" requirement
# is visible and locked regardless of upstream default drift.
HESSIAN_NUM_BATCHES = 5
HESSIAN_MAX_ITER = 100
HESSIAN_TOL = 1e-3

# Verdict thresholds (Part 2), applied to the reported number, never hidden:
# ptq_amplification <= this counts as "approximately 1" -> fp32_intrinsic.
AMP_INTRINSIC_MAX = 1.5
# ptq_amplification >= this multiple of the model's median amplification
# (and above AMP_INTRINSIC_MAX) -> quant_induced. Otherwise -> mixed.
AMP_FAR_ABOVE_MEDIAN_MULT = 3.0

# Banked-FP32-profile reconciliation (Part 2): a layer's relative difference
# between the banked profile and this run's fresh fp32_unfused trace counts
# as agreement if <= RECONCILE_TOLERANCE; the profiles are called matching
# overall if at least RECONCILE_MATCH_FRACTION of common layers agree.
RECONCILE_TOLERANCE = 0.3
RECONCILE_MATCH_FRACTION = 0.8

TRACES_FIELDNAMES = [
    "model", "dataset", "variant", "probe_seed",
    "canonical_layer", "quantized_layer", "weight_shape", "hessian_trace",
]
COMPARISON_FIELDNAMES = [
    "model", "dataset", "canonical_layer",
    "trace_fp32_unfused", "trace_fp32_fused", "trace_ptq", "trace_fp32_fused_qat", "trace_qat",
    "fusion_ratio", "ptq_amplification", "qat_amplification", "elev_ptq", "elev_qat", "verdict",
]
SUMMARY_FIELDNAMES = [
    "model", "dataset", "spearman_fp32_ptq", "spearman_fp32_qat",
    "conv1_fp32", "conv1_ptq", "conv1_ptq_amplification", "conv1_amp_rank",
    "top_amplified_layer", "banked_fp32_matches", "note",
]


class QuantInducedTraceError(RuntimeError):
    pass


class QuantInducedMappingError(QuantInducedTraceError):
    """Part 0 gate failure -- bijection or shape check did not pass."""
    pass


# ---------------------------------------------------------------------------
# Part 0: forward-order enumeration, position+shape mapping gate
# ---------------------------------------------------------------------------

def _weight_layers_in_forward_order(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def _build_layer_mapping(model_name: str, unfused_model: nn.Module, quantized_model: nn.Module) -> list[dict]:
    unfused_layers = _weight_layers_in_forward_order(unfused_model)
    quant_layers = _weight_layers_in_forward_order(quantized_model)

    n = max(len(unfused_layers), len(quant_layers))
    table = []
    ok = len(unfused_layers) == len(quant_layers)
    for i in range(n):
        if i >= len(unfused_layers) or i >= len(quant_layers):
            ok = False
            table.append({
                "position": i,
                "canonical_name": unfused_layers[i][0] if i < len(unfused_layers) else "<missing>",
                "quantized_name": quant_layers[i][0] if i < len(quant_layers) else "<missing>",
                "unfused_shape": None, "quantized_shape": None, "shape_match": False,
            })
            continue
        u_name, u_mod = unfused_layers[i]
        q_name, q_mod = quant_layers[i]
        u_shape, q_shape = tuple(u_mod.weight.shape), tuple(q_mod.weight.shape)
        match = u_shape == q_shape
        ok = ok and match
        table.append({
            "position": i, "canonical_name": u_name, "quantized_name": q_name,
            "unfused_shape": u_shape, "quantized_shape": q_shape, "shape_match": match,
        })

    header = f"{'pos':>4} {'canonical_name':<40} {'quantized_name':<40} {'unfused_shape':<22} {'quantized_shape':<22} match"
    lines = [
        f"{r['position']:>4} {r['canonical_name']:<40} {r['quantized_name']:<40} "
        f"{str(r['unfused_shape']):<22} {str(r['quantized_shape']):<22} {r['shape_match']}"
        for r in table
    ]
    logger.info(
        f"[QuantInducedTrace] {model_name}: Part 0 layer mapping table "
        f"({len(unfused_layers)} unfused vs {len(quant_layers)} quantized conv/linear layers):\n"
        + header + "\n" + "\n".join(lines)
    )

    if not ok:
        raise QuantInducedMappingError(
            f"{model_name}: Part 0 mapping gate FAILED -- bijection/shape check did not pass "
            f"(see table above). Refusing fuzzy name fallback."
        )

    logger.info(f"[QuantInducedTrace] {model_name}: Part 0 gate PASSED -- {len(table)} layers bijective and shape-matched")
    return table


# ---------------------------------------------------------------------------
# Model construction / loading
# ---------------------------------------------------------------------------

def _load_unfused_fp32(model_name: str, ckpt_path: str, num_classes: int, channels: int, image_size: int, device: torch.device) -> nn.Module:
    model = build_model(num_classes=num_classes, model_name=model_name, channels=channels, image_size=image_size)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def _build_quant_skeleton(model_name: str, num_classes: int, channels: int, image_size: int) -> nn.Module:
    model = build_model(num_classes=num_classes, model_name=model_name, channels=channels, image_size=image_size)
    fuse_model_architectures(model, model_name)
    replace_layers_for_quantization(model)
    return model


def _load_quantized(model_name: str, ckpt_path: str, num_classes: int, channels: int, image_size: int, device: torch.device) -> nn.Module:
    model = _build_quant_skeleton(model_name, num_classes, channels, image_size)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def _make_fused_fp32(model: nn.Module, label: str) -> nn.Module:
    # Both quantizers -> Identity, on every quantized layer. Verified
    # immediately after -- fail loudly naming model/stage (via `label`)
    # rather than silently tracing a still-quantized model.
    weight_layers = _disable_weight_quant(model)
    act_layers = _disable_activation_quant(model)
    _verify_identity_swap(model, "weight_fake_quant", weight_layers, label)
    _verify_identity_swap(model, "act_fake_quant", act_layers, label)
    return model


# ---------------------------------------------------------------------------
# Loader (eval-mode, num_workers=0, pin_memory=False, shuffle=False)
# ---------------------------------------------------------------------------

def _build_hessian_loader(dataset_name: str) -> tuple[DataLoader, int]:
    _, val_loader, num_classes = get_data_loaders(dataset_name)
    hessian_loader = DataLoader(
        val_loader.dataset, batch_size=HESSIAN_BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    return hessian_loader, num_classes


def _trace_variant(model: nn.Module, hessian_loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    # Probe seed reset immediately before this call, not once per model --
    # the config-lock every cross-variant ratio in this mode depends on.
    torch.manual_seed(PROBE_SEED)
    return compute_layerwise_hessian_trace_pyhessian(
        model, hessian_loader, criterion, device,
        num_batches=HESSIAN_NUM_BATCHES, max_iter=HESSIAN_MAX_ITER, tol=HESSIAN_TOL,
    )


# ---------------------------------------------------------------------------
# CSV writing helpers
# ---------------------------------------------------------------------------

def _nan_to_blank(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


def _write_variant_traces(
    model_name: str, dataset_name: str, variant: str,
    mapping: list[dict], trace_dict: dict[str, float], key_kind: str, csv_path: str,
) -> None:
    for row in mapping:
        name_key = row["canonical_name"] if key_kind == "canonical" else row["quantized_name"]
        trace_val = trace_dict.get(f"{name_key}.weight")
        if trace_val is None:
            logger.warning(
                f"[QuantInducedTrace] {model_name}/{dataset_name} variant={variant}: no trace "
                f"found for '{name_key}.weight' -- layer omitted from this variant's trace rows"
            )
            continue
        _append_row(csv_path, {
            "model": model_name, "dataset": dataset_name, "variant": variant, "probe_seed": PROBE_SEED,
            "canonical_layer": row["canonical_name"], "quantized_layer": row["quantized_name"],
            "weight_shape": str(row["unfused_shape"]), "hessian_trace": trace_val,
        }, TRACES_FIELDNAMES)


# ---------------------------------------------------------------------------
# Part 2: decomposition and classification
# ---------------------------------------------------------------------------

def _classify_verdict(amp: float, median_amp: float) -> str:
    if isinstance(amp, float) and math.isnan(amp):
        return "undetermined"
    if amp <= AMP_INTRINSIC_MAX:
        return "fp32_intrinsic"
    if not (isinstance(median_amp, float) and math.isnan(median_amp)) and amp >= AMP_FAR_ABOVE_MEDIAN_MULT * median_amp:
        return "quant_induced"
    return "mixed"


def _decompose_and_classify(
    model_name: str, dataset_name: str, mapping: list[dict], traces: dict[str, dict[str, float]], comparison_csv_path: str,
) -> list[dict]:
    fp32_unfused = traces.get("fp32_unfused", {})
    fp32_fused = traces.get("fp32_fused", {})
    ptq = traces.get("ptq", {})
    fp32_fused_qat = traces.get("fp32_fused_qat", {})
    qat = traces.get("qat", {})

    per_layer = []
    for row in mapping:
        canon, quant = row["canonical_name"], row["quantized_name"]
        t_unfused = fp32_unfused.get(f"{canon}.weight")
        t_fused = fp32_fused.get(f"{quant}.weight")
        t_ptq = ptq.get(f"{quant}.weight")
        t_fused_qat = fp32_fused_qat.get(f"{quant}.weight")
        t_qat = qat.get(f"{quant}.weight")

        fusion_ratio = _safe_div(t_fused, t_unfused) if t_fused is not None and t_unfused is not None else float("nan")
        ptq_amp = _safe_div(t_ptq, t_fused) if t_ptq is not None and t_fused is not None else float("nan")
        qat_amp = _safe_div(t_qat, t_fused_qat) if t_qat is not None and t_fused_qat is not None else float("nan")

        per_layer.append({
            "canonical_layer": canon, "quantized_layer": quant,
            "trace_fp32_unfused": t_unfused, "trace_fp32_fused": t_fused, "trace_ptq": t_ptq,
            "trace_fp32_fused_qat": t_fused_qat, "trace_qat": t_qat,
            "fusion_ratio": fusion_ratio, "ptq_amplification": ptq_amp, "qat_amplification": qat_amp,
        })

    ptq_amps = [r["ptq_amplification"] for r in per_layer if not math.isnan(r["ptq_amplification"])]
    median_ptq_amp = statistics.median(ptq_amps) if ptq_amps else float("nan")
    ptq_trace_vals = [r["trace_ptq"] for r in per_layer if r["trace_ptq"] is not None]
    median_ptq_trace = statistics.median(ptq_trace_vals) if ptq_trace_vals else float("nan")
    qat_trace_vals = [r["trace_qat"] for r in per_layer if r["trace_qat"] is not None]
    median_qat_trace = statistics.median(qat_trace_vals) if qat_trace_vals else float("nan")

    logger.info(
        f"[QuantInducedTrace] {model_name}/{dataset_name}: median ptq_amplification={median_ptq_amp:.4g} "
        f"across {len(ptq_amps)} layers -- verdict thresholds: <= {AMP_INTRINSIC_MAX} -> fp32_intrinsic, "
        f">= {AMP_FAR_ABOVE_MEDIAN_MULT}x median (and > {AMP_INTRINSIC_MAX}) -> quant_induced, else mixed"
    )

    for r in per_layer:
        r["elev_ptq"] = _safe_div(r["trace_ptq"], median_ptq_trace) if r["trace_ptq"] is not None else float("nan")
        r["elev_qat"] = _safe_div(r["trace_qat"], median_qat_trace) if r["trace_qat"] is not None else float("nan")
        r["verdict"] = _classify_verdict(r["ptq_amplification"], median_ptq_amp)

        _append_row(comparison_csv_path, {
            "model": model_name, "dataset": dataset_name, "canonical_layer": r["canonical_layer"],
            "trace_fp32_unfused": _nan_to_blank(r["trace_fp32_unfused"]),
            "trace_fp32_fused": _nan_to_blank(r["trace_fp32_fused"]),
            "trace_ptq": _nan_to_blank(r["trace_ptq"]),
            "trace_fp32_fused_qat": _nan_to_blank(r["trace_fp32_fused_qat"]),
            "trace_qat": _nan_to_blank(r["trace_qat"]),
            "fusion_ratio": _nan_to_blank(r["fusion_ratio"]),
            "ptq_amplification": _nan_to_blank(r["ptq_amplification"]),
            "qat_amplification": _nan_to_blank(r["qat_amplification"]),
            "elev_ptq": _nan_to_blank(r["elev_ptq"]), "elev_qat": _nan_to_blank(r["elev_qat"]),
            "verdict": r["verdict"],
        }, COMPARISON_FIELDNAMES)
        logger.info(
            f"[QuantInducedTrace] {model_name}/{dataset_name} layer={r['canonical_layer']}: "
            f"fusion_ratio={r['fusion_ratio']:.4g} ptq_amplification={r['ptq_amplification']:.4g} "
            f"qat_amplification={r['qat_amplification']:.4g} verdict={r['verdict']}"
        )

    return per_layer


# ---------------------------------------------------------------------------
# Banked FP32 profile reconciliation
# ---------------------------------------------------------------------------

def _load_banked_fp32_profile(path: str, model_name: str, dataset_name: str) -> dict[str, float] | None:
    if not path or not os.path.exists(path):
        logger.warning(f"[QuantInducedTrace] banked FP32 profile not found at {path!r}")
        return None
    df = pd.read_csv(path)
    trace_col = "trace" if "trace" in df.columns else ("hessian_trace" if "hessian_trace" in df.columns else None)
    if trace_col is None or not {"model", "dataset", "layer"}.issubset(df.columns):
        logger.warning(
            f"[QuantInducedTrace] banked FP32 profile at {path} missing expected columns "
            f"(need model, dataset, layer, trace|hessian_trace); columns={list(df.columns)}"
        )
        return None

    subset = df[(df["model"] == model_name) & (df["dataset"] == dataset_name)]
    if "stage" in subset.columns:
        subset = subset[subset["stage"] == "FP32"]
    if "init" in subset.columns:
        subset = subset[subset["init"] == "trained_fp32"]
    if subset.empty:
        return None

    subset = subset.copy()
    subset["layer_name"] = subset["layer"].apply(lambda s: s[:-len(".weight")] if isinstance(s, str) and s.endswith(".weight") else s)
    return dict(zip(subset["layer_name"], subset[trace_col]))


def _reconcile_banked_profile(
    model_name: str, dataset_name: str, comparison_rows: list[dict], banked_profile_path: str | None,
) -> tuple[str, str]:
    if not banked_profile_path:
        return "not_provided", ""

    label = f"{model_name}/{dataset_name}"
    banked = _load_banked_fp32_profile(banked_profile_path, model_name, dataset_name)
    if not banked:
        return "not_provided", f"banked profile path given but no matching rows for {label}"

    diffs = []
    mismatched = []
    for r in comparison_rows:
        layer = r["canonical_layer"]
        u = r["trace_fp32_unfused"]
        if layer not in banked or u is None:
            continue
        b = banked[layer]
        rel = abs(b - u) / max(abs(b), abs(u), 1e-9)
        diffs.append(rel)
        if rel > RECONCILE_TOLERANCE:
            mismatched.append((layer, b, u, rel))

    if not diffs:
        return "not_provided", f"banked profile given but no overlapping layers for {label}"

    frac_ok = sum(1 for rel in diffs if rel <= RECONCILE_TOLERANCE) / len(diffs)
    if frac_ok >= RECONCILE_MATCH_FRACTION:
        note = (
            f"banked FP32 profile AGREES with this run's fp32_unfused on {frac_ok*100:.0f}% of "
            f"{len(diffs)} common layers (tolerance {RECONCILE_TOLERANCE*100:.0f}% relative diff) -- "
            f"the established 'FP32 conv1 14x' claim is misattributed and should read PTQ."
        )
        matches = "yes"
    else:
        mismatch_desc = "; ".join(f"{l}: banked={b:.4g} vs unfused={u:.4g} ({rel*100:.0f}%)" for l, b, u, rel in mismatched[:10])
        note = (
            f"banked FP32 profile DISAGREES with this run's fp32_unfused on {(1-frac_ok)*100:.0f}% of "
            f"{len(diffs)} common layers (tolerance {RECONCILE_TOLERANCE*100:.0f}%) -- this looks like a "
            f"config/computation difference between the two runs, not a finding. Mismatched layers: {mismatch_desc}"
        )
        matches = "no"

    logger.info(f"[QuantInducedTrace] {label}: banked-profile reconciliation -- {note}")
    return matches, note


# ---------------------------------------------------------------------------
# Summary: conv1 spotlight, Spearman, reconciliation
# ---------------------------------------------------------------------------

def _write_summary(
    model_name: str, dataset_name: str, comparison_rows: list[dict], banked_profile_path: str | None, summary_csv_path: str,
) -> None:
    label = f"{model_name}/{dataset_name}"
    is_resnet = "resnet" in model_name

    spearman_fp32_ptq = float("nan")
    spearman_fp32_qat = float("nan")
    if is_resnet:
        common_ptq = [(r["trace_fp32_unfused"], r["trace_ptq"]) for r in comparison_rows
                      if r["trace_fp32_unfused"] is not None and r["trace_ptq"] is not None]
        if len(common_ptq) >= 3:
            spearman_fp32_ptq, _ = spearmanr(*zip(*common_ptq))
        common_qat = [(r["trace_fp32_unfused"], r["trace_qat"]) for r in comparison_rows
                      if r["trace_fp32_unfused"] is not None and r["trace_qat"] is not None]
        if len(common_qat) >= 3:
            spearman_fp32_qat, _ = spearmanr(*zip(*common_qat))
    else:
        logger.info(f"[QuantInducedTrace] {label}: cnn has too few layers to correlate -- Spearman not computed")

    conv1_row = next((r for r in comparison_rows if r["canonical_layer"] == "conv1"), None)
    conv1_fp32 = conv1_row["trace_fp32_unfused"] if conv1_row else None
    conv1_ptq = conv1_row["trace_ptq"] if conv1_row else None
    conv1_ptq_amp = conv1_row["ptq_amplification"] if conv1_row else float("nan")

    ranked = sorted(
        (r for r in comparison_rows if not (isinstance(r["ptq_amplification"], float) and math.isnan(r["ptq_amplification"]))),
        key=lambda r: r["ptq_amplification"], reverse=True,
    )
    conv1_amp_rank = None
    top_amplified_layer = ranked[0]["canonical_layer"] if ranked else ""
    if conv1_row is not None:
        for i, r in enumerate(ranked, start=1):
            if r["canonical_layer"] == "conv1":
                conv1_amp_rank = i
                break

    if conv1_row is not None and ranked:
        if conv1_amp_rank == 1:
            logger.info(
                f"[QuantInducedTrace] {label}: conv1 IS the most quantization-amplified layer "
                f"(ptq_amplification={conv1_ptq_amp:.4g}, rank 1/{len(ranked)})"
            )
        else:
            logger.info(
                f"[QuantInducedTrace] {label}: conv1 is NOT the most amplified layer -- rank "
                f"{conv1_amp_rank}/{len(ranked)} (top={top_amplified_layer}, conv1 "
                f"ptq_amplification={conv1_ptq_amp:.4g}) -- amplification is broadly distributed, "
                f"conv1's dominance looks like a global quantization effect rather than conv1-specific"
            )
    elif conv1_row is None:
        logger.info(f"[QuantInducedTrace] {label}: no layer named 'conv1' in the canonical mapping -- spotlight skipped")

    banked_fp32_matches, reconcile_note = _reconcile_banked_profile(model_name, dataset_name, comparison_rows, banked_profile_path)

    _append_row(summary_csv_path, {
        "model": model_name, "dataset": dataset_name,
        "spearman_fp32_ptq": _nan_to_blank(spearman_fp32_ptq), "spearman_fp32_qat": _nan_to_blank(spearman_fp32_qat),
        "conv1_fp32": _nan_to_blank(conv1_fp32), "conv1_ptq": _nan_to_blank(conv1_ptq),
        "conv1_ptq_amplification": _nan_to_blank(conv1_ptq_amp), "conv1_amp_rank": _nan_to_blank(conv1_amp_rank),
        "top_amplified_layer": top_amplified_layer, "banked_fp32_matches": banked_fp32_matches,
        "note": reconcile_note,
    }, SUMMARY_FIELDNAMES)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_one_model(
    model_name: str, dataset_name: str, specs: dict, num_classes: int, device: torch.device,
    hessian_loader: DataLoader, fp32_models_dir: str, quant_dir: str, banked_profile_path: str | None,
    traces_csv: str, comparison_csv: str, summary_csv: str,
) -> None:
    channels, image_size = specs["channels"], specs["image_size"]
    criterion = nn.CrossEntropyLoss()
    label = f"{model_name}/{dataset_name}"

    # ---- Part 0: mapping gate (checkpoint-independent) ----
    unfused_skeleton = build_model(num_classes=num_classes, model_name=model_name, channels=channels, image_size=image_size)
    quant_skeleton = _build_quant_skeleton(model_name, num_classes, channels, image_size)
    mapping = _build_layer_mapping(model_name, unfused_skeleton, quant_skeleton)
    del unfused_skeleton, quant_skeleton

    # ---- checkpoint resolution (robust, as in P1) ----
    try:
        fp32_ckpt = _resolve_checkpoint_robust(fp32_models_dir, {"model": model_name, "dataset": dataset_name})
    except FileNotFoundError as exc:
        logger.warning(f"[QuantInducedTrace] {label}: FP32 baseline checkpoint missing ({exc}) -- skipping model (required reference)")
        return
    except WeightAblationCheckpointError as exc:
        logger.error(f"[QuantInducedTrace] {label}: FP32 baseline checkpoint resolution AMBIGUOUS/NEAR-MISS -- {exc}")
        return

    stage_ckpts: dict[str, str] = {}
    for stage in ("PTQ", "QAT"):
        try:
            stage_ckpts[stage] = _resolve_checkpoint_robust(quant_dir, {"stage": stage, "model": model_name, "dataset": dataset_name})
            logger.info(f"[QuantInducedTrace] {label}: resolved {stage} checkpoint -> {stage_ckpts[stage]}")
        except FileNotFoundError as exc:
            logger.warning(f"[QuantInducedTrace] {label}: {stage} checkpoint missing ({exc}) -- skipping {stage} stage")
        except WeightAblationCheckpointError as exc:
            logger.error(f"[QuantInducedTrace] {label}: {stage} checkpoint resolution AMBIGUOUS/NEAR-MISS -- {exc} -- skipping {stage} stage")

    # ---- Part 1: trace the variants ----
    traces: dict[str, dict[str, float]] = {}

    model = _load_unfused_fp32(model_name, fp32_ckpt, num_classes, channels, image_size, device)
    traces["fp32_unfused"] = _trace_variant(model, hessian_loader, criterion, device)
    _write_variant_traces(model_name, dataset_name, "fp32_unfused", mapping, traces["fp32_unfused"], "canonical", traces_csv)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if "PTQ" in stage_ckpts:
        ptq_model = _load_quantized(model_name, stage_ckpts["PTQ"], num_classes, channels, image_size, device)
        traces["ptq"] = _trace_variant(ptq_model, hessian_loader, criterion, device)
        _write_variant_traces(model_name, dataset_name, "ptq", mapping, traces["ptq"], "quantized", traces_csv)
        del ptq_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        fused_model = _load_quantized(model_name, stage_ckpts["PTQ"], num_classes, channels, image_size, device)
        _make_fused_fp32(fused_model, f"{label} fp32_fused")
        traces["fp32_fused"] = _trace_variant(fused_model, hessian_loader, criterion, device)
        _write_variant_traces(model_name, dataset_name, "fp32_fused", mapping, traces["fp32_fused"], "quantized", traces_csv)
        del fused_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "QAT" in stage_ckpts:
        qat_model = _load_quantized(model_name, stage_ckpts["QAT"], num_classes, channels, image_size, device)
        traces["qat"] = _trace_variant(qat_model, hessian_loader, criterion, device)
        _write_variant_traces(model_name, dataset_name, "qat", mapping, traces["qat"], "quantized", traces_csv)
        del qat_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        fused_qat_model = _load_quantized(model_name, stage_ckpts["QAT"], num_classes, channels, image_size, device)
        _make_fused_fp32(fused_qat_model, f"{label} fp32_fused_qat")
        traces["fp32_fused_qat"] = _trace_variant(fused_qat_model, hessian_loader, criterion, device)
        _write_variant_traces(model_name, dataset_name, "fp32_fused_qat", mapping, traces["fp32_fused_qat"], "quantized", traces_csv)
        del fused_qat_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "ptq" not in traces and "qat" not in traces:
        logger.warning(f"[QuantInducedTrace] {label}: neither PTQ nor QAT checkpoint available -- decomposition will be all-NaN, only fp32_unfused recorded")

    # ---- Part 2: decomposition + reconciliation ----
    comparison_rows = _decompose_and_classify(model_name, dataset_name, mapping, traces, comparison_csv)
    _write_summary(model_name, dataset_name, comparison_rows, banked_profile_path, summary_csv)


def run_quant_induced_trace(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    banked_fp32_profile: str | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[QuantInducedTrace] CUDA not available -- falling back to CPU, this will be slow.")
    _enable_determinism()
    logger.info(f"[QuantInducedTrace] device={device} probe_seed={PROBE_SEED} banked_fp32_profile={banked_fp32_profile}")

    fp32_models_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)
    quant_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)

    os.makedirs(CSV_DIR, exist_ok=True)
    traces_csv = os.path.join(CSV_DIR, "quant_induced_traces.csv")
    comparison_csv = os.path.join(CSV_DIR, "quant_induced_comparison.csv")
    summary_csv = os.path.join(CSV_DIR, "quant_induced_summary.csv")

    for dataset_name in DATASETS:
        specs = DATASET_SPECS[dataset_name]
        try:
            hessian_loader, num_classes = _build_hessian_loader(dataset_name)
        except Exception as exc:
            logger.warning(f"[QuantInducedTrace] {dataset_name}: could not load dataset ({exc}) -- skipping")
            continue

        for model_name in ORDERED_MODELS:
            logger.info(f"[QuantInducedTrace] === {model_name}/{dataset_name} ===")
            try:
                _run_one_model(
                    model_name, dataset_name, specs, num_classes, device,
                    hessian_loader, fp32_models_dir, quant_dir, banked_fp32_profile,
                    traces_csv, comparison_csv, summary_csv,
                )
            except QuantInducedMappingError as exc:
                if model_name in REQUIRED_MODELS:
                    logger.error(f"[QuantInducedTrace] {model_name}/{dataset_name}: REQUIRED model failed the Part 0 mapping gate -- {exc}")
                else:
                    logger.warning(f"[QuantInducedTrace] {model_name}/{dataset_name}: optional model did not fuse cleanly (Part 0 gate failed) -- skipping. {exc}")
            except DiagnoseActivationsError as exc:
                logger.error(f"[QuantInducedTrace] {model_name}/{dataset_name}: Identity-swap verification FAILED -- {exc}")
            except Exception as exc:
                logger.error(f"[QuantInducedTrace] FAILED {model_name}/{dataset_name}: {exc}", exc_info=True)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    logger.info("[QuantInducedTrace] === Quant-Induced-Trace complete ===")
