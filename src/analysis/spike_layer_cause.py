"""
spike_layer_cause.py -- identifies the "spike layer" (the layer whose
quantization state most matters to accuracy) per model x dataset, then
attributes its size-independent curvature excess to architectural
descriptors, using CIFAR10-vs-IMAGENET100 as a resolution contrast and a
KFAC-style forward/backward split as a source-localisation test.

Motivation / reframing: the earlier plan hardcoded conv1 as "the" spike
layer. weight_ablation_canonical.py's results show the spike position is
architecture-dependent -- conv1 in the resnets, fc1 in the cnn (per
quant_induced_trace.py's CIFAR10 findings), unknown for the IMAGENET100
variants. This module makes NO assumption about which layer that is; it is
selected per model x dataset from actual data (Part 1) two independent ways
-- highest fused-basis Tr(H) (spike_by_trace) and highest |weight_damage_pts|
(spike_by_damage) -- and if they disagree, the mechanism study (Parts 3-6)
runs on both, since the disagreement is itself hypothesis-discriminating
information, not noise to average away.

Three hypotheses under test, discriminated by dataset resolution and A/G
(input/output-gradient) source:
  H1 spatial extent / weight-sharing multiplicity (conv only; forward/A term;
     predicts excess scales with input resolution).
  H2 fan-in (dataset-independent; predicts excess concentrated in low-fan-in
     layers regardless of resolution).
  H3 input-covariance conditioning (predicts excess tracks input statistics
     -- dataset-dependent for the stem, dataset-independent for the head;
     carried by Tr(A)).
H1 and H2 predict different spike positions (stem vs. lowest-fan-in-whatever-
depth), so Part 1's two-way spike selection is itself a discriminator.

Seven parts, per model x dataset (cnn, resnet18_no_weights,
resnet50_no_weights x CIFAR10, IMAGENET100), variants fp32_unfused,
fp32_fused, ptq (QAT excluded -- a training regime, not an architectural
property):

  Part 0 (gate): extends the frozen trace_config.json (written by
    --relock-traces) with an IMAGENET100 canonical entry under a new
    "datasets" key -- appended, the original CIFAR10-implicit top-level keys
    (data/estimator/basis/...) that weight_ablation_canonical.py's
    _log_trace_config already reads are left byte-for-byte untouched. Probe
    count/image count are matched WITHIN a dataset; IMAGENET100's reduced
    budget relative to CIFAR10 (224x224 HVP cost) is recorded in the note
    field, not hidden. Verified by reading the file back and asserting both
    dataset keys are present.
  Part 1 (spike selection): spike_by_trace = argmax fused-basis Tr(H) (this
    module's own Part 2 trace, computed under the frozen/extended config --
    NOT re-read from a possibly differently-configured banked
    canonical_traces.csv, for full internal consistency across Parts 1-6).
    spike_by_damage = argmax |weight_damage_pts|, reusing an existing
    weight_ablation_canonical-style PTQ damage CSV if one is found for that
    model/dataset (glob over results/*/csv/), else computed fresh via
    weight_ablation_canonical.py's own _run_isolation_sweep (reused
    unchanged) and banked to spike_layer_cause_ptq_damage.csv for reuse.
  Part 2 (per-layer traces): Tr(H) via compute_layerwise_hessian_trace_
    pyhessian (unchanged), per layer x variant; trace_per_param = trace_raw
    / numel; log-log regression of trace_raw vs numel (slope, R^2) to check
    the ~size-invariant raw budget seen on CIFAR10 holds on IMAGENET100 too.
  Part 3 (size-independent residual): log-log fit of trace_per_param vs
    numel; each layer's residual above that fit is the quantity Part 4
    attributes.
  Part 4 (architectural descriptors + attribution): fan_in/fan_out/kernel/
    input_map/output_map/numel extracted via the SAME forward pass as Part
    5's KFAC hooks (architecture-only, but dataset-dependent for the stem --
    conv1 is a 3x3/stride1/no-maxpool stem on CIFAR10 vs 7x7/stride2/maxpool
    on IMAGENET100, see src/model_cnn/pretrained_resnet18.py::
    _adapt_first_conv). Part 3's residual is regressed (Spearman +
    numel-controlled partial Pearson correlation) against output_map,
    fan_in, and Tr(A_prev) per dataset.
  Part 5 (KFAC A/G split): one forward+backward pass on the canonical batch
    (autograd required -- no torch.no_grad()) with forward-pre hooks (layer
    input) and full-backward hooks (grad w.r.t. layer output). Conv layers
    use patch-unfolded input covariance (KFC, Grosse & Martens 2016) via
    F.unfold with the layer's own kernel/stride/padding/dilation -- this is
    where weight-sharing multiplicity enters, so naive per-channel
    covariance would blur H1 and H3 together. Linear layers use the raw
    input-feature covariance. Only traces are needed (never the full A/G
    matrices), so Tr(A) = mean over (samples x spatial positions) of
    ||patch||^2, and Tr(G) analogously over grad_output. predicted_per_param
    ~= [Tr(A)/fan_in] * [Tr(G)/fan_out] (Part 5's own approximation, not the
    exact KFAC Fisher block) is compared against Part 2/3's measured
    trace_per_param as an honesty check on the approximation.
  Part 6 (resolution contrast): per model, compares the spike layer's Part 3
    residual and Part 5 Tr(A) between CIFAR10 and IMAGENET100 (fp32_unfused
    variant -- the clean architectural signal, not quantization-noised);
    reports resolution_contrast_ratio and which hypothesis (H1/H2/H3/mixed/
    none) the contrast supports, with explicit, logged thresholds -- never a
    hidden cutoff. When spike_by_trace and spike_by_damage disagree, this
    (and Part 4's attribution) runs once per candidate spike layer.

Reuses (does not duplicate): compute_layerwise_hessian_trace_pyhessian and
pyhessian._single_batch (src/analysis/pyhessian.py, unchanged); the Part 0
name/shape mapping gate and model-construction helpers (_build_layer_mapping,
_load_unfused_fp32, _build_quant_skeleton, _load_quantized, _make_fused_fp32,
PROBE_SEED -- src/analysis/quant_induced_trace.py); the checkpoint loader,
Identity-swap helpers, FP32 checkpoint-dir resolver and _append_row CSV
writer (src/analysis/diagnose_activations.py); the robust checkpoint
resolver and eval-loader builder (src/analysis/_ablation_common.py);
weight_ablation_canonical.py's own _run_isolation_sweep (unchanged) for
spike_by_damage's fallback computation; _resolve_trace_config_path (src/
analysis/weight_ablation_canonical.py); and _safe_div / _enable_determinism
(src/analysis/random_init_control.py).

Analysis only -- no torchao/deployment code. Runs as a single local process
(`python -m src.main --spike-layer-cause ...`), no SLURM/torchrun required;
prefers CUDA (A100 in production). eval() mode, deterministic algorithms,
fixed probe seed, shuffle=False -- but NOT torch.no_grad(), since Part 5's
KFAC measurement needs a real backward pass.
"""

import os
import glob
import json
import math
import logging
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import spearmanr, pearsonr, linregress
import pandas as pd

from src.model_cnn.train import build_model
from src.analysis.diagnose_activations import (
    _resolve_fp32_models_dir,
    _load_fp32_reference,
    _append_row,
    DiagnoseActivationsError,
)
from src.analysis._ablation_common import (
    _resolve_checkpoint_robust,
    WeightAblationCheckpointError,
    _build_eval_loader,
)
from src.analysis.weight_ablation_canonical import (
    _run_isolation_sweep,
    _resolve_trace_config_path,
)
from src.analysis.quant_induced_trace import (
    _build_layer_mapping,
    _load_unfused_fp32,
    _build_quant_skeleton,
    _load_quantized,
    _make_fused_fp32,
    QuantInducedMappingError,
    PROBE_SEED,
)
from src.analysis.pyhessian import compute_layerwise_hessian_trace_pyhessian, _single_batch, _disable_inplace_ops
from src.analysis.random_init_control import _safe_div, _enable_determinism
from src.quantization.deploy_fbgemm import _resolve_checkpoint_dir
from src.utility.config import CSV_DIR, DATASET_SPECS, RESULTS_DIR
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

DATASETS = ["CIFAR10", "IMAGENET100"]
# resnet50 first -- it carries the headline conv1 spike this mode investigates.
ORDERED_MODELS = ["resnet50_no_weights", "resnet18_no_weights", "cnn"]
VARIANTS = ["fp32_unfused", "fp32_fused", "ptq"]

# Per-dataset Hessian-trace estimator config (Part 0). CIFAR10 mirrors the
# frozen canonical config (relock_traces.py: batch=16, num_batches=5 ->
# 80 images, max_iter=100). IMAGENET100 is reduced -- 224x224 HVP cost is far
# higher per iteration than 32x32 -- and that reduction is recorded in
# trace_config.json's note field (Part 0), not hidden. Matched WITHIN each
# dataset across every model/variant in this module.
DATASET_TRACE_CONFIG = {
    "CIFAR10":     {"batch_size": 16, "num_batches": 5, "max_iter": 100, "tol": 1e-3},
    "IMAGENET100": {"batch_size": 8,  "num_batches": 3, "max_iter": 30,  "tol": 1e-3},
}

# Part 6 verdict thresholds, applied to the reported ratio, never hidden.
RESOLUTION_RATIO_ELEVATED = 1.5   # ratio >= this on IMAGENET100/CIFAR10 -> "substantially larger"
RESOLUTION_RATIO_SIMILAR_LO = 1.0 / 1.5
RESOLUTION_RATIO_SIMILAR_HI = 1.5

SPIKE_SELECTION_FIELDNAMES = ["model", "dataset", "spike_by_trace", "spike_by_damage", "agreement"]
TRACES_FIELDNAMES = [
    "model", "dataset", "variant", "layer", "layer_type", "numel", "trace_raw", "trace_per_param",
    "elev_raw", "elev_per_param", "raw_vs_numel_slope", "raw_vs_numel_r2",
]
RESIDUAL_FIELDNAMES = ["model", "dataset", "variant", "layer", "per_param_residual", "matched_count_ratio"]
DESCRIPTORS_FIELDNAMES = [
    "model", "dataset", "layer", "layer_type", "fan_in", "fan_out", "kh", "kw",
    "input_map", "output_map", "numel",
]
KFAC_FIELDNAMES = [
    "model", "dataset", "variant", "layer", "tr_A", "tr_G",
    "A_per_infan", "G_per_outfan", "predicted_per_param", "measured_per_param",
]
ATTRIBUTION_FIELDNAMES = [
    "model", "dataset", "spike_layer", "descriptor", "spearman_vs_residual", "partial_corr_controlling_numel",
    "spike_A_share", "spike_G_share", "resolution_contrast_ratio", "hypothesis_supported",
]
# This module's own fallback PTQ-damage bank (distinct filename/schema from
# weight_ablation_canonical.py's own CSVs -- never written to that filename,
# to avoid corrupting its richer schema with this module's leaner one).
DAMAGE_FIELDNAMES = ["model", "dataset", "stage", "layer", "weight_damage_pts", "abs_weight_damage_pts"]
SELF_COMPUTED_DAMAGE_FILENAME = "spike_layer_cause_ptq_damage.csv"


class SpikeLayerCauseError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Part 0: extend trace_config.json with an IMAGENET100 entry
# ---------------------------------------------------------------------------

def _extend_trace_config_with_imagenet100(canonical_traces_csv: str) -> dict:
    path = _resolve_trace_config_path(canonical_traces_csv)
    if not os.path.exists(path):
        raise SpikeLayerCauseError(
            f"trace_config.json not found at {path} (sibling of {canonical_traces_csv}) -- "
            f"run --relock-traces first to freeze the CIFAR10 canonical config."
        )
    with open(path) as f:
        config = json.load(f)

    if "datasets" not in config:
        # First extension -- snapshot the existing (CIFAR10-implicit) top-level
        # keys under datasets.CIFAR10 for symmetry, WITHOUT touching those
        # top-level keys themselves (weight_ablation_canonical.py's
        # _log_trace_config reads them directly and must keep working).
        config["datasets"] = {
            "CIFAR10": {
                "batch_size": config["data"]["batch_size"],
                "num_batches": config["data"]["num_batches"],
                "num_images": config["data"]["num_images"],
                "max_iter": config["estimator"]["max_iter"],
                "tol": config["estimator"]["tol"],
                "note": "original frozen CIFAR10 config (relock_traces.py Part 0) -- referenced here, not duplicated logic; top-level keys are the source of truth and are unchanged by this extension.",
            }
        }

    cfg = DATASET_TRACE_CONFIG["IMAGENET100"]
    imagenet_num_images = cfg["batch_size"] * cfg["num_batches"]
    cifar_cfg = config["datasets"]["CIFAR10"]
    config["datasets"]["IMAGENET100"] = {
        "batch_size": cfg["batch_size"], "num_batches": cfg["num_batches"],
        "num_images": imagenet_num_images, "max_iter": cfg["max_iter"], "tol": cfg["tol"],
        "normalization": (
            "standard IMAGENET100 mean/std (src/utility/utils.py _norm); val transform only "
            "(Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize, no augmentation)"
        ),
        "note": (
            f"probe count (max_iter={cfg['max_iter']}) and image count ({imagenet_num_images}) reduced "
            f"from CIFAR10's (max_iter={cifar_cfg['max_iter']}, num_images={cifar_cfg['num_images']}) -- "
            "224x224 HVP cost is far higher per iteration than 32x32; this is a recorded cross-dataset "
            "difference (spike_layer_cause.py Part 0), not a silent one. Matched WITHIN IMAGENET100 across "
            "every model/variant in this module -- spike_layer_cause.py's cross-dataset comparisons "
            "(Part 6) use ratios/residuals, not raw trace magnitudes, precisely because of this difference."
        ),
    }

    with open(path, "w") as f:
        json.dump(config, f, indent=2)

    with open(path) as f:
        verify = json.load(f)
    assert "CIFAR10" in verify.get("datasets", {}) and "IMAGENET100" in verify.get("datasets", {}), (
        f"Part 0 verification FAILED -- both dataset keys must be present in {path} after write, "
        f"got datasets={list(verify.get('datasets', {}).keys())}"
    )
    logger.info(
        f"[SpikeLayerCause] Part 0: trace_config.json extended with IMAGENET100 entry -> {path} "
        f"(CIFAR10 entry preserved; verified by read-back)"
    )
    return verify


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _build_hessian_loader(dataset_name: str) -> tuple[DataLoader, int]:
    cfg = DATASET_TRACE_CONFIG[dataset_name]
    _, val_loader, num_classes = get_data_loaders(dataset_name)
    loader = DataLoader(
        val_loader.dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, pin_memory=False,
    )
    return loader, num_classes


def _trace_variant(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, cfg: dict) -> dict[str, float]:
    torch.manual_seed(PROBE_SEED)
    return compute_layerwise_hessian_trace_pyhessian(
        model, loader, criterion, device, num_batches=cfg["num_batches"], max_iter=cfg["max_iter"], tol=cfg["tol"],
    )


# ---------------------------------------------------------------------------
# Part 5 (+ Part 4 descriptors): KFAC forward/backward hooks
# ---------------------------------------------------------------------------

def _fan_in_out(weight: torch.Tensor) -> tuple[int, int]:
    out_c, in_c = weight.shape[0], weight.shape[1]
    receptive = 1
    for s in weight.shape[2:]:
        receptive *= s
    return in_c * receptive, out_c * receptive


def _measure_kfac_and_descriptors(
    model: nn.Module, mapping_names: list[str], inputs: torch.Tensor, targets: torch.Tensor,
    criterion: nn.Module, device: torch.device,
) -> dict[str, dict]:
    """
    One forward+backward pass on (inputs, targets) with forward-pre hooks
    (layer input, for Tr(A)) and full-backward hooks (grad w.r.t. layer
    output, for Tr(G)) on every named layer. Conv layers use patch-unfolded
    input covariance (KFC); linear layers use the raw input-feature
    covariance. Does NOT use torch.no_grad() -- a real backward pass is
    required. Returns per layer: layer_type, fan_in, fan_out, kh, kw,
    input_map, output_map, numel, tr_A, tr_G, A_per_infan, G_per_outfan.
    """
    model.eval()
    # torchvision's resnet18/50 BasicBlock/Bottleneck use an inplace residual
    # add (out += identity) directly in forward() -- not toggleable via any
    # module .inplace flag, so _disable_inplace_ops (reused below for the
    # inplace=True ReLUs) cannot fix this part. register_full_backward_hook
    # is module-boundary-based and cannot coexist with that inplace add
    # ("Output 0 of BackwardHookFunction is a view and is being modified
    # inplace"). The standard workaround -- used here -- is to capture the
    # gradient via Tensor.register_hook() on the output tensor itself from a
    # plain register_forward_hook, which ties the hook to that tensor's own
    # autograd node rather than to module-level bookkeeping, and is robust
    # to whatever happens to the tensor's storage afterward.
    _disable_inplace_ops(model)
    named = dict(model.named_modules())
    layers = [(name, named[name]) for name in mapping_names if name in named]

    input_store: dict[str, torch.Tensor] = {}
    grad_store: dict[str, torch.Tensor] = {}
    handles = []

    def make_fwd_hook(name):
        def hook(module, inp):
            input_store[name] = inp[0].detach()
        return hook

    def make_output_grad_hook(name):
        def hook(module, inp, output):
            def grad_hook(grad):
                grad_store[name] = grad.detach()
            output.register_hook(grad_hook)
        return hook

    for name, module in layers:
        handles.append(module.register_forward_pre_hook(make_fwd_hook(name)))
        handles.append(module.register_forward_hook(make_output_grad_hook(name)))

    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)

    results: dict[str, dict] = {}
    for name, module in layers:
        x = input_store.get(name)
        g = grad_store.get(name)
        if x is None or g is None:
            logger.warning(
                f"[SpikeLayerCause] layer={name}: missing hook capture "
                f"(input_captured={x is not None}, grad_captured={g is not None}) -- skipped"
            )
            continue

        weight = module.weight
        numel = weight.numel()

        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise SpikeLayerCauseError(
                    f"{name}: grouped conv (groups={module.groups}) not supported by this patch-based "
                    f"(KFC) implementation -- every conv in this project's resnet18/50/cnn is groups=1, "
                    f"so a non-1 value here means an unexpected architecture change, not a case to silently paper over."
                )
            layer_type = "conv"
            kh, kw = module.kernel_size
            input_map = x.shape[2] * x.shape[3]
            output_map = g.shape[2] * g.shape[3]
            # (N, fan_in, L) patches -- L == output_map by construction (same
            # kernel/stride/padding/dilation as the conv itself).
            patches = F.unfold(x, kernel_size=module.kernel_size, dilation=module.dilation, padding=module.padding, stride=module.stride)
            tr_A = patches.pow(2).sum(dim=1).mean().item()
            g_flat = g.flatten(2)  # (N, out_channels, L_out)
            tr_G = g_flat.pow(2).sum(dim=1).mean().item()
        else:
            assert isinstance(module, nn.Linear), f"{name}: expected Conv2d or Linear, got {type(module).__name__}"
            layer_type = "linear"
            kh, kw = 1, 1
            input_map, output_map = 1, 1
            tr_A = x.pow(2).sum(dim=1).mean().item()
            tr_G = g.pow(2).sum(dim=1).mean().item()

        fan_in, fan_out = _fan_in_out(weight)
        results[name] = {
            "layer_type": layer_type, "fan_in": fan_in, "fan_out": fan_out, "kh": kh, "kw": kw,
            "input_map": input_map, "output_map": output_map, "numel": numel,
            "tr_A": tr_A, "tr_G": tr_G,
            "A_per_infan": _safe_div(tr_A, fan_in), "G_per_outfan": _safe_div(tr_G, fan_out),
        }

    return results


# ---------------------------------------------------------------------------
# Part 2 + Part 3: per-layer traces, log-log regressions, size-independent residual
# ---------------------------------------------------------------------------

def _loglog_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """log10(y) ~ log10(x). Returns (slope, intercept, r2); NaNs if <3 usable (positive) points."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and x > 0 and y > 0]
    if len(pairs) < 3:
        return float("nan"), float("nan"), float("nan")
    lx = [math.log10(x) for x, _ in pairs]
    ly = [math.log10(y) for _, y in pairs]
    result = linregress(lx, ly)
    return result.slope, result.intercept, result.rvalue ** 2


def _write_variant_trace_and_residual_rows(
    model_name: str, dataset_name: str, variant: str, mapping_names: list[str],
    trace_dict: dict[str, float], numel_map: dict[str, int], layer_type_map: dict[str, str],
    traces_csv: str, residual_csv: str,
) -> dict[str, dict]:
    layer_data: dict[str, dict] = {}
    for name in mapping_names:
        trace_val = trace_dict.get(f"{name}.weight")
        numel = numel_map.get(name)
        if trace_val is None or not numel:
            continue
        layer_data[name] = {"trace_raw": trace_val, "numel": numel, "trace_per_param": trace_val / numel}

    if not layer_data:
        logger.warning(f"[SpikeLayerCause] {model_name}/{dataset_name} variant={variant}: no layers with both trace and numel -- skipping")
        return {}

    numels = [d["numel"] for d in layer_data.values()]
    raws = [d["trace_raw"] for d in layer_data.values()]
    per_params = [d["trace_per_param"] for d in layer_data.values()]

    slope_raw, _intercept_raw, r2_raw = _loglog_fit(numels, raws)
    slope_pp, intercept_pp, r2_pp = _loglog_fit(numels, per_params)

    median_raw = statistics.median(raws)
    median_pp = statistics.median(per_params)
    n_fit = sum(1 for n, r in zip(numels, raws) if n > 0 and r > 0)
    matched_ratio = n_fit / len(layer_data)

    for name, d in layer_data.items():
        layer_type = layer_type_map.get(name, "")
        _append_row(traces_csv, {
            "model": model_name, "dataset": dataset_name, "variant": variant, "layer": name,
            "layer_type": layer_type, "numel": d["numel"], "trace_raw": d["trace_raw"],
            "trace_per_param": d["trace_per_param"],
            "elev_raw": _safe_div(d["trace_raw"], median_raw), "elev_per_param": _safe_div(d["trace_per_param"], median_pp),
            "raw_vs_numel_slope": slope_raw, "raw_vs_numel_r2": r2_raw,
        }, TRACES_FIELDNAMES)

        if d["numel"] > 0 and d["trace_per_param"] > 0 and slope_pp == slope_pp:
            residual = math.log10(d["trace_per_param"]) - (slope_pp * math.log10(d["numel"]) + intercept_pp)
        else:
            residual = float("nan")
        d["residual"] = residual

        _append_row(residual_csv, {
            "model": model_name, "dataset": dataset_name, "variant": variant, "layer": name,
            "per_param_residual": residual, "matched_count_ratio": matched_ratio,
        }, RESIDUAL_FIELDNAMES)

    logger.info(
        f"[SpikeLayerCause] {model_name}/{dataset_name} variant={variant}: {len(layer_data)} layers -- "
        f"raw_vs_numel(slope={slope_raw:.3f}, r2={r2_raw:.3f}) trace_per_param_vs_numel(slope={slope_pp:.3f}, r2={r2_pp:.3f})"
    )
    return layer_data


# ---------------------------------------------------------------------------
# Part 1: spike selection
# ---------------------------------------------------------------------------

def _spike_by_trace(trace_rows_fp32_fused: dict[str, dict]) -> str | None:
    if not trace_rows_fp32_fused:
        return None
    return max(trace_rows_fp32_fused.keys(), key=lambda n: trace_rows_fp32_fused[n]["trace_raw"])


def _spike_by_damage(damage: dict[str, float]) -> str | None:
    if not damage:
        return None
    return max(damage.keys(), key=lambda n: abs(damage[n]))


def _find_existing_damage(model_name: str, dataset_name: str) -> tuple[dict[str, float], str] | None:
    for fname in ("weight_ablation_canonical_v2.csv", "weight_ablation_canonical.csv", SELF_COMPUTED_DAMAGE_FILENAME):
        for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "csv", fname)), reverse=True):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if not {"model", "dataset", "stage", "layer", "weight_damage_pts"}.issubset(df.columns):
                continue
            subset = df[(df["model"] == model_name) & (df["dataset"] == dataset_name) & (df["stage"] == "PTQ")]
            if not subset.empty:
                return dict(zip(subset["layer"], subset["weight_damage_pts"])), path
    return None


def _compute_damage_via_isolation(
    model_name: str, dataset_name: str, fp32_ckpt: str, ptq_ckpt: str,
    num_classes: int, channels: int, image_size: int, eval_loader: DataLoader, device: torch.device,
    mapping_names: list[str],
) -> dict[str, float]:
    from src.main import evaluate
    label = f"{model_name}/{dataset_name} (spike-layer-cause PTQ damage)"
    fp32_model = _load_fp32_reference(model_name, fp32_ckpt, num_classes, channels, image_size).to(device)
    with torch.no_grad():
        fp32_acc = evaluate(fp32_model, eval_loader, device)
    del fp32_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    damage = _run_isolation_sweep(
        model_name, dataset_name, "PTQ", ptq_ckpt, num_classes, channels, image_size,
        eval_loader, device, fp32_acc, mapping_names,
    )
    logger.info(f"[SpikeLayerCause] {label}: computed PTQ isolation damage fresh ({len(mapping_names)} layers, fp32_acc={fp32_acc:.2f}%)")
    return damage


def _get_damage_by_layer(
    model_name: str, dataset_name: str, fp32_ckpt: str, ptq_ckpt: str | None,
    num_classes: int, channels: int, image_size: int, device: torch.device,
    mapping_names: list[str], damage_csv: str,
) -> dict[str, float]:
    found = _find_existing_damage(model_name, dataset_name)
    if found is not None:
        damage, path = found
        logger.info(f"[SpikeLayerCause] {model_name}/{dataset_name}: reusing existing PTQ weight-ablation damage from {path} ({len(damage)} layers)")
        return damage
    if ptq_ckpt is None:
        logger.warning(f"[SpikeLayerCause] {model_name}/{dataset_name}: no existing damage data AND no PTQ checkpoint -- spike_by_damage unavailable")
        return {}
    eval_loader, _ = _build_eval_loader(dataset_name)
    damage = _compute_damage_via_isolation(
        model_name, dataset_name, fp32_ckpt, ptq_ckpt, num_classes, channels, image_size, eval_loader, device, mapping_names,
    )
    for name, d in damage.items():
        _append_row(damage_csv, {
            "model": model_name, "dataset": dataset_name, "stage": "PTQ", "layer": name,
            "weight_damage_pts": d, "abs_weight_damage_pts": abs(d),
        }, DAMAGE_FIELDNAMES)
    return damage


def _write_spike_selection(model_name: str, dataset_name: str, spike_trace: str | None, spike_damage: str | None, csv_path: str) -> None:
    agreement = "n/a" if (spike_trace is None or spike_damage is None) else ("yes" if spike_trace == spike_damage else "no")
    _append_row(csv_path, {
        "model": model_name, "dataset": dataset_name,
        "spike_by_trace": spike_trace or "", "spike_by_damage": spike_damage or "", "agreement": agreement,
    }, SPIKE_SELECTION_FIELDNAMES)
    logger.info(f"[SpikeLayerCause] {model_name}/{dataset_name}: spike_by_trace={spike_trace} spike_by_damage={spike_damage} agreement={agreement}")


# ---------------------------------------------------------------------------
# Part 4: attribution regression (per dataset)
# ---------------------------------------------------------------------------

def _partial_corr(x: list[float], y: list[float], z: list[float]) -> float:
    """Pearson partial correlation of x,y controlling for z, via linear residualization."""
    x_arr, y_arr, z_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
    x_arr, y_arr, z_arr = x_arr[mask], y_arr[mask], z_arr[mask]
    if len(x_arr) < 4:
        return float("nan")

    def _resid(a, b):
        design = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    rx, ry = _resid(x_arr, z_arr), _resid(y_arr, z_arr)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    r, _ = pearsonr(rx, ry)
    return r


def _run_attribution_regression(
    mapping_names: list[str], residual_by_layer: dict[str, float],
    descriptors: dict[str, dict], kfac: dict[str, dict],
) -> dict[str, dict]:
    """Per descriptor (output_map, fan_in, tr_A_prev): Spearman vs. residual
    (bivariate, unadjusted) and partial Pearson correlation controlling for
    log(numel). Descriptors are the layer's OWN structural properties;
    tr_A_prev is Part 5's own Tr(A) for that layer (named "_prev" per the
    spec's phrasing -- Tr(A) is defined over that layer's own input, i.e.
    the previous layer's output)."""
    layers = [n for n in mapping_names if n in residual_by_layer and n in descriptors and n in kfac and residual_by_layer[n] == residual_by_layer[n]]
    if len(layers) < 4:
        return {}

    residuals = [residual_by_layer[n] for n in layers]
    log_numel = [math.log10(descriptors[n]["numel"]) for n in layers]
    descriptor_series = {
        "output_map": [descriptors[n]["output_map"] for n in layers],
        "fan_in": [descriptors[n]["fan_in"] for n in layers],
        "tr_A_prev": [kfac[n]["tr_A"] for n in layers],
    }

    result = {}
    for desc_name, values in descriptor_series.items():
        rho, _p = spearmanr(values, residuals)
        pc = _partial_corr(values, residuals, log_numel)
        result[desc_name] = {"spearman_vs_residual": rho, "partial_corr_controlling_numel": pc}
    return result


# ---------------------------------------------------------------------------
# Part 6: resolution contrast + attribution CSV
# ---------------------------------------------------------------------------

def _spike_ag_share(spike_layer: str, kfac: dict[str, dict]) -> tuple[float, float]:
    """
    Elevation of the spike layer's A/G-per-fan factors above the model's
    median layer (in log space), normalised into a share -- which factor
    carries the spike's elevation above a typical layer. Not a probability;
    can exceed 1 or go negative (a factor that works AGAINST the spike's
    elevation), which is itself informative.
    """
    a_vals = [v["A_per_infan"] for v in kfac.values() if v["A_per_infan"] == v["A_per_infan"] and v["A_per_infan"] > 0]
    g_vals = [v["G_per_outfan"] for v in kfac.values() if v["G_per_outfan"] == v["G_per_outfan"] and v["G_per_outfan"] > 0]
    spike = kfac.get(spike_layer)
    if spike is None or not a_vals or not g_vals:
        return float("nan"), float("nan")
    a_spike, g_spike = spike["A_per_infan"], spike["G_per_outfan"]
    if not (a_spike > 0 and g_spike > 0):
        return float("nan"), float("nan")
    elev_a = math.log10(a_spike) - math.log10(statistics.median(a_vals))
    elev_g = math.log10(g_spike) - math.log10(statistics.median(g_vals))
    denom = elev_a + elev_g
    if abs(denom) < 1e-9:
        return float("nan"), float("nan")
    return elev_a / denom, elev_g / denom


def _classify_hypothesis(resid_ratio: float, spike_A_share: float, spike_G_share: float, layer_type: str) -> str:
    if resid_ratio != resid_ratio:  # NaN
        return "undetermined"
    if spike_G_share == spike_G_share and spike_A_share == spike_A_share and spike_G_share > spike_A_share:
        return "mixed (backward/G-dominated -- neither H1 nor H3 alone)"
    if resid_ratio >= RESOLUTION_RATIO_ELEVATED:
        return "H1" if layer_type == "conv" else "H3"
    if RESOLUTION_RATIO_SIMILAR_LO <= resid_ratio <= RESOLUTION_RATIO_SIMILAR_HI:
        return "H2"
    return "mixed"


def _write_attribution_for_spike(
    model_name: str, spike_layer: str,
    per_dataset_regression: dict[str, dict[str, dict]],
    per_dataset_residual: dict[str, dict[str, float]],
    per_dataset_kfac: dict[str, dict[str, dict]],
    per_dataset_layer_type: dict[str, dict[str, str]],
    csv_path: str,
) -> None:
    datasets_present = [d for d in DATASETS if spike_layer in per_dataset_residual.get(d, {})]

    resid_ratio = float("nan")
    if "CIFAR10" in datasets_present and "IMAGENET100" in datasets_present:
        r_cifar = per_dataset_residual["CIFAR10"][spike_layer]
        r_imagenet = per_dataset_residual["IMAGENET100"][spike_layer]
        # residuals are log10-space and can be <=0; compare via the DIFFERENCE
        # exponentiated back to a ratio-like quantity so a sign flip doesn't
        # produce a meaningless negative "ratio".
        resid_ratio = 10 ** (r_imagenet - r_cifar) if (r_cifar == r_cifar and r_imagenet == r_imagenet) else float("nan")

    layer_type = ""
    for d in datasets_present:
        layer_type = per_dataset_layer_type.get(d, {}).get(spike_layer, layer_type)

    for dataset_name in datasets_present:
        regression = per_dataset_regression.get(dataset_name, {})
        kfac = per_dataset_kfac.get(dataset_name, {})
        a_share, g_share = _spike_ag_share(spike_layer, kfac) if kfac else (float("nan"), float("nan"))
        hypothesis = _classify_hypothesis(resid_ratio, a_share, g_share, layer_type)

        for desc_name in ("output_map", "fan_in", "tr_A_prev"):
            stats = regression.get(desc_name, {"spearman_vs_residual": float("nan"), "partial_corr_controlling_numel": float("nan")})
            _append_row(csv_path, {
                "model": model_name, "dataset": dataset_name, "spike_layer": spike_layer, "descriptor": desc_name,
                "spearman_vs_residual": stats["spearman_vs_residual"], "partial_corr_controlling_numel": stats["partial_corr_controlling_numel"],
                "spike_A_share": a_share, "spike_G_share": g_share,
                "resolution_contrast_ratio": resid_ratio, "hypothesis_supported": hypothesis,
            }, ATTRIBUTION_FIELDNAMES)

    logger.info(
        f"[SpikeLayerCause] {model_name}: spike={spike_layer} layer_type={layer_type or '?'} "
        f"resolution_contrast_ratio(IMAGENET100/CIFAR10 residual)={resid_ratio:.4g} -- hypothesis_supported={_classify_hypothesis(resid_ratio, float('nan'), float('nan'), layer_type)}"
    )


# ---------------------------------------------------------------------------
# Per model x dataset orchestration
# ---------------------------------------------------------------------------

def _run_model_dataset(
    model_name: str, dataset_name: str, specs: dict, num_classes: int, device: torch.device,
    fp32_ckpt: str, ptq_ckpt: str | None,
    traces_csv: str, residual_csv: str, descriptors_csv: str, kfac_csv: str, damage_csv: str,
) -> dict | None:
    channels, image_size = specs["channels"], specs["image_size"]
    cfg = DATASET_TRACE_CONFIG[dataset_name]
    criterion = nn.CrossEntropyLoss()
    label = f"{model_name}/{dataset_name}"

    unfused_skel = build_model(num_classes=num_classes, model_name=model_name, channels=channels, image_size=image_size)
    quant_skel = _build_quant_skeleton(model_name, num_classes, channels, image_size)
    mapping = _build_layer_mapping(model_name, unfused_skel, quant_skel)   # raises QuantInducedMappingError
    del unfused_skel, quant_skel
    mapping_names = [row["canonical_name"] for row in mapping]

    try:
        hessian_loader, _ = _build_hessian_loader(dataset_name)
    except Exception as exc:
        logger.error(f"[SpikeLayerCause] {label}: could not build hessian loader ({exc}) -- skipping")
        return None
    batch = _single_batch(hessian_loader, cfg["num_batches"], device)
    if batch is None:
        logger.error(f"[SpikeLayerCause] {label}: empty hessian loader -- skipping")
        return None
    inputs, targets = batch

    variant_builders = {"fp32_unfused": lambda: _load_unfused_fp32(model_name, fp32_ckpt, num_classes, channels, image_size, device)}
    if ptq_ckpt is not None:
        variant_builders["ptq"] = lambda: _load_quantized(model_name, ptq_ckpt, num_classes, channels, image_size, device)
        variant_builders["fp32_fused"] = lambda: _make_fused_fp32(
            _load_quantized(model_name, ptq_ckpt, num_classes, channels, image_size, device), f"{label} fp32_fused",
        )
    else:
        logger.warning(f"[SpikeLayerCause] {label}: PTQ checkpoint missing -- fp32_fused/ptq variants skipped, fp32_unfused only")

    trace_by_variant: dict[str, dict[str, float]] = {}
    kfac_by_variant: dict[str, dict[str, dict]] = {}
    descriptors: dict[str, dict] | None = None
    numel_map: dict[str, int] | None = None
    layer_type_map: dict[str, str] | None = None

    for variant in VARIANTS:
        builder = variant_builders.get(variant)
        if builder is None:
            continue
        model = builder()
        trace_by_variant[variant] = _trace_variant(model, hessian_loader, criterion, device, cfg)
        kfac = _measure_kfac_and_descriptors(model, mapping_names, inputs, targets, criterion, device)
        kfac_by_variant[variant] = kfac
        if variant == "fp32_unfused":
            descriptors = {n: {k: v[k] for k in ("layer_type", "fan_in", "fan_out", "kh", "kw", "input_map", "output_map", "numel")} for n, v in kfac.items()}
            numel_map = {n: v["numel"] for n, v in kfac.items()}
            layer_type_map = {n: v["layer_type"] for n, v in kfac.items()}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if descriptors is None or numel_map is None or layer_type_map is None:
        logger.error(f"[SpikeLayerCause] {label}: fp32_unfused variant failed -- cannot extract descriptors, skipping")
        return None

    for name in mapping_names:
        d = descriptors.get(name)
        if d is None:
            continue
        _append_row(descriptors_csv, {"model": model_name, "dataset": dataset_name, "layer": name, **d}, DESCRIPTORS_FIELDNAMES)

    trace_rows_by_variant: dict[str, dict[str, dict]] = {}
    residual_by_variant: dict[str, dict[str, float]] = {}
    for variant, trace_dict in trace_by_variant.items():
        rows = _write_variant_trace_and_residual_rows(
            model_name, dataset_name, variant, mapping_names, trace_dict, numel_map, layer_type_map, traces_csv, residual_csv,
        )
        trace_rows_by_variant[variant] = rows
        residual_by_variant[variant] = {n: d["residual"] for n, d in rows.items()}

    for variant, kfac in kfac_by_variant.items():
        trace_rows = trace_rows_by_variant.get(variant, {})
        for name, k in kfac.items():
            measured = trace_rows.get(name, {}).get("trace_per_param")
            a_ok = k["A_per_infan"] == k["A_per_infan"]
            g_ok = k["G_per_outfan"] == k["G_per_outfan"]
            predicted = k["A_per_infan"] * k["G_per_outfan"] if (a_ok and g_ok) else float("nan")
            _append_row(kfac_csv, {
                "model": model_name, "dataset": dataset_name, "variant": variant, "layer": name,
                "tr_A": k["tr_A"], "tr_G": k["tr_G"], "A_per_infan": k["A_per_infan"], "G_per_outfan": k["G_per_outfan"],
                "predicted_per_param": predicted, "measured_per_param": measured if measured is not None else "",
            }, KFAC_FIELDNAMES)

    spike_trace = _spike_by_trace(trace_rows_by_variant.get("fp32_fused") or trace_rows_by_variant.get("fp32_unfused") or {})

    damage = _get_damage_by_layer(model_name, dataset_name, fp32_ckpt, ptq_ckpt, num_classes, channels, image_size, device, mapping_names, damage_csv)
    spike_damage = _spike_by_damage(damage)

    regression = _run_attribution_regression(
        mapping_names, residual_by_variant.get("fp32_unfused", {}), descriptors, kfac_by_variant.get("fp32_unfused", {}),
    )

    return {
        "mapping_names": mapping_names, "spike_trace": spike_trace, "spike_damage": spike_damage,
        "residual_fp32_unfused": residual_by_variant.get("fp32_unfused", {}),
        "kfac_fp32_unfused": kfac_by_variant.get("fp32_unfused", {}),
        "layer_type_map": layer_type_map, "regression": regression,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_spike_layer_cause(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    canonical_traces_csv: str,
    imagenet100_checkpoint_dir: str | None = None,
) -> None:
    """
    imagenet100_checkpoint_dir: this module's whole point is a cross-dataset
    (CIFAR10 vs IMAGENET100) comparison, but in practice no single run_id in
    this project has banked freshly-trained checkpoints for BOTH datasets at
    once (each training run tends to focus on one) -- --checkpoint-dir /
    --load-run-id alone cannot express "use run A for CIFAR10, run B for
    IMAGENET100". When given, this overrides the checkpoint DIRECTORY (not
    --load-run-id) used for IMAGENET100 only; CIFAR10 always resolves via
    --checkpoint-dir/--load-run-id as every other mode does. Omit to use the
    same source for both datasets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[SpikeLayerCause] CUDA not available -- falling back to CPU, this will be slow.")
    _enable_determinism()
    logger.info(f"[SpikeLayerCause] device={device} canonical_traces_csv={canonical_traces_csv} imagenet100_checkpoint_dir={imagenet100_checkpoint_dir}")

    _extend_trace_config_with_imagenet100(canonical_traces_csv)

    checkpoint_dirs_by_dataset = {
        "CIFAR10": (_resolve_fp32_models_dir(checkpoint_dir, load_run_id), _resolve_checkpoint_dir(checkpoint_dir, load_run_id)),
    }
    if imagenet100_checkpoint_dir:
        checkpoint_dirs_by_dataset["IMAGENET100"] = (_resolve_fp32_models_dir(imagenet100_checkpoint_dir, None), _resolve_checkpoint_dir(imagenet100_checkpoint_dir, None))
    else:
        checkpoint_dirs_by_dataset["IMAGENET100"] = checkpoint_dirs_by_dataset["CIFAR10"]

    os.makedirs(CSV_DIR, exist_ok=True)
    spike_selection_csv = os.path.join(CSV_DIR, "spike_selection.csv")
    traces_csv = os.path.join(CSV_DIR, "spike_layer_traces.csv")
    residual_csv = os.path.join(CSV_DIR, "spike_layer_residual.csv")
    descriptors_csv = os.path.join(CSV_DIR, "spike_layer_descriptors.csv")
    kfac_csv = os.path.join(CSV_DIR, "spike_layer_kfac.csv")
    attribution_csv = os.path.join(CSV_DIR, "spike_layer_attribution.csv")
    damage_csv = os.path.join(CSV_DIR, SELF_COMPUTED_DAMAGE_FILENAME)

    for model_name in ORDERED_MODELS:
        per_dataset_regression: dict[str, dict] = {}
        per_dataset_residual: dict[str, dict[str, float]] = {}
        per_dataset_kfac: dict[str, dict[str, dict]] = {}
        per_dataset_layer_type: dict[str, dict[str, str]] = {}
        spikes_seen: set[str] = set()

        for dataset_name in DATASETS:
            label = f"{model_name}/{dataset_name}"
            logger.info(f"[SpikeLayerCause] === {label} ===")
            specs = DATASET_SPECS[dataset_name]
            fp32_models_dir, quant_dir = checkpoint_dirs_by_dataset[dataset_name]

            try:
                fp32_ckpt = _resolve_checkpoint_robust(fp32_models_dir, {"model": model_name, "dataset": dataset_name})
            except (FileNotFoundError, WeightAblationCheckpointError) as exc:
                logger.warning(f"[SpikeLayerCause] {label}: FP32 baseline checkpoint unresolvable ({exc}) -- skipping dataset for this model")
                continue

            ptq_ckpt = None
            try:
                ptq_ckpt = _resolve_checkpoint_robust(quant_dir, {"stage": "PTQ", "model": model_name, "dataset": dataset_name})
            except FileNotFoundError as exc:
                logger.warning(f"[SpikeLayerCause] {label}: PTQ checkpoint missing ({exc}) -- fp32_fused/ptq variants and spike_by_damage will be degraded")
            except WeightAblationCheckpointError as exc:
                logger.error(f"[SpikeLayerCause] {label}: PTQ checkpoint AMBIGUOUS/NEAR-MISS -- {exc} -- proceeding without it")

            num_classes = specs["num_classes"]

            try:
                result = _run_model_dataset(
                    model_name, dataset_name, specs, num_classes, device, fp32_ckpt, ptq_ckpt,
                    traces_csv, residual_csv, descriptors_csv, kfac_csv, damage_csv,
                )
            except QuantInducedMappingError as exc:
                logger.error(f"[SpikeLayerCause] {label}: Part 0 mapping gate FAILED -- {exc} -- skipping")
                continue
            except DiagnoseActivationsError as exc:
                logger.error(f"[SpikeLayerCause] {label}: Identity-swap verification FAILED -- {exc} -- skipping")
                continue
            except Exception as exc:
                logger.error(f"[SpikeLayerCause] FAILED {label}: {exc}", exc_info=True)
                continue
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if result is None:
                continue

            _write_spike_selection(model_name, dataset_name, result["spike_trace"], result["spike_damage"], spike_selection_csv)
            per_dataset_regression[dataset_name] = result["regression"]
            per_dataset_residual[dataset_name] = result["residual_fp32_unfused"]
            per_dataset_kfac[dataset_name] = result["kfac_fp32_unfused"]
            per_dataset_layer_type[dataset_name] = result["layer_type_map"]
            for s in (result["spike_trace"], result["spike_damage"]):
                if s is not None:
                    spikes_seen.add(s)

        # ---- Part 6: resolution contrast + attribution, once per candidate spike ----
        for spike_layer in spikes_seen:
            _write_attribution_for_spike(
                model_name, spike_layer, per_dataset_regression, per_dataset_residual,
                per_dataset_kfac, per_dataset_layer_type, attribution_csv,
            )

    logger.info("[SpikeLayerCause] === Spike-Layer-Cause complete ===")
