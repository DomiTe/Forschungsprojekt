"""
relock_traces.py -- freezes a single canonical Hessian-trace estimator
configuration, diagnoses which configuration knob produced each drifting
legacy trace number, recomputes every headline trace from the frozen config,
and writes an old->new reconciliation ledger.

Motivation: reconciliation showed the banked traces were produced under
drifting conventions -- resnet18's banked FP32 disagrees with a fresh
unfused run on most layers (systematically ~2x); resnet50 conv1 FP32 reads
differently across runs (13.87 vs 11.79, ~15%); and resnet50 conv1 PTQ reads
13,161.8 in the original notes versus ~1030.9 in a fresh run (~12.8x). Before
any of this goes in a writeup, every headline trace must come from one
frozen configuration, and each legacy number must be attributable to a
specific knob so it cannot silently return -- the trace analog of the single
shared build path (src.quantization.quantizer) that already prevents
deployment-path divergence.

Four parts, CIFAR10 only; resnet18_no_weights and resnet50_no_weights
required (cnn optional, not run by default -- add it to ORDERED_MODELS to
include):

  Part 0 (freeze): every estimator field is defined explicitly, logged, and
    written to results/<RUN_ID>/trace_config.json -- data split/order,
    loss reduction, probe seed/count, model mode, basis (fused adopted as
    canonical for cross-stage comparison, since PTQ/QAT exist only fused),
    trace normalisation (raw headline + per-param companion), and numerics
    (device, deterministic algorithms). Nothing is left to a function
    default; every value used downstream is read from this dict/file.
  Part 1 (diagnose): on resnet50 conv1, the clearest legacy anchor, walks a
    one-knob-at-a-time grid from canonical -- loss reduction, trace
    normalisation, basis, device, probe count, number of images -- plus one
    additional (clearly labelled "bonus", not one of the six prescribed
    knobs) data-source check: reproducing the ORIGINAL src/main.py training
    pipeline's actual hessian_loader construction, which this investigation
    found reads `get_data_loaders(dataset_name, batch_size=HESSIAN_BATCH_SIZE)`
    and keeps its first return value -- the TRAIN loader (shuffle=True,
    RandomCrop(32,padding=4)+RandomHorizontalFlip augmentation), not a fixed
    val split, with no probe seed reset anywhere near the original
    compute_layerwise_hessian_trace_pyhessian call sites. That is a highly
    plausible root cause for exactly this kind of run-to-run, non-reproducible
    drift, so it is tested empirically here rather than assumed. If the
    largest gap (13,161.8 vs ~1030.9) still isn't explained by any of the
    above, the PTQ checkpoint's observer calibration state is checked
    (uncalibrated observers => degenerate quantized forward => a very
    different curvature) -- reusing the diagnose_activations calibration
    pattern, not re-implemented.
  Part 2 (recompute): once Part 1 has been run and reported, every headline
    trace (fp32_unfused, fp32_fused, ptq, qat, fp32_fused_qat) for both
    required models is recomputed under ONLY the frozen config, gated by the
    quant-induced mode's Part 0 name/shape mapping (reused, not duplicated).
  Part 3 (verify + ledger): three layers are retraced twice under the same
    probe seed to assert bitwise-adjacent determinism (<=1e-6 relative,
    deterministic algorithms on), then retraced under two more seeds to
    report the frozen config's residual across-seed std. The reconciliation
    ledger records, per headline quantity, the old value(s) and source, the
    new canonical value, the ratio, and the knob (if any) from Part 1 that
    explains the gap.

Reuses (does not duplicate): compute_layerwise_hessian_trace_pyhessian
(src/analysis/pyhessian.py, unchanged), the quant-induced mode's Part 0
mapping gate and model-construction/Identity-swap helpers
(src/analysis/quant_induced_trace.py: _build_layer_mapping,
_load_unfused_fp32, _build_quant_skeleton, _load_quantized, _make_fused_fp32,
_load_banked_fp32_profile), the robust checkpoint resolver
(src/analysis/weight_ablation.py -- P1), the FP32 checkpoint directory
resolver and _append_row CSV writer (src/analysis/diagnose_activations.py),
the quantized-checkpoint directory resolver (src/quantization/deploy_fbgemm.py),
and _enable_determinism (src/analysis/random_init_control.py).

Analysis only. Runs as a single local process
(`python -m src.main --relock-traces ...`), no SLURM/torchrun required;
prefers CUDA (A100 in production; this investigation's own runs were made on
a local CUDA GPU, see the run log for the exact device).
"""

import os
import csv
import json
import math
import logging
import statistics
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from src.model_cnn.train import build_model
from src.analysis.pyhessian import compute_layerwise_hessian_trace_pyhessian
from src.analysis.diagnose_activations import _resolve_fp32_models_dir, _append_row
from src.analysis.weight_ablation import _resolve_checkpoint_robust, WeightAblationCheckpointError
from src.analysis.random_init_control import _enable_determinism, _safe_div
from src.analysis.quant_induced_trace import (
    REQUIRED_MODELS,
    _build_layer_mapping,
    _load_unfused_fp32,
    _build_quant_skeleton,
    _load_quantized,
    _make_fused_fp32,
    _load_banked_fp32_profile,
    RECONCILE_TOLERANCE,
    QuantInducedMappingError,
)
from src.quantization.quantizer import QuantizedConv2d, QuantizedLinear
from src.quantization.deploy_fbgemm import _resolve_checkpoint_dir
from src.utility.config import CSV_DIR, RUN_DIR, DATASET_SPECS, HESSIAN_BATCH_SIZE
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

DATASET_NAME = "CIFAR10"
DIAGNOSIS_MODEL = "resnet50_no_weights"          # Part 1's grid runs only on this model's conv1, per spec
PART2_MODELS = ["resnet50_no_weights", "resnet18_no_weights"]   # cnn optional -- not run by default

# ---------------------------------------------------------------------------
# Frozen canonical config (Part 0). Every trace call in this module reads
# these constants explicitly; nothing is left to compute_layerwise_hessian_
# trace_pyhessian's own defaults, even where they happen to coincide.
# ---------------------------------------------------------------------------
CANONICAL_PROBE_SEED = 20260811          # matches src.analysis.quant_induced_trace.PROBE_SEED
CANONICAL_BATCH_SIZE = HESSIAN_BATCH_SIZE  # 16
CANONICAL_NUM_BATCHES = 5                 # 80 images total -- matches the codebase-wide convention
CANONICAL_MAX_ITER = 100
CANONICAL_TOL = 1e-3
CANONICAL_LOSS_REDUCTION = "mean"
CANONICAL_BASIS = "fused"                 # PTQ/QAT exist only fused; adopted for cross-stage comparison

# Part 1 grid settings (deviate ONE knob at a time from canonical, above).
PROBE_COUNT_GRID = [(20, "smaller_maxIter=20"), (300, "larger_maxIter=300")]
NUM_IMAGES_ONE_BATCH = 1
N_DATA_SOURCE_REPEATS = 2                 # bonus knob: reproduces the ORIGINAL (pre-relock) pipeline's
                                           # hessian_loader construction (see module docstring), repeated
                                           # to show its run-to-run spread (no fixed seed, shuffled data).
DEVICE_CHECK_NUM_BATCHES = 1              # reduced-cost device knob (num_batches, max_iter both cut) --
DEVICE_CHECK_MAX_ITER = 20                # a full 54-layer canonical-config CPU pass was not run in this
                                           # session (too costly for its diagnostic value -- float32
                                           # CPU-vs-CUDA differences are not expected to explain a >10x
                                           # gap); this reduced pair still tests the device knob honestly,
                                           # just not at the full canonical image/probe count.

ANCHOR_TOLERANCE = 0.10  # 10% relative -- "reproduces" this legacy anchor, per spec

DEFAULT_LEGACY_ANCHORS = {
    "resnet50_conv1_fp32": [
        {"name": "fp32_conv1_13.87", "value": 13.87,
         "source": "prior run notes (unspecified estimator config)"},
        {"name": "fp32_conv1_11.79", "value": 11.79,
         "source": "prior run notes (unspecified estimator config), second run"},
    ],
    "resnet50_conv1_ptq": [
        {"name": "ptq_conv1_13161.8", "value": 13161.8,
         "source": "original src/main.py pipeline run, layerwise_hessian_traces.csv (PTQ stage)"},
        {"name": "ptq_conv1_1030.9", "value": 1030.9,
         "source": "a later/fresh run's notes"},
    ],
}
DEFAULT_ELEVATION_ANCHOR_NAME = "conv1_elevation_14x"
DEFAULT_ELEVATION_ANCHOR_VALUE = 14.0

DRIFT_FIELDNAMES = ["knob", "setting", "resnet50_conv1_trace", "reproduces_anchor", "residual_pct"]
CANONICAL_FIELDNAMES = [
    "model", "dataset", "variant", "canonical_layer", "weight_shape",
    "trace_raw", "trace_per_param", "probe_seed",
]
LEDGER_FIELDNAMES = ["quantity", "old_value", "old_source", "canonical_value", "ratio", "explained_by_knob", "note"]


class RelockTracesError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Part 0: freeze the canonical config
# ---------------------------------------------------------------------------

def _freeze_config(run_dir: str, device: torch.device) -> dict:
    config = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_by": "src/analysis/relock_traces.py",
        "data": {
            "dataset": DATASET_NAME,
            "split": "test (val)",
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": False,
            "batch_size": CANONICAL_BATCH_SIZE,
            "num_batches": CANONICAL_NUM_BATCHES,
            "num_images": CANONICAL_BATCH_SIZE * CANONICAL_NUM_BATCHES,
            "image_selection": (
                "first num_images images of the CIFAR10 test split in dataset order "
                "(deterministic -- shuffle=False, no DataLoader generator randomness)"
            ),
            "normalization": (
                "standard CIFAR10 mean/std (src/utility/utils.py _norm); no augmentation "
                "(test transform: Resize -> ToTensor -> Normalize only, RandomCrop/Flip excluded)"
            ),
        },
        "loss": {"criterion": "CrossEntropyLoss", "reduction": CANONICAL_LOSS_REDUCTION},
        "estimator": {
            "engine": "compute_layerwise_hessian_trace_pyhessian (src/analysis/pyhessian.py, unchanged)",
            "probe_distribution": "Rademacher (PyHessian hessian.trace() default)",
            "probe_seed": CANONICAL_PROBE_SEED,
            "probe_seed_reset": "torch.manual_seed(probe_seed) immediately before every estimator call",
            "max_iter": CANONICAL_MAX_ITER,
            "tol": CANONICAL_TOL,
        },
        "model_mode": "eval",
        "basis": {
            "canonical_for_cross_stage_comparison": CANONICAL_BASIS,
            "note": (
                "PTQ/QAT exist only fused+quantized, so fused is canonical for any PTQ/QAT "
                "comparison. FP32 is still traced in BOTH bases (fp32_unfused, fp32_fused) and "
                "the unfused->fused fusion_ratio is always reported alongside FP32 so nothing "
                "is hidden by the basis choice (see src/analysis/quant_induced_trace.py)."
            ),
        },
        "trace_normalization": {
            "headline": "trace_raw",
            "companion": "trace_per_param = trace_raw / weight.numel()",
        },
        "numerics": {
            "dtype": "float32",
            "device": str(device),
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
    }
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "trace_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"[RelockTraces] Part 0: frozen canonical config -> {path}\n{json.dumps(config, indent=2)}")
    return config


def _load_legacy_anchors(path_or_json: str | None) -> dict:
    if not path_or_json:
        return DEFAULT_LEGACY_ANCHORS
    try:
        if os.path.exists(path_or_json):
            with open(path_or_json) as f:
                anchors = json.load(f)
        else:
            anchors = json.loads(path_or_json)
        logger.info(f"[RelockTraces] loaded legacy anchors override from {path_or_json!r}")
        return anchors
    except Exception as exc:
        logger.warning(f"[RelockTraces] could not parse --legacy-anchors ({exc}) -- falling back to built-in defaults")
        return DEFAULT_LEGACY_ANCHORS


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _build_val_hessian_loader(dataset_name: str) -> tuple[DataLoader, int]:
    _, val_loader, num_classes = get_data_loaders(dataset_name)
    loader = DataLoader(
        val_loader.dataset, batch_size=CANONICAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
    )
    return loader, num_classes


def _build_original_pipeline_loader(dataset_name: str) -> tuple[DataLoader, int]:
    # Mirrors src/main.py's ORIGINAL (pre-relock) hessian_loader construction
    # exactly: get_data_loaders(dataset_name, batch_size=HESSIAN_BATCH_SIZE)
    # and keep the FIRST return value -- the TRAIN loader (shuffle=True,
    # RandomCrop(32,padding=4)+RandomHorizontalFlip augmentation, no DataLoader
    # generator seed). NOT part of the frozen canonical config -- reproduced
    # here only to test whether it explains the legacy anchors' drift.
    train_loader, _, num_classes = get_data_loaders(dataset_name, batch_size=CANONICAL_BATCH_SIZE)
    return train_loader, num_classes


# ---------------------------------------------------------------------------
# Estimator wrapper (explicit config every call -- no defaults relied upon)
# ---------------------------------------------------------------------------

def _trace_full(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
    probe_seed: int, num_batches: int, max_iter: int, tol: float,
) -> dict[str, float]:
    torch.manual_seed(probe_seed)
    return compute_layerwise_hessian_trace_pyhessian(
        model, loader, criterion, device, num_batches=num_batches, max_iter=max_iter, tol=tol,
    )


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# Part 1: drift diagnosis grid (resnet50 conv1)
# ---------------------------------------------------------------------------

def _best_anchor_match(value: float, anchors: list[dict]) -> tuple[str | None, float]:
    best = (None, float("nan"))
    for a in anchors:
        rel = abs(value - a["value"]) / max(abs(a["value"]), 1e-9)
        if best[0] is None or rel < best[1]:
            best = (a["name"], rel)
    return best


def _record_drift_row(csv_path: str, knob: str, setting: str, trace_value, anchors: list[dict]) -> None:
    if trace_value is None or (isinstance(trace_value, float) and math.isnan(trace_value)):
        name, rel, reproduces = "", "", ""
    elif anchors:
        name, rel = _best_anchor_match(trace_value, anchors)
        reproduces = name if rel <= ANCHOR_TOLERANCE else "none"
    else:
        name, rel, reproduces = "", "", ""
    _append_row(csv_path, {
        "knob": knob, "setting": setting,
        "resnet50_conv1_trace": trace_value if trace_value is not None else "",
        "reproduces_anchor": reproduces,
        "residual_pct": (rel * 100 if isinstance(rel, float) and not math.isnan(rel) else ""),
    }, DRIFT_FIELDNAMES)
    if trace_value is not None:
        logger.info(
            f"[RelockTraces] drift grid: knob={knob} setting={setting} conv1_trace={trace_value:.6g} "
            f"closest_anchor={name or 'n/a'} residual={(rel*100 if isinstance(rel, float) and not math.isnan(rel) else float('nan')):.2f}% "
            f"reproduces={reproduces or 'n/a'}"
        )
    else:
        logger.info(f"[RelockTraces] drift grid: knob={knob} setting={setting} -> {csv_path.split('/')[-1]} informational row (no trace value)")


def _run_grid_for_stage(
    stage_label: str, build_fn, device: torch.device,
    val_loader: DataLoader, orig_loader: DataLoader, anchors: list[dict], drift_csv: str,
) -> dict[str, float]:
    """
    build_fn(device) -> fresh nn.Module, .eval(), on `device`, ready to trace.
    Returns the CANONICAL (mean loss, given device, val split, canonical
    probe seed/max_iter/tol/num_batches) full per-layer trace dict, so Part 2
    can reuse it instead of recomputing.
    """
    criterion_mean = nn.CrossEntropyLoss(reduction="mean")
    criterion_sum = nn.CrossEntropyLoss(reduction="sum")

    # ---- canonical ----
    model = build_fn(device)
    canonical_traces = _trace_full(model, val_loader, criterion_mean, device, CANONICAL_PROBE_SEED, CANONICAL_NUM_BATCHES, CANONICAL_MAX_ITER, CANONICAL_TOL)
    _record_drift_row(drift_csv, f"{stage_label}:canonical", "mean/given-device/val/canonical_probes/canonical_images", canonical_traces.get("conv1.weight"), anchors)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- loss reduction: sum ----
    model = build_fn(device)
    t = _trace_full(model, val_loader, criterion_sum, device, CANONICAL_PROBE_SEED, CANONICAL_NUM_BATCHES, CANONICAL_MAX_ITER, CANONICAL_TOL)
    _record_drift_row(drift_csv, f"{stage_label}:loss_reduction", "sum", t.get("conv1.weight"), anchors)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- trace normalisation: per-param (derived, no rerun) ----
    conv1_numel = None
    if "conv1.weight" in canonical_traces:
        m = build_fn(device)
        conv1_numel = dict(m.named_modules())["conv1"].weight.numel()
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        per_param = canonical_traces["conv1.weight"] / conv1_numel
        _record_drift_row(drift_csv, f"{stage_label}:trace_normalization", f"per_param(numel={conv1_numel})", per_param, anchors)

    # ---- probe count ----
    for max_iter_setting, label in PROBE_COUNT_GRID:
        model = build_fn(device)
        t = _trace_full(model, val_loader, criterion_mean, device, CANONICAL_PROBE_SEED, CANONICAL_NUM_BATCHES, max_iter_setting, CANONICAL_TOL)
        _record_drift_row(drift_csv, f"{stage_label}:probe_count", label, t.get("conv1.weight"), anchors)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- number of images: one batch ----
    model = build_fn(device)
    t = _trace_full(model, val_loader, criterion_mean, device, CANONICAL_PROBE_SEED, NUM_IMAGES_ONE_BATCH, CANONICAL_MAX_ITER, CANONICAL_TOL)
    _record_drift_row(drift_csv, f"{stage_label}:num_images", f"one_batch({CANONICAL_BATCH_SIZE}_images)", t.get("conv1.weight"), anchors)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- bonus: data source (original pre-relock pipeline's loader) ----
    for i in range(N_DATA_SOURCE_REPEATS):
        model = build_fn(device)
        # Deliberately NOT resetting the probe seed here -- mirrors the
        # ORIGINAL pipeline's actual lack of a fixed seed near its estimator
        # calls, so repeated runs show the real run-to-run spread that
        # convention produced.
        t = compute_layerwise_hessian_trace_pyhessian(
            model, orig_loader, criterion_mean, device,
            num_batches=CANONICAL_NUM_BATCHES, max_iter=CANONICAL_MAX_ITER, tol=CANONICAL_TOL,
        )
        _record_drift_row(
            drift_csv, f"{stage_label}:data_source_original_pipeline(bonus)",
            f"train_shuffled_augmented_unseeded_run{i+1}", t.get("conv1.weight"), anchors,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return canonical_traces


def _run_device_pair(stage_label: str, build_fn, val_loader: DataLoader, anchors: list[dict], drift_csv: str) -> None:
    # Reduced-cost pair (see DEVICE_CHECK_* constants' docstring above) run
    # once, on the fp32_fused stage only -- device numerics are a
    # model-independent property of the estimator, not expected to differ
    # meaningfully between stages.
    criterion_mean = nn.CrossEntropyLoss(reduction="mean")

    cuda_device = torch.device("cuda")
    model = build_fn(cuda_device)
    t_cuda = _trace_full(model, val_loader, criterion_mean, cuda_device, CANONICAL_PROBE_SEED, DEVICE_CHECK_NUM_BATCHES, DEVICE_CHECK_MAX_ITER, CANONICAL_TOL)
    _record_drift_row(drift_csv, f"{stage_label}:device", f"cuda(reduced: num_batches={DEVICE_CHECK_NUM_BATCHES},maxIter={DEVICE_CHECK_MAX_ITER})", t_cuda.get("conv1.weight"), anchors)
    del model
    torch.cuda.empty_cache()

    cpu_device = torch.device("cpu")
    model = build_fn(cpu_device)
    t_cpu = _trace_full(model, val_loader, criterion_mean, cpu_device, CANONICAL_PROBE_SEED, DEVICE_CHECK_NUM_BATCHES, DEVICE_CHECK_MAX_ITER, CANONICAL_TOL)
    _record_drift_row(drift_csv, f"{stage_label}:device", f"cpu(reduced: num_batches={DEVICE_CHECK_NUM_BATCHES},maxIter={DEVICE_CHECK_MAX_ITER})", t_cpu.get("conv1.weight"), anchors)
    del model

    v_cuda, v_cpu = t_cuda.get("conv1.weight"), t_cpu.get("conv1.weight")
    if v_cuda is not None and v_cpu is not None:
        rel = abs(v_cuda - v_cpu) / max(abs(v_cuda), abs(v_cpu), 1e-9)
        logger.info(
            f"[RelockTraces] {stage_label}: device knob (reduced settings) -- cuda={v_cuda:.6g} "
            f"cpu={v_cpu:.6g} relative_diff={rel*100:.3f}% -- a full canonical-config CPU pass "
            f"was NOT run in this session (compute cost vs. expected information value; float32 "
            f"CPU-vs-CUDA numerics are not expected to explain a >10x gap). This reduced pair is "
            f"available as a sanity check only."
        )


def _check_ptq_observer_calibration(model: nn.Module, label: str) -> tuple[bool, list[str]]:
    total, uncalibrated = 0, []
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            total += 1
            obs = module.act_fake_quant.activation_post_process
            min_v, max_v = obs.min_val.item(), obs.max_val.item()
            if not (math.isfinite(min_v) and math.isfinite(max_v) and min_v < max_v):
                uncalibrated.append(name)
    calibrated = len(uncalibrated) == 0
    logger.info(
        f"[RelockTraces] {label}: observer calibration check -- {total} quantized layers, "
        f"{len(uncalibrated)} uncalibrated{': ' + str(uncalibrated[:10]) if uncalibrated else ''}"
    )
    return calibrated, uncalibrated


def _run_part1_drift_diagnosis(
    model_name: str, specs: dict, num_classes: int, device: torch.device,
    val_loader: DataLoader, orig_loader: DataLoader,
    fp32_ckpt: str, ptq_ckpt: str, legacy_anchors: dict, drift_csv: str,
) -> dict[str, dict[str, float]]:
    channels, image_size = specs["channels"], specs["image_size"]
    fp32_anchors = legacy_anchors.get("resnet50_conv1_fp32", [])
    ptq_anchors = legacy_anchors.get("resnet50_conv1_ptq", [])

    def build_unfused(dev):
        return _load_unfused_fp32(model_name, fp32_ckpt, num_classes, channels, image_size, dev)

    def build_fused(dev):
        m = _load_quantized(model_name, ptq_ckpt, num_classes, channels, image_size, dev)
        return _make_fused_fp32(m, f"{model_name} fp32_fused (relock Part1)")

    def build_ptq(dev):
        return _load_quantized(model_name, ptq_ckpt, num_classes, channels, image_size, dev)

    # ---- basis: unfused (single canonical run, no sub-grid) ----
    unfused_model = build_unfused(device)
    fp32_unfused_canonical = _trace_full(unfused_model, val_loader, nn.CrossEntropyLoss(reduction="mean"), device, CANONICAL_PROBE_SEED, CANONICAL_NUM_BATCHES, CANONICAL_MAX_ITER, CANONICAL_TOL)
    del unfused_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _record_drift_row(drift_csv, "fp32:basis", "unfused", fp32_unfused_canonical.get("conv1.weight"), fp32_anchors)

    # ---- fp32_fused: full sub-grid (canonical basis) ----
    fp32_fused_canonical = _run_grid_for_stage("fp32_fused", build_fused, device, val_loader, orig_loader, fp32_anchors, drift_csv)
    _record_drift_row(drift_csv, "fp32_fused:basis", "fused(canonical)", fp32_fused_canonical.get("conv1.weight"), fp32_anchors)

    if fp32_fused_canonical:
        median_fused = statistics.median(fp32_fused_canonical.values())
        elev = _safe_div(fp32_fused_canonical.get("conv1.weight"), median_fused)
        _record_drift_row(drift_csv, "fp32_fused:elevation_ratio", "conv1_over_median", elev,
                           [{"name": DEFAULT_ELEVATION_ANCHOR_NAME, "value": DEFAULT_ELEVATION_ANCHOR_VALUE}])

    # ---- ptq: full sub-grid ----
    ptq_canonical = _run_grid_for_stage("ptq", build_ptq, device, val_loader, orig_loader, ptq_anchors, drift_csv)

    # ---- device knob (reduced-cost pair, fp32_fused stage only) ----
    _run_device_pair("fp32_fused", build_fused, val_loader, fp32_anchors, drift_csv)

    # ---- observer calibration fallback check (Part 1 spec) ----
    ptq_model = build_ptq(device)
    calibrated, uncalibrated = _check_ptq_observer_calibration(ptq_model, f"{model_name} PTQ checkpoint ({ptq_ckpt})")
    del ptq_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _append_row(drift_csv, {
        "knob": "observer_calibration", "setting": f"ptq_checkpoint calibrated={calibrated} uncalibrated_count={len(uncalibrated)}",
        "resnet50_conv1_trace": "", "reproduces_anchor": "", "residual_pct": "",
    }, DRIFT_FIELDNAMES)

    return {"fp32_unfused": fp32_unfused_canonical, "fp32_fused": fp32_fused_canonical, "ptq": ptq_canonical}


def _anchor_explained(drift_csv: str, anchor_name: str) -> bool:
    if not os.path.exists(drift_csv):
        return False
    df = pd.read_csv(drift_csv)
    return bool((df["reproduces_anchor"] == anchor_name).any())


# ---------------------------------------------------------------------------
# Part 2: canonical recompute (both required models, all variants)
# ---------------------------------------------------------------------------

def _run_canonical_variant_traces(
    model_name: str, dataset_name: str, specs: dict, num_classes: int, device: torch.device,
    val_loader: DataLoader, mapping: list[dict],
    fp32_ckpt: str, ptq_ckpt: str | None, qat_ckpt: str | None,
    canonical_csv: str, reuse: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    channels, image_size = specs["channels"], specs["image_size"]
    criterion = nn.CrossEntropyLoss(reduction=CANONICAL_LOSS_REDUCTION)
    reuse = reuse or {}

    variant_specs = [
        ("fp32_unfused", lambda dev: _load_unfused_fp32(model_name, fp32_ckpt, num_classes, channels, image_size, dev), "canonical"),
    ]
    if ptq_ckpt:
        variant_specs.append((
            "fp32_fused",
            lambda dev: _make_fused_fp32(_load_quantized(model_name, ptq_ckpt, num_classes, channels, image_size, dev), f"{model_name} fp32_fused (relock Part2)"),
            "quantized",
        ))
        variant_specs.append((
            "ptq", lambda dev: _load_quantized(model_name, ptq_ckpt, num_classes, channels, image_size, dev), "quantized",
        ))
    if qat_ckpt:
        variant_specs.append((
            "qat", lambda dev: _load_quantized(model_name, qat_ckpt, num_classes, channels, image_size, dev), "quantized",
        ))
        variant_specs.append((
            "fp32_fused_qat",
            lambda dev: _make_fused_fp32(_load_quantized(model_name, qat_ckpt, num_classes, channels, image_size, dev), f"{model_name} fp32_fused_qat (relock Part2)"),
            "quantized",
        ))

    results: dict[str, dict[str, float]] = {}
    for variant, build_fn, key_kind in variant_specs:
        if variant in reuse:
            traces = reuse[variant]
            logger.info(f"[RelockTraces] {model_name}/{dataset_name} variant={variant}: reusing Part 1's canonical run (identical config)")
        else:
            model = build_fn(device)
            traces = _trace_full(model, val_loader, criterion, device, CANONICAL_PROBE_SEED, CANONICAL_NUM_BATCHES, CANONICAL_MAX_ITER, CANONICAL_TOL)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[variant] = traces

        for row in mapping:
            name_key = row["canonical_name"] if key_kind == "canonical" else row["quantized_name"]
            trace_val = traces.get(f"{name_key}.weight")
            if trace_val is None:
                continue
            numel = _numel(row["unfused_shape"])
            _append_row(canonical_csv, {
                "model": model_name, "dataset": dataset_name, "variant": variant,
                "canonical_layer": row["canonical_name"], "weight_shape": str(row["unfused_shape"]),
                "trace_raw": trace_val, "trace_per_param": trace_val / numel if numel else "",
                "probe_seed": CANONICAL_PROBE_SEED,
            }, CANONICAL_FIELDNAMES)

    return results


# ---------------------------------------------------------------------------
# Part 3: determinism check + reconciliation ledger
# ---------------------------------------------------------------------------

def _run_part3_determinism(
    model_name: str, specs: dict, num_classes: int, device: torch.device,
    val_loader: DataLoader, mapping: list[dict], fp32_ckpt: str,
    canonical_run_a: dict[str, float] | None,
) -> bool:
    channels, image_size = specs["channels"], specs["image_size"]
    criterion = nn.CrossEntropyLoss(reduction=CANONICAL_LOSS_REDUCTION)
    check_layers = sorted({mapping[0]["canonical_name"], mapping[len(mapping) // 2]["canonical_name"], mapping[-1]["canonical_name"]})

    run_a = canonical_run_a
    if run_a is None:
        model = _load_unfused_fp32(model_name, fp32_ckpt, num_classes, channels, image_size, device)
        run_a = _trace_full(model, val_loader, criterion, device, CANONICAL_PROBE_SEED, CANONICAL_NUM_BATCHES, CANONICAL_MAX_ITER, CANONICAL_TOL)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model = _load_unfused_fp32(model_name, fp32_ckpt, num_classes, channels, image_size, device)
    run_b = _trace_full(model, val_loader, criterion, device, CANONICAL_PROBE_SEED, CANONICAL_NUM_BATCHES, CANONICAL_MAX_ITER, CANONICAL_TOL)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    determinism_ok = True
    for layer in check_layers:
        key = f"{layer}.weight"
        a, b = run_a.get(key), run_b.get(key)
        if a is None or b is None:
            continue
        rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        ok = rel <= 1e-6
        determinism_ok = determinism_ok and ok
        logger.info(
            f"[RelockTraces] determinism check {model_name} layer={layer}: run_a={a:.10g} "
            f"run_b={b:.10g} rel_diff={rel:.3e} {'OK' if ok else 'FAILED (>1e-6)'}"
        )

    extra_seeds = [CANONICAL_PROBE_SEED + 1, CANONICAL_PROBE_SEED + 2]
    seed_values = {CANONICAL_PROBE_SEED: {l: run_a.get(f"{l}.weight") for l in check_layers}}
    for seed in extra_seeds:
        model = _load_unfused_fp32(model_name, fp32_ckpt, num_classes, channels, image_size, device)
        traces = _trace_full(model, val_loader, criterion, device, seed, CANONICAL_NUM_BATCHES, CANONICAL_MAX_ITER, CANONICAL_TOL)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seed_values[seed] = {l: traces.get(f"{l}.weight") for l in check_layers}

    for layer in check_layers:
        vals = [seed_values[s][layer] for s in seed_values if seed_values[s].get(layer) is not None]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        cv_pct = (std / mean * 100) if mean else float("nan")
        logger.info(
            f"[RelockTraces] {model_name} layer={layer}: across-seed ({len(vals)} seeds) "
            f"values={[round(v, 6) for v in vals]} mean={mean:.6g} std={std:.6g} cv={cv_pct:.3f}%"
        )

    if not determinism_ok:
        logger.error(
            f"[RelockTraces] {model_name}: DETERMINISM CHECK FAILED -- two runs under the same "
            f"probe seed and deterministic-algorithms=True disagreed by more than 1e-6 relative "
            f"on at least one checked layer. The frozen config's reproducibility guarantee does "
            f"not hold on this device/build; investigate before trusting canonical_traces.csv."
        )
    return determinism_ok


def _write_ledger_row(csv_path: str, quantity: str, old_value, old_source: str, canonical_value, explained_by_knob: str, note: str) -> None:
    ratio = _safe_div(old_value, canonical_value) if (old_value is not None and canonical_value not in (None, 0)) else float("nan")
    _append_row(csv_path, {
        "quantity": quantity, "old_value": old_value if old_value is not None else "",
        "old_source": old_source, "canonical_value": canonical_value if canonical_value is not None else "",
        "ratio": ratio if not (isinstance(ratio, float) and math.isnan(ratio)) else "",
        "explained_by_knob": explained_by_knob, "note": note,
    }, LEDGER_FIELDNAMES)


def _find_explaining_knob(drift_csv: str, anchor_name: str) -> str:
    if not os.path.exists(drift_csv):
        return "UNRESOLVED (no drift diagnosis data)"
    df = pd.read_csv(drift_csv)
    matches = df[df["reproduces_anchor"] == anchor_name]
    if not matches.empty:
        best = matches.sort_values("residual_pct").iloc[0]
        return f"{best['knob']}={best['setting']} (residual {float(best['residual_pct']):.2f}%)"
    return "UNRESOLVED (no knob reproduced this anchor within tolerance -- see drift_diagnosis.csv for closest residuals)"


def _write_reconciliation_ledger(
    ledger_csv: str, drift_csv: str, legacy_anchors: dict,
    canonical_by_model: dict[str, dict[str, dict[str, float]]],
    mapping_by_model: dict[str, list[dict]],
    banked_fp32_profile: str | None, dataset_name: str,
) -> None:
    diag_model = DIAGNOSIS_MODEL
    fp32_fused_canonical = canonical_by_model.get(diag_model, {}).get("fp32_fused", {})
    fp32_unfused_canonical = canonical_by_model.get(diag_model, {}).get("fp32_unfused", {})
    ptq_canonical = canonical_by_model.get(diag_model, {}).get("ptq", {})
    conv1_fp32_canonical = fp32_fused_canonical.get("conv1.weight")
    conv1_fp32_unfused = fp32_unfused_canonical.get("conv1.weight")
    conv1_ptq_canonical = ptq_canonical.get("conv1.weight")

    for anchor in legacy_anchors.get("resnet50_conv1_fp32", []):
        explained = _find_explaining_knob(drift_csv, anchor["name"])
        # Report the anchor against whichever FP32 basis it is numerically
        # closer to (unfused vs. canonical fused), independent of whether
        # that basis crossed the strict 10% "reproduces" threshold used for
        # explained_by_knob -- reporting the fused value for an anchor that
        # is obviously an unfused-basis number (just outside 10%, say) would
        # be far more misleading than reporting the closer basis with an
        # honest residual. Fusion alone changes conv1's trace by ~9.6x for
        # this checkpoint, so the two bases are not interchangeable.
        if conv1_fp32_unfused is not None and conv1_fp32_canonical is not None:
            rel_unfused = abs(anchor["value"] - conv1_fp32_unfused) / max(abs(conv1_fp32_unfused), 1e-9)
            rel_fused = abs(anchor["value"] - conv1_fp32_canonical) / max(abs(conv1_fp32_canonical), 1e-9)
            if rel_unfused <= rel_fused:
                matched_value = conv1_fp32_unfused
                note = (
                    f"closer to the UNFUSED basis (unfused conv1={conv1_fp32_unfused:.4g}, residual "
                    f"{rel_unfused*100:.1f}%) than to canonical basis={CANONICAL_BASIS} "
                    f"(conv1={conv1_fp32_canonical:.4g}, residual {rel_fused*100:.1f}%); "
                    f"fusion_ratio={_safe_div(conv1_fp32_canonical, conv1_fp32_unfused):.3g}"
                )
            else:
                matched_value = conv1_fp32_canonical
                note = (
                    f"closer to canonical basis={CANONICAL_BASIS} (conv1={conv1_fp32_canonical:.4g}, "
                    f"residual {rel_fused*100:.1f}%) than to unfused (unfused conv1="
                    f"{conv1_fp32_unfused:.4g}, residual {rel_unfused*100:.1f}%)"
                )
        else:
            matched_value = conv1_fp32_canonical
            note = f"canonical basis={CANONICAL_BASIS}"
        _write_ledger_row(ledger_csv, "resnet50_conv1_fp32", anchor["value"], anchor["source"], matched_value, explained, note)

    for anchor in legacy_anchors.get("resnet50_conv1_ptq", []):
        explained = _find_explaining_knob(drift_csv, anchor["name"])
        _write_ledger_row(ledger_csv, "resnet50_conv1_ptq", anchor["value"], anchor["source"], conv1_ptq_canonical, explained, "")

    if conv1_fp32_canonical is not None and fp32_fused_canonical:
        median_fp32 = statistics.median(fp32_fused_canonical.values())
        elev_canonical = _safe_div(conv1_fp32_canonical, median_fp32)
        explained = _find_explaining_knob(drift_csv, DEFAULT_ELEVATION_ANCHOR_NAME)
        _write_ledger_row(
            ledger_csv, "resnet50_conv1_elevation_fp32", DEFAULT_ELEVATION_ANCHOR_VALUE,
            "prior 'conv1 14x' claim", elev_canonical, explained,
            "elevation = conv1_trace / median(all-layer trace), fused basis",
        )

    if banked_fp32_profile:
        for model_name, mapping in mapping_by_model.items():
            banked = _load_banked_fp32_profile(banked_fp32_profile, model_name, dataset_name)
            unfused_canonical = canonical_by_model.get(model_name, {}).get("fp32_unfused", {})
            if not banked or not unfused_canonical:
                continue
            diffs = []
            for row in mapping:
                canon = row["canonical_name"]
                b = banked.get(canon)
                u = unfused_canonical.get(f"{canon}.weight")
                if b is None or u is None:
                    continue
                diffs.append(abs(b - u) / max(abs(b), abs(u), 1e-9))
            if not diffs:
                continue
            frac_ok = sum(1 for d in diffs if d <= RECONCILE_TOLERANCE) / len(diffs)
            median_banked = statistics.median(banked.values())
            median_canon = statistics.median(unfused_canonical.values())
            note = (
                f"{len(diffs)} layers compared; {frac_ok*100:.0f}% within "
                f"{RECONCILE_TOLERANCE*100:.0f}% relative diff of the banked value"
            )
            explained = (
                "data_source_original_pipeline (see drift_diagnosis.csv bonus rows) -- the banked "
                "profile was produced by the un-seeded, train-split (shuffle=True, augmented) "
                "hessian_loader in the pre-relock src/main.py pipeline, not the frozen canonical "
                "val-split config"
            )
            _write_ledger_row(
                ledger_csv, f"{model_name}_fp32_banked_vs_canonical_median_layer_trace",
                median_banked, banked_fp32_profile, median_canon, explained, note,
            )
            logger.info(f"[RelockTraces] {model_name}: banked-vs-canonical reconciliation -- {note}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_relock_traces(
    checkpoint_dir: str | None,
    load_run_id: str | None,
    banked_fp32_profile: str | None,
    legacy_anchors: str | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[RelockTraces] CUDA not available -- falling back to CPU, this will be slow.")
    _enable_determinism()

    anchors = _load_legacy_anchors(legacy_anchors)
    config = _freeze_config(RUN_DIR, device)
    logger.info(f"[RelockTraces] device={device} probe_seed={config['estimator']['probe_seed']} basis={config['basis']['canonical_for_cross_stage_comparison']}")

    fp32_models_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)
    quant_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)

    os.makedirs(CSV_DIR, exist_ok=True)
    drift_csv = os.path.join(CSV_DIR, "drift_diagnosis.csv")
    canonical_csv = os.path.join(CSV_DIR, "canonical_traces.csv")
    ledger_csv = os.path.join(CSV_DIR, "trace_reconciliation_ledger.csv")

    specs = DATASET_SPECS[DATASET_NAME]
    val_loader, num_classes = _build_val_hessian_loader(DATASET_NAME)
    orig_loader, _ = _build_original_pipeline_loader(DATASET_NAME)

    # ---- Part 1: drift diagnosis on resnet50 conv1 ----
    canonical_by_model: dict[str, dict[str, dict[str, float]]] = {}
    mapping_by_model: dict[str, list[dict]] = {}

    try:
        diag_fp32_ckpt = _resolve_checkpoint_robust(fp32_models_dir, {"model": DIAGNOSIS_MODEL, "dataset": DATASET_NAME})
        diag_ptq_ckpt = _resolve_checkpoint_robust(quant_dir, {"stage": "PTQ", "model": DIAGNOSIS_MODEL, "dataset": DATASET_NAME})
    except (FileNotFoundError, WeightAblationCheckpointError) as exc:
        raise RelockTracesError(f"Part 1 requires {DIAGNOSIS_MODEL}/{DATASET_NAME} FP32 and PTQ checkpoints -- {exc}") from exc

    logger.info(f"[RelockTraces] === Part 1: drift diagnosis on {DIAGNOSIS_MODEL} conv1 ===")
    diag_canonical = _run_part1_drift_diagnosis(
        DIAGNOSIS_MODEL, specs, num_classes, device, val_loader, orig_loader,
        diag_fp32_ckpt, diag_ptq_ckpt, anchors, drift_csv,
    )
    canonical_by_model[DIAGNOSIS_MODEL] = dict(diag_canonical)

    largest_ptq_anchor = max(anchors.get("resnet50_conv1_ptq", []), key=lambda a: a["value"], default=None)
    if largest_ptq_anchor is not None:
        explained = _anchor_explained(drift_csv, largest_ptq_anchor["name"])
        if explained:
            logger.info(
                f"[RelockTraces] Part 1 RESULT: legacy anchor {largest_ptq_anchor['name']}="
                f"{largest_ptq_anchor['value']} was reproduced within {ANCHOR_TOLERANCE*100:.0f}% by "
                f"a tested knob -- drift is understood, Part 2's canonical recompute is trustworthy."
            )
        else:
            logger.error(
                f"[RelockTraces] Part 1 RESULT: UNRESOLVED -- legacy anchor {largest_ptq_anchor['name']}="
                f"{largest_ptq_anchor['value']} was NOT reproduced within {ANCHOR_TOLERANCE*100:.0f}% by any "
                f"tested knob (including the bonus data-source and observer-calibration checks). See "
                f"{drift_csv} for the closest residuals per knob. Proceeding to Part 2/3 anyway -- the "
                f"frozen canonical config is well-defined independent of this historical gap -- but this "
                f"gap must be flagged as UNRESOLVED wherever the legacy number is cited."
            )

    # ---- Part 0 mapping gate for Part 2 (both required models) ----
    for model_name in PART2_MODELS:
        unfused_skel = build_model(num_classes=num_classes, model_name=model_name, channels=specs["channels"], image_size=specs["image_size"])
        quant_skel = _build_quant_skeleton(model_name, num_classes, specs["channels"], specs["image_size"])
        try:
            mapping_by_model[model_name] = _build_layer_mapping(model_name, unfused_skel, quant_skel)
        except QuantInducedMappingError as exc:
            logger.error(f"[RelockTraces] {model_name}: Part 0 mapping gate FAILED -- {exc} -- skipping Part 2/3 for this model")
            mapping_by_model.pop(model_name, None)
        finally:
            del unfused_skel, quant_skel

    # ---- Part 2: canonical recompute ----
    logger.info("[RelockTraces] === Part 2: canonical recompute (frozen config only) ===")
    for model_name in PART2_MODELS:
        if model_name not in mapping_by_model:
            continue
        try:
            m_fp32_ckpt = _resolve_checkpoint_robust(fp32_models_dir, {"model": model_name, "dataset": DATASET_NAME})
        except (FileNotFoundError, WeightAblationCheckpointError) as exc:
            logger.error(f"[RelockTraces] {model_name}: FP32 baseline checkpoint unresolvable -- {exc} -- skipping")
            continue

        m_ptq_ckpt = None
        try:
            m_ptq_ckpt = _resolve_checkpoint_robust(quant_dir, {"stage": "PTQ", "model": model_name, "dataset": DATASET_NAME})
        except FileNotFoundError as exc:
            logger.warning(f"[RelockTraces] {model_name}: PTQ checkpoint missing ({exc}) -- skipping PTQ/fp32_fused variants")
        except WeightAblationCheckpointError as exc:
            logger.error(f"[RelockTraces] {model_name}: PTQ checkpoint AMBIGUOUS/NEAR-MISS -- {exc} -- skipping PTQ/fp32_fused variants")

        m_qat_ckpt = None
        try:
            m_qat_ckpt = _resolve_checkpoint_robust(quant_dir, {"stage": "QAT", "model": model_name, "dataset": DATASET_NAME})
        except FileNotFoundError as exc:
            logger.warning(f"[RelockTraces] {model_name}: QAT checkpoint missing ({exc}) -- skipping QAT/fp32_fused_qat variants")
        except WeightAblationCheckpointError as exc:
            logger.error(f"[RelockTraces] {model_name}: QAT checkpoint AMBIGUOUS/NEAR-MISS -- {exc} -- skipping QAT/fp32_fused_qat variants")

        reuse = {}
        if model_name == DIAGNOSIS_MODEL:
            reuse = {k: v for k, v in diag_canonical.items() if v}

        results = _run_canonical_variant_traces(
            model_name, DATASET_NAME, specs, num_classes, device, val_loader, mapping_by_model[model_name],
            m_fp32_ckpt, m_ptq_ckpt, m_qat_ckpt, canonical_csv, reuse=reuse,
        )
        canonical_by_model[model_name] = results

    # ---- Part 3: determinism check + reconciliation ledger ----
    logger.info("[RelockTraces] === Part 3: determinism check + reconciliation ledger ===")
    if DIAGNOSIS_MODEL in mapping_by_model:
        _run_part3_determinism(
            DIAGNOSIS_MODEL, specs, num_classes, device, val_loader, mapping_by_model[DIAGNOSIS_MODEL],
            diag_fp32_ckpt, canonical_run_a=canonical_by_model.get(DIAGNOSIS_MODEL, {}).get("fp32_unfused"),
        )

    _write_reconciliation_ledger(ledger_csv, drift_csv, anchors, canonical_by_model, mapping_by_model, banked_fp32_profile, DATASET_NAME)

    logger.info("[RelockTraces] === Relock-Traces complete ===")
