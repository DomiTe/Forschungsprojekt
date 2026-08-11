"""
random_init_control.py -- P2 Method 1: random-init control for the layer-wise
weight-Hessian trace.

Motivation: the trained FP32 baseline shows conv1 elevated (~14x on
resnet50). That elevation is ambiguous between "conv1 has high curvature
because of where it sits architecturally -- spatial extent, fan-in,
weight-sharing multiplicity" and "because of what it learned". This module
runs the identical trace estimator (compute_layerwise_hessian_trace_pyhessian,
reused unchanged -- never duplicated) on random weights and compares the
per-layer profile against the trained-FP32 profile, to see whether the spike
is already there before any training happens.

Three parts, CIFAR10 only; models cnn, resnet18_no_weights,
resnet50_no_weights (resnet50 first -- it carries the headline spike):

  Part 0 (gate): for each random-init draw, confirm the model is genuinely
    untrained with a concrete positive check -- it must evaluate at ~chance
    top-1 accuracy on the CIFAR10 test set. Aborts that architecture if the
    accuracy is materially above chance (a checkpoint would have to have
    leaked into what should be an untrained build_model() call).
  Part 1: >=3 independent init seeds per architecture. The init seed is
    reset immediately before build_model(); a single fixed probe seed is
    reset immediately before every compute_layerwise_hessian_trace_pyhessian
    call (random-init AND the trained-FP32 recompute), so the Hutchinson
    probe draws are identical across all of them -- the across-seed std this
    produces reflects the weight draw alone, not probe noise.
  Part 2: align random-init and trained-FP32 traces by module qualified name
    (both come from unfused build_model, so they must match 1:1 -- asserted,
    not assumed), compute the per-layer ratio, Spearman rho of the profile
    shape (resnets only -- cnn has ~6 weight layers, reported but not leaned
    on), and for the spike layer (identified data-driven as the layer with
    the largest trace-over-median-trace ratio in the trained profile, not
    hardcoded to "conv1") the continuous quantity
    fraction_present_at_init = elev_over_median_random / elev_over_median_trained,
    which classifies the elevation as architectural / learned / mixed.

The banked FP32 Hessian profile (layerwise_hessian_traces.csv, written by the
main training pipeline) does not record its estimator config (data split,
batch size, probe count, eval/train mode, probe seed), so per this mode's
controlled-comparison requirement it is never reused -- the trained-FP32
profile is always recomputed in this same run with the identical config used
for the random-init sweep. Reuses (does not duplicate) the FP32 checkpoint
resolution / loading helpers already established in
src/analysis/diagnose_activations.py.

BN handling: primary control is (a) BN left at its default random-init
running stats (mean 0, var 1 -- eval-mode BN is then effectively identity).
If a model's verdict under (a) is not "architectural", the BN-populated
control (b) is additionally run (forward a few hundred CIFAR images in train
mode first, to populate real running stats) so the gap cannot be
attributable to BN miscalibration rather than architecture -- written to
sibling CSVs suffixed "_bn_populated" (see run_random_init_control), and as
an additional bn_mode="populated" row in random_init_summary.csv.

Analysis only -- no quantization/PTQ/QAT/deployment code; reuses
compute_layerwise_hessian_trace_pyhessian unchanged. Runs as a single local
process (`python -m src.main --random-init-control ...`), no
SLURM/torchrun required; prefers CUDA.

Sources: Sagun, Bottou & LeCun, "Eigenvalues of the Hessian in Deep
Learning: Singularity and Beyond" (arXiv:1611.07476); Sagun, Evci, Guney,
Dauphin & Bottou, "Empirical Analysis of the Hessian of Over-Parametrized
Neural Networks" (arXiv:1706.04454) -- both study eigenvalue/trace structure
at and after initialisation. Karakida, Akaho & Amari, "Universal Statistics
of Fisher Information in Deep Neural Networks: Mean Field Approach"
(AISTATS 2019) derives mean-field Fisher predictions for random weights,
the regime random-init is a direct instance of.
"""

import os
import csv
import math
import statistics
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from src.model_cnn.train import build_model
from src.analysis.pyhessian import compute_layerwise_hessian_trace_pyhessian
from src.analysis.diagnose_activations import (
    _resolve_fp32_models_dir,
    _fp32_checkpoint_path,
    _load_fp32_reference,
)
from src.utility.config import CSV_DIR, DATASET_SPECS, HESSIAN_BATCH_SIZE, TEST_BATCH_SIZE
from src.utility.utils import get_data_loaders

logger = logging.getLogger(__name__)

DATASETS = ["CIFAR10", "IMAGENET100"]
# resnet50 first -- it carries the headline conv1 spike this control targets.
ORDERED_MODELS = ["resnet50_no_weights", "resnet18_no_weights", "cnn"]

# >=3 independent random-init draws per architecture, per spec (P2 Method 1).
# Drop to 2 only if resnet50's wall time genuinely threatens the budget --
# never drop the probe count instead, that would break the config match
# against the trained-FP32 recompute.
INIT_SEEDS = [0, 1, 2]

# Fixed and reset immediately before every estimator call (random-init AND
# trained-FP32), so Hutchinson probe draws are identical everywhere and the
# across-seed std reflects only the weight draw, not probe noise.
PROBE_SEED = 20260810

# compute_layerwise_hessian_trace_pyhessian's own defaults -- passed
# explicitly (not left implicit) so the "identical estimator config between
# regimes" requirement is visible and locked regardless of upstream default
# drift.
HESSIAN_NUM_BATCHES = 5
HESSIAN_MAX_ITER = 100
HESSIAN_TOL = 1e-3

# Positive check that a "random-init" model is genuinely untrained: abort if
# top-1 accuracy on the CIFAR10 test set exceeds chance (100/num_classes) by
# more than this multiplier. A degenerate random-init model that always
# predicts one class already scores ~chance on the (balanced) CIFAR10 test
# set, so this catches a leaked checkpoint without false-positiving on
# ordinary random-init noise.
CHANCE_ACC_MULTIPLIER = 2.0

# BN-populated control (b): number of (shuffle=False) batches forwarded in
# train() mode to populate running stats before switching back to eval().
# HESSIAN_BATCH_SIZE(=16) * 20 = 320 images -- "a few hundred" per spec.
BN_POPULATE_BATCHES = 20

# Classification thresholds for fraction_present_at_init, applied to the
# reported number (never a hidden cutoff) -- most of the elevation already
# present at init (>=0.7) -> architectural; near-flat at init (<=0.3) ->
# learned; in between -> mixed.
FRACTION_ARCHITECTURAL_MIN = 0.7
FRACTION_LEARNED_MAX = 0.3

TRACES_FIELDNAMES = ["model", "dataset", "init", "seed", "layer", "hessian_trace"]
COMPARISON_FIELDNAMES = [
    "model", "dataset", "layer", "trace_random_mean", "trace_random_std",
    "trace_trained_fp32", "ratio_trained_over_random", "elev_over_median_random",
    "elev_over_median_trained", "classification",
]
SUMMARY_FIELDNAMES = [
    "model", "dataset", "n_seeds", "spearman_profile_rho", "spearman_p",
    "spike_layer", "elev_random", "elev_trained", "verdict", "bn_mode",
]


class RandomInitControlError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Determinism (double-backward through cuDNN convolutions can otherwise add
# noise that inflates the across-seed std on an A100). warn_only=True: some
# double-backward ops genuinely have no deterministic kernel, and this mode
# must not hard-crash on that -- it should just log and proceed.
# ---------------------------------------------------------------------------

def _enable_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Data loaders -- eval() mode, num_workers=0, pin_memory=False, shuffle=False,
# built once from the CIFAR10 test split and reused (never re-shuffled) for
# every estimator call and every chance-accuracy check.
# ---------------------------------------------------------------------------

def _build_loaders(dataset_name: str) -> tuple[DataLoader, DataLoader, int]:
    _, val_loader, num_classes = get_data_loaders(dataset_name)
    hessian_loader = DataLoader(
        val_loader.dataset, batch_size=HESSIAN_BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    chance_loader = DataLoader(
        val_loader.dataset, batch_size=TEST_BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    return hessian_loader, chance_loader, num_classes


# ---------------------------------------------------------------------------
# Part 0: positive untrained-model check
# ---------------------------------------------------------------------------

def _assert_untrained(model: nn.Module, chance_loader: DataLoader, device: torch.device, num_classes: int, label: str) -> float:
    from src.main import evaluate  # deferred: avoids importing the full training pipeline at module load

    acc = evaluate(model, chance_loader, device)
    chance_acc = 100.0 / num_classes
    threshold = chance_acc * CHANCE_ACC_MULTIPLIER
    logger.info(f"[RandomInitControl] {label}: chance-check acc={acc:.2f}% (chance={chance_acc:.2f}%, threshold<{threshold:.2f}%)")
    if acc > threshold:
        raise RandomInitControlError(
            f"{label}: random-init model evaluates at {acc:.2f}% top-1, above the abort threshold "
            f"{threshold:.2f}% ({CHANCE_ACC_MULTIPLIER}x chance={chance_acc:.2f}%) -- this looks like "
            f"a checkpoint leaked into what should be an untrained build_model() call. Aborting this "
            f"architecture rather than trusting the trace."
        )
    return acc


def _populate_bn_stats(model: nn.Module, loader: DataLoader, device: torch.device, num_batches: int) -> None:
    model.train()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(inputs.to(device))
    model.eval()


# ---------------------------------------------------------------------------
# Trained-FP32 profile (always recomputed -- see module docstring)
# ---------------------------------------------------------------------------

def _get_trained_fp32_traces(
    model_name: str, dataset_name: str, specs: dict, num_classes: int, device: torch.device,
    hessian_loader: DataLoader, criterion: nn.Module, fp32_models_dir: str, traces_csv_path: str,
) -> dict[str, float]:
    logger.info(
        f"[RandomInitControl] {model_name}/{dataset_name}: banked FP32 Hessian profile "
        f"(layerwise_hessian_traces.csv) does not record its estimator config -- recomputing the "
        f"trained-FP32 profile in this run with the identical config used for the random-init "
        f"sweep, so this is a controlled comparison rather than two settings compared by accident."
    )
    ckpt_path = _fp32_checkpoint_path(fp32_models_dir, model_name, dataset_name)
    model = _load_fp32_reference(model_name, ckpt_path, num_classes, specs["channels"], specs["image_size"])
    model = model.to(device)
    model.eval()

    torch.manual_seed(PROBE_SEED)
    traces = compute_layerwise_hessian_trace_pyhessian(
        model, hessian_loader, criterion, device,
        num_batches=HESSIAN_NUM_BATCHES, max_iter=HESSIAN_MAX_ITER, tol=HESSIAN_TOL,
    )
    for layer, trace_val in traces.items():
        _append_row(traces_csv_path, {
            "model": model_name, "dataset": dataset_name, "init": "trained_fp32",
            "seed": "", "layer": layer, "hessian_trace": trace_val,
        }, TRACES_FIELDNAMES)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return traces


# ---------------------------------------------------------------------------
# Part 1: random-init sweep (multiple seeds, fixed probe seed)
# ---------------------------------------------------------------------------

def _run_random_seeds(
    model_name: str, dataset_name: str, specs: dict, num_classes: int, device: torch.device,
    hessian_loader: DataLoader, chance_loader: DataLoader, criterion: nn.Module,
    seeds: list[int], bn_mode: str, traces_csv_path: str,
) -> dict[str, list[float]]:
    channels, image_size = specs["channels"], specs["image_size"]
    per_layer_traces: dict[str, list[float]] = {}

    for seed in seeds:
        label = f"{model_name}/{dataset_name} seed={seed} bn={bn_mode}"

        # Init seed governs build_model()'s weight draw only.
        torch.manual_seed(seed)
        model = build_model(
            num_classes=num_classes, model_name=model_name, channels=channels, image_size=image_size,
        ).to(device)
        model.eval()

        if bn_mode == "populated":
            _populate_bn_stats(model, hessian_loader, device, BN_POPULATE_BATCHES)

        _assert_untrained(model, chance_loader, device, num_classes, label)

        # Reset immediately before the estimator call (not once per loop) so
        # every seed sees identical Hutchinson probe draws.
        torch.manual_seed(PROBE_SEED)
        traces = compute_layerwise_hessian_trace_pyhessian(
            model, hessian_loader, criterion, device,
            num_batches=HESSIAN_NUM_BATCHES, max_iter=HESSIAN_MAX_ITER, tol=HESSIAN_TOL,
        )
        for layer, trace_val in traces.items():
            _append_row(traces_csv_path, {
                "model": model_name, "dataset": dataset_name, "init": "random",
                "seed": seed, "layer": layer, "hessian_trace": trace_val,
            }, TRACES_FIELDNAMES)
            per_layer_traces.setdefault(layer, []).append(trace_val)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return per_layer_traces


# ---------------------------------------------------------------------------
# Part 2: alignment, comparison, classification
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    if b == 0 or (isinstance(b, float) and math.isnan(b)):
        return float("nan")
    return a / b


def _classify(fraction: float) -> str:
    if isinstance(fraction, float) and math.isnan(fraction):
        return "undetermined"
    if fraction >= FRACTION_ARCHITECTURAL_MIN:
        return "architectural"
    if fraction <= FRACTION_LEARNED_MAX:
        return "learned"
    return "mixed"


def _compare_and_classify(
    model_name: str, dataset_name: str,
    random_traces: dict[str, list[float]], trained_traces: dict[str, float],
    comparison_csv_path: str,
) -> dict[str, dict]:
    random_layers = set(random_traces.keys())
    trained_layers = set(trained_traces.keys())
    mismatch = random_layers.symmetric_difference(trained_layers)
    if mismatch:
        logger.error(
            f"[RandomInitControl] {model_name}/{dataset_name}: layer name MISMATCH between "
            f"random-init and trained-FP32 traces: {sorted(mismatch)} -- both come from unfused "
            f"build_model() and should match 1:1, this indicates a real bug."
        )
        raise RandomInitControlError(
            f"{model_name}/{dataset_name}: layer name mismatch between random-init and "
            f"trained-FP32 profiles: {sorted(mismatch)}"
        )

    layers = sorted(random_layers)

    trace_random_mean = {l: statistics.mean(random_traces[l]) for l in layers}
    trace_random_std = {
        l: (statistics.stdev(random_traces[l]) if len(random_traces[l]) >= 2 else 0.0)
        for l in layers
    }
    trace_trained = {l: trained_traces[l] for l in layers}

    median_random = statistics.median(trace_random_mean.values())
    median_trained = statistics.median(trace_trained.values())

    per_layer: dict[str, dict] = {}
    for l in layers:
        elev_r = _safe_div(trace_random_mean[l], median_random)
        elev_t = _safe_div(trace_trained[l], median_trained)
        ratio = _safe_div(trace_trained[l], trace_random_mean[l])
        fraction = _safe_div(elev_r, elev_t)
        classification = _classify(fraction)

        per_layer[l] = {
            "trace_random_mean": trace_random_mean[l],
            "trace_random_std": trace_random_std[l],
            "trace_trained_fp32": trace_trained[l],
            "ratio_trained_over_random": ratio,
            "elev_over_median_random": elev_r,
            "elev_over_median_trained": elev_t,
            "fraction_present_at_init": fraction,
            "classification": classification,
        }

        _append_row(comparison_csv_path, {
            "model": model_name, "dataset": dataset_name, "layer": l,
            "trace_random_mean": trace_random_mean[l],
            "trace_random_std": trace_random_std[l],
            "trace_trained_fp32": trace_trained[l],
            "ratio_trained_over_random": ratio,
            "elev_over_median_random": elev_r,
            "elev_over_median_trained": elev_t,
            "classification": classification,
        }, COMPARISON_FIELDNAMES)

    return per_layer


def _write_summary(
    model_name: str, dataset_name: str, n_seeds: int, per_layer: dict[str, dict],
    bn_mode: str, summary_csv_path: str,
) -> str:
    layers = sorted(per_layer.keys())
    random_profile = [per_layer[l]["trace_random_mean"] for l in layers]
    trained_profile = [per_layer[l]["trace_trained_fp32"] for l in layers]

    if len(layers) >= 3:
        rho, p_value = spearmanr(random_profile, trained_profile)
    else:
        rho, p_value = float("nan"), float("nan")
        logger.warning(
            f"[RandomInitControl] {model_name}/{dataset_name}: only {len(layers)} layers -- "
            f"Spearman correlation not meaningful (need >= 3), reporting NaN"
        )
    if len(layers) < 7:
        logger.info(
            f"[RandomInitControl] {model_name}/{dataset_name}: {len(layers)} weight layers -- "
            f"reporting the per-layer profile but not leaning on the Spearman rho (meaningful "
            f"only for the resnets, per spec)"
        )

    # Spike layer identified data-driven: the layer with the largest
    # elevation-over-median in the trained profile (not hardcoded to a name).
    spike_layer = max(layers, key=lambda l: per_layer[l]["elev_over_median_trained"])
    elev_random = per_layer[spike_layer]["elev_over_median_random"]
    elev_trained = per_layer[spike_layer]["elev_over_median_trained"]
    fraction = per_layer[spike_layer]["fraction_present_at_init"]
    verdict = per_layer[spike_layer]["classification"]

    logger.info(
        f"[RandomInitControl] {model_name}/{dataset_name} (bn={bn_mode}): spike_layer={spike_layer} "
        f"elev_random={elev_random:.3f} elev_trained={elev_trained:.3f} "
        f"fraction_present_at_init={fraction:.3f} verdict={verdict} "
        f"spearman_rho={rho:.4f} p={p_value:.4g} n_layers={len(layers)}"
    )

    _append_row(summary_csv_path, {
        "model": model_name, "dataset": dataset_name, "n_seeds": n_seeds,
        "spearman_profile_rho": rho, "spearman_p": p_value, "spike_layer": spike_layer,
        "elev_random": elev_random, "elev_trained": elev_trained,
        "verdict": verdict, "bn_mode": bn_mode,
    }, SUMMARY_FIELDNAMES)

    return verdict


# ---------------------------------------------------------------------------
# CSV (append mode -- one row written to disk immediately after computation)
# ---------------------------------------------------------------------------

def _append_row(path: str, row: dict, fieldnames: list[str]) -> None:
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _sibling_path(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_one_model(
    model_name: str, dataset_name: str, specs: dict, num_classes: int, device: torch.device,
    hessian_loader: DataLoader, chance_loader: DataLoader, fp32_models_dir: str,
    traces_csv: str, comparison_csv: str, summary_csv: str,
) -> None:
    criterion = nn.CrossEntropyLoss()

    trained_traces = _get_trained_fp32_traces(
        model_name, dataset_name, specs, num_classes, device,
        hessian_loader, criterion, fp32_models_dir, traces_csv,
    )
    if not trained_traces:
        raise RandomInitControlError(f"{model_name}/{dataset_name}: empty trained-FP32 trace")

    random_traces = _run_random_seeds(
        model_name, dataset_name, specs, num_classes, device,
        hessian_loader, chance_loader, criterion, INIT_SEEDS, "default", traces_csv,
    )
    per_layer = _compare_and_classify(model_name, dataset_name, random_traces, trained_traces, comparison_csv)
    verdict = _write_summary(model_name, dataset_name, len(INIT_SEEDS), per_layer, "default", summary_csv)

    if verdict != "architectural":
        logger.info(
            f"[RandomInitControl] {model_name}/{dataset_name}: verdict={verdict} under default "
            f"(uncalibrated) BN -- also running the BN-populated control, since the gap could "
            f"otherwise be attributable to BN miscalibration rather than architecture."
        )
        traces_csv_bn = _sibling_path(traces_csv, "_bn_populated")
        comparison_csv_bn = _sibling_path(comparison_csv, "_bn_populated")
        random_traces_bn = _run_random_seeds(
            model_name, dataset_name, specs, num_classes, device,
            hessian_loader, chance_loader, criterion, INIT_SEEDS, "populated", traces_csv_bn,
        )
        per_layer_bn = _compare_and_classify(model_name, dataset_name, random_traces_bn, trained_traces, comparison_csv_bn)
        _write_summary(model_name, dataset_name, len(INIT_SEEDS), per_layer_bn, "populated", summary_csv)


def run_random_init_control(
    checkpoint_dir: str | None, load_run_id: str | None, datasets: list[str] | None = None,
) -> None:
    """datasets restricts the sweep to a subset of DATASETS (e.g. one dataset,
    for a parallel per-dataset analysis run) -- default None means every
    dataset in DATASETS."""
    datasets = datasets if datasets is not None else DATASETS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("[RandomInitControl] CUDA not available -- falling back to CPU, this will be slow.")
    _enable_determinism()
    logger.info(f"[RandomInitControl] device={device} init_seeds={INIT_SEEDS} probe_seed={PROBE_SEED}")

    fp32_models_dir = _resolve_fp32_models_dir(checkpoint_dir, load_run_id)

    os.makedirs(CSV_DIR, exist_ok=True)
    traces_csv = os.path.join(CSV_DIR, "random_init_traces.csv")
    comparison_csv = os.path.join(CSV_DIR, "random_init_comparison.csv")
    summary_csv = os.path.join(CSV_DIR, "random_init_summary.csv")

    for dataset_name in datasets:
        specs = DATASET_SPECS[dataset_name]
        try:
            hessian_loader, chance_loader, num_classes = _build_loaders(dataset_name)
        except Exception as exc:
            logger.warning(f"[RandomInitControl] {dataset_name}: could not load dataset ({exc}) -- skipping")
            continue

        for model_name in ORDERED_MODELS:
            logger.info(f"[RandomInitControl] === {model_name}/{dataset_name} ===")
            try:
                _run_one_model(
                    model_name, dataset_name, specs, num_classes, device,
                    hessian_loader, chance_loader, fp32_models_dir,
                    traces_csv, comparison_csv, summary_csv,
                )
            except Exception as exc:
                logger.error(f"[RandomInitControl] FAILED {model_name}/{dataset_name}: {exc}", exc_info=True)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    logger.info("[RandomInitControl] === Random-Init-Control complete ===")
