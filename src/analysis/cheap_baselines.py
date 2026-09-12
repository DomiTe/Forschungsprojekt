"""
cheap_baselines.py -- Task 2: three zero/first-order per-layer sensitivity
baselines that require no Hessian-vector product, computed on the same
probe set and loss reduction as trace_config.json, then scored against
abs_loss_damage with the identical Spearman/bootstrap/rank-1/top-k*
methodology as S_raw (src/analysis/predictor_eval.py, shared with Table 6
and Task 1). Answers whether the HVP cost S_raw requires is justified, or
whether a baseline this cheap matches or beats it.

Baselines (weight-only, matching S_raw/S_pert's scope):
  weight magnitude: ||W_l||_F. No probe data needed.
  gradient norm: ||dL/dW_l||_F from one backward pass over the full probe
    set concatenated into a single batch, CrossEntropyLoss(reduction="mean")
    -- same reduction trace_config.json declares for the Hessian estimator.
  empirical Fisher diagonal: sum over the probe set of the elementwise
    squared per-sample gradient, summed over W_l's elements. Requires a true
    per-sample gradient, which a single mean-reduced backward pass over a
    batch cannot give (it returns the mean gradient, not per-sample
    gradients) -- so this loops the probe set at batch size 1.

Caveat (project spec, weight-quantization-damage context): near a converged
minimum the loss gradient is close to zero, so gradient-norm and Fisher are
expected to be noise-dominated baselines; this is checked in the results,
not assumed.

Model basis: the same fp32_fused basis S_raw/S_pert/S_hawq are computed on
(PTQ checkpoint skeleton, weight_fake_quant/act_fake_quant swapped to
Identity) -- reuses _build_quant_skeleton/_load_quantized/_make_fused_fp32/
_weight_layers_in_forward_order (src/analysis/quant_induced_trace.py) and
_checkpoint_path (src/quantization/deploy_fbgemm.py) rather than
reconstructing that basis or the checkpoint path convention again.

Probe set: mirrors relock_traces.py's _build_val_hessian_loader (val/test
split, shuffle=False, first num_batches*batch_size images in dataset order)
using the per-dataset batch_size/num_batches frozen in trace_config.json,
bypassing get_data_loaders' DATASET_SPECS lookup (only CIFAR10 is currently
uncommented there) by calling the private per-dataset loader functions in
src/utility/utils.py directly with the dataset's trace_config.json image
size.

Documentation: torch.autograd
(https://docs.pytorch.org/docs/stable/autograd.html) for per-parameter
.grad; scipy.stats.spearmanr, as in predictor_eval.py.

Run: python -m src.analysis.cheap_baselines
"""

import csv
import os
import logging

import torch
import torch.nn as nn

from src.analysis.predictor_eval import (
    REPO_ROOT,
    COMBINATIONS,
    MODEL_LABEL,
    load_combo_frame,
    bootstrap_spearman_ci,
    rank1_and_topk,
)
from src.analysis.quant_induced_trace import (
    _build_quant_skeleton,
    _load_quantized,
    _make_fused_fp32,
    _weight_layers_in_forward_order,
)
from src.quantization.deploy_fbgemm import _checkpoint_path
from src.utility.utils import _get_cifar10_loaders, _get_imagenet100_loaders

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(REPO_ROOT, "results", "20260813_053524_36857", "quantized_models")
STAGE = "PTQ"

PROBE_CONFIG = {
    "CIFAR10": {"image_size": 32, "channels": 3, "num_classes": 10, "batch_size": 16, "num_batches": 5},
    "IMAGENET100": {"image_size": 224, "channels": 3, "num_classes": 100, "batch_size": 8, "num_batches": 3},
}

OUT_DIR = os.path.join(REPO_ROOT, "results", "review_response", "csv")
OUT_PATH = os.path.join(OUT_DIR, "cheap_baselines_loss.csv")


def _build_probe_batch(dataset: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = PROBE_CONFIG[dataset]
    if dataset == "CIFAR10":
        _, val_loader, _ = _get_cifar10_loaders(cfg["image_size"])
    elif dataset == "IMAGENET100":
        _, val_loader, _ = _get_imagenet100_loaders(cfg["image_size"])
    else:
        raise ValueError(dataset)
    loader = torch.utils.data.DataLoader(
        val_loader.dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, pin_memory=False,
    )
    images, labels = [], []
    for i, (x, y) in enumerate(loader):
        if i >= cfg["num_batches"]:
            break
        images.append(x)
        labels.append(y)
    return torch.cat(images).to(device), torch.cat(labels).to(device)


def _load_fused_fp32_model(model_name: str, dataset: str, device: torch.device) -> nn.Module:
    cfg = PROBE_CONFIG[dataset]
    ckpt_path = _checkpoint_path(CHECKPOINT_DIR, STAGE, model_name, dataset)
    model = _load_quantized(model_name, ckpt_path, cfg["num_classes"], cfg["channels"], cfg["image_size"], device)
    _make_fused_fp32(model, f"{model_name}/{dataset} {STAGE} fp32_fused (cheap_baselines)")
    return model


def compute_baselines(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> dict[str, dict[str, float]]:
    """weight magnitude, gradient norm (aggregate probe batch), empirical
    Fisher diagonal (per-sample loop) -- per Conv2d/Linear layer."""
    layers = _weight_layers_in_forward_order(model)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    weight_magnitude = {name: m.weight.detach().norm().item() for name, m in layers}

    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    grad_norm = {name: m.weight.grad.detach().norm().item() for name, m in layers}

    fisher_diag = {name: 0.0 for name, _ in layers}
    n = images.shape[0]
    for i in range(n):
        model.zero_grad(set_to_none=True)
        logits_i = model(images[i : i + 1])
        loss_i = criterion(logits_i, labels[i : i + 1])
        loss_i.backward()
        for name, m in layers:
            fisher_diag[name] += m.weight.grad.detach().pow(2).sum().item()

    return {"weight_magnitude": weight_magnitude, "grad_norm": grad_norm, "fisher_diag": fisher_diag}


FIELDNAMES = [
    "model", "dataset", "predictor", "n_layers",
    "rho", "p", "ci_low", "ci_high", "n_bootstrap", "n_bootstrap_requested",
    "true_top_layer", "rank_of_true_top", "is_rank1", "k_star", "top_k_overlap",
]


def run() -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for model_name, dataset in COMBINATIONS:
        damage_frame = load_combo_frame(model_name, dataset, stage=STAGE)
        layer_order = damage_frame["layer"].tolist()
        damages = dict(zip(damage_frame["layer"], damage_frame["abs_loss_damage"]))
        raw_scores = dict(zip(damage_frame["layer"], damage_frame["hessian_trace_fused"]))

        model = _load_fused_fp32_model(model_name, dataset, device)
        images, labels = _build_probe_batch(dataset, device)
        baselines = compute_baselines(model, images, labels)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        predictors = dict(baselines)
        predictors["raw_trh"] = raw_scores

        for predictor, score_map in predictors.items():
            scores = [score_map[l] for l in layer_order]
            damage_vals = [damages[l] for l in layer_order]
            stats = bootstrap_spearman_ci(scores, damage_vals)
            rank_stats = rank1_and_topk(layer_order, scores, damage_vals)
            rows.append({
                "model": model_name, "dataset": dataset, "predictor": predictor,
                "n_layers": rank_stats["n_layers"],
                "rho": stats["rho"], "p": stats["p"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                "n_bootstrap": stats["n_bootstrap"], "n_bootstrap_requested": stats["n_bootstrap_requested"],
                "true_top_layer": rank_stats["true_top_layer"],
                "rank_of_true_top": rank_stats["rank_of_true_top"],
                "is_rank1": rank_stats["is_rank1"],
                "k_star": rank_stats["k_star"],
                "top_k_overlap": f"{rank_stats['top_k_overlap']}/{rank_stats['k_star']}",
            })
    return rows


def main() -> None:
    rows = run()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
