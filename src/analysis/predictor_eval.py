"""
predictor_eval.py -- shared Spearman/bootstrap/rank-1/top-k* evaluation for
testing whether a per-layer sensitivity score predicts per-layer PoT
quantization damage (abs_loss_damage).

Used by conv1_excluded_control.py (conv1-excluded re-analysis of Table 6) and
cheap_baselines.py (zero/first-order baselines vs. S_raw), so the evaluation
logic matches Table 6's methodology exactly and is not reimplemented per
script or per combination.

Point estimate and p-value: scipy.stats.spearmanr
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).
Confidence interval: a percentile bootstrap over layers (paired resampling of
score/damage pairs with replacement), the same resampling scheme as
scipy.stats.bootstrap(paired=True, method="percentile")
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
implemented directly (not via scipy.stats.bootstrap) so degenerate resamples
-- a resample with constant score or constant damage, for which Spearman rho
is undefined -- can be dropped and counted explicitly.
"""

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODELS = ["cnn", "resnet18_no_weights", "resnet50_no_weights"]
DATASETS = ["IMAGENET100", "CIFAR10"]  # Table 6 row order
COMBINATIONS = [(model, dataset) for dataset in DATASETS for model in MODELS]

MODEL_LABEL = {"cnn": "CNN", "resnet18_no_weights": "ResNet-18", "resnet50_no_weights": "ResNet-50"}

# S_raw, S_pert, S_hawq columns in weight_ablation_canonical_v2(_imagnet).csv
SCORE_COLUMNS = {"raw_trh": "hessian_trace_fused", "dwsq": "delta_w_sq", "trh_dwsq": "trh_times_dwsq"}
PREDICTOR_LABEL = {"raw_trh": "S_raw", "dwsq": "S_pert", "trh_dwsq": "S_hawq"}

CANON_PATHS = {
    "CIFAR10": "results/20260816_230437_38678/csv/weight_ablation_canonical_v2.csv",
    "IMAGENET100": "results/20260816_083054_38677/csv/weight_ablation_canonical_v2_imagnet.csv",
}
DAMAGE_PATH = "results/20260822_131637_61872/csv/weight_ablation_loss_damage.csv"


def load_combo_frame(model: str, dataset: str, stage: str = "PTQ") -> pd.DataFrame:
    """Per-layer S_raw/S_pert/S_hawq joined with abs_loss_damage for one combination."""
    canon = pd.read_csv(os.path.join(REPO_ROOT, CANON_PATHS[dataset]))
    canon = canon[(canon.model == model) & (canon.dataset == dataset) & (canon.stage == stage)]
    damage = pd.read_csv(os.path.join(REPO_ROOT, DAMAGE_PATH))
    damage = damage[(damage.model == model) & (damage.dataset == dataset) & (damage.stage == stage)]
    merged = canon.merge(damage[["layer", "abs_loss_damage"]], on="layer", how="inner", validate="one_to_one")
    if not (len(merged) == len(canon) == len(damage)):
        raise ValueError(
            f"layer set mismatch for {model}/{dataset}/{stage}: "
            f"canon={len(canon)} damage={len(damage)} merged={len(merged)}"
        )
    return merged.reset_index(drop=True)


def bootstrap_spearman_ci(scores, damages, n_resamples: int = 2000, seed: int = 42, confidence: float = 0.95) -> dict:
    scores = np.asarray(scores, dtype=float)
    damages = np.asarray(damages, dtype=float)
    n = len(scores)
    rho, p = spearmanr(scores, damages)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        s, d = scores[idx], damages[idx]
        if np.all(s == s[0]) or np.all(d == d[0]):
            continue
        r, _ = spearmanr(s, d)
        if not np.isnan(r):
            rhos.append(r)
    alpha = 1 - confidence
    if rhos:
        ci_low, ci_high = np.percentile(rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    else:
        ci_low, ci_high = float("nan"), float("nan")
    return {
        "rho": rho,
        "p": p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_layers": n,
        "n_bootstrap": len(rhos),
        "n_bootstrap_requested": n_resamples,
        "seed": seed,
    }


def rank1_and_topk(layers, scores, damages) -> dict:
    """Rank of the true top-damage layer under `scores` (1 = highest score),
    and top-k* overlap with k* = ceil(0.1 * n_layers)."""
    n = len(layers)
    k_star = max(1, math.ceil(0.1 * n))
    order_by_score = sorted(range(n), key=lambda i: -scores[i])
    true_top_idx = max(range(n), key=lambda i: damages[i])
    rank_of_true_top = order_by_score.index(true_top_idx) + 1
    topk_by_score = set(order_by_score[:k_star])
    topk_by_damage = set(sorted(range(n), key=lambda i: -damages[i])[:k_star])
    overlap = len(topk_by_score & topk_by_damage)
    return {
        "n_layers": n,
        "k_star": k_star,
        "true_top_layer": layers[true_top_idx],
        "rank_of_true_top": rank_of_true_top,
        "is_rank1": rank_of_true_top == 1,
        "top_k_overlap": overlap,
    }


def evaluate(df: pd.DataFrame, score_col: str, n_resamples: int = 2000, seed: int = 42) -> dict:
    layers = df["layer"].tolist()
    scores = df[score_col].tolist()
    damages = df["abs_loss_damage"].tolist()
    stats = bootstrap_spearman_ci(scores, damages, n_resamples=n_resamples, seed=seed)
    rank_stats = rank1_and_topk(layers, scores, damages)
    return {**stats, **rank_stats}
