"""
conv1_excluded_control.py -- Task 1 re-analysis: recomputes Table 6 (S_raw,
S_pert, S_hawq vs. abs_loss_damage, PTQ) after removing conv1 from each
model's layer set, and reports the delta against the with-conv1 numbers.

conv1 dominates the with-conv1 rank-1 successes in Table 6 (it is the true
top-damage layer in 4 of 6 combinations). This control asks whether the
predictive signal survives once that one layer is taken out of contention.

Statistical caveat (stated again in the logged output, not just here): after
removing conv1 the CNN has n=5 layers. A percentile bootstrap over 5 items
is noise -- the CNN row is reported for completeness only and must not be
read as evidence either way. n=20 (ResNet-18) and n=53 (ResNet-50) are the
combinations this control actually speaks to.

Reuses (does not duplicate): src/analysis/predictor_eval.py for the
Spearman/bootstrap/rank-1/top-k* evaluation, identical to Table 6's
methodology and to cheap_baselines.py's (Task 2).

Run: python -m src.analysis.conv1_excluded_control
"""

import csv
import os
import logging

from src.analysis.predictor_eval import (
    REPO_ROOT,
    COMBINATIONS,
    MODEL_LABEL,
    SCORE_COLUMNS,
    PREDICTOR_LABEL,
    load_combo_frame,
    evaluate,
)

logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(REPO_ROOT, "results", "review_response", "csv")
OUT_PATH = os.path.join(OUT_DIR, "conv1_excluded_control_loss.csv")

FIELDNAMES = [
    "model", "dataset", "predictor",
    "n_layers_with_conv1", "rho_with_conv1", "ci_low_with_conv1", "ci_high_with_conv1",
    "rank1_with_conv1", "topk_overlap_with_conv1", "k_star_with_conv1",
    "n_layers_no_conv1", "rho_no_conv1", "ci_low_no_conv1", "ci_high_no_conv1",
    "rank1_no_conv1", "topk_overlap_no_conv1", "k_star_no_conv1",
    "delta_rho", "true_top_layer_no_conv1",
]


def run() -> list[dict]:
    rows = []
    for model, dataset in COMBINATIONS:
        df = load_combo_frame(model, dataset, stage="PTQ")
        df_no_conv1 = df[df.layer != "conv1"].reset_index(drop=True)
        for predictor, col in SCORE_COLUMNS.items():
            with_c1 = evaluate(df, col)
            no_c1 = evaluate(df_no_conv1, col)
            rows.append({
                "model": model, "dataset": dataset, "predictor": predictor,
                "n_layers_with_conv1": with_c1["n_layers"],
                "rho_with_conv1": with_c1["rho"],
                "ci_low_with_conv1": with_c1["ci_low"],
                "ci_high_with_conv1": with_c1["ci_high"],
                "rank1_with_conv1": with_c1["is_rank1"],
                "topk_overlap_with_conv1": f"{with_c1['top_k_overlap']}/{with_c1['k_star']}",
                "k_star_with_conv1": with_c1["k_star"],
                "n_layers_no_conv1": no_c1["n_layers"],
                "rho_no_conv1": no_c1["rho"],
                "ci_low_no_conv1": no_c1["ci_low"],
                "ci_high_no_conv1": no_c1["ci_high"],
                "rank1_no_conv1": no_c1["is_rank1"],
                "topk_overlap_no_conv1": f"{no_c1['top_k_overlap']}/{no_c1['k_star']}",
                "k_star_no_conv1": no_c1["k_star"],
                "delta_rho": no_c1["rho"] - with_c1["rho"],
                "true_top_layer_no_conv1": no_c1["true_top_layer"],
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

    logger.info(f"{'dataset':<12}{'model':<12}{'pred':<9}"
                f"{'n(w/o1)':>8}{'rho(w/o1)':>11}{'CI':>20}"
                f"{'rank1':>7}{'topk*':>8}   n(no1) rho(no1)  CI                  rank1  topk*   d_rho")
    for r in rows:
        model_lbl = MODEL_LABEL[r["model"]]
        pred_lbl = PREDICTOR_LABEL[r["predictor"]]
        logger.info(
            f"{r['dataset']:<12}{model_lbl:<12}{pred_lbl:<9}"
            f"{r['n_layers_with_conv1']:>8}{r['rho_with_conv1']:>11.3f}"
            f"  [{r['ci_low_with_conv1']:.2f},{r['ci_high_with_conv1']:.2f}]"
            f"{str(r['rank1_with_conv1']):>7}{r['topk_overlap_with_conv1']:>8}   "
            f"{r['n_layers_no_conv1']:>5}{r['rho_no_conv1']:>9.3f}"
            f"  [{r['ci_low_no_conv1']:.2f},{r['ci_high_no_conv1']:.2f}]"
            f"{str(r['rank1_no_conv1']):>7}{r['topk_overlap_no_conv1']:>7}"
            f"{r['delta_rho']:>8.3f}"
        )


if __name__ == "__main__":
    main()
