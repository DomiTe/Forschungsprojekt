"""Rebuild paper/figures/main_figure.pdf with the per-panel Spearman annotation
moved out of the plot area (it used to float as a fixed textbox in the
upper-left of every panel, overlapping scattered data points, most visibly
in the CIFAR10 row). The annotation is now part of each panel's title block
(above the axes, in the reserved title margin) instead of sitting on top of
the data.

Data sources, all PTQ stage only:
  - per-layer metric values (Sraw/Spert/Shawq): weight_ablation_canonical_v2*.csv
  - per-layer abs_loss_damage: weight_ablation_loss_damage.csv
  - Spearman rho + 2.5/97.5% bootstrap CI (B=2000, seed=42): bootstrap_ci_spearman_loss.csv
    (reused as-is so the annotated numbers match Table 1 / the surrounding
    prose exactly, rather than being recomputed with a different bootstrap
    draw that would only agree to ~1 decimal place).
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/home/dominic/Desktop/HTW-Master/3rd-Semester/Forschungsprojekt"

LOSS_DAMAGE_CSV = f"{REPO}/results/20260822_131637_61872/csv/weight_ablation_loss_damage.csv"
CANON_CIFAR_CSV = f"{REPO}/results/20260816_230437_38678/csv/weight_ablation_canonical_v2.csv"
CANON_IMAGENET_CSV = f"{REPO}/results/20260816_083054_38677/csv/weight_ablation_canonical_v2_imagnet.csv"
BOOTSTRAP_CSV = f"{REPO}/results/review_response/csv/bootstrap_ci_spearman_loss.csv"
OUT_PDF = f"{REPO}/paper/figures/main_figure.pdf"

MODEL_ORDER = ["cnn", "resnet18_no_weights", "resnet50_no_weights"]
MODEL_LABEL = {"cnn": "CNN", "resnet18_no_weights": "ResNet-18", "resnet50_no_weights": "ResNet-50"}
DATASET_ORDER = ["IMAGENET100", "CIFAR10"]
DATASET_LABEL = {"IMAGENET100": "ImageNet100", "CIFAR10": "CIFAR10"}
PRED_TO_METRIC = {"raw_trh": "Sraw", "dwsq": "Spert", "trh_dwsq": "Shawq"}
METRIC_ORDER = ["Sraw", "Spert", "Shawq"]


def load_metric_values():
    values = {}
    for path in (CANON_CIFAR_CSV, CANON_IMAGENET_CSV):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["stage"] != "PTQ":
                    continue
                key = (row["model"], row["dataset"], row["layer"])
                values[key] = {
                    "Sraw": float(row["hessian_trace_fused"]),
                    "Spert": float(row["delta_w_sq"]),
                    "Shawq": float(row["trh_times_dwsq"]),
                }
    return values


def load_damage():
    damage = {}
    with open(LOSS_DAMAGE_CSV) as f:
        for row in csv.DictReader(f):
            if row["stage"] != "PTQ":
                continue
            key = (row["model"], row["dataset"], row["layer"])
            damage[key] = float(row["abs_loss_damage"])
    return damage


def load_bootstrap():
    stats = {}
    with open(BOOTSTRAP_CSV) as f:
        for row in csv.DictReader(f):
            metric = PRED_TO_METRIC[row["predictor"]]
            key = (row["model"], row["dataset"], metric)
            stats[key] = (float(row["rho"]), float(row["ci_low"]), float(row["ci_high"]))
    return stats


def main():
    metric_values = load_metric_values()
    damage = load_damage()
    bootstrap = load_bootstrap()

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), layout="constrained")

    colors = {"Sraw": "#4C72B0", "Spert": "#DD8452", "Shawq": "#55A868"}
    markers = {"Sraw": "o", "Spert": "^", "Shawq": "D"}
    greek = {"Sraw": "raw", "Spert": "pert", "Shawq": "hawq"}

    for col, model in enumerate(MODEL_ORDER):
        for row, dataset in enumerate(DATASET_ORDER):
            ax = axes[row, col]
            keys = [k for k in metric_values if k[0] == model and k[1] == dataset and k in damage]
            layers = [k[2] for k in keys]
            y = np.array([damage[k] for k in keys])

            top_idx = int(np.argmax(y))
            top_layer = layers[top_idx]

            rho_lines = []
            for metric in METRIC_ORDER:
                x = np.array([metric_values[k][metric] for k in keys])
                ax.scatter(
                    x, y, s=20, marker=markers[metric], color=colors[metric],
                    alpha=0.75, linewidths=0.3, edgecolors="white", zorder=2,
                )
                rho, lo, hi = bootstrap[(model, dataset, metric)]
                sig = lo > 0 or hi < 0
                rho_txt = f"{rho:.2f}"
                ci_txt = f"[{lo:.2f},{hi:.2f}]"
                if sig:
                    rho_lines.append(
                        r"$S_{%s}\ \rho=\mathbf{%s\ %s}$" % (greek[metric], rho_txt, ci_txt)
                    )
                else:
                    rho_lines.append(r"$S_{%s}\ \rho=%s\ %s$" % (greek[metric], rho_txt, ci_txt))

            xt = np.array([metric_values[k]["Sraw"] for k in keys])[top_idx]
            yt = y[top_idx]
            ax.scatter(
                [xt], [yt], s=100, marker="D", facecolors="none",
                edgecolors="black", linewidths=1.3, zorder=3,
            )
            if top_layer != "conv1":
                ax.annotate(
                    top_layer, (xt, yt), textcoords="offset points",
                    xytext=(5, 5), fontsize=6, ha="left",
                )

            ax.set_xscale("log")
            ax.tick_params(labelsize=6.5)
            ax.margins(y=0.18)

            if row == 0:
                title = MODEL_LABEL[model] + "\n" + "\n".join(rho_lines)
            else:
                title = "\n".join(rho_lines)
            ax.set_title(title, fontsize=6.3, linespacing=1.6, loc="left")

            if col == 0:
                ax.set_ylabel(f"{DATASET_LABEL[dataset]}\nabs_loss_damage", fontsize=7.5)
            if row == 1:
                ax.set_xlabel("metric value (log)", fontsize=7.5)

    legend_elems = [
        Line2D([0], [0], marker=markers[m], color="w", markerfacecolor=colors[m],
               markersize=6, label=f"$S_{{{greek[m]}}}$")
        for m in METRIC_ORDER
    ]
    fig.legend(
        handles=legend_elems, loc="outside upper center", ncol=3, frameon=False,
        fontsize=8.5, handletextpad=0.4, columnspacing=1.4,
    )

    fig.savefig(OUT_PDF)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
