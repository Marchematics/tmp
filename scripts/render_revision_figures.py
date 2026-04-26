from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data_raw" / "a_project_2" / "outputs" / "paper_revision_figures"


def _style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 220,
        }
    )


def render_method_pipeline():
    fig, ax = plt.subplots(figsize=(10.8, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, 0.27, 0.16, 0.46, "#dceeff", "Frozen OVD\ndetector + NMS", "image, prompt"),
        (0.22, 0.27, 0.17, 0.46, "#fbe7c6", "Fit split\nlearns $\\phi(s)$", "TP / null matches"),
        (0.43, 0.27, 0.17, 0.46, "#e9f7df", "Calibration split\nnull scores only", "$\\{S_j^{\\mathrm{cal}}\}$"),
        (0.64, 0.27, 0.15, 0.46, "#f6def5", "Self-normalized\ne-value", "$e(S)$"),
        (0.83, 0.27, 0.15, 0.46, "#ffe3dc", "Per-family\ne-BH", "released boxes"),
    ]

    for x, y, w, h, color, title, subtitle in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center", fontsize=8.7, color="#333333")

    arrows = [
        ((0.18, 0.50), (0.22, 0.50)),
        ((0.39, 0.50), (0.43, 0.50)),
        ((0.60, 0.50), (0.64, 0.50)),
        ((0.79, 0.50), (0.83, 0.50)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#444444",
            )
        )

    ax.text(0.715, 0.82, r"$e(S)=\frac{(m+1)\phi(S)}{\phi(S)+\sum_j \phi(S_j^{\mathrm{cal}})}$", ha="center", va="center", fontsize=11)
    ax.text(
        0.50,
        0.08,
        "Family-level fit / calibration / test splitting preserves within-family dependence and keeps validity exact under exchangeability.",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / "method_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def render_coco_alpha_curve():
    base = ROOT / "data_raw" / "a_project_2" / "outputs" / "coco_groundingdino_gonogo_1000_mix_formal_hist_smooth0"
    mean_df = pd.read_csv(base / "mean_results_by_alpha.csv")
    oracle_df = pd.read_csv(base / "score_oracle_mean_by_alpha.csv")

    betting = mean_df[mean_df["method"] == "betting"].sort_values("alpha")
    binary = mean_df[mean_df["method"] == "binary"].sort_values("alpha")
    oracle = oracle_df.sort_values("alpha")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8), sharex=True)
    colors = {"betting": "#1f77b4", "binary": "#d62728", "oracle": "#2ca02c"}
    markers = {"betting": "o", "binary": "s", "oracle": "^"}

    for ax, metric, ylabel in [
        (axes[0], "recall", "Recall"),
        (axes[1], "fdp", "Empirical FDP"),
    ]:
        ax.plot(betting["alpha"], betting[metric], color=colors["betting"], marker=markers["betting"], linewidth=2, label="Betting e-BH")
        ax.plot(binary["alpha"], binary[metric], color=colors["binary"], marker=markers["binary"], linewidth=2, linestyle="--", label="Binary e-BH")
        ax.plot(oracle["alpha"], oracle[metric], color=colors["oracle"], marker=markers["oracle"], linewidth=2, linestyle="-.", label="Score oracle")
        ax.set_xlabel(r"Target level $\alpha$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_xticks([0.05, 0.10, 0.15, 0.20])

    axes[0].set_ylim(0, max(oracle["recall"]) * 1.15)
    axes[1].set_ylim(0, 0.23)
    axes[0].legend(loc="upper left", ncol=1, frameon=True)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT_DIR / "coco_alpha_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    render_method_pipeline()
    render_coco_alpha_curve()


if __name__ == "__main__":
    main()
