#!/usr/bin/env python3
"""Generate K (family size) distribution figure for paper."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_hallucination_fdr_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "font.family": "serif",
})

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
FIGURE_DIR = OUTPUT_DIR / "paper_figures"

CONFIGS = [
    ("COCO/GDino", "outputs/coco_groundingdino_gonogo_1000_mix_analysis/candidate_table.csv", "#2166ac"),
    ("COCO/OWL-ViT", "outputs/coco_owlvit_val_1000_mix_analysis/candidate_table.csv", "#4393c3"),
    ("VOC/GDino", "outputs/voc_1000_gdino_analysis/candidate_table.csv", "#d6604d"),
    ("VOC/OWL-ViT", "outputs/voc_1000_owlvit_analysis/candidate_table.csv", "#f4a582"),
]

# Also get per-K FDP/recall from strata results
STRATA_CONFIGS = [
    ("COCO/GDino", "outputs/coco_gdino_1000_20seed_formal_hist"),
    ("VOC/GDino", "outputs/voc_gdino_1000_20seed_formal_hist"),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel (a): K distribution (CDF)
    ax = axes[0]
    for label, path, color in CONFIGS:
        df = pd.read_csv(OUTPUT_DIR.parent / path)
        K = df.groupby("family_id").size()
        K_sorted = np.sort(K.values)
        cdf = np.arange(1, len(K_sorted) + 1) / len(K_sorted)
        ax.plot(K_sorted, cdf, color=color, linewidth=1.5, label=label)

    # Structural impossibility region: K < ceil(alpha*(m+1)/phi_max)
    # For alpha=0.10 and typical m~500-2000, even K=1 could theoretically reject
    # But practically, small K families rarely produce rejections
    ax.axvline(x=5, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.annotate("$K=5$", xy=(5.5, 0.15), fontsize=9, color="gray")

    ax.set_xlabel("Family size $K$ (candidates per family)")
    ax.set_ylabel("Cumulative fraction of families")
    ax.set_title("(a) Family size distribution")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 300)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    # Panel (b): FDP and recall by K stratum
    ax = axes[1]
    alpha = 0.10
    colors_fdp = {"COCO/GDino": "#2166ac", "VOC/GDino": "#d6604d"}
    colors_rec = {"COCO/GDino": "#4393c3", "VOC/GDino": "#f4a582"}
    strata_order = ["K<=5", "5<K<=30", "K>30"]
    x_pos = np.arange(len(strata_order))
    width = 0.35

    for i, (label, out_dir) in enumerate(STRATA_CONFIGS):
        strata = pd.read_csv(OUTPUT_DIR.parent / out_dir / "family_strata_results.csv")
        strata_a10 = strata[(strata["method"] == "betting") & (strata["alpha"] == alpha)]

        fdps = []
        recalls = []
        for s in strata_order:
            sub = strata_a10[strata_a10["stratum"] == s]
            fdps.append(sub["fdp"].mean() * 100)
            recalls.append(sub["recall"].mean() * 100)

        offset = (i - 0.5) * width
        bars = ax.bar(x_pos + offset, fdps, width * 0.9, color=colors_fdp[label],
                       alpha=0.8, label=f"{label} FDP")
        # Recall as text above bars
        for j, (f, r) in enumerate(zip(fdps, recalls)):
            ax.annotate(f"R={r:.0f}%", xy=(x_pos[j] + offset, f + 0.3),
                        fontsize=7.5, ha="center", va="bottom", color=colors_fdp[label])

    ax.axhline(y=alpha * 100, color="gray", linestyle=":", linewidth=1.5, label=r"$\alpha$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strata_order)
    ax.set_xlabel("Family size stratum")
    ax.set_ylabel("Pooled FDP (%)")
    ax.set_title(f"(b) FDP by family size ($\\alpha={alpha}$)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"fig_k_distribution.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {FIGURE_DIR}/fig_k_distribution.{{pdf,png}}")


if __name__ == "__main__":
    main()
