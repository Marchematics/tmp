#!/usr/bin/env python3
"""Generate publication-quality contamination sweep figure for paper.
Shows COCO and VOC side-by-side for cross-validation."""
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
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "serif",
})

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
FIGURE_DIR = OUTPUT_DIR / "paper_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Reference points from real non-exhaustive datasets
LVIS_PHI_MAX = 31.7
LVIS_FDP = 0.176
OI_PHI_MAX = 37.1
OI_FDP = 0.323
ALPHA = 0.10

DATASETS = [
    ("COCO", OUTPUT_DIR / "contamination_sweep_coco_gdino", "#2166ac", "o", "-"),
    ("VOC", OUTPUT_DIR / "contamination_sweep_voc_gdino", "#d6604d", "D", "--"),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    # ─── Panel (a): φ_max vs η_actual ───
    ax = axes[0]
    for name, sweep_dir, color, marker, ls in DATASETS:
        summary = pd.read_csv(sweep_dir / "contamination_sweep_summary.csv")
        eta = summary["eta_actual_mean"].values
        pos = eta > 0
        ax.plot(
            eta[pos], summary["phi_max_mean"].values[pos],
            marker=marker, linestyle=ls, color=color, markersize=5, linewidth=1.5,
            label=rf"{name} $\phi_{{\max}}$", zorder=5,
        )

    # Theoretical ceiling
    eta_theory = np.linspace(0.003, 0.055, 300)
    ax.plot(eta_theory, 1.0 / eta_theory, "--", color="#b2182b", linewidth=2,
            label=r"Theorem 2: $1/\eta$", zorder=4)

    # LVIS / OI reference
    ax.scatter([1/LVIS_PHI_MAX], [LVIS_PHI_MAX], marker="o", s=70, color="#e66101",
               zorder=6, label="LVIS")
    ax.scatter([1/OI_PHI_MAX], [OI_PHI_MAX], marker="D", s=70, color="#5e3c99",
               zorder=6, label="OI V7")

    ax.set_xlabel(r"Null-pool contamination $\eta$")
    ax.set_ylabel(r"$\phi_{\max}$")
    ax.set_title("(a) Density-ratio ceiling")
    ax.set_yscale("log")
    ax.set_xlim(-0.002, 0.055)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    # ─── Panel (b): FDP vs η_actual ───
    ax = axes[1]
    for name, sweep_dir, color, marker, ls in DATASETS:
        summary = pd.read_csv(sweep_dir / "contamination_sweep_summary.csv")
        eta = summary["eta_actual_mean"].values
        ax.plot(
            eta, summary["fdp_true_mean"],
            marker=marker, linestyle=ls, color=color, markersize=4, linewidth=1.5,
            label=f"{name} true FDP", zorder=5,
        )
    # Show COCO apparent FDP (one dataset enough to show the divergence)
    coco_summary = pd.read_csv(DATASETS[0][1] / "contamination_sweep_summary.csv")
    ax.plot(
        coco_summary["eta_actual_mean"], coco_summary["fdp_contaminated_mean"],
        "o:", color="#999999", markersize=4, linewidth=1.2,
        label="Apparent FDP (COCO)", zorder=3,
    )
    ax.axhline(y=ALPHA, color="gray", linestyle="--", linewidth=1.5, label=rf"$\alpha={ALPHA}$")

    # LVIS / OI as reference
    ax.scatter([0.050], [LVIS_FDP], marker="o", s=70, color="#e66101", zorder=6,
               label=f"LVIS ({LVIS_FDP:.0%})")
    ax.scatter([0.052], [OI_FDP], marker="D", s=70, color="#5e3c99", zorder=6,
               label=f"OI V7 ({OI_FDP:.0%})")

    ax.set_xlabel(r"Null-pool contamination $\eta$")
    ax.set_ylabel("FDP")
    ax.set_title("(b) FDR validity under contamination")
    ax.set_ylim(-0.02, 0.55)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.grid(alpha=0.2)

    # ─── Panel (c): Recall vs η_actual ───
    ax = axes[2]
    for name, sweep_dir, color, marker, ls in DATASETS:
        summary = pd.read_csv(sweep_dir / "contamination_sweep_summary.csv")
        eta = summary["eta_actual_mean"].values
        ax.plot(
            eta, summary["recall_mean"],
            marker=marker, linestyle=ls, color=color, markersize=5, linewidth=1.5,
            label=f"{name} recall", zorder=5,
        )
    ax.set_xlabel(r"Null-pool contamination $\eta$")
    ax.set_ylabel("Recall")
    ax.set_title("(c) Power collapse")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout(w_pad=2.5)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"fig_contamination_sweep.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {FIGURE_DIR}/fig_contamination_sweep.{{pdf,png}}")


if __name__ == "__main__":
    main()
