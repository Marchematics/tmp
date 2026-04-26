#!/usr/bin/env python3
"""Generate Figure 1: alpha-sweep risk curves for 6 dataset-detector configs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "outputs" / "paper_ground_truth_table_2026-04-14.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data from ground truth CSV
rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["table"] == "table1_main" and row["dataset"] in ("COCO val", "VOC 2012"):
            rows.append(row)

# Organize by (dataset, detector)
configs = {}
for row in rows:
    key = (row["dataset"], row["detector"])
    if key not in configs:
        configs[key] = {"alphas": [], "fdp": [], "fdp_std": [], "recall": [], "recall_std": []}
    configs[key]["alphas"].append(float(row["alpha"]))
    configs[key]["fdp"].append(float(row["fdp_pooled_mean"]))
    configs[key]["fdp_std"].append(float(row["fdp_pooled_std"]))
    configs[key]["recall"].append(float(row["recall_mean"]))

# Style mapping
style_map = {
    ("COCO val", "GDino"):      {"color": "#1f77b4", "marker": "o", "ls": "-",  "label": "COCO / GDino"},
    ("COCO val", "OWL-ViT"):    {"color": "#ff7f0e", "marker": "o", "ls": "-",  "label": "COCO / OWL-ViT"},
    ("COCO val", "YOLO-World"):  {"color": "#2ca02c", "marker": "o", "ls": "-",  "label": "COCO / YOLO-World"},
    ("VOC 2012", "GDino"):      {"color": "#d62728", "marker": "D", "ls": "--", "label": "VOC / GDino"},
    ("VOC 2012", "OWL-ViT"):    {"color": "#9467bd", "marker": "D", "ls": "--", "label": "VOC / OWL-ViT"},
    ("VOC 2012", "YOLO-World"):  {"color": "#8c564b", "marker": "D", "ls": "--", "label": "VOC / YOLO-World"},
}

# Plot order for legend consistency
plot_order = [
    ("COCO val", "GDino"),
    ("COCO val", "OWL-ViT"),
    ("COCO val", "YOLO-World"),
    ("VOC 2012", "GDino"),
    ("VOC 2012", "OWL-ViT"),
    ("VOC 2012", "YOLO-World"),
]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)

# (a) FDP
ax = axes[0]
alphas_ref = [0.05, 0.10, 0.15, 0.20]
ax.plot(alphas_ref, alphas_ref, "k--", alpha=0.25, linewidth=0.8, label="$y = x$")
for key in plot_order:
    if key not in configs:
        continue
    d = configs[key]
    s = style_map[key]
    alphas_arr = np.array(d["alphas"])
    fdp_arr = np.array(d["fdp"])
    fdp_std_arr = np.array(d["fdp_std"])
    ax.fill_between(alphas_arr, fdp_arr - fdp_std_arr, fdp_arr + fdp_std_arr,
                    color=s["color"], alpha=0.12)
    ax.plot(d["alphas"], d["fdp"],
            color=s["color"], marker=s["marker"], ls=s["ls"],
            markersize=5, linewidth=1.5, label=s["label"])
ax.set_ylabel("Empirical FDP", fontsize=8)
ax.set_title("(a) Empirical FDP", fontsize=9)
ax.set_xticks(alphas_ref)
ax.set_ylim(-0.003, 0.10)
ax.tick_params(labelsize=7)

# (b) Recall
ax = axes[1]
for key in plot_order:
    if key not in configs:
        continue
    d = configs[key]
    s = style_map[key]
    ax.plot(d["alphas"], d["recall"],
            color=s["color"], marker=s["marker"], ls=s["ls"],
            markersize=5, linewidth=1.5, label=s["label"])
ax.set_ylabel("Recall", fontsize=8)
ax.set_title("(b) Recall", fontsize=9)
ax.set_xticks(alphas_ref)
ax.tick_params(labelsize=7)

# Shared x-label
for a in axes:
    a.set_xlabel(r"Target level $\alpha$", fontsize=8)

# Collect legend handles from the FDP subplot (which has y=x + all 6 configs)
handles, labels = axes[0].get_legend_handles_labels()

# Place legend OUTSIDE the figure at the bottom, with enough room
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.02), frameon=False, columnspacing=0.8,
           handletextpad=0.4)

# Adjust layout: leave space at bottom for legend
fig.subplots_adjust(bottom=0.28, wspace=0.3)

fig.savefig(OUT_DIR / "fig1_alpha_sweep.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT_DIR / "fig1_alpha_sweep.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved to {OUT_DIR / 'fig1_alpha_sweep.pdf'}")
print(f"Saved to {OUT_DIR / 'fig1_alpha_sweep.png'}")
