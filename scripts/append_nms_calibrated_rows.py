"""Append score-adaptive calibrated NMS-aware rows to the main CSV.

Rows get table='nms_aware_calibrated' with method='betting_eBH_nmsaware_sa_iou050'.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CSV = PROJECT / "outputs/paper_ground_truth_table_2026-04-14.csv"

SOURCES = [
    # (src_rel, dataset, detector, iou, R_stat_override, method_tag_override)
    ("outputs/coco_gdino_nmsaware_calibrated_20seed", "COCO", "GDino", 0.50, None, None),
    ("outputs/voc_gdino_nmsaware_calibrated_20seed", "VOC", "GDino", 0.50, None, None),
    ("outputs/coco_train_gdino_nmsaware_calibrated_20seed", "COCO-train", "GDino", 0.50, None, None),
    ("outputs/coco_regionclip_nmsaware_calibrated_20seed", "COCO", "RegionCLIP", 0.50, None, None),
    ("outputs/coco_owlvit_nmsaware_calibrated_20seed", "COCO", "OWL-ViT", 0.50, None, None),
    ("outputs/lvis_gdino_nmsaware_calibrated_20seed", "LVIS", "GDino", 0.50, None, None),
    # IoU sweep ablations (τ=0.30 stronger compression, τ=0.70 degenerate)
    ("outputs/coco_gdino_nmsaware_calibrated_20seed_iou030", "COCO", "GDino", 0.30, None, None),
    ("outputs/coco_gdino_nmsaware_calibrated_20seed_iou070", "COCO", "GDino", 0.70, None, None),
    ("outputs/voc_gdino_nmsaware_calibrated_20seed_iou030", "VOC", "GDino", 0.30, None, None),
    ("outputs/voc_gdino_nmsaware_calibrated_20seed_iou070", "VOC", "GDino", 0.70, None, None),
    # R-statistic ablations (COCO/GDino τ=0.50)
    ("outputs/coco_gdino_nmsaware_calibrated_20seed_sumphi", "COCO", "GDino", 0.50, "sumphi",
     "betting_eBH_nmsaware_sa_sumphi_iou050"),
    ("outputs/coco_gdino_nmsaware_calibrated_20seed_top2mean", "COCO", "GDino", 0.50, "top2mean",
     "betting_eBH_nmsaware_sa_top2meanphi_iou050"),
    # Mondrian stratification (COCO/GDino τ=0.50, max_phi + per-cluster-size cal pool)
    ("outputs/coco_gdino_nmsaware_calibrated_20seed_mondrian", "COCO", "GDino", 0.50, "mondrian",
     "betting_eBH_nmsaware_sa_maxphi_mondrian_iou050"),
    # Cross-detector IoU sweep
    ("outputs/coco_regionclip_nmsaware_calibrated_20seed_iou030", "COCO", "RegionCLIP", 0.30, None, None),
    ("outputs/coco_regionclip_nmsaware_calibrated_20seed_iou070", "COCO", "RegionCLIP", 0.70, None, None),
    ("outputs/coco_owlvit_nmsaware_calibrated_20seed_iou030", "COCO", "OWL-ViT", 0.30, None, None),
    ("outputs/coco_owlvit_nmsaware_calibrated_20seed_iou070", "COCO", "OWL-ViT", 0.70, None, None),
    # VOC + COCO-train R-statistic ablations (confirming max_phi ranking)
    ("outputs/voc_gdino_nmsaware_calibrated_20seed_sumphi", "VOC", "GDino", 0.50, "sumphi",
     "betting_eBH_nmsaware_sa_sumphi_iou050"),
    ("outputs/voc_gdino_nmsaware_calibrated_20seed_top2mean", "VOC", "GDino", 0.50, "top2mean",
     "betting_eBH_nmsaware_sa_top2meanphi_iou050"),
    ("outputs/coco_train_gdino_nmsaware_calibrated_20seed_sumphi", "COCO-train", "GDino", 0.50, "sumphi",
     "betting_eBH_nmsaware_sa_sumphi_iou050"),
    ("outputs/coco_train_gdino_nmsaware_calibrated_20seed_top2mean", "COCO-train", "GDino", 0.50, "top2mean",
     "betting_eBH_nmsaware_sa_top2meanphi_iou050"),
    # RegionCLIP extreme compression (τ=0.10, 0.20)
    ("outputs/coco_regionclip_nmsaware_calibrated_20seed_iou010", "COCO", "RegionCLIP", 0.10, None, None),
    ("outputs/coco_regionclip_nmsaware_calibrated_20seed_iou020", "COCO", "RegionCLIP", 0.20, None, None),
]

SCHEMA = [
    "table", "dataset", "detector", "alpha", "method",
    "fdp_pooled_mean", "fdp_pooled_std", "fdp_family_mean",
    "recall_mean", "rejections_mean",
    "phi_max_median", "loo_mean", "n_seeds", "source",
    "pooled_bound_eps", "pooled_bound_delta", "pooled_bound_covered",
    "pooled_bound_eps_stratified",
]


def main() -> None:
    df_main = pd.read_csv(CSV)
    before = len(df_main)
    df_main = df_main[df_main["table"] != "nms_aware_calibrated"].copy()
    print(f"main: {before} rows; dropped {before-len(df_main)} stale calibrated rows")

    new_rows = []
    for src_rel, dataset, detector, iou, R_override, method_override in SOURCES:
        summary = PROJECT / src_rel / "summary.csv"
        if not summary.exists():
            print(f"  [skip] {summary}")
            continue
        df = pd.read_csv(summary)
        agg = df.groupby("alpha").agg(
            fdp_pooled_mean=("sa_fdp_pooled", "mean"),
            fdp_pooled_std=("sa_fdp_pooled", "std"),
            fdp_family_mean=("sa_family_fdp_mean", "mean"),
            recall_mean=("sa_recall_object", "mean"),
            rejections_mean=("sa_rej", "mean"),
            n_seeds=("seed", "count"),
        ).reset_index()
        agg["table"] = "nms_aware_calibrated"
        agg["dataset"] = dataset
        agg["detector"] = detector
        if method_override is not None:
            agg["method"] = method_override
        else:
            agg["method"] = f"betting_eBH_nmsaware_sa_maxphi_iou{int(iou*100):03d}"
        agg["phi_max_median"] = np.nan
        agg["loo_mean"] = np.nan
        agg["source"] = src_rel
        agg["pooled_bound_eps"] = np.nan
        agg["pooled_bound_delta"] = np.nan
        agg["pooled_bound_covered"] = ""
        agg["pooled_bound_eps_stratified"] = np.nan
        agg = agg[SCHEMA]
        new_rows.append(agg)
        print(f"  +{len(agg)} rows from {src_rel}")

    if not new_rows:
        return

    new_df = pd.concat(new_rows, ignore_index=True)
    for c in ["fdp_pooled_mean", "fdp_pooled_std", "fdp_family_mean", "recall_mean"]:
        new_df[c] = new_df[c].astype(float).round(4)
    new_df["rejections_mean"] = new_df["rejections_mean"].astype(float).round(2)
    new_df["alpha"] = new_df["alpha"].astype(float).round(3)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CSV.with_suffix(f".bak.{ts}.csv")
    shutil.copy(CSV, backup)
    combined = pd.concat([df_main, new_df], ignore_index=True)
    combined.to_csv(CSV, index=False)
    print(f"appended {len(new_df)} rows; total {len(combined)}; backup: {backup}")
    print("\n=== new rows preview ===")
    print(new_df.to_string(index=False))


if __name__ == "__main__":
    main()
