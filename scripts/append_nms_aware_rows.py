"""Append NMS-aware rows to paper_ground_truth_table_2026-04-14.csv.

Reads summary.csv from each outputs/*_nmsaware_*/ directory, aggregates per
(source, alpha), and appends rows with table='nms_aware_evalue' into the
main CSV following the 18-col schema.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CSV = PROJECT / "outputs/paper_ground_truth_table_2026-04-14.csv"

# (source_dir, dataset, detector, iou)
SOURCES = [
    ("outputs/coco_gdino_1000_nmsaware_20seed_iou050", "COCO", "GDino", 0.50),
    ("outputs/coco_gdino_1000_nmsaware_20seed_iou030_full", "COCO", "GDino", 0.30),
    ("outputs/coco_gdino_1000_nmsaware_20seed_iou070_full", "COCO", "GDino", 0.70),
    ("outputs/voc_gdino_1000_nmsaware_20seed_iou050", "VOC", "GDino", 0.50),
    ("outputs/lvis_gdino_1000_nmsaware_5seed_iou050", "LVIS", "GDino", 0.50),
    ("outputs/coco_regionclip_1000_nmsaware_20seed_iou050", "COCO", "RegionCLIP", 0.50),
    ("outputs/coco_owlvit_1000_nmsaware_20seed_iou050", "COCO", "OWL-ViT", 0.50),
    ("outputs/coco_train_gdino_1000_nmsaware_20seed_iou050", "COCO-train", "GDino", 0.50),
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
    print(f"main CSV rows: {len(df_main)} cols: {list(df_main.columns)}")
    # Drop any existing nms_aware_evalue rows so this script is idempotent
    before = len(df_main)
    df_main = df_main[df_main["table"] != "nms_aware_evalue"].copy()
    print(f"dropped {before - len(df_main)} existing nms_aware_evalue rows")

    new_rows = []
    for src_rel, dataset, detector, iou in SOURCES:
        summary = PROJECT / src_rel / "summary.csv"
        if not summary.exists():
            print(f"  [skip] no summary: {summary}")
            continue
        df = pd.read_csv(summary)
        # Aggregate per alpha
        agg = df.groupby("alpha").agg(
            fdp_pooled_mean=("nms_fdp_pooled", "mean"),
            fdp_pooled_std=("nms_fdp_pooled", "std"),
            fdp_family_mean=("nms_family_fdp_mean", "mean"),
            # Use HONEST object-level recall (apples-to-apples with baseline on
            # the same NMS cluster graph). The previous "recall_members" metric
            # double-counted TP candidates within clusters and inflated gains.
            recall_mean=("nms_recall_object", "mean"),
            rejections_mean=("nms_rej", "mean"),
            n_seeds=("seed", "count"),
        ).reset_index()
        agg["table"] = "nms_aware_evalue"
        agg["dataset"] = dataset
        agg["detector"] = detector
        agg["method"] = f"betting_eBH_nmsaware_iou{int(iou*100):03d}"
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
        print("no rows to append")
        return

    new_df = pd.concat(new_rows, ignore_index=True)
    # Round numeric for readability
    for c in ["fdp_pooled_mean", "fdp_pooled_std", "fdp_family_mean",
              "recall_mean"]:
        new_df[c] = new_df[c].astype(float).round(4)
    new_df["rejections_mean"] = new_df["rejections_mean"].astype(float).round(2)
    new_df["alpha"] = new_df["alpha"].astype(float).round(3)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CSV.with_suffix(f".bak.{ts}.csv")
    shutil.copy(CSV, backup)
    combined = pd.concat([df_main, new_df], ignore_index=True)
    combined.to_csv(CSV, index=False)
    print(f"\nappended {len(new_df)} rows; new total {len(combined)}")
    print(f"backup: {backup}")
    print(f"wrote : {CSV}")

    print("\n=== new rows preview ===")
    print(new_df.to_string(index=False))


if __name__ == "__main__":
    main()
