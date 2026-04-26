#!/usr/bin/env python3
"""Object-level vs candidate-level comparison table (E14).

Reads existing ground truth CSV, joins:
  - candidate-level metrics from table1_main (betting_eBH)
  - object-level (cluster) metrics from nms_aware_calibrated (default IoU=0.50)

Outputs a paper-ready comparison table.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT = PROJECT_ROOT / "outputs" / "paper_ground_truth_table_2026-04-14.csv"
OUT = PROJECT_ROOT / "outputs" / "object_vs_candidate_table.csv"


def main() -> None:
    df = pd.read_csv(GT)
    cand = df[
        (df["table"] == "table1_main")
        & (df["method"] == "betting_eBH")
    ][["dataset", "detector", "alpha", "fdp_pooled_mean", "recall_mean", "rejections_mean"]]
    cand = cand.rename(columns={
        "fdp_pooled_mean": "candidate_fdp",
        "recall_mean": "candidate_recall",
        "rejections_mean": "candidate_rejections",
    })

    obj = df[
        (df["table"] == "nms_aware_calibrated")
        & (df["method"] == "betting_eBH_nmsaware_sa_maxphi_iou050")
    ][["dataset", "detector", "alpha", "fdp_pooled_mean", "recall_mean", "rejections_mean"]]
    obj = obj.rename(columns={
        "fdp_pooled_mean": "object_fdp",
        "recall_mean": "object_recall",
        "rejections_mean": "object_rejections",
    })

    # Normalize dataset name (cand uses "COCO val" / "VOC 2012", obj uses "COCO" / "VOC")
    obj["dataset"] = obj["dataset"].replace({"COCO": "COCO val", "VOC": "VOC 2012"})

    merged = cand.merge(obj, on=["dataset", "detector", "alpha"], how="inner")
    merged["recall_gain_x"] = merged["object_recall"] / merged["candidate_recall"]
    merged = merged.sort_values(["dataset", "detector", "alpha"])

    print(merged.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    merged.to_csv(OUT, index=False)
    print(f"\nWrote {len(merged)} rows to {OUT}")


if __name__ == "__main__":
    main()
