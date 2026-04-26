#!/usr/bin/env python3
"""Raw detector baseline (E1): no e-BH, no filtering.

For each (dataset, detector), evaluate the detector's raw candidate output
on the same 60/20/20 family-level test split as the main pipeline (20 seeds).

Reports (aligned with main pipeline's candidate-level metric):
  - raw_pooled_fdp  = #FP / #total candidates  (the FDP reviewer cares about)
  - raw_recall      = 1.0  (raw detector keeps all TP candidates by definition)
  - raw_rejections  = #total candidates
  - obj_recall_proxy = min(1, #TP / #GT_present) (object-level proxy)

Output schema matches `paper_ground_truth_table_2026-04-14.csv` table2_baselines.
Method name: ``raw_detector``. Alpha is replicated across {0.05, 0.10, 0.15, 0.20}
for join compatibility (raw output is alpha-independent).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

CONFIGS = [
    # (dataset_label, detector_label, analysis_dir)
    ("COCO val", "GDino",      "coco_groundingdino_gonogo_1000_mix_analysis"),
    ("COCO val", "OWL-ViT",    "coco_owlvit_val_1000_mix_analysis"),
    ("COCO val", "YOLO-World", "coco_yoloworld_val_1000_mix_analysis"),
    ("VOC 2012", "GDino",      "voc_1000_gdino_analysis"),
    ("VOC 2012", "OWL-ViT",    "voc_1000_owlvit_analysis"),
    ("VOC 2012", "YOLO-World", "voc_yoloworld_1000_mix_analysis"),
]
ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 20
SPLIT_RATIOS = (0.6, 0.2, 0.2)


def split_test_families(df: pd.DataFrame, *, seed: int) -> set:
    """Mirror split_families() in run_formal_betting_pipeline.py."""
    rng = np.random.default_rng(seed)
    families = df["family_id"].drop_duplicates().to_numpy()
    rng.shuffle(families)
    n_total = len(families)
    n_fit = int(round(SPLIT_RATIOS[0] * n_total))
    n_cal = int(round(SPLIT_RATIOS[1] * n_total))
    n_fit = max(1, min(n_fit, n_total - 2))
    n_cal = max(1, min(n_cal, n_total - n_fit - 1))
    return set(families[n_fit + n_cal :])


def evaluate_config(analysis_dir: Path) -> dict:
    cand_path = analysis_dir / "candidate_table.csv"
    if not cand_path.exists():
        raise FileNotFoundError(cand_path)
    df = pd.read_csv(cand_path)

    # Per-family GT count (only present families have non-zero GT)
    family_gt = (
        df.drop_duplicates("family_id")
        .set_index("family_id")[["gt_count", "is_prompt_absent"]]
    )

    rejections, fdp_pooled, obj_recall = [], [], []
    for seed in range(N_SEEDS):
        test_fams = split_test_families(df, seed=seed)
        sub = df[df["family_id"].isin(test_fams)]
        gt_test = family_gt.loc[list(test_fams)]
        total_gt = int(gt_test["gt_count"].sum())
        n_cand = len(sub)
        n_fp = int((~sub["is_tp"]).sum())
        n_tp = int(sub["is_tp"].sum())
        rejections.append(n_cand)
        fdp_pooled.append(n_fp / max(n_cand, 1))
        obj_recall.append(min(1.0, n_tp / max(total_gt, 1)))

    return {
        "rejections_mean": float(np.mean(rejections)),
        "fdp_pooled_mean": float(np.mean(fdp_pooled)),
        "fdp_pooled_std": float(np.std(fdp_pooled, ddof=1)),
        "recall_mean": 1.0,  # candidate-level by definition (no filter)
        "obj_recall_proxy": float(np.mean(obj_recall)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(OUTPUTS / "raw_detector_baseline.csv"),
        help="Output CSV path",
    )
    args = ap.parse_args()

    rows = []
    for dataset, detector, dirname in CONFIGS:
        analysis_dir = OUTPUTS / dirname
        try:
            stats = evaluate_config(analysis_dir)
        except FileNotFoundError as e:
            print(f"[skip] {dataset}/{detector}: {e}")
            continue
        for alpha in ALPHAS:
            rows.append({
                "table": "table2_baselines",
                "dataset": dataset,
                "detector": detector,
                "alpha": alpha,
                "method": "raw_detector",
                "fdp_pooled_mean": stats["fdp_pooled_mean"],
                "fdp_pooled_std": stats["fdp_pooled_std"],
                "fdp_family_mean": "",
                "recall_mean": stats["recall_mean"],
                "rejections_mean": stats["rejections_mean"],
                "phi_max_median": "",
                "loo_mean": "",
                "n_seeds": N_SEEDS,
                "source": f"outputs/{dirname}",
                "pooled_bound_eps": "",
                "pooled_bound_delta": "",
                "pooled_bound_covered": "",
                "pooled_bound_eps_stratified": "",
            })
        print(
            f"{dataset:10s} {detector:11s}  rej={stats['rejections_mean']:8.1f}  "
            f"FDP={stats['fdp_pooled_mean']:.4f}±{stats['fdp_pooled_std']:.4f}  "
            f"obj_recall_proxy={stats['obj_recall_proxy']:.4f}"
        )

    out_path = Path(args.out)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
