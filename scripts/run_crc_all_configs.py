#!/usr/bin/env python3
"""Driver: run CRC baseline on all 6 configs × 4 alphas × 20 seeds.

Wraps run_crc_scoregate_baseline.py for each config; uses --risk family_crc
(the canonical CRC-style risk control). Aggregates per-seed CSVs into rows
ready to append to paper_ground_truth_table_2026-04-14.csv.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_crc_scoregate_baseline.py"
OUTPUTS = PROJECT_ROOT / "outputs"

CONFIGS = [
    ("COCO val", "GDino",      "coco_groundingdino_gonogo_1000_mix_analysis"),
    ("COCO val", "OWL-ViT",    "coco_owlvit_val_1000_mix_analysis"),
    ("COCO val", "YOLO-World", "coco_yoloworld_val_1000_mix_analysis"),
    ("VOC 2012", "GDino",      "voc_1000_gdino_analysis"),
    ("VOC 2012", "OWL-ViT",    "voc_1000_owlvit_analysis"),
    ("VOC 2012", "YOLO-World", "voc_yoloworld_1000_mix_analysis"),
]
SEEDS_STR = ",".join(str(s) for s in range(20))


def run_one(analysis_dir: Path, out_dir: Path) -> Path:
    cand_path = analysis_dir / "candidate_table.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "seed_alpha_results.csv"
    if summary_csv.exists():
        print(f"  [cached] {summary_csv}")
        return summary_csv
    cmd = [
        "python", str(SCRIPT),
        "--candidate-table", str(cand_path),
        "--out-dir", str(out_dir),
        "--seeds", SEEDS_STR,
        "--risk", "family_crc",
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return summary_csv


def aggregate(summary_csv: Path, dataset: str, detector: str, source: str) -> list[dict]:
    df = pd.read_csv(summary_csv)
    rows = []
    for alpha, sub in df.groupby("alpha"):
        rows.append({
            "table": "table2_baselines",
            "dataset": dataset,
            "detector": detector,
            "alpha": float(alpha),
            "method": "crc_family",
            "fdp_pooled_mean": float(sub["fdp"].mean()),
            "fdp_pooled_std": float(sub["fdp"].std(ddof=1)),
            "fdp_family_mean": float(sub["mean_family_fdp"].mean()),
            "recall_mean": float(sub["recall"].mean()),
            "rejections_mean": float(sub["total_rejections"].mean()),
            "phi_max_median": "",
            "loo_mean": "",
            "n_seeds": 20,
            "source": source,
            "pooled_bound_eps": "",
            "pooled_bound_delta": "",
            "pooled_bound_covered": "",
            "pooled_bound_eps_stratified": "",
        })
    return rows


def main() -> None:
    all_rows = []
    for dataset, detector, dirname in CONFIGS:
        analysis_dir = OUTPUTS / dirname
        if not (analysis_dir / "candidate_table.csv").exists():
            print(f"[skip] {dataset}/{detector}: no candidate_table")
            continue
        out_dir = OUTPUTS / f"{dirname.rsplit('_analysis',1)[0]}_crc_baseline"
        print(f"\n{dataset} / {detector} -> {out_dir.name}")
        try:
            summary_csv = run_one(analysis_dir, out_dir)
        except subprocess.CalledProcessError as e:
            print(f"  [fail] {e}")
            continue
        rows = aggregate(summary_csv, dataset, detector, f"outputs/{out_dir.name}")
        for r in rows:
            print(
                f"  α={r['alpha']:.2f}  rej={r['rejections_mean']:6.1f}  "
                f"FDP={r['fdp_pooled_mean']:.4f}  recall={r['recall_mean']:.4f}"
            )
        all_rows.extend(rows)

    out_csv = OUTPUTS / "crc_baseline_all_configs.csv"
    if not all_rows:
        print("No rows produced")
        return
    fieldnames = list(all_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
