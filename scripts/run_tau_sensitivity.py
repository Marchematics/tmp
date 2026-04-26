#!/usr/bin/env python3
"""IoU threshold sensitivity analysis.

Reads an existing candidate_table.csv (which has max_iou column),
re-labels is_tp with tau in {0.3, 0.5, 0.7}, then runs the formal
betting pipeline on each.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def relabel_candidate_table(input_csv: str, tau: float, output_csv: str):
    df = pd.read_csv(input_csv)
    df["is_tp"] = df["max_iou"] >= tau
    df["is_null"] = ~df["is_tp"]
    df.to_csv(output_csv, index=False)
    n_tp = df["is_tp"].sum()
    n_null = df["is_null"].sum()
    print(f"tau={tau}: {n_tp} TP, {n_null} null out of {len(df)} candidates")
    return output_csv


def run_formal_pipeline(candidate_csv: str, out_dir: str, n_seeds: int = 5):
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_formal_betting_pipeline.py"),
        "--candidate-table", candidate_csv,
        "--out-dir", out_dir,
        "--alphas", "0.05,0.1,0.15,0.2",
        "--seeds", ",".join(str(i) for i in range(n_seeds)),
        "--split-ratios", "0.6,0.2,0.2",
        "--estimator", "hist",
        "--num-bins", "80",
        "--smoothing", "0",
        "--cal-null-sizes", "all",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=str, required=True,
                        help="Original candidate_table.csv with max_iou column")
    parser.add_argument("--out-base", type=str, required=True,
                        help="Base output directory")
    parser.add_argument("--taus", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    taus = [float(x) for x in args.taus.split(",")]
    results = []

    for tau in taus:
        out_dir = os.path.join(args.out_base, f"tau_{tau:.1f}")
        os.makedirs(out_dir, exist_ok=True)

        # Re-label
        relabeled_csv = os.path.join(out_dir, "candidate_table.csv")
        relabel_candidate_table(args.candidate_csv, tau, relabeled_csv)

        # Run formal pipeline
        pipeline_out = os.path.join(out_dir, "formal_hist")
        run_formal_pipeline(relabeled_csv, pipeline_out, args.n_seeds)

        # Read results
        mean_csv = os.path.join(pipeline_out, "mean_results_by_alpha.csv")
        if os.path.exists(mean_csv):
            df = pd.read_csv(mean_csv)
            sub = df[(df["alpha"] == 0.10) & (df["method"] == "betting")]
            if len(sub) > 0:
                row = sub.iloc[0]
                results.append({
                    "tau": tau,
                    "fdp": row.get("fdp", 0),
                    "recall": row.get("recall", 0),
                    "rejections": row.get("total_rejections", 0),
                })

    if results:
        print("\n=== Tau sensitivity summary at alpha=0.10 ===")
        for r in results:
            print(f"tau={r['tau']:.1f}: FDP={r['fdp']*100:.2f}%, Recall={r['recall']*100:.1f}%, Rej={r['rejections']:.0f}")

        # Also read phi_max from seed summaries
        for r in results:
            tau = r["tau"]
            seed_dir = os.path.join(args.out_base, f"tau_{tau:.1f}", "formal_hist", "seed_0")
            summary_path = os.path.join(seed_dir, "summary.json")
            if os.path.exists(summary_path):
                import json
                with open(summary_path) as f:
                    d = json.load(f)
                r["phi_max"] = d.get("phi_max", "?")
                print(f"tau={tau:.1f}: phi_max={r['phi_max']}")


if __name__ == "__main__":
    main()
