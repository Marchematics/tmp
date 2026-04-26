#!/usr/bin/env python3
"""Compute Benjamini-Yekutieli (BY) baseline from existing formal pipeline data.

BY = BH with alpha replaced by alpha / sum(1/k for k=1..K).
Reads the same candidate_table + uses the same 3-split logic as the formal pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ovd_hallucination_fdr.evalues import conformal_pvalues  # noqa: E402


def harmonic_sum(K: int) -> float:
    return sum(1.0 / k for k in range(1, K + 1))


def by_procedure(pvalues: np.ndarray, alpha: float) -> list[int]:
    """Benjamini-Yekutieli at level alpha under arbitrary dependence."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = int(p.size)
    if m == 0:
        return []
    H_m = harmonic_sum(m)
    adjusted_alpha = alpha / H_m
    # Standard BH at adjusted level
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = adjusted_alpha * np.arange(1, m + 1) / m
    below = np.where(sorted_p <= thresholds)[0]
    if below.size == 0:
        return []
    cutoff = float(sorted_p[below[-1]])
    return [int(idx) for idx, value in enumerate(p) if float(value) <= cutoff]


def run_by_on_config(candidate_csv: str, n_seeds: int, alphas: list[float],
                     split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)):
    df = pd.read_csv(candidate_csv)
    results = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        families = df["family_id"].unique()
        rng.shuffle(families)

        n = len(families)
        n_fit = int(n * split_ratios[0])
        n_cal = int(n * split_ratios[1])
        fit_families = set(families[:n_fit])
        cal_families = set(families[n_fit:n_fit + n_cal])
        test_families = set(families[n_fit + n_cal:])

        # Calibration null scores
        cal_df = df[df["family_id"].isin(cal_families)]
        cal_null_scores = cal_df[cal_df["is_null"] == True]["score"].to_numpy(dtype=float)  # noqa

        if len(cal_null_scores) == 0:
            continue

        # Test families
        test_df = df[df["family_id"].isin(test_families)]

        for alpha in alphas:
            total_rej = 0
            total_fp = 0
            total_tp = 0
            n_test_fam = 0
            family_fdps = []

            for fam_id, group in test_df.groupby("family_id"):
                n_test_fam += 1
                scores = group["score"].to_numpy(dtype=float)
                is_tp = group["is_tp"].to_numpy(dtype=bool)

                # Conformal p-values
                pvals = conformal_pvalues(scores, cal_null_scores)

                # BY
                rejected = by_procedure(pvals, alpha)

                rej_count = len(rejected)
                if rej_count > 0:
                    fp = sum(1 for idx in rejected if not is_tp[idx])
                    tp = sum(1 for idx in rejected if is_tp[idx])
                    fdp = fp / rej_count
                else:
                    fp = 0
                    tp = 0
                    fdp = 0.0

                total_rej += rej_count
                total_fp += fp
                total_tp += tp
                family_fdps.append(fdp)

            pooled_fdp = total_fp / max(total_fp + total_tp, 1)
            mean_family_fdp = np.mean(family_fdps) if family_fdps else 0.0
            total_tp_in_test = int(test_df["is_tp"].sum())
            recall = total_tp / max(total_tp_in_test, 1)

            results.append({
                "seed": seed,
                "alpha": alpha,
                "method": "pvalue_by",
                "total_rejections": total_rej,
                "fp": total_fp,
                "tp": total_tp,
                "pooled_fdp": pooled_fdp,
                "mean_family_fdp": mean_family_fdp,
                "recall": recall,
                "n_test_families": n_test_fam,
                "n_cal_null": len(cal_null_scores),
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=str, required=True)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--alphas", type=str, default="0.05,0.1,0.15,0.2")
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    results = run_by_on_config(args.candidate_csv, args.n_seeds, alphas)
    results.to_csv(args.out_csv, index=False)

    # Print summary at alpha=0.10
    sub = results[results["alpha"] == 0.10]
    if len(sub) > 0:
        print(f"\n=== BY baseline at alpha=0.10, {len(sub)} seeds ===")
        print(f"Rejections: {sub['total_rejections'].mean():.1f} +/- {sub['total_rejections'].std():.1f}")
        print(f"Pooled FDP: {sub['pooled_fdp'].mean()*100:.2f} +/- {sub['pooled_fdp'].std()*100:.2f}%")
        print(f"Recall:     {sub['recall'].mean()*100:.2f} +/- {sub['recall'].std()*100:.2f}%")
        print(f"FP:         {sub['fp'].mean():.1f} +/- {sub['fp'].std():.1f}")


if __name__ == "__main__":
    main()
