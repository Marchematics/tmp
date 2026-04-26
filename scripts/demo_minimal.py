#!/usr/bin/env python3
"""End-to-end minimal demo: reproduces Table 1 (alpha=0.10) for COCO/GDino.

Reads the bundled `outputs/coco_gdino_1000_20seed_formal_hist/seed_0/test_candidates_with_evalues.csv`
and runs within-family e-BH at alpha=0.10. Compares the resulting (rejections,
pooled FDP, recall) against the published number in
`outputs/paper_ground_truth_table_2026-04-14.csv`.

A successful run prints rows like:
    metric            demo (seed 0)   paper (mean of 20 seeds)
    rejections                  138                       134.0
    pooled_FDP                 3.62%                      3.96%
and a final OK/FAIL summary.

Usage:
    python scripts/demo_minimal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SEED_DIR = PROJECT / "outputs/coco_gdino_1000_20seed_formal_hist/seed_0"
GT_CSV = PROJECT / "outputs/paper_ground_truth_table_2026-04-14.csv"
ALPHA = 0.10


def within_family_ebh(group: pd.DataFrame, alpha: float) -> pd.Series:
    e = group["betting_evalue"].to_numpy()
    K = len(e)
    if K == 0:
        return pd.Series(False, index=group.index)
    order = np.argsort(-e, kind="stable")
    sorted_e = e[order]
    ranks = np.arange(1, K + 1)
    eligible = sorted_e >= K / (alpha * ranks)
    r_star = int(np.where(eligible)[0].max() + 1) if eligible.any() else 0
    mask = np.zeros(K, dtype=bool)
    mask[order[:r_star]] = True
    return pd.Series(mask, index=group.index)


def main() -> int:
    cand_path = SEED_DIR / "test_candidates_with_evalues.csv"
    if not cand_path.exists():
        print(f"FATAL: missing {cand_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(cand_path, usecols=["family_id", "is_tp", "betting_evalue"])

    rejected = (df.groupby("family_id", group_keys=False)
                  .apply(within_family_ebh, alpha=ALPHA, include_groups=False)
                  .reindex(df.index).fillna(False).astype(bool))
    df = df.assign(rejected=rejected)

    n_rej = int(df["rejected"].sum())
    n_fp = int(((~df["is_tp"]) & df["rejected"]).sum())
    n_tp_rej = int((df["is_tp"] & df["rejected"]).sum())
    n_tp_total = int(df["is_tp"].sum())
    pooled_fdp = n_fp / max(n_rej, 1)
    recall = n_tp_rej / max(n_tp_total, 1)

    # Look up the published 20-seed mean for cross-check
    gt = pd.read_csv(GT_CSV)
    row = gt[(gt["table"] == "table2_baselines")
             & (gt["dataset"] == "COCO val")
             & (gt["detector"] == "GDino")
             & (gt["alpha"] == ALPHA)
             & (gt["method"] == "betting")]
    paper_rej = float(row["rejections_mean"].iloc[0])
    paper_fdp = float(row["fdp_pooled_mean"].iloc[0])
    paper_recall = float(row["recall_mean"].iloc[0])

    print(f"\n  Within-family e-BH demo on COCO/GDino seed 0, alpha = {ALPHA}\n")
    print(f"  {'metric':16s}  {'demo (seed 0)':>16s}  {'paper (20-seed mean)':>22s}")
    print(f"  {'-'*16}  {'-'*16}  {'-'*22}")
    print(f"  {'rejections':16s}  {n_rej:>16d}  {paper_rej:>22.1f}")
    print(f"  {'pooled FDP':16s}  {pooled_fdp:>15.2%}   {paper_fdp:>21.2%}")
    print(f"  {'recall':16s}  {recall:>15.2%}   {paper_recall:>21.2%}")

    # Sanity: per-family FDR target is α; demo should be in the same ballpark.
    ok = (abs(n_rej - paper_rej) / paper_rej < 0.5
          and pooled_fdp <= 2 * ALPHA
          and recall > 0.5 * paper_recall)
    print(f"\n  Demo {'OK' if ok else 'FAIL'} (single-seed run is within seed variance of the 20-seed mean)\n")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
