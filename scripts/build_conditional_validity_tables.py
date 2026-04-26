#!/usr/bin/env python3
"""Conditional validity diagnostic tables (A1-A3, Phase 1 of Round 2 sprint).

Produces three stratified tables sharing the same underlying e-BH evaluation:

  A1) score_quantile_table.csv  — bin candidates by their own score (5 quantile bins
                                    defined on fit-split null scores per config).
  A2) object_size_table.csv     — bin candidates by box area (small/medium/large,
                                    COCO area thresholds: ≤32², 32²-96², >96²).
  A3) per_category_table.csv    — group families by category (COCO/VOC GDino only).

For each (config, alpha, bin) row, report:
  n_candidates, n_rejections, pooled_fdp, recall, loo_mean_e (nulls only),
  pass_rate (fraction of 20 seeds with pooled_fdp <= alpha), and where well-defined,
  per_family_fdp.

Reads precomputed seed_X/test_candidates_with_evalues.csv and runs within-family
e-BH (same logic as build_k_stratified_table.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

CONFIGS = [
    ("COCO val", "GDino",      "coco_gdino_1000_20seed_formal_hist"),
    ("COCO val", "OWL-ViT",    "coco_owlvit_1000_20seed_formal_hist"),
    ("COCO val", "YOLO-World", "coco_yoloworld_val_1000_20seed_formal_hist"),
    ("VOC 2012", "GDino",      "voc_gdino_1000_20seed_formal_hist"),
    ("VOC 2012", "OWL-ViT",    "voc_owlvit_1000_20seed_formal_hist"),
    ("VOC 2012", "YOLO-World", "voc_yoloworld_1000_20seed_formal_hist"),
]
ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 20

# COCO standard area thresholds (https://cocodataset.org/#detection-eval)
SIZE_BINS = [
    ("small",  lambda area: area <= 32 ** 2),
    ("medium", lambda area: (area > 32 ** 2) & (area <= 96 ** 2)),
    ("large",  lambda area: area > 96 ** 2),
]

# Score quantile bins: 5 equal-frequency bins based on test-set null scores per config.
# We define them as labels; actual edges computed per-config.
SCORE_BINS = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]

USECOLS = [
    "family_id", "category_id", "category_name", "score",
    "x1", "y1", "x2", "y2", "is_tp", "is_null", "betting_evalue",
]


def within_family_ebh(group: pd.DataFrame, alpha: float) -> pd.Series:
    e = group["betting_evalue"].to_numpy()
    K = len(e)
    if K == 0:
        return pd.Series(False, index=group.index)
    order = np.argsort(-e, kind="stable")
    sorted_e = e[order]
    ranks = np.arange(1, K + 1)
    threshold = K / (alpha * ranks)
    eligible = sorted_e >= threshold
    r_star = int(np.where(eligible)[0].max() + 1) if eligible.any() else 0
    mask = np.zeros(K, dtype=bool)
    mask[order[:r_star]] = True
    return pd.Series(mask, index=group.index)


def annotate_rejections(df_seed: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rej = (
        df_seed.groupby("family_id", group_keys=False)
        .apply(within_family_ebh, alpha=alpha)
    )
    return df_seed.assign(rejected=rej.reindex(df_seed.index).fillna(False).astype(bool))


def stratum_metrics(df_stratum: pd.DataFrame) -> dict:
    n_cand = len(df_stratum)
    n_rej = int(df_stratum["rejected"].sum())
    n_fp = int(((~df_stratum["is_tp"]) & df_stratum["rejected"]).sum())
    n_tp_rej = int((df_stratum["is_tp"] & df_stratum["rejected"]).sum())
    n_tp_total = int(df_stratum["is_tp"].sum())
    is_null = df_stratum["is_null"].to_numpy()
    e = df_stratum["betting_evalue"].to_numpy()
    loo_mean = float(e[is_null].mean()) if is_null.any() else float("nan")
    pooled_fdp = n_fp / max(n_rej, 1)
    recall = n_tp_rej / max(n_tp_total, 1)
    return {
        "n_candidates": n_cand,
        "n_rejections": n_rej,
        "pooled_fdp": pooled_fdp,
        "recall": recall,
        "loo_mean": loo_mean,
        "n_tp_total": n_tp_total,
    }


def per_family_fdp(df_stratum: pd.DataFrame) -> float:
    """Per-family FDP averaged over families with at least one rejection.
    Only valid when stratum cleanly partitions families (e.g., per-category)."""
    fam_with_rej = (
        df_stratum.groupby("family_id")["rejected"].sum() > 0
    )
    fams_rej = fam_with_rej[fam_with_rej].index
    if len(fams_rej) == 0:
        return 0.0
    sub = df_stratum[df_stratum["family_id"].isin(fams_rej)]
    return float(
        sub.groupby("family_id")
        .apply(lambda g: ((~g["is_tp"]) & g["rejected"]).sum() / max(g["rejected"].sum(), 1))
        .mean()
    )


def aggregate_seeds(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    metric_cols = [c for c in raw.columns if c not in ("seed", "alpha", "bin", "n_tp_total")]
    agg = (
        raw.groupby(["alpha", "bin"])
        .agg({c: "mean" for c in metric_cols})
        .reset_index()
    )
    pass_rate = (
        raw.assign(passed=raw["pooled_fdp"] <= raw["alpha"])
        .groupby(["alpha", "bin"])["passed"].mean().reset_index()
        .rename(columns={"passed": "pass_rate"})
    )
    agg = agg.merge(pass_rate, on=["alpha", "bin"])
    return agg


def process_config_score_quantile(config_dir: Path, dataset: str, detector: str) -> pd.DataFrame:
    """A1: score quantile bins. Edges defined on each seed's own test-null scores."""
    rows = []
    for seed in range(N_SEEDS):
        cand_path = config_dir / f"seed_{seed}" / "test_candidates_with_evalues.csv"
        if not cand_path.exists():
            continue
        df = pd.read_csv(cand_path, usecols=USECOLS)
        null_scores = df.loc[df["is_null"], "score"].to_numpy()
        if len(null_scores) < 5:
            continue
        edges = np.quantile(null_scores, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        edges[0] = -np.inf
        edges[-1] = np.inf
        df["score_bin"] = pd.cut(df["score"], bins=edges, labels=SCORE_BINS, include_lowest=True)
        for alpha in ALPHAS:
            df_a = annotate_rejections(df, alpha)
            for bin_label in SCORE_BINS:
                sub = df_a[df_a["score_bin"] == bin_label]
                if len(sub) == 0:
                    continue
                m = stratum_metrics(sub)
                rows.append({"seed": seed, "alpha": alpha, "bin": bin_label, **m})
    agg = aggregate_seeds(rows)
    if not agg.empty:
        agg.insert(0, "dataset", dataset)
        agg.insert(1, "detector", detector)
    return agg


def process_config_object_size(config_dir: Path, dataset: str, detector: str) -> pd.DataFrame:
    """A2: COCO-standard size bins (small / medium / large)."""
    rows = []
    for seed in range(N_SEEDS):
        cand_path = config_dir / f"seed_{seed}" / "test_candidates_with_evalues.csv"
        if not cand_path.exists():
            continue
        df = pd.read_csv(cand_path, usecols=USECOLS)
        df["box_area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
        df["size_bin"] = "?"
        for label, predicate in SIZE_BINS:
            df.loc[predicate(df["box_area"]), "size_bin"] = label
        for alpha in ALPHAS:
            df_a = annotate_rejections(df, alpha)
            for bin_label, _ in SIZE_BINS:
                sub = df_a[df_a["size_bin"] == bin_label]
                if len(sub) == 0:
                    continue
                m = stratum_metrics(sub)
                rows.append({"seed": seed, "alpha": alpha, "bin": bin_label, **m})
    agg = aggregate_seeds(rows)
    if not agg.empty:
        agg.insert(0, "dataset", dataset)
        agg.insert(1, "detector", detector)
    return agg


def process_config_per_category(config_dir: Path, dataset: str, detector: str) -> pd.DataFrame:
    """A3: per-category. Family-clean (each family is single category)."""
    rows = []
    for seed in range(N_SEEDS):
        cand_path = config_dir / f"seed_{seed}" / "test_candidates_with_evalues.csv"
        if not cand_path.exists():
            continue
        df = pd.read_csv(cand_path, usecols=USECOLS)
        for alpha in ALPHAS:
            df_a = annotate_rejections(df, alpha)
            for cat_name, sub in df_a.groupby("category_name"):
                if len(sub) == 0:
                    continue
                m = stratum_metrics(sub)
                m["per_family_fdp"] = per_family_fdp(sub)
                rows.append({"seed": seed, "alpha": alpha, "bin": str(cat_name), **m})
    agg = aggregate_seeds(rows)
    if not agg.empty:
        agg.insert(0, "dataset", dataset)
        agg.insert(1, "detector", detector)
    return agg


def main() -> None:
    out_score, out_size, out_cat = [], [], []
    for dataset, detector, dirname in CONFIGS:
        config_dir = OUTPUTS / dirname
        if not config_dir.exists():
            print(f"[skip] {dataset}/{detector}: dir missing")
            continue
        print(f"\n=== {dataset} / {detector} ===")
        # A1
        sq = process_config_score_quantile(config_dir, dataset, detector)
        if not sq.empty:
            out_score.append(sq)
            sub = sq[sq["alpha"] == 0.10]
            print(f"  [score_quantile] {len(sub)} bins at α=0.10")
            for _, r in sub.iterrows():
                print(
                    f"    {r['bin']:10s}  n={r['n_candidates']:5.0f}  rej={r['n_rejections']:5.1f}"
                    f"  pooled_fdp={r['pooled_fdp']:.4f}  recall={r['recall']:.3f}"
                    f"  LOO={r['loo_mean']:.3f}  pass={r['pass_rate']:.2f}"
                )
        # A2
        os_ = process_config_object_size(config_dir, dataset, detector)
        if not os_.empty:
            out_size.append(os_)
            sub = os_[os_["alpha"] == 0.10]
            print(f"  [object_size] {len(sub)} bins at α=0.10")
            for _, r in sub.iterrows():
                print(
                    f"    {r['bin']:8s}  n={r['n_candidates']:5.0f}  rej={r['n_rejections']:5.1f}"
                    f"  pooled_fdp={r['pooled_fdp']:.4f}  recall={r['recall']:.3f}"
                    f"  LOO={r['loo_mean']:.3f}  pass={r['pass_rate']:.2f}"
                )
        # A3 — only COCO/GDino + VOC/GDino (clean configs)
        if detector == "GDino":
            pc = process_config_per_category(config_dir, dataset, detector)
            if not pc.empty:
                out_cat.append(pc)
                sub = pc[pc["alpha"] == 0.10]
                # Print summary stats
                if len(sub) > 0:
                    print(f"  [per_category] {len(sub)} categories at α=0.10")
                    print(
                        f"    pooled_fdp range: [{sub['pooled_fdp'].min():.4f}, "
                        f"{sub['pooled_fdp'].max():.4f}]; mean={sub['pooled_fdp'].mean():.4f}"
                    )
                    n_fail = int((sub["pooled_fdp"] > 0.10).sum())
                    print(f"    {n_fail}/{len(sub)} categories with pooled_FDP > 0.10")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if out_score:
        pd.concat(out_score, ignore_index=True).to_csv(OUTPUTS / "score_quantile_table.csv", index=False)
        print(f"\nWrote {OUTPUTS / 'score_quantile_table.csv'}")
    if out_size:
        pd.concat(out_size, ignore_index=True).to_csv(OUTPUTS / "object_size_table.csv", index=False)
        print(f"Wrote {OUTPUTS / 'object_size_table.csv'}")
    if out_cat:
        pd.concat(out_cat, ignore_index=True).to_csv(OUTPUTS / "per_category_table.csv", index=False)
        print(f"Wrote {OUTPUTS / 'per_category_table.csv'}")


if __name__ == "__main__":
    main()
