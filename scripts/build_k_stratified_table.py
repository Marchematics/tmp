#!/usr/bin/env python3
"""Large-K explicit stratified table (E7).

For each config × α × K-bin, report:
  family_share, mean_K, num_test_families, total_rejections,
  per_family_fdp, pooled_fdp, recall, pass_rate (FDP <= α)

K bins: {1, 2-5, 6-10, 11-20, >20}

Reads pre-computed `seed_X/test_candidates_with_evalues.csv` from
`<config>_20seed_formal_hist/` directories. Re-runs e-BH from `betting_evalue`
column for each (seed, alpha) and re-bins by family_size.
"""
from __future__ import annotations

import argparse
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

K_BINS = [
    ("K=1",    lambda k: k == 1),
    ("2-5",    lambda k: 2 <= k <= 5),
    ("6-10",   lambda k: 6 <= k <= 10),
    ("11-20",  lambda k: 11 <= k <= 20),
    ("K>20",   lambda k: k > 20),
]


def within_family_ebh(group: pd.DataFrame, alpha: float) -> pd.Series:
    """Self-normalized within-family e-BH. The betting_evalue column already
    includes the (m+1) factor, so the within-family threshold is K/(alpha*r)."""
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


def evaluate_seed(df_seed: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Run within-family e-BH on one seed, return per-family stats."""
    rej_per_family = (
        df_seed.groupby("family_id", group_keys=False)
        .apply(within_family_ebh, alpha=alpha)
    )
    df = df_seed.assign(rejected=rej_per_family.reindex(df_seed.index).fillna(False))
    fam = (
        df.groupby("family_id")
        .agg(
            K=("family_id", "size"),
            n_rej=("rejected", "sum"),
            n_tp_rej=("is_tp", lambda s: int((s & df.loc[s.index, "rejected"]).sum())),
            n_fp_rej=("is_tp", lambda s: int((~s & df.loc[s.index, "rejected"]).sum())),
            n_tp_total=("is_tp", "sum"),
        )
        .reset_index()
    )
    return fam


def bin_label(k: int) -> str:
    for label, predicate in K_BINS:
        if predicate(k):
            return label
    return "?"


def aggregate_config(config_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in range(N_SEEDS):
        cand_path = config_dir / f"seed_{seed}" / "test_candidates_with_evalues.csv"
        if not cand_path.exists():
            print(f"  [skip] missing seed {seed}")
            continue
        df = pd.read_csv(
            cand_path,
            usecols=["family_id", "is_tp", "betting_evalue"],
        )
        for alpha in ALPHAS:
            fam = evaluate_seed(df, alpha)
            fam["bin"] = fam["K"].apply(bin_label)
            for bin_label_str, _ in K_BINS:
                sub = fam[fam["bin"] == bin_label_str]
                if len(sub) == 0:
                    continue
                tot_rej = int(sub["n_rej"].sum())
                tot_fp = int(sub["n_fp_rej"].sum())
                tot_tp = int(sub["n_tp_rej"].sum())
                tot_tp_avail = int(sub["n_tp_total"].sum())
                # per-family FDP (mean across families with rejections)
                fam_with_rej = sub[sub["n_rej"] > 0]
                if len(fam_with_rej) > 0:
                    pf_fdp = (
                        fam_with_rej["n_fp_rej"] / fam_with_rej["n_rej"]
                    ).mean()
                else:
                    pf_fdp = 0.0
                rows.append({
                    "seed": seed,
                    "alpha": alpha,
                    "bin": bin_label_str,
                    "n_families": len(sub),
                    "mean_K": float(sub["K"].mean()),
                    "total_rejections": tot_rej,
                    "fdp_pooled": tot_fp / max(tot_rej, 1),
                    "per_family_fdp": pf_fdp,
                    "recall": tot_tp / max(tot_tp_avail, 1),
                })
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    # Aggregate over seeds
    agg = (
        raw.groupby(["alpha", "bin"])
        .agg(
            n_families_mean=("n_families", "mean"),
            mean_K=("mean_K", "mean"),
            total_rejections_mean=("total_rejections", "mean"),
            fdp_pooled_mean=("fdp_pooled", "mean"),
            fdp_pooled_std=("fdp_pooled", "std"),
            per_family_fdp_mean=("per_family_fdp", "mean"),
            recall_mean=("recall", "mean"),
            pass_rate=("fdp_pooled", lambda s: float((s <= s.name[0]).mean())
                                   if isinstance(s.name, tuple) else 0.0),
        )
        .reset_index()
    )
    # Pass rate: fraction of seeds where pooled_fdp <= alpha
    pass_rate = (
        raw.assign(passed=raw["fdp_pooled"] <= raw["alpha"])
        .groupby(["alpha", "bin"])["passed"].mean().reset_index()
        .rename(columns={"passed": "pass_rate"})
    )
    agg = agg.drop(columns=["pass_rate"]).merge(pass_rate, on=["alpha", "bin"])
    # Family share per alpha
    totals = agg.groupby("alpha")["n_families_mean"].transform("sum")
    agg["family_share"] = agg["n_families_mean"] / totals
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(OUTPUTS / "k_stratified_table.csv"),
    )
    args = ap.parse_args()

    all_rows = []
    for dataset, detector, dirname in CONFIGS:
        config_dir = OUTPUTS / dirname
        if not config_dir.exists():
            print(f"[skip] {dataset}/{detector}: dir missing")
            continue
        print(f"\n{dataset} / {detector}")
        agg = aggregate_config(config_dir)
        if agg.empty:
            continue
        agg.insert(0, "dataset", dataset)
        agg.insert(1, "detector", detector)
        all_rows.append(agg)
        # quick print: alpha=0.10 row
        sub = agg[agg["alpha"] == 0.10]
        for _, r in sub.iterrows():
            print(
                f"  α=0.10  {r['bin']:6s}  share={r['family_share']:.2%}  "
                f"K̄={r['mean_K']:5.1f}  rej={r['total_rejections_mean']:6.1f}  "
                f"pooled_FDP={r['fdp_pooled_mean']:.4f}  "
                f"recall={r['recall_mean']:.3f}  pass={r['pass_rate']:.2f}"
            )
    if not all_rows:
        print("No data!")
        return
    out_df = pd.concat(all_rows, ignore_index=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
