#!/usr/bin/env python3
"""Adaptive p-value FDR baselines (E4): Storey-BH and BKY two-stage.

Both are within-family adaptive procedures applied to existing conformal
p-values (column ``conformal_pvalue`` in seed_X/test_candidates_with_evalues.csv).

For each (config, seed, alpha):
  1. Within each family, apply the adaptive BH variant.
  2. Aggregate rejections, compute pooled FDP and recall.
Output: rows ready to append to paper_ground_truth_table_2026-04-14.csv
        with method ∈ {storey_bh, bky_2stage}.
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
    ("COCO val", "GDino",      "coco_gdino_1000_20seed_formal_hist"),
    ("COCO val", "OWL-ViT",    "coco_owlvit_1000_20seed_formal_hist"),
    ("COCO val", "YOLO-World", "coco_yoloworld_val_1000_20seed_formal_hist"),
    ("VOC 2012", "GDino",      "voc_gdino_1000_20seed_formal_hist"),
    ("VOC 2012", "OWL-ViT",    "voc_owlvit_1000_20seed_formal_hist"),
    ("VOC 2012", "YOLO-World", "voc_yoloworld_1000_20seed_formal_hist"),
]
ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 20
LAMBDA = 0.5  # Storey null-proportion threshold


def bh_reject(p: np.ndarray, alpha: float) -> np.ndarray:
    """Standard Benjamini–Hochberg. Returns boolean mask aligned with input."""
    K = len(p)
    if K == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, K + 1)
    eligible = sorted_p <= alpha * ranks / K
    if not eligible.any():
        r = 0
    else:
        r = int(np.where(eligible)[0].max() + 1)
    mask = np.zeros(K, dtype=bool)
    mask[order[:r]] = True
    return mask


def storey_bh(p: np.ndarray, alpha: float, lam: float = LAMBDA) -> np.ndarray:
    """Storey adaptive BH: estimate pi0 then run BH at alpha/pi0_hat."""
    K = len(p)
    if K == 0:
        return np.zeros(0, dtype=bool)
    pi0 = (np.sum(p > lam) + 1) / ((1 - lam) * K)
    pi0 = min(pi0, 1.0)
    if pi0 <= 0:
        return bh_reject(p, alpha)
    return bh_reject(p, alpha / pi0)


def bky_two_stage(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini–Krieger–Yekutieli two-stage adaptive linear step-up."""
    K = len(p)
    if K == 0:
        return np.zeros(0, dtype=bool)
    alpha1 = alpha / (1 + alpha)
    R1 = int(bh_reject(p, alpha1).sum())
    m0_hat = max(K - R1, 1)
    return bh_reject(p, alpha * K / m0_hat)


METHODS = {
    "storey_bh": storey_bh,
    "bky_2stage": bky_two_stage,
}


def evaluate_family(group: pd.DataFrame, alpha: float, fn) -> pd.Series:
    p = group["conformal_pvalue"].to_numpy()
    return pd.Series(fn(p, alpha), index=group.index)


def evaluate_seed(df_seed: pd.DataFrame, alpha: float, fn) -> dict:
    rej = (
        df_seed.groupby("family_id", group_keys=False)
        .apply(evaluate_family, alpha=alpha, fn=fn)
    )
    rejected = rej.reindex(df_seed.index).fillna(False).to_numpy()
    is_tp = df_seed["is_tp"].to_numpy()
    n_rej = int(rejected.sum())
    n_fp = int((rejected & ~is_tp).sum())
    n_tp_rej = int((rejected & is_tp).sum())
    n_tp_total = int(is_tp.sum())
    # per-family fdp (mean across families with rejections)
    df = df_seed.assign(rejected=rejected)
    fam = df.groupby("family_id").agg(
        n_rej=("rejected", "sum"),
        n_fp=("rejected", lambda s: int((s & ~df.loc[s.index, "is_tp"]).sum())),
    )
    with_rej = fam[fam["n_rej"] > 0]
    pf_fdp = (
        (with_rej["n_fp"] / with_rej["n_rej"]).mean()
        if len(with_rej) > 0
        else 0.0
    )
    return {
        "rejections": n_rej,
        "fdp_pooled": n_fp / max(n_rej, 1),
        "fdp_family": float(pf_fdp),
        "recall": n_tp_rej / max(n_tp_total, 1),
    }


def aggregate_config(config_dir: Path, dataset: str, detector: str) -> list[dict]:
    rows_per_method: dict[str, list[dict]] = {m: [] for m in METHODS}
    for seed in range(N_SEEDS):
        path = config_dir / f"seed_{seed}" / "test_candidates_with_evalues.csv"
        if not path.exists():
            continue
        df = pd.read_csv(
            path,
            usecols=["family_id", "is_tp", "conformal_pvalue"],
        )
        for alpha in ALPHAS:
            for name, fn in METHODS.items():
                stats = evaluate_seed(df, alpha, fn)
                rows_per_method[name].append({"seed": seed, "alpha": alpha, **stats})

    out_rows = []
    for method_name, raw in rows_per_method.items():
        if not raw:
            continue
        rdf = pd.DataFrame(raw)
        for alpha, sub in rdf.groupby("alpha"):
            out_rows.append({
                "table": "table2_baselines",
                "dataset": dataset,
                "detector": detector,
                "alpha": float(alpha),
                "method": method_name,
                "fdp_pooled_mean": float(sub["fdp_pooled"].mean()),
                "fdp_pooled_std": float(sub["fdp_pooled"].std(ddof=1)),
                "fdp_family_mean": float(sub["fdp_family"].mean()),
                "recall_mean": float(sub["recall"].mean()),
                "rejections_mean": float(sub["rejections"].mean()),
                "phi_max_median": "",
                "loo_mean": "",
                "n_seeds": N_SEEDS,
                "source": f"outputs/{config_dir.name}",
                "pooled_bound_eps": "",
                "pooled_bound_delta": "",
                "pooled_bound_covered": "",
                "pooled_bound_eps_stratified": "",
            })
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUTS / "adaptive_fdr_baselines.csv"))
    args = ap.parse_args()

    all_rows = []
    for dataset, detector, dirname in CONFIGS:
        config_dir = OUTPUTS / dirname
        if not config_dir.exists():
            print(f"[skip] {dataset}/{detector}: dir missing")
            continue
        print(f"\n{dataset} / {detector}")
        rows = aggregate_config(config_dir, dataset, detector)
        # Print α=0.10 summary
        for r in rows:
            if r["alpha"] != 0.10:
                continue
            print(
                f"  {r['method']:12s}  rej={r['rejections_mean']:6.1f}  "
                f"FDP={r['fdp_pooled_mean']:.4f}  "
                f"per-fam={r['fdp_family_mean']:.4f}  "
                f"recall={r['recall_mean']:.3f}"
            )
        all_rows.extend(rows)

    if not all_rows:
        print("No data!")
        return
    fieldnames = list(all_rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
