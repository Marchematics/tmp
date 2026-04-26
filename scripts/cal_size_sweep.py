#!/usr/bin/env python3
"""Calibration pool size sweep (E6).

For each m ∈ {50, 100, 200, 500, 1000, 2000, all}, subsample the calibration
null pool to m candidates and run within-family e-BH on test families.
Reports LOO mean, pooled FDP, recall, and no-rejection rate as a function
of m. Demonstrates sample efficiency of the wrapper.

Run on COCO/GDino (representative single config); easy to extend to all 6.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAND_PATH = (
    PROJECT_ROOT / "outputs"
    / "coco_groundingdino_gonogo_1000_mix_analysis"
    / "candidate_table.csv"
)
OUT_CSV = PROJECT_ROOT / "outputs" / "cal_size_sweep.csv"

M_VALUES = [50, 100, 200, 500, 1000, 2000, None]  # None = all
ALPHA = 0.10
N_SEEDS = 20
N_BINS = 80


def split_families(family_ids, seed):
    rng = np.random.default_rng(seed)
    fams = family_ids.copy()
    rng.shuffle(fams)
    n = len(fams)
    n_fit = int(round(0.6 * n))
    n_cal = int(round(0.2 * n))
    return (set(fams[:n_fit]),
            set(fams[n_fit:n_fit+n_cal]),
            set(fams[n_fit+n_cal:]))


def fit_phi(fit_null, fit_tp):
    edges = np.linspace(min(fit_null.min(), fit_tp.min() if len(fit_tp) else 0),
                        max(fit_null.max(), fit_tp.max() if len(fit_tp) else 1) + 1e-9,
                        N_BINS + 1)
    nc, _ = np.histogram(fit_null, bins=edges)
    tc, _ = np.histogram(fit_tp, bins=edges)
    p0 = nc.astype(float) / max(nc.sum(), 1)
    p1 = tc.astype(float) / max(tc.sum(), 1)
    phi = np.where(p0 > 0, p1 / np.maximum(p0, 1e-12), 0.0)
    expected = float((p0 * phi).sum())
    if expected > 0:
        phi /= expected
    return edges, phi


def apply_phi(s, edges, phi):
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(phi) - 1)
    return phi[idx]


def within_family_ebh(e, alpha):
    K = len(e)
    if K == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-e, kind="stable")
    se = e[order]
    ranks = np.arange(1, K + 1)
    eligible = se >= K / (alpha * ranks)
    r = int(np.where(eligible)[0].max() + 1) if eligible.any() else 0
    mask = np.zeros(K, dtype=bool)
    mask[order[:r]] = True
    return mask


def run_seed(df, seed: int, m: int | None) -> dict:
    fams = df["family_id"].drop_duplicates().to_numpy()
    fit_set, cal_set, test_set = split_families(fams, seed)
    fit = df[df["family_id"].isin(fit_set)]
    cal = df[df["family_id"].isin(cal_set)]
    test = df[df["family_id"].isin(test_set)]

    edges, phi = fit_phi(
        fit[fit["is_null"]]["score"].to_numpy(),
        fit[fit["is_tp"]]["score"].to_numpy(),
    )

    cal_null_scores = cal[cal["is_null"]]["score"].to_numpy()
    if m is not None and m < len(cal_null_scores):
        rng = np.random.default_rng(seed * 1000 + m)
        idx = rng.choice(len(cal_null_scores), size=m, replace=False)
        cal_null_scores = cal_null_scores[idx]

    cal_phi = apply_phi(cal_null_scores, edges, phi)
    sum_phi_cal = float(cal_phi.sum())
    m_cal = len(cal_phi)

    test_scores = test["score"].to_numpy()
    test_phi = apply_phi(test_scores, edges, phi)
    denom = test_phi + sum_phi_cal
    e = np.zeros_like(test_phi)
    valid = denom > 0
    e[valid] = (m_cal + 1) * test_phi[valid] / denom[valid]

    # LOO mean over null test candidates
    is_null = test["is_null"].to_numpy()
    loo_mean = float(e[is_null].mean()) if is_null.any() else float("nan")

    # Within-family e-BH at α
    fam_ids = test["family_id"].to_numpy()
    is_tp = test["is_tp"].to_numpy()
    unique_fams, inverse = np.unique(fam_ids, return_inverse=True)
    rejected = np.zeros(len(test), dtype=bool)
    fam_has_rej = 0
    for i in range(len(unique_fams)):
        idx = np.where(inverse == i)[0]
        mask = within_family_ebh(e[idx], ALPHA)
        if mask.any():
            rejected[idx[mask]] = True
            fam_has_rej += 1

    n_rej = int(rejected.sum())
    n_fp = int((rejected & ~is_tp).sum())
    n_tp = int((rejected & is_tp).sum())
    n_tp_total = int(is_tp.sum())
    no_rej_rate = 1 - fam_has_rej / max(len(unique_fams), 1)

    return {
        "rejections": n_rej,
        "fdp_pooled": n_fp / max(n_rej, 1),
        "recall": n_tp / max(n_tp_total, 1),
        "loo_mean": loo_mean,
        "no_rejection_rate": no_rej_rate,
        "m_cal_actual": m_cal,
    }


def main() -> None:
    print(f"Loading {CAND_PATH}")
    df = pd.read_csv(CAND_PATH)

    rows = []
    for m in M_VALUES:
        m_label = "all" if m is None else str(m)
        seed_results = [run_seed(df, s, m) for s in range(N_SEEDS)]
        sdf = pd.DataFrame(seed_results)
        agg = {
            "m_target": m_label,
            "m_actual_mean": float(sdf["m_cal_actual"].mean()),
            "rejections_mean": float(sdf["rejections"].mean()),
            "fdp_pooled_mean": float(sdf["fdp_pooled"].mean()),
            "fdp_pooled_std": float(sdf["fdp_pooled"].std(ddof=1)),
            "recall_mean": float(sdf["recall"].mean()),
            "recall_std": float(sdf["recall"].std(ddof=1)),
            "loo_mean": float(sdf["loo_mean"].mean()),
            "loo_std": float(sdf["loo_mean"].std(ddof=1)),
            "no_rejection_rate_mean": float(sdf["no_rejection_rate"].mean()),
            "pass_rate": float((sdf["fdp_pooled"] <= ALPHA).mean()),
        }
        rows.append(agg)
        print(
            f"m={m_label:5s}  rej={agg['rejections_mean']:6.1f}  "
            f"FDP={agg['fdp_pooled_mean']:.4f}±{agg['fdp_pooled_std']:.4f}  "
            f"recall={agg['recall_mean']:.3f}±{agg['recall_std']:.3f}  "
            f"LOO={agg['loo_mean']:.3f}±{agg['loo_std']:.3f}  "
            f"no-rej={agg['no_rejection_rate_mean']:.2%}  "
            f"pass={agg['pass_rate']:.2f}"
        )

    fields = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
