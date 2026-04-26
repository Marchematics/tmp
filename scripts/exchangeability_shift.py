#!/usr/bin/env python3
"""Exchangeability shift stress test (E8 / shift A).

Setup:
  Calibration null pool has its top-ρ% highest-score candidates removed
  before computing the e-values. This breaks the (cal, test) null
  exchangeability assumption: test null candidates are then drawn from
  a higher-score distribution than the contaminated cal pool.

For each ρ ∈ {0, 5, 10, 20, 30}%, report:
  - LOO mean (the diagnostic — should drift away from 1 under shift)
  - pooled FDP, per-family FDP, recall, pass rate
  - rejections

Run on COCO/GDino at α=0.10 (representative); 20 seeds.
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
OUT_CSV = PROJECT_ROOT / "outputs" / "shift_stress.csv"

RHO_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]
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


def run_seed(df, seed: int, rho: float) -> dict:
    fams = df["family_id"].drop_duplicates().to_numpy()
    fit_set, cal_set, test_set = split_families(fams, seed)
    fit = df[df["family_id"].isin(fit_set)]
    cal = df[df["family_id"].isin(cal_set)]
    test = df[df["family_id"].isin(test_set)]

    edges, phi = fit_phi(
        fit[fit["is_null"]]["score"].to_numpy(),
        fit[fit["is_tp"]]["score"].to_numpy(),
    )

    # Apply shift: drop top-rho fraction of cal nulls (by score)
    cal_null = cal[cal["is_null"]].sort_values("score", ascending=False).reset_index(drop=True)
    n_drop = int(round(rho * len(cal_null)))
    cal_null_kept = cal_null.iloc[n_drop:]
    cal_null_scores = cal_null_kept["score"].to_numpy()

    cal_phi = apply_phi(cal_null_scores, edges, phi)
    sum_phi_cal = float(cal_phi.sum())
    m_cal = len(cal_phi)

    test_scores = test["score"].to_numpy()
    test_phi = apply_phi(test_scores, edges, phi)
    denom = test_phi + sum_phi_cal
    e = np.zeros_like(test_phi)
    valid = denom > 0
    e[valid] = (m_cal + 1) * test_phi[valid] / denom[valid]

    is_null = test["is_null"].to_numpy()
    is_tp = test["is_tp"].to_numpy()
    loo_mean = float(e[is_null].mean()) if is_null.any() else float("nan")

    fam_ids = test["family_id"].to_numpy()
    unique_fams, inverse = np.unique(fam_ids, return_inverse=True)
    rejected = np.zeros(len(test), dtype=bool)
    for i in range(len(unique_fams)):
        idx = np.where(inverse == i)[0]
        m = within_family_ebh(e[idx], ALPHA)
        if m.any():
            rejected[idx[m]] = True

    n_rej = int(rejected.sum())
    n_fp = int((rejected & ~is_tp).sum())
    n_tp = int((rejected & is_tp).sum())
    n_tp_total = int(is_tp.sum())

    if n_rej > 0:
        rej_idx = np.where(rejected)[0]
        fams_rej = fam_ids[rej_idx]
        fp_rej = (~is_tp)[rej_idx].astype(int)
        fdp_fam = float(pd.DataFrame({"f": fams_rej, "fp": fp_rej}).groupby("f")["fp"].mean().mean())
    else:
        fdp_fam = 0.0

    return {
        "rejections": n_rej,
        "fdp_pooled": n_fp / max(n_rej, 1),
        "fdp_family": fdp_fam,
        "recall": n_tp / max(n_tp_total, 1),
        "loo_mean": loo_mean,
        "m_cal_after_shift": m_cal,
    }


def main() -> None:
    print(f"Loading {CAND_PATH}")
    df = pd.read_csv(CAND_PATH)

    rows = []
    for rho in RHO_VALUES:
        seed_results = [run_seed(df, s, rho) for s in range(N_SEEDS)]
        sdf = pd.DataFrame(seed_results)
        agg = {
            "rho": rho,
            "m_cal_mean": float(sdf["m_cal_after_shift"].mean()),
            "rejections_mean": float(sdf["rejections"].mean()),
            "fdp_pooled_mean": float(sdf["fdp_pooled"].mean()),
            "fdp_pooled_std": float(sdf["fdp_pooled"].std(ddof=1)),
            "fdp_family_mean": float(sdf["fdp_family"].mean()),
            "recall_mean": float(sdf["recall"].mean()),
            "loo_mean": float(sdf["loo_mean"].mean()),
            "loo_std": float(sdf["loo_mean"].std(ddof=1)),
            "pass_pooled": float((sdf["fdp_pooled"] <= ALPHA).mean()),
            "pass_family": float((sdf["fdp_family"] <= ALPHA).mean()),
        }
        rows.append(agg)
        print(
            f"ρ={rho:5.2f}  rej={agg['rejections_mean']:6.1f}  "
            f"FDP_pool={agg['fdp_pooled_mean']:.4f}  per-fam={agg['fdp_family_mean']:.4f}  "
            f"recall={agg['recall_mean']:.3f}  "
            f"LOO={agg['loo_mean']:.3f}±{agg['loo_std']:.3f}  "
            f"pass={agg['pass_pooled']:.2f}/{agg['pass_family']:.2f}"
        )

    fields = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
