#!/usr/bin/env python3
"""B1: Fit-split-only adaptive score floor (Phase 2 of Round 2 sprint).

For each (config, seed, alpha), compares three modes on the test split:
  Mode 1 — no floor (raw e-BH).
  Mode 2 — fixed floor s_floor = 0.40 (existing baseline).
  Mode 3 — adaptive floor selected on fit split with TP retention ≥ rho.

The adaptive floor is selected ONLY from fit-split data (no test labels touched).
The floor is a deterministic threshold on score, so filtering test candidates by
`score >= s_floor*` does not invalidate the e-values. Per-family e-BH is applied
to the surviving candidates within each test family.

Outputs `outputs/adaptive_floor_results.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

OUTPUTS = PROJECT_ROOT / "outputs"

CONFIGS = [
    ("COCO val", "GDino",      "coco_groundingdino_gonogo_1000_mix_analysis",
                                "coco_gdino_1000_20seed_formal_hist"),
    ("COCO val", "OWL-ViT",    "coco_owlvit_val_1000_mix_analysis",
                                "coco_owlvit_1000_20seed_formal_hist"),
    ("COCO val", "YOLO-World", "coco_yoloworld_val_1000_mix_analysis",
                                "coco_yoloworld_val_1000_20seed_formal_hist"),
    ("VOC 2012", "GDino",      "voc_1000_gdino_analysis",
                                "voc_gdino_1000_20seed_formal_hist"),
    ("VOC 2012", "OWL-ViT",    "voc_1000_owlvit_analysis",
                                "voc_owlvit_1000_20seed_formal_hist"),
    ("VOC 2012", "YOLO-World", "voc_yoloworld_1000_mix_analysis",
                                "voc_yoloworld_1000_20seed_formal_hist"),
]
ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 20
RHO_VALUES = [0.80, 0.85, 0.90]
SPLIT_RATIOS = (0.6, 0.2, 0.2)
FLOOR_GRID = np.arange(0.0, 0.85, 0.05)


def split_families_repro(df: pd.DataFrame, seed: int) -> pd.Series:
    """Reproduces split_families() from run_formal_betting_pipeline.py."""
    rng = np.random.default_rng(seed)
    families = df["family_id"].drop_duplicates().to_numpy()
    rng.shuffle(families)
    n_total = len(families)
    n_fit = int(round(SPLIT_RATIOS[0] * n_total))
    n_cal = int(round(SPLIT_RATIOS[1] * n_total))
    n_fit = max(1, min(n_fit, n_total - 2))
    n_cal = max(1, min(n_cal, n_total - n_fit - 1))
    fit_set = set(families[:n_fit])
    cal_set = set(families[n_fit:n_fit + n_cal])
    split = np.where(df["family_id"].isin(fit_set), "fit", "test")
    split = np.where(df["family_id"].isin(cal_set), "cal", split)
    return pd.Series(split, index=df.index)


def select_score_floor_on_fit(fit_df: pd.DataFrame, min_tp_retention: float) -> float:
    """Pick the floor that maximizes (TP-vs-null score separation) / mean_K_eff,
    subject to TP retention >= min_tp_retention. Tie-break: prefer higher floor
    (more aggressive filtering)."""
    if "is_tp" not in fit_df.columns or "is_null" not in fit_df.columns:
        return 0.0
    total_tp = max(int(fit_df["is_tp"].sum()), 1)
    best_floor = 0.0
    best_proxy = -np.inf
    for floor in FLOOR_GRID:
        kept = fit_df[fit_df["score"] >= floor]
        n_kept = len(kept)
        if n_kept == 0:
            continue
        tp_retention = int(kept["is_tp"].sum()) / total_tp
        if tp_retention < min_tp_retention:
            continue
        n_fams = kept["family_id"].nunique()
        if n_fams == 0:
            continue
        k_eff = n_kept / n_fams
        tp_scores = kept.loc[kept["is_tp"], "score"]
        null_scores = kept.loc[kept["is_null"], "score"]
        if len(tp_scores) == 0 or len(null_scores) == 0:
            continue
        score_sep = float(tp_scores.mean() - null_scores.mean())
        recall_proxy = score_sep / max(k_eff, 1e-6)
        if recall_proxy >= best_proxy - 1e-12:
            best_proxy = recall_proxy
            best_floor = float(floor)
    return best_floor


def within_family_ebh_mask(e: np.ndarray, alpha: float) -> np.ndarray:
    K = len(e)
    if K == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-e, kind="stable")
    sorted_e = e[order]
    ranks = np.arange(1, K + 1)
    threshold = K / (alpha * ranks)
    eligible = sorted_e >= threshold
    r_star = int(np.where(eligible)[0].max() + 1) if eligible.any() else 0
    mask = np.zeros(K, dtype=bool)
    mask[order[:r_star]] = True
    return mask


def evaluate_filtered(test_df: pd.DataFrame, floor: float, alpha: float) -> dict:
    """Apply floor (filter by score), run within-family e-BH on surviving candidates,
    return aggregate metrics. Recall denominator is total TP in test (pre-filter)."""
    total_tp_pre = int(test_df["is_tp"].sum())
    if floor > 0:
        survived = test_df[test_df["score"] >= floor]
    else:
        survived = test_df
    n_rej_total = 0
    n_fp_total = 0
    n_tp_total = 0
    pf_fdps = []
    for _, group in survived.groupby("family_id", sort=False):
        e = group["betting_evalue"].to_numpy(dtype=float)
        mask = within_family_ebh_mask(e, alpha)
        if mask.any():
            n_rej_fam = int(mask.sum())
            is_tp_arr = group["is_tp"].to_numpy()
            n_fp_fam = int(((~is_tp_arr) & mask).sum())
            n_tp_fam = int((is_tp_arr & mask).sum())
            n_rej_total += n_rej_fam
            n_fp_total += n_fp_fam
            n_tp_total += n_tp_fam
            pf_fdps.append(n_fp_fam / n_rej_fam)
    return {
        "rejections": n_rej_total,
        "pooled_fdp": n_fp_total / max(n_rej_total, 1),
        "per_family_fdp": float(np.mean(pf_fdps)) if pf_fdps else 0.0,
        "recall": n_tp_total / max(total_tp_pre, 1),
        "n_survived": int(len(survived)),
        "tp_retention": int(survived["is_tp"].sum()) / max(total_tp_pre, 1),
    }


def process_config(dataset: str, detector: str,
                   cand_dirname: str, formal_dirname: str) -> pd.DataFrame:
    cand_path = OUTPUTS / cand_dirname / "candidate_table.csv"
    formal_dir = OUTPUTS / formal_dirname
    if not cand_path.exists():
        print(f"  [skip] {cand_path} missing")
        return pd.DataFrame()
    if not formal_dir.exists():
        print(f"  [skip] {formal_dir} missing")
        return pd.DataFrame()
    cand_df = pd.read_csv(cand_path)
    rows = []
    for seed in range(N_SEEDS):
        seed_dir = formal_dir / f"seed_{seed}"
        ev_path = seed_dir / "test_candidates_with_evalues.csv"
        if not ev_path.exists():
            continue
        ev_df = pd.read_csv(ev_path)
        # Reproduce the split that produced this seed's outputs
        split = split_families_repro(cand_df, seed)
        fit_df = cand_df[split == "fit"].reset_index(drop=True)

        for rho in RHO_VALUES:
            adaptive_floor = select_score_floor_on_fit(fit_df, rho)
            for alpha in ALPHAS:
                # Mode 1 — no floor
                m1 = evaluate_filtered(ev_df, 0.0, alpha)
                m1.update({"mode": "no_floor", "floor": 0.0,
                           "seed": seed, "alpha": alpha, "rho": rho})
                rows.append(m1)
                # Mode 2 — fixed floor 0.40
                m2 = evaluate_filtered(ev_df, 0.40, alpha)
                m2.update({"mode": "fixed_0.40", "floor": 0.40,
                           "seed": seed, "alpha": alpha, "rho": rho})
                rows.append(m2)
                # Mode 3 — adaptive
                m3 = evaluate_filtered(ev_df, adaptive_floor, alpha)
                m3.update({"mode": "adaptive_fit", "floor": adaptive_floor,
                           "seed": seed, "alpha": alpha, "rho": rho})
                rows.append(m3)
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    agg = (
        raw.groupby(["mode", "rho", "alpha"])
        .agg(
            floor_mean=("floor", "mean"),
            rejections_mean=("rejections", "mean"),
            pooled_fdp_mean=("pooled_fdp", "mean"),
            pooled_fdp_std=("pooled_fdp", "std"),
            per_family_fdp_mean=("per_family_fdp", "mean"),
            recall_mean=("recall", "mean"),
            tp_retention_mean=("tp_retention", "mean"),
        )
        .reset_index()
    )
    pass_rate = (
        raw.assign(passed=raw["pooled_fdp"] <= raw["alpha"])
        .groupby(["mode", "rho", "alpha"])["passed"].mean().reset_index()
        .rename(columns={"passed": "pass_rate"})
    )
    agg = agg.merge(pass_rate, on=["mode", "rho", "alpha"])
    agg.insert(0, "dataset", dataset)
    agg.insert(1, "detector", detector)
    return agg


def main() -> None:
    all_rows = []
    for dataset, detector, cand_dir, formal_dir in CONFIGS:
        print(f"\n=== {dataset} / {detector} ===")
        agg = process_config(dataset, detector, cand_dir, formal_dir)
        if agg.empty:
            continue
        all_rows.append(agg)
        sub = agg[(agg["alpha"] == 0.10) & (agg["rho"] == 0.85)]
        for _, r in sub.iterrows():
            print(
                f"  α=0.10 ρ=0.85 {r['mode']:13s}"
                f"  floor={r['floor_mean']:.3f}  rej={r['rejections_mean']:6.1f}"
                f"  pooled_FDP={r['pooled_fdp_mean']:.4f}  recall={r['recall_mean']:.3f}"
                f"  pass={r['pass_rate']:.2f}  TPret={r['tp_retention_mean']:.3f}"
            )

    if not all_rows:
        print("No data!")
        return
    out = pd.concat(all_rows, ignore_index=True)
    out_path = OUTPUTS / "adaptive_floor_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
