#!/usr/bin/env python3
"""Prompt-bank scaling experiment (E5/C1).

Goal: show that as the prompt bank grows, pooled-template calibration
suffers from score-mixing compression (Proposition 3); per-template
calibration restores FDR control.

Setup:
  - Reuses the freeform-2000 GDino candidate table (8 templates already
    available; no detector re-inference needed).
  - For T ∈ {1, 2, 4, 8}: take top-T templates by family count, restrict
    to families using those templates.
  - Compare two calibration modes per (T, seed, alpha):
      pooled_template      — single 80-bin histogram phi over all cal candidates
      per_template         — each template fits its own 80-bin histogram phi
  - Run within-family e-BH using betting e-values
    E[k] = (m+1) * phi(s_k) / (sum_phi_cal + phi(s_k)).
  - Report pooled FDP, per-family FDP, recall, phi_max, LOO mean.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAND_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "coco_gdino_freeform_2000"
    / "analysis"
    / "candidate_table.csv"
)
OUT_CSV = PROJECT_ROOT / "outputs" / "prompt_bank_scaling.csv"

T_VALUES = [1, 2, 3, 4, 8]  # T=8 uses all 8 templates (present + absent)
ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 20
N_BINS = 80
SPLIT_RATIOS = (0.6, 0.2, 0.2)


def split_families(family_ids: np.ndarray, seed: int) -> tuple[set, set, set]:
    rng = np.random.default_rng(seed)
    fams = family_ids.copy()
    rng.shuffle(fams)
    n = len(fams)
    n_fit = int(round(SPLIT_RATIOS[0] * n))
    n_cal = int(round(SPLIT_RATIOS[1] * n))
    n_fit = max(1, min(n_fit, n - 2))
    n_cal = max(1, min(n_cal, n - n_fit - 1))
    return (
        set(fams[:n_fit]),
        set(fams[n_fit : n_fit + n_cal]),
        set(fams[n_fit + n_cal :]),
    )


def fit_phi_hist(
    fit_null_scores: np.ndarray,
    fit_tp_scores: np.ndarray,
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit 80-bin histogram density ratio. Returns (bin_edges, phi_per_bin).

    phi(s) = p1(s)/p0(s) on the bin containing s. Mirrors the main pipeline.
    """
    all_scores = np.concatenate([fit_null_scores, fit_tp_scores])
    if len(all_scores) == 0:
        return np.array([0.0, 1.0]), np.array([1.0])
    edges = np.linspace(all_scores.min(), all_scores.max() + 1e-9, n_bins + 1)
    null_counts, _ = np.histogram(fit_null_scores, bins=edges)
    tp_counts, _ = np.histogram(fit_tp_scores, bins=edges)
    p0 = null_counts.astype(float) / max(null_counts.sum(), 1)
    p1 = tp_counts.astype(float) / max(tp_counts.sum(), 1)
    # add tiny smoothing to avoid divide-by-zero
    phi = np.where(p0 > 0, p1 / np.maximum(p0, 1e-12), 0.0)
    # Renormalise so fit-null E[phi] = 1 (canonical betting form):
    # E_null[phi] = sum_b p0[b] * phi[b]
    expected_null = float((p0 * phi).sum())
    if expected_null > 0:
        phi = phi / expected_null
    return edges, phi


def apply_phi(scores: np.ndarray, edges: np.ndarray, phi: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, len(phi) - 1)
    return phi[idx]


def within_family_ebh(evalues: np.ndarray, alpha: float) -> np.ndarray:
    K = len(evalues)
    if K == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-evalues, kind="stable")
    sorted_e = evalues[order]
    ranks = np.arange(1, K + 1)
    eligible = sorted_e >= K / (alpha * ranks)
    r = int(np.where(eligible)[0].max() + 1) if eligible.any() else 0
    mask = np.zeros(K, dtype=bool)
    mask[order[:r]] = True
    return mask


def run_seed(
    df_subset: pd.DataFrame,
    *,
    seed: int,
    mode: str,  # "pooled_template" or "per_template"
) -> dict:
    fams = df_subset["family_id"].drop_duplicates().to_numpy()
    fit_set, cal_set, test_set = split_families(fams, seed=seed)

    fit = df_subset[df_subset["family_id"].isin(fit_set)]
    cal = df_subset[df_subset["family_id"].isin(cal_set)]
    test = df_subset[df_subset["family_id"].isin(test_set)]

    out = {}
    out["n_test_families"] = len(test_set)

    # --- Fit phi (per mode) ---
    if mode == "pooled_template":
        phi_by_template = None
        edges, phi = fit_phi_hist(
            fit[fit["is_null"]]["score"].to_numpy(),
            fit[fit["is_tp"]]["score"].to_numpy(),
        )
        cal_phi = apply_phi(cal[cal["is_null"]]["score"].to_numpy(), edges, phi)
        sum_phi_cal = float(cal_phi.sum())
        m_cal = len(cal_phi)
        # Apply phi to all test candidates uniformly
        test_phi = apply_phi(test["score"].to_numpy(), edges, phi)
        phi_max = float(test_phi.max()) if len(test_phi) else 0.0
    else:  # per_template
        phi_by_template = {}
        for tpl, fit_t in fit.groupby("template"):
            edges_t, phi_t = fit_phi_hist(
                fit_t[fit_t["is_null"]]["score"].to_numpy(),
                fit_t[fit_t["is_tp"]]["score"].to_numpy(),
            )
            phi_by_template[tpl] = (edges_t, phi_t)
        # cal sum_phi and m per template
        per_tpl_sum_phi = {}
        per_tpl_m = {}
        for tpl, cal_t in cal.groupby("template"):
            cn = cal_t[cal_t["is_null"]]["score"].to_numpy()
            edges_t, phi_t = phi_by_template.get(tpl, (np.array([0.0, 1.0]), np.array([1.0])))
            cn_phi = apply_phi(cn, edges_t, phi_t)
            per_tpl_sum_phi[tpl] = float(cn_phi.sum())
            per_tpl_m[tpl] = len(cn_phi)
        # Apply phi per template to test
        test_phi_arr = np.zeros(len(test))
        scores = test["score"].to_numpy()
        templates = test["template"].to_numpy()
        for i, (s, tpl) in enumerate(zip(scores, templates)):
            edges_t, phi_t = phi_by_template.get(tpl, (np.array([0.0, 1.0]), np.array([1.0])))
            test_phi_arr[i] = apply_phi(np.array([s]), edges_t, phi_t)[0]
        test_phi = test_phi_arr
        phi_max = float(test_phi.max()) if len(test_phi) else 0.0

    # --- Compute self-normalised e-values ---
    evalues = np.zeros(len(test))
    if mode == "pooled_template":
        denom = test_phi + sum_phi_cal
        valid = denom > 0
        evalues[valid] = (m_cal + 1) * test_phi[valid] / denom[valid]
    else:
        sum_phi_arr = np.array([per_tpl_sum_phi.get(t, 0.0) for t in templates])
        m_arr = np.array([per_tpl_m.get(t, 0) for t in templates])
        denom = test_phi + sum_phi_arr
        valid = denom > 0
        evalues[valid] = (m_arr[valid] + 1) * test_phi[valid] / denom[valid]

    # --- LOO mean (sanity) ---
    null_mask = test["is_null"].to_numpy()
    out["loo_mean"] = float(evalues[null_mask].mean()) if null_mask.any() else float("nan")
    out["phi_max"] = phi_max

    # --- e-BH at each alpha within family ---
    fam_ids = test["family_id"].to_numpy()
    is_tp = test["is_tp"].to_numpy()
    is_null = test["is_null"].to_numpy()
    unique_fams, inverse = np.unique(fam_ids, return_inverse=True)
    out["per_alpha"] = {}
    for alpha in ALPHAS:
        rejected = np.zeros(len(test), dtype=bool)
        for i in range(len(unique_fams)):
            idx = np.where(inverse == i)[0]
            mask = within_family_ebh(evalues[idx], alpha)
            if mask.any():
                rejected[idx[mask]] = True
        n_rej = int(rejected.sum())
        n_fp = int((rejected & ~is_tp).sum())
        n_tp = int((rejected & is_tp).sum())
        # per-family FDP
        rej_idx = np.where(rejected)[0]
        if len(rej_idx) > 0:
            fam_rej = pd.Series(fam_ids[rej_idx])
            fam_fp = pd.Series((~is_tp[rej_idx]).astype(int))
            grp = pd.DataFrame({"f": fam_rej, "fp": fam_fp}).groupby("f")["fp"].mean()
            pf_fdp = float(grp.mean())
        else:
            pf_fdp = 0.0
        out["per_alpha"][alpha] = {
            "rejections": n_rej,
            "fdp_pooled": n_fp / max(n_rej, 1),
            "fdp_family": pf_fdp,
            "recall": n_tp / max(int(is_tp.sum()), 1),
        }
    return out


def main() -> None:
    print(f"Loading {CAND_TABLE}")
    df = pd.read_csv(CAND_TABLE)
    df["template"] = df.apply(
        lambda r: r["prompt"].replace(r["category_name"], "{cat}"),
        axis=1,
    )
    # Templates split into present-only (with TPs) and absent-only.
    # In freeform_2000, each template is either all-present or all-absent.
    present_templates = (
        df[df["is_prompt_absent"] == False]["template"].drop_duplicates().tolist()
    )
    absent_templates = (
        df[df["is_prompt_absent"] == True]["template"].drop_duplicates().tolist()
    )
    fam_counts = df.drop_duplicates("family_id").groupby("template").size()
    present_order = sorted(present_templates, key=lambda t: -fam_counts[t])
    absent_order = sorted(absent_templates, key=lambda t: -fam_counts[t])
    print(f"Present: {present_order}")
    print(f"Absent:  {absent_order}")

    rows = []
    for T in T_VALUES:
        if T <= 4:
            # Present-only templates: top T by family count
            template_subset = present_order[:T]
        else:
            # T=8: all present + absent templates (full freeform setup)
            template_subset = present_order + absent_order
        df_subset = df[df["template"].isin(template_subset)]
        n_fams_total = df_subset["family_id"].nunique()
        cal_per_template = (
            df_subset[df_subset["is_null"]]
            .drop_duplicates(["family_id", "candidate_idx"])
            .groupby("template")
            .size()
            .mean()
        )
        print(f"\nT={T}: families={n_fams_total}, mean cal_null/template ≈ {cal_per_template:.0f}")
        for mode in ("pooled_template", "per_template"):
            seed_results = []
            for seed in range(N_SEEDS):
                seed_results.append(run_seed(df_subset, seed=seed, mode=mode))
            # Aggregate
            for alpha in ALPHAS:
                fdps = [r["per_alpha"][alpha]["fdp_pooled"] for r in seed_results]
                pffs = [r["per_alpha"][alpha]["fdp_family"] for r in seed_results]
                recs = [r["per_alpha"][alpha]["recall"] for r in seed_results]
                rejs = [r["per_alpha"][alpha]["rejections"] for r in seed_results]
                rows.append({
                    "T": T,
                    "calibration_mode": mode,
                    "alpha": alpha,
                    "n_test_families": float(np.mean([r["n_test_families"] for r in seed_results])),
                    "loo_mean": float(np.mean([r["loo_mean"] for r in seed_results])),
                    "phi_max": float(np.mean([r["phi_max"] for r in seed_results])),
                    "rejections_mean": float(np.mean(rejs)),
                    "fdp_pooled_mean": float(np.mean(fdps)),
                    "fdp_pooled_std": float(np.std(fdps, ddof=1)),
                    "fdp_family_mean": float(np.mean(pffs)),
                    "recall_mean": float(np.mean(recs)),
                    "pass_rate": float(np.mean([f <= alpha for f in fdps])),
                })
            # Print α=0.10 summary
            r10 = next(x for x in rows[-len(ALPHAS):] if x["alpha"] == 0.10)
            print(
                f"  {mode:18s} α=0.10  rej={r10['rejections_mean']:6.1f}  "
                f"FDP={r10['fdp_pooled_mean']:.4f}  per-fam={r10['fdp_family_mean']:.4f}  "
                f"recall={r10['recall_mean']:.4f}  φmax={r10['phi_max']:.2f}  "
                f"LOO={r10['loo_mean']:.3f}"
            )

    # Save
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
