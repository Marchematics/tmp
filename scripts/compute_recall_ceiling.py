#!/usr/bin/env python3
"""Compute recall ceiling and efficiency ratio for the paper.

For each family, the activation threshold T_r = K*C_m / (α*r*(m+1) - K)
determines the minimum φ(S) needed for the r-th rejection. A TP with
φ(S) < T_1 is *never* rejected by any e-BH procedure using this φ model.

Two ceilings are computed:
  1. T_1-ceiling (theorem-level): count TPs with φ ≥ T_1 for their family.
  2. Tight ceiling (per-family): for each family, find max r s.t. r candidates
     have φ ≥ T_r, then count TPs among those top-r candidates.

Efficiency = actual_recall / recall_ceiling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def reconstruct_phi(scores: np.ndarray, bins_df: pd.DataFrame) -> np.ndarray:
    """Map scores to φ values using the histogram density ratio model."""
    edges = np.concatenate([bins_df["left"].values, [bins_df["right"].values[-1]]])
    phi_vals = bins_df["phi"].values
    bin_ids = np.searchsorted(edges, scores, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, len(phi_vals) - 1)
    return phi_vals[bin_ids]


def activation_threshold(K: int, C_m: float, m: int, alpha: float, r: int) -> float:
    """T_r = K * C_m / (α * r * (m+1) - K). Returns inf if denominator ≤ 0."""
    denom = alpha * r * (m + 1) - K
    if denom <= 0:
        return float("inf")
    return K * C_m / denom


def compute_tight_ceiling_family(phi_sorted_desc: np.ndarray, is_tp_sorted: np.ndarray,
                                  K: int, C_m: float, m: int, alpha: float) -> int:
    """For a single family, find max reachable rejections and count reachable TPs."""
    max_r = 0
    for r in range(1, K + 1):
        T_r = activation_threshold(K, C_m, m, alpha, r)
        # Need r candidates with φ ≥ T_r (the r-th largest must exceed T_r)
        if r <= len(phi_sorted_desc) and phi_sorted_desc[r - 1] >= T_r:
            max_r = r
        # NO break: threshold K/(αr) decreases with r, so later r may pass
    # Among top max_r candidates, count TPs
    return int(is_tp_sorted[:max_r].sum())


def process_seed(pipeline_dir: Path, seed: int, alpha: float,
                 seed_meta: dict) -> dict:
    """Process one seed: compute recall ceiling and efficiency."""
    seed_dir = pipeline_dir / f"seed_{seed}"

    # Load density ratio bins
    bins_df = pd.read_csv(seed_dir / "density_ratio_bins.csv")

    # Load test candidates
    test_df = pd.read_csv(seed_dir / "test_candidates_with_evalues.csv")

    # Get calibration parameters
    C_m = seed_meta["sum_phi_cal"]
    m = seed_meta["num_cal_null"]
    phi_max = seed_meta["density_ratio"]["max_phi"]

    # Compute φ for all test candidates
    scores = test_df["score"].to_numpy(dtype=float)
    phi = reconstruct_phi(scores, bins_df)
    test_df["phi"] = phi

    # --- T_1-based ceiling ---
    # For each candidate, compute T_1 for its family
    K_vals = test_df["family_size"].to_numpy(dtype=int)
    T1_vals = np.array([activation_threshold(int(K), C_m, m, alpha, 1) for K in K_vals])
    test_df["T1"] = T1_vals
    test_df["reachable_T1"] = phi >= T1_vals

    tp_mask = test_df["is_tp"].astype(bool)
    n_tp_total = int(tp_mask.sum())

    if n_tp_total == 0:
        return {"seed": seed, "n_tp": 0, "error": "no_tp"}

    n_tp_reachable_T1 = int((tp_mask & test_df["reachable_T1"]).sum())
    ceiling_T1 = n_tp_reachable_T1 / n_tp_total

    # --- Tight ceiling (per-family) ---
    n_tp_reachable_tight = 0
    for fam_id, group in test_df.groupby("family_id", sort=False):
        K = len(group)
        phi_fam = group["phi"].to_numpy(dtype=float)
        is_tp_fam = group["is_tp"].to_numpy(dtype=bool)
        # Sort by phi descending
        order = np.argsort(-phi_fam)
        phi_sorted = phi_fam[order]
        is_tp_sorted = is_tp_fam[order]
        n_tp_reachable_tight += compute_tight_ceiling_family(
            phi_sorted, is_tp_sorted, K, C_m, m, alpha
        )
    ceiling_tight = n_tp_reachable_tight / n_tp_total

    # --- Actual recall (from e-BH rejections) ---
    # A candidate is rejected if it was selected by e-BH
    # Re-run e-BH logic: for each family, sort by betting_evalue desc,
    # find max k s.t. (k * e_{(k)} / K) >= 1/alpha
    n_tp_rejected = 0
    n_rejected_total = 0
    for fam_id, group in test_df.groupby("family_id", sort=False):
        K = len(group)
        evalues = group["betting_evalue"].to_numpy(dtype=float)
        is_tp_fam = group["is_tp"].to_numpy(dtype=bool)
        order = np.argsort(-evalues)
        evalues_sorted = evalues[order]
        is_tp_sorted = is_tp_fam[order]

        # e-BH: find max k s.t. e_{(k)} >= K/(alpha*k)
        selected_k = 0
        for k in range(1, K + 1):
            if evalues_sorted[k - 1] >= K / (alpha * k):
                selected_k = k
            # NO break: threshold decreases with k, later k may still pass
        if selected_k > 0:
            n_tp_rejected += int(is_tp_sorted[:selected_k].sum())
            n_rejected_total += selected_k

    actual_recall = n_tp_rejected / n_tp_total

    # --- Score-conditional ceilings ---
    score_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    score_cond = {}
    for thr in score_thresholds:
        tp_above = tp_mask & (test_df["score"] >= thr)
        n_above = int(tp_above.sum())
        if n_above == 0:
            continue
        n_reachable = int((tp_above & test_df["reachable_T1"]).sum())
        score_cond[f"n_tp_score>={thr}"] = n_above
        score_cond[f"ceiling_score>={thr}"] = n_reachable / n_above

    # --- Load pipeline official recall for cross-check ---
    official_recall = None
    results_csv = pipeline_dir / "seed_alpha_results.csv"
    if results_csv.exists():
        res_df = pd.read_csv(results_csv)
        mask = (res_df["seed"] == seed) & (res_df["method"] == "betting")
        alpha_mask = mask & (np.abs(res_df["alpha"] - alpha) < 1e-6)
        if alpha_mask.any():
            official_recall = float(res_df.loc[alpha_mask, "recall"].iloc[0])

    return {
        "seed": seed,
        "n_tp": n_tp_total,
        "n_tp_reachable_T1": n_tp_reachable_T1,
        "n_tp_reachable_tight": n_tp_reachable_tight,
        "n_tp_rejected": n_tp_rejected,
        "n_rejected_total": n_rejected_total,
        "ceiling_T1": ceiling_T1,
        "ceiling_tight": ceiling_tight,
        "actual_recall": actual_recall,
        "official_recall": official_recall,
        "efficiency_T1": actual_recall / max(ceiling_T1, 1e-12),
        "efficiency_tight": actual_recall / max(ceiling_tight, 1e-12),
        "C_m": C_m,
        "m": m,
        "phi_max": phi_max,
        **score_cond,
    }


def main():
    p = argparse.ArgumentParser(description="Compute recall ceiling and efficiency")
    p.add_argument("--pipeline-dirs", type=str, nargs="+", required=True,
                   help="Pipeline output directories")
    p.add_argument("--labels", type=str, nargs="+", default=None,
                   help="Labels for each pipeline dir")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seeds", type=str, default="0-19",
                   help="Seed range, e.g. '0-19' or '0,1,2,3,4'")
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    # Parse seeds
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    labels = args.labels or [Path(d).name for d in args.pipeline_dirs]

    all_rows = []
    for label, pdir_str in zip(labels, args.pipeline_dirs):
        pdir = Path(pdir_str)
        # Load seed summaries
        with open(pdir / "seed_summaries.json") as f:
            seed_summaries = json.load(f)

        print(f"\n{'='*60}")
        print(f"  {label}  (alpha={args.alpha})")
        print(f"{'='*60}")

        for seed in seeds:
            meta = seed_summaries[str(seed)]
            result = process_seed(pdir, seed, args.alpha, meta)
            result["config"] = label
            all_rows.append(result)

        # Seed-averaged summary
        seed_results = [r for r in all_rows if r["config"] == label]
        valid = [r for r in seed_results if "error" not in r]
        if valid:
            c_T1 = np.mean([r["ceiling_T1"] for r in valid])
            c_tight = np.mean([r["ceiling_tight"] for r in valid])
            act = np.mean([r["actual_recall"] for r in valid])
            eff_T1 = np.mean([r["efficiency_T1"] for r in valid])
            eff_tight = np.mean([r["efficiency_tight"] for r in valid])
            off_vals = [r["official_recall"] for r in valid
                        if r.get("official_recall") is not None]
            off = np.mean(off_vals) if off_vals else float("nan")
            print(f"\n  Mean over {len(valid)} seeds:")
            print(f"    Pipeline official recall: {off:.3f}  ({off*100:.1f}%)")
            print(f"    Recomputed recall:        {act:.3f}  ({act*100:.1f}%)")
            print(f"    Attainable ceiling (T1):  {c_T1:.3f}  ({c_T1*100:.1f}%)")
            print(f"    Attainable ceiling (tight): {c_tight:.3f}  ({c_tight*100:.1f}%)")
            print(f"    Efficiency (T1):          {eff_T1:.3f}  ({eff_T1*100:.1f}%)")
            print(f"    Efficiency (tight):       {eff_tight:.3f}  ({eff_tight*100:.1f}%)")
            if off_vals:
                print(f"    Official vs tight ceiling: {off/max(c_tight,1e-12):.3f}")

            # Score-conditional
            for thr in [0.0, 0.3, 0.5]:
                key = f"ceiling_score>={thr}"
                vals = [r.get(key) for r in valid if r.get(key) is not None]
                if vals:
                    print(f"    Ceiling (score>={thr}):    {np.mean(vals):.3f}")

    # Save
    df = pd.DataFrame(all_rows)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved to {args.out_csv}")
    else:
        out_path = PROJECT_ROOT / "outputs" / "recall_ceiling_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
