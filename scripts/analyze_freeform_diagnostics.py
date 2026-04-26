#!/usr/bin/env python3
"""Prompt-conditional φ_max / C_m / T_1 diagnostic for free-form pilot.

For each seed, this script:
1. Re-fits the density ratio model on the fit split
2. Computes per-template φ_max on calibration nulls
3. Computes per-template C_m (sum of φ on calibration nulls)
4. Computes T_1(C_m) = K * C_m / (α*(m+1) - K) per template
5. Reports whether φ_max < T_1 (power collapse predicted by Theorem 1)

Output: a CSV diagnostic table and summary JSON.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_formal_betting_pipeline import (
    fit_density_ratio_model,
    split_families,
    compute_self_normalized_evalues,
)


def extract_template(prompt: str) -> str:
    patterns = [
        (r"^a visible (.+) in the scene$", "a visible {cat} in the scene"),
        (r"^a small (.+) near the center$", "a small {cat} near the center"),
        (r"^a photo of a (.+)$", "a photo of a {cat}"),
        (r"^a (.+) in the scene$", "a {cat} in the scene"),
        (r"^a (.+) object$", "a {cat} object"),
        (r"^the (.+)$", "the {cat}"),
    ]
    for pat, tmpl in patterns:
        if re.match(pat, prompt):
            return tmpl
    return "other"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--alphas", type=str, default="0.05,0.1,0.15,0.2")
    parser.add_argument("--num-bins", type=int, default=80)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--clip-phi", type=float, default=1_000_000.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    df = pd.read_csv(args.candidate_table)
    for col in ["is_tp", "is_null", "is_prompt_absent"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().eq("true")
    df["template"] = df["prompt"].apply(extract_template)

    all_rows = []
    all_global_rows = []

    for seed in seeds:
        split = split_families(df, seed=seed, ratios=(0.6, 0.2, 0.2))
        df_s = df.copy()
        df_s["split"] = split

        fit_df = df_s[df_s["split"] == "fit"].copy()
        cal_df = df_s[df_s["split"] == "cal"].copy()
        test_df = df_s[df_s["split"] == "test"].copy()

        model, _, ratio_summary = fit_density_ratio_model(
            fit_df,
            estimator="hist",
            num_bins=args.num_bins,
            smoothing=args.smoothing,
            clip_phi=args.clip_phi,
            max_logreg_iter=1000,
        )

        # Global calibration null pool
        cal_null = cal_df[cal_df["is_null"]].copy()
        cal_null_phi = model.phi(cal_null["score"].to_numpy(dtype=float))
        global_C_m = float(np.sum(cal_null_phi))
        global_m = int(cal_null_phi.size)
        global_phi_max = float(np.max(cal_null_phi)) if len(cal_null_phi) > 0 else 0.0

        # Per-template diagnostics on calibration nulls
        cal_null = cal_null.assign(phi=cal_null_phi)
        cal_null["template"] = cal_null["prompt"].apply(extract_template)

        for tmpl, grp in cal_null.groupby("template"):
            tmpl_phi = grp["phi"].to_numpy()
            tmpl_C_m = float(np.sum(tmpl_phi))
            tmpl_m = int(len(tmpl_phi))
            tmpl_phi_max = float(np.max(tmpl_phi)) if len(tmpl_phi) > 0 else 0.0

            # Test families for this template
            test_tmpl = test_df[test_df["template"] == tmpl]
            K_values = test_tmpl.groupby("family_id").size()
            mean_K = float(K_values.mean()) if len(K_values) > 0 else 0.0
            median_K = float(K_values.median()) if len(K_values) > 0 else 0.0

            for alpha in alphas:
                # T_1 using GLOBAL C_m (since we pool across templates)
                denom_global = alpha * (global_m + 1) - mean_K
                T1_global = mean_K * global_C_m / denom_global if denom_global > 0 else float("inf")

                # T_1 using per-template C_m (hypothetical Mondrian)
                denom_tmpl = alpha * (tmpl_m + 1) - mean_K
                T1_tmpl = mean_K * tmpl_C_m / denom_tmpl if denom_tmpl > 0 else float("inf")

                # φ_max on fit split (the actual peak the model can produce)
                fit_phi_max = ratio_summary.get("max_phi", 0.0)

                all_rows.append({
                    "seed": seed,
                    "template": tmpl,
                    "alpha": alpha,
                    "cal_null_count": tmpl_m,
                    "tmpl_phi_max": tmpl_phi_max,
                    "tmpl_C_m": tmpl_C_m,
                    "global_phi_max": global_phi_max,
                    "global_C_m": global_C_m,
                    "global_m": global_m,
                    "fit_phi_max": fit_phi_max,
                    "mean_K": mean_K,
                    "median_K": median_K,
                    "T1_global": T1_global,
                    "T1_tmpl": T1_tmpl,
                    "phi_max_below_T1_global": fit_phi_max < T1_global,
                    "phi_max_below_T1_tmpl": fit_phi_max < T1_tmpl,
                })

        # Global summary
        for alpha in alphas:
            test_K = test_df.groupby("family_id").size()
            mean_K_all = float(test_K.mean())
            denom = alpha * (global_m + 1) - mean_K_all
            T1 = mean_K_all * global_C_m / denom if denom > 0 else float("inf")

            all_global_rows.append({
                "seed": seed,
                "alpha": alpha,
                "global_m": global_m,
                "global_C_m": global_C_m,
                "global_phi_max": global_phi_max,
                "fit_phi_max": ratio_summary.get("max_phi", 0.0),
                "mean_K": mean_K_all,
                "T1": T1,
                "phi_max_below_T1": ratio_summary.get("max_phi", 0.0) < T1,
            })

    result_df = pd.DataFrame(all_rows)
    global_df = pd.DataFrame(all_global_rows)

    result_df.to_csv(args.out_dir / "prompt_conditional_diagnostics.csv", index=False)
    global_df.to_csv(args.out_dir / "global_diagnostics.csv", index=False)

    # Summary: mean over seeds
    print("\n=== GLOBAL DIAGNOSTICS (mean over seeds) ===")
    gm = global_df.groupby("alpha").agg(
        global_m=("global_m", "mean"),
        global_C_m=("global_C_m", "mean"),
        fit_phi_max=("fit_phi_max", "mean"),
        mean_K=("mean_K", "mean"),
        T1=("T1", "mean"),
        pct_below_T1=("phi_max_below_T1", "mean"),
    ).reset_index()
    print(gm.to_string(index=False))

    print("\n=== PER-TEMPLATE DIAGNOSTICS at alpha=0.10 (mean over seeds) ===")
    tmpl_summary = (
        result_df[result_df["alpha"] == 0.10]
        .groupby("template")
        .agg(
            cal_null_count=("cal_null_count", "mean"),
            fit_phi_max=("fit_phi_max", "mean"),
            mean_K=("mean_K", "mean"),
            T1_global=("T1_global", "mean"),
            pct_below_T1=("phi_max_below_T1_global", "mean"),
        )
        .sort_values("T1_global", ascending=False)
        .reset_index()
    )
    print(tmpl_summary.to_string(index=False))

    # Save summary JSON
    summary = {
        "description": "Prompt-conditional activation frontier diagnostic for free-form pilot",
        "interpretation": (
            "phi_max_below_T1 = True means Theorem 1 predicts zero rejections for that template. "
            "When this holds for most/all templates, the overall power collapse is explained by "
            "the activation frontier, not by a validity failure."
        ),
        "global_mean": gm.to_dict("records"),
        "per_template_alpha010": tmpl_summary.to_dict("records"),
    }
    with open(args.out_dir / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
