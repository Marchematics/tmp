#!/usr/bin/env python3
"""Per-template density-ratio fitting mitigation for free-form power collapse.

Hypothesis: the global density-ratio model compresses φ_max because it mixes
heterogeneous score distributions across templates. Fitting separate models
per template should recover higher φ_max and potentially break the
activation-frontier barrier.

Strategy:
  1. Extract template from each prompt
  2. For each seed: split families → fit/cal/test
  3. Fit a SEPARATE density-ratio model per template (on fit split)
  4. Compute per-template C_m on cal nulls using the template-specific model
  5. Compute self-normalized e-values using template-specific φ + C_m
  6. Run per-family e-BH as usual
  7. Compare with global model results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_formal_betting_pipeline import (
    DensityRatioModel,
    fit_density_ratio_model,
    split_families,
    compute_self_normalized_evalues,
    ebh_summary,
)
from ovd_hallucination_fdr.evalues import e_bh


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--alphas", type=str, default="0.05,0.1,0.15,0.2")
    parser.add_argument("--num-bins", type=int, default=80)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--clip-phi", type=float, default=1_000_000.0)
    parser.add_argument("--min-fit-samples", type=int, default=50,
                        help="Minimum null+tp samples in fit split to fit per-template model")
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
    all_diagnostic_rows = []

    for seed in seeds:
        split = split_families(df, seed=seed, ratios=(0.6, 0.2, 0.2))
        df_s = df.copy()
        df_s["split"] = split

        fit_df = df_s[df_s["split"] == "fit"].copy()
        cal_df = df_s[df_s["split"] == "cal"].copy()
        test_df = df_s[df_s["split"] == "test"].copy()

        # --- Global model (baseline) ---
        global_model, _, global_summary = fit_density_ratio_model(
            fit_df, estimator="hist", num_bins=args.num_bins,
            smoothing=args.smoothing, clip_phi=args.clip_phi, max_logreg_iter=1000,
        )
        cal_null_global = cal_df[cal_df["is_null"]].copy()
        cal_null_phi_global = global_model.phi(cal_null_global["score"].to_numpy(dtype=float))
        global_C_m = float(np.sum(cal_null_phi_global))
        global_m = int(len(cal_null_phi_global))

        # Global e-values on test
        test_phi_global = global_model.phi(test_df["score"].to_numpy(dtype=float))
        test_eval_global = compute_self_normalized_evalues(test_phi_global, global_C_m, global_m)

        # --- Per-template models ---
        templates = df["template"].unique()
        template_models = {}
        template_cal_stats = {}

        for tmpl in templates:
            fit_tmpl = fit_df[fit_df["template"] == tmpl]
            n_null_fit = int(fit_tmpl["is_null"].sum())
            n_tp_fit = int(fit_tmpl["is_tp"].sum())

            if n_null_fit >= args.min_fit_samples and n_tp_fit >= 5 and n_tp_fit > 0:
                try:
                    tmpl_model, _, tmpl_summary = fit_density_ratio_model(
                        fit_tmpl, estimator="hist", num_bins=min(args.num_bins, max(20, n_null_fit // 5)),
                        smoothing=args.smoothing, clip_phi=args.clip_phi, max_logreg_iter=1000,
                    )
                    template_models[tmpl] = tmpl_model
                except Exception as e:
                    print(f"  [seed={seed}] Template '{tmpl}': fit failed ({e}), using global")
                    template_models[tmpl] = global_model
            else:
                print(f"  [seed={seed}] Template '{tmpl}': too few samples (null={n_null_fit}, tp={n_tp_fit}), using global")
                template_models[tmpl] = global_model

            # Per-template cal stats
            cal_tmpl_null = cal_df[(cal_df["template"] == tmpl) & cal_df["is_null"]]
            if len(cal_tmpl_null) > 0:
                model_for_tmpl = template_models[tmpl]
                cal_phi_tmpl = model_for_tmpl.phi(cal_tmpl_null["score"].to_numpy(dtype=float))
                template_cal_stats[tmpl] = {
                    "C_m": float(np.sum(cal_phi_tmpl)),
                    "m": int(len(cal_phi_tmpl)),
                    "phi_max_cal": float(np.max(cal_phi_tmpl)),
                }
            else:
                template_cal_stats[tmpl] = {"C_m": global_C_m, "m": global_m, "phi_max_cal": 0.0}

        # Compute per-template e-values on test
        test_phi_pertemplate = np.zeros(len(test_df), dtype=float)
        test_eval_pertemplate_local = np.zeros(len(test_df), dtype=float)  # per-template C_m
        test_eval_pertemplate_global = np.zeros(len(test_df), dtype=float)  # global C_m

        for tmpl in templates:
            mask = (test_df["template"] == tmpl).to_numpy()
            if not np.any(mask):
                continue
            model_t = template_models[tmpl]
            scores_t = test_df.loc[mask, "score"].to_numpy(dtype=float)
            phi_t = model_t.phi(scores_t)
            test_phi_pertemplate[mask] = phi_t

            # Strategy A: per-template φ + per-template C_m
            stats_t = template_cal_stats[tmpl]
            test_eval_pertemplate_local[mask] = compute_self_normalized_evalues(
                phi_t, stats_t["C_m"], stats_t["m"]
            )

            # Strategy B: per-template φ + global C_m
            test_eval_pertemplate_global[mask] = compute_self_normalized_evalues(
                phi_t, global_C_m, global_m
            )

        # Diagnostic: per-template φ_max comparison
        for tmpl in templates:
            model_t = template_models[tmpl]
            fit_tmpl = fit_df[fit_df["template"] == tmpl]
            if len(fit_tmpl) > 0:
                all_phi = model_t.phi(fit_tmpl["score"].to_numpy(dtype=float))
                fit_phi_max = float(np.max(all_phi))
            else:
                fit_phi_max = 0.0

            stats_t = template_cal_stats[tmpl]
            is_own_model = model_t is not global_model

            all_diagnostic_rows.append({
                "seed": seed,
                "template": tmpl,
                "own_model": is_own_model,
                "fit_phi_max_pertemplate": fit_phi_max,
                "fit_phi_max_global": global_summary.get("max_phi", 0.0),
                "phi_max_ratio": fit_phi_max / max(global_summary.get("max_phi", 1.0), 1e-9),
                "cal_C_m_local": stats_t["C_m"],
                "cal_m_local": stats_t["m"],
                "cal_C_m_global": global_C_m,
                "cal_m_global": global_m,
            })

        # Run e-BH for each strategy
        strategies = {
            "global": test_eval_global,
            "pertemplate_phi_local_Cm": test_eval_pertemplate_local,
            "pertemplate_phi_global_Cm": test_eval_pertemplate_global,
        }

        test_work = test_df.copy()
        for strategy_name, evalues in strategies.items():
            test_work["evalue"] = evalues
            for alpha in alphas:
                summary = ebh_summary(test_work, evalue_col="evalue", alpha=alpha)
                all_rows.append({
                    "seed": seed,
                    "alpha": alpha,
                    "strategy": strategy_name,
                    "rejections": summary["total_rejections"],
                    "fdp": summary["fdp"],
                    "recall": summary["recall"],
                    "mean_family_fdp": summary["mean_family_fdp"],
                    "max_evalue": float(np.max(evalues)) if len(evalues) > 0 else 0.0,
                })

    # Summarize
    result_df = pd.DataFrame(all_rows)
    diag_df = pd.DataFrame(all_diagnostic_rows)

    result_df.to_csv(args.out_dir / "mitigation_results.csv", index=False)
    diag_df.to_csv(args.out_dir / "mitigation_diagnostics.csv", index=False)

    # Print summary
    print("\n=== MITIGATION RESULTS (mean over seeds) ===")
    summary = (
        result_df.groupby(["strategy", "alpha"])
        .agg(
            rejections=("rejections", "mean"),
            fdp=("fdp", "mean"),
            recall=("recall", "mean"),
            max_evalue=("max_evalue", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\n=== PER-TEMPLATE φ_max DIAGNOSTIC (mean over seeds) ===")
    diag_summary = (
        diag_df.groupby("template")
        .agg(
            own_model_pct=("own_model", "mean"),
            phi_max_pertemplate=("fit_phi_max_pertemplate", "mean"),
            phi_max_global=("fit_phi_max_global", "mean"),
            phi_max_ratio=("phi_max_ratio", "mean"),
            cal_C_m_local=("cal_C_m_local", "mean"),
            cal_m_local=("cal_m_local", "mean"),
        )
        .sort_values("phi_max_pertemplate", ascending=False)
        .reset_index()
    )
    print(diag_summary.to_string(index=False))

    # Save summary JSON
    summary_json = {
        "description": "Per-template density-ratio fitting mitigation for free-form power collapse",
        "strategies": {
            "global": "Global density-ratio model + global calibration pool (baseline)",
            "pertemplate_phi_local_Cm": "Per-template density-ratio model + per-template calibration pool",
            "pertemplate_phi_global_Cm": "Per-template density-ratio model + global calibration pool",
        },
        "results_mean": summary.to_dict("records"),
        "phi_max_diagnostic": diag_summary.to_dict("records"),
    }
    with open(args.out_dir / "mitigation_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=str)

    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
