#!/usr/bin/env python3
"""Best mitigation configs with clip sensitivity + full alpha curve + LOO."""
from __future__ import annotations

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
    ebh_summary,
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


def run_pertemplate(df, seeds, alphas, estimator, num_bins, smoothing, clip_phi, min_fit=50):
    templates = df["template"].unique()
    all_rows = []

    for seed in seeds:
        split = split_families(df, seed=seed, ratios=(0.6, 0.2, 0.2))
        df_s = df.assign(split=split)
        fit_df = df_s[df_s["split"] == "fit"]
        cal_df = df_s[df_s["split"] == "cal"]
        test_df = df_s[df_s["split"] == "test"].copy()

        # Global baseline
        global_model, _, _ = fit_density_ratio_model(
            fit_df, estimator="hist", num_bins=80, smoothing=1.0,
            clip_phi=clip_phi, max_logreg_iter=1000,
        )
        cal_null_g = cal_df[cal_df["is_null"]]
        g_phi = global_model.phi(cal_null_g["score"].to_numpy(dtype=float))
        g_Cm, g_m = float(np.sum(g_phi)), len(g_phi)

        # Per-template
        tmpl_models = {}
        tmpl_stats = {}
        for tmpl in templates:
            fit_t = fit_df[fit_df["template"] == tmpl]
            n_null = int(fit_t["is_null"].sum())
            n_tp = int(fit_t["is_tp"].sum())
            if n_null >= min_fit and n_tp >= 5:
                try:
                    bins = min(num_bins, max(10, n_null // 5)) if estimator == "hist" else num_bins
                    m, _, _ = fit_density_ratio_model(
                        fit_t, estimator=estimator, num_bins=bins,
                        smoothing=smoothing, clip_phi=clip_phi, max_logreg_iter=1000,
                    )
                    tmpl_models[tmpl] = m
                except Exception:
                    tmpl_models[tmpl] = global_model
            else:
                tmpl_models[tmpl] = global_model

            cal_t = cal_df[(cal_df["template"] == tmpl) & cal_df["is_null"]]
            if len(cal_t) > 0:
                phi_c = tmpl_models[tmpl].phi(cal_t["score"].to_numpy(dtype=float))
                tmpl_stats[tmpl] = {"C_m": float(np.sum(phi_c)), "m": len(phi_c),
                                    "phi_max": float(np.max(phi_c))}
            else:
                tmpl_stats[tmpl] = {"C_m": g_Cm, "m": g_m, "phi_max": 0.0}

        # e-values
        test_ev = np.zeros(len(test_df), dtype=float)
        test_ev_g = np.zeros(len(test_df), dtype=float)
        for tmpl in templates:
            mask = (test_df["template"] == tmpl).to_numpy()
            if not np.any(mask):
                continue
            scores = test_df.loc[mask, "score"].to_numpy(dtype=float)
            phi_t = tmpl_models[tmpl].phi(scores)
            st = tmpl_stats[tmpl]
            test_ev[mask] = compute_self_normalized_evalues(phi_t, st["C_m"], st["m"])
            phi_g = global_model.phi(scores)
            test_ev_g[mask] = compute_self_normalized_evalues(phi_g, g_Cm, g_m)

        for name, ev in [("pertemplate", test_ev), ("global", test_ev_g)]:
            test_df["evalue"] = ev
            for alpha in alphas:
                s = ebh_summary(test_df, evalue_col="evalue", alpha=alpha)
                all_rows.append({
                    "seed": seed, "alpha": alpha, "strategy": name,
                    "rejections": s["total_rejections"], "fdp": s["fdp"],
                    "recall": s["recall"], "mean_family_fdp": s["mean_family_fdp"],
                    "pass": 1 if s["fdp"] <= alpha else 0,
                })

    return pd.DataFrame(all_rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    alphas = [0.05, 0.10, 0.15, 0.20]

    df = pd.read_csv(args.candidate_table)
    for col in ["is_tp", "is_null", "is_prompt_absent"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().eq("true")
    df["template"] = df["prompt"].apply(extract_template)

    configs = [
        # Best configs from sweep, with clip sensitivity
        ("isotonic_clip500", "isotonic", 80, 1.0, 500.0),
        ("isotonic_clip1000", "isotonic", 80, 1.0, 1000.0),
        ("isotonic_noclip", "isotonic", 80, 1.0, 1_000_000.0),
        ("hist80_sm01_clip500", "hist", 80, 0.1, 500.0),
        ("hist80_sm01_noclip", "hist", 80, 0.1, 1_000_000.0),
        ("logreg_noclip", "logreg", 80, 1.0, 1_000_000.0),
    ]

    all_summary = []
    for label, est, bins, sm, clip in configs:
        print(f"Running {label}...")
        res = run_pertemplate(df, seeds, alphas, est, bins, sm, clip)
        res["config"] = label
        res.to_csv(args.out_dir / f"best_{label}.csv", index=False)

        # Summary at α=0.10
        for strat in ["pertemplate", "global"]:
            sub = res[(res["strategy"] == strat) & (res["alpha"] == 0.10)]
            if len(sub) == 0:
                continue
            all_summary.append({
                "config": label, "strategy": strat,
                "rejections": sub["rejections"].mean(),
                "fdp": sub["fdp"].mean(),
                "recall": sub["recall"].mean(),
                "pass_rate": sub["pass"].mean(),
                "mean_family_fdp": sub["mean_family_fdp"].mean(),
            })

        # Full alpha curve for pertemplate
        sub_pt = res[res["strategy"] == "pertemplate"]
        for alpha in alphas:
            sub_a = sub_pt[sub_pt["alpha"] == alpha]
            all_summary.append({
                "config": f"{label}_alpha", "strategy": f"α={alpha}",
                "rejections": sub_a["rejections"].mean(),
                "fdp": sub_a["fdp"].mean(),
                "recall": sub_a["recall"].mean(),
                "pass_rate": sub_a["pass"].mean(),
                "mean_family_fdp": sub_a["mean_family_fdp"].mean(),
            })

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(args.out_dir / "best_configs_summary.csv", index=False)

    print("\n========= COMPARISON at α=0.10 =========")
    comp = summary_df[~summary_df["config"].str.contains("_alpha")]
    print(comp.to_string(index=False))

    print("\n========= ALPHA CURVES (isotonic_clip500) =========")
    iso500 = summary_df[summary_df["config"] == "isotonic_clip500_alpha"]
    print(iso500.to_string(index=False))

    print("\n========= ALPHA CURVES (hist80_sm01_noclip) =========")
    hist01 = summary_df[summary_df["config"] == "hist80_sm01_noclip_alpha"]
    print(hist01.to_string(index=False))


if __name__ == "__main__":
    main()
