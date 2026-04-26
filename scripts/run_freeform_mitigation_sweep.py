#!/usr/bin/env python3
"""Sweep estimators and parameters for per-template mitigation.

Tests: hist (various bins), isotonic, logreg — all with per-template fitting.
Only uses the theoretically valid Strategy A (per-template φ + per-template C_m).
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def normalize_template_from_category(row) -> str:
    prompt = str(row["prompt"])
    category = str(row.get("category_name", ""))
    if category and category in prompt:
        return prompt.replace(category, "{category}")
    return extract_template(prompt)


def stable_index(text: str, modulus: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def load_template_pool(template_dir: Path, count: int) -> list[str]:
    if count == 7:
        return [
            "a visible {category} in the scene",
            "a photo of a {category}",
            "a {category} object",
            "the {category} in this image",
            "a photo containing a {category}",
            "a small {category} near the center",
            "a {category} in the scene",
        ]
    path = template_dir / f"freeform_{count}.txt"
    templates = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(templates) < count:
        raise ValueError(f"{path} has {len(templates)} templates, expected at least {count}")
    templates = templates[:count]
    missing = [template for template in templates if "{category}" not in template]
    if missing:
        raise ValueError(f"Templates must contain {{category}}; bad example: {missing[0]}")
    return templates


def apply_template_pool(df: pd.DataFrame, templates: list[str]) -> pd.DataFrame:
    out = df.copy()
    family_base = out[["family_id", "category_name"]].drop_duplicates("family_id").copy()
    family_base["template"] = family_base["family_id"].map(lambda item: templates[stable_index(str(item), len(templates))])
    family_base["prompt"] = family_base.apply(
        lambda row: str(row["template"]).format(category=str(row["category_name"])),
        axis=1,
    )
    mapping_template = dict(zip(family_base["family_id"], family_base["template"]))
    mapping_prompt = dict(zip(family_base["family_id"], family_base["prompt"]))
    out["template"] = out["family_id"].map(mapping_template)
    out["prompt"] = out["family_id"].map(mapping_prompt)
    return out


def run_one_config(df, seeds, alphas, estimator, num_bins, smoothing, clip_phi, min_fit):
    """Run per-template mitigation with one config. Returns (results_rows, diag_rows)."""
    all_rows = []
    all_diag = []
    templates = df["template"].unique()

    for seed in seeds:
        split = split_families(df, seed=seed, ratios=(0.6, 0.2, 0.2))
        df_s = df.assign(split=split)
        fit_df = df_s[df_s["split"] == "fit"]
        cal_df = df_s[df_s["split"] == "cal"]
        test_df = df_s[df_s["split"] == "test"].copy()

        # Global baseline
        try:
            global_model, _, g_sum = fit_density_ratio_model(
                fit_df, estimator="hist", num_bins=80, smoothing=1.0,
                clip_phi=clip_phi, max_logreg_iter=1000,
            )
        except Exception:
            continue
        cal_null_g = cal_df[cal_df["is_null"]]
        g_phi = global_model.phi(cal_null_g["score"].to_numpy(dtype=float))
        g_Cm = float(np.sum(g_phi))
        g_m = len(g_phi)

        # Per-template models
        tmpl_models = {}
        tmpl_cal_stats = {}
        for tmpl in templates:
            fit_t = fit_df[fit_df["template"] == tmpl]
            n_null = int(fit_t["is_null"].sum())
            n_tp = int(fit_t["is_tp"].sum())
            if n_null >= min_fit and n_tp >= 5:
                try:
                    bins_use = min(num_bins, max(10, n_null // 5)) if estimator == "hist" else num_bins
                    m, _, ms = fit_density_ratio_model(
                        fit_t, estimator=estimator, num_bins=bins_use,
                        smoothing=smoothing, clip_phi=clip_phi, max_logreg_iter=1000,
                    )
                    tmpl_models[tmpl] = m
                except Exception:
                    tmpl_models[tmpl] = global_model
            else:
                tmpl_models[tmpl] = global_model

            cal_t_null = cal_df[(cal_df["template"] == tmpl) & cal_df["is_null"]]
            if len(cal_t_null) > 0:
                model_t = tmpl_models[tmpl]
                phi_t = model_t.phi(cal_t_null["score"].to_numpy(dtype=float))
                tmpl_cal_stats[tmpl] = {"C_m": float(np.sum(phi_t)), "m": len(phi_t)}
            else:
                tmpl_cal_stats[tmpl] = {"C_m": g_Cm, "m": g_m}

        # Compute per-template e-values (Strategy A only)
        test_evals = np.zeros(len(test_df), dtype=float)
        test_evals_global = np.zeros(len(test_df), dtype=float)
        for tmpl in templates:
            mask = (test_df["template"] == tmpl).to_numpy()
            if not np.any(mask):
                continue
            model_t = tmpl_models[tmpl]
            scores_t = test_df.loc[mask, "score"].to_numpy(dtype=float)
            phi_t = model_t.phi(scores_t)
            stats_t = tmpl_cal_stats[tmpl]
            test_evals[mask] = compute_self_normalized_evalues(phi_t, stats_t["C_m"], stats_t["m"])

            # Global baseline
            phi_g = global_model.phi(scores_t)
            test_evals_global[mask] = compute_self_normalized_evalues(phi_g, g_Cm, g_m)

        # φ_max diagnostic per template
        for tmpl in templates:
            model_t = tmpl_models[tmpl]
            fit_t = fit_df[fit_df["template"] == tmpl]
            cal_t = cal_df[cal_df["template"] == tmpl]
            if len(fit_t) > 0:
                all_phi = model_t.phi(fit_t["score"].to_numpy(dtype=float))
                phi_max_t = float(np.max(all_phi))
            else:
                phi_max_t = 0.0
            all_diag.append({
                "seed": seed, "template": tmpl,
                "own_model": tmpl_models[tmpl] is not global_model,
                "phi_max": phi_max_t,
                "num_fit_families": int(fit_t["family_id"].nunique()),
                "num_cal_families": int(cal_t["family_id"].nunique()),
                "num_fit_candidates": int(len(fit_t)),
                "num_cal_candidates": int(len(cal_t)),
            })

        # e-BH
        for strategy_name, ev in [("pertemplate", test_evals), ("global", test_evals_global)]:
            test_df["evalue"] = ev
            for alpha in alphas:
                s = ebh_summary(test_df, evalue_col="evalue", alpha=alpha)
                all_rows.append({
                    "seed": seed, "alpha": alpha, "strategy": strategy_name,
                    "rejections": s["total_rejections"], "fdp": s["fdp"],
                    "recall": s["recall"], "mean_family_fdp": s["mean_family_fdp"],
                })

    return pd.DataFrame(all_rows), pd.DataFrame(all_diag)


def plot_template_count_sweep(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.0, 4.4))
    sub = summary_df[summary_df["strategy"] == "pertemplate"].sort_values("template_count")
    ax1.plot(sub["mean_template_cal_families"], sub["fdp"], marker="o", label="FDP", color="#b2182b")
    ax1.axhline(0.10, color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.65)
    ax1.set_xlabel("Mean calibration families per template")
    ax1.set_ylabel("FDP", color="#b2182b")
    ax1.tick_params(axis="y", labelcolor="#b2182b")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(sub["mean_template_cal_families"], sub["recall"], marker="s", label="Recall", color="#2166ac")
    ax2.set_ylabel("Recall", color="#2166ac")
    ax2.tick_params(axis="y", labelcolor="#2166ac")

    for _, row in sub.iterrows():
        ax1.annotate(str(int(row["template_count"])), (row["mean_template_cal_families"], row["fdp"]), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument(
        "--template-counts",
        type=str,
        default="",
        help="Optional comma-separated template pool sizes, e.g. 7,50,100,200. Enables template-count sweep mode.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "templates",
        help="Directory containing freeform_50.txt/freeform_100.txt/freeform_200.txt.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    alphas = [0.05, 0.10, 0.15, 0.20]

    df = pd.read_csv(args.candidate_table)
    for col in ["is_tp", "is_null", "is_prompt_absent"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().eq("true")
    if args.template_counts.strip():
        counts = [int(item.strip()) for item in args.template_counts.split(",") if item.strip()]
        all_raw = []
        all_summary = []
        all_diag = []
        for count in counts:
            print(f"\n--- Template count: {count} ---")
            templates = load_template_pool(args.template_dir, count)
            df_count = apply_template_pool(df, templates)
            res_df, diag_df = run_one_config(
                df_count,
                seeds,
                alphas,
                estimator="hist",
                num_bins=80,
                smoothing=1.0,
                clip_phi=1_000_000.0,
                min_fit=20,
            )
            res_df["template_count"] = count
            diag_df["template_count"] = count
            all_raw.append(res_df)
            all_diag.append(diag_df)

            cal_fam_mean = float(diag_df.groupby("seed")["num_cal_families"].mean().mean()) if len(diag_df) else 0.0
            fit_fam_mean = float(diag_df.groupby("seed")["num_fit_families"].mean().mean()) if len(diag_df) else 0.0
            for strategy in ["pertemplate", "global"]:
                sub = res_df[(res_df["strategy"] == strategy) & (res_df["alpha"] == 0.10)]
                if len(sub) == 0:
                    continue
                all_summary.append(
                    {
                        "template_count": count,
                        "strategy": strategy,
                        "alpha": 0.10,
                        "mean_template_fit_families": fit_fam_mean,
                        "mean_template_cal_families": cal_fam_mean,
                        "rejections": float(sub["rejections"].mean()),
                        "fdp": float(sub["fdp"].mean()),
                        "recall": float(sub["recall"].mean()),
                        "mean_family_fdp": float(sub["mean_family_fdp"].mean()),
                        "phi_max_mean": float(diag_df["phi_max"].mean()) if len(diag_df) else 0.0,
                        "phi_max_p95": float(diag_df["phi_max"].quantile(0.95)) if len(diag_df) else 0.0,
                        "phi_max_max": float(diag_df["phi_max"].max()) if len(diag_df) else 0.0,
                    }
                )
        raw_df = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
        diag_df = pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame()
        summary_df = pd.DataFrame(all_summary)
        raw_df.to_csv(args.out_dir / "template_count_sweep_raw.csv", index=False)
        diag_df.to_csv(args.out_dir / "template_count_sweep_diagnostics.csv", index=False)
        summary_df.to_csv(args.out_dir / "template_count_sweep_results.csv", index=False)
        plot_template_count_sweep(summary_df, args.out_dir / "template_count_sweep_curve.pdf")
        print("\n========= TEMPLATE COUNT RESULTS at alpha=0.10 =========")
        print(summary_df.to_string(index=False))
        print(f"\nWrote template-count sweep to {args.out_dir}")
        return

    df["template"] = df.apply(normalize_template_from_category, axis=1)

    configs = [
        {"estimator": "hist", "num_bins": 40, "smoothing": 1.0, "label": "hist_40"},
        {"estimator": "hist", "num_bins": 80, "smoothing": 1.0, "label": "hist_80"},
        {"estimator": "hist", "num_bins": 160, "smoothing": 1.0, "label": "hist_160"},
        {"estimator": "hist", "num_bins": 80, "smoothing": 0.1, "label": "hist_80_sm01"},
        {"estimator": "isotonic", "num_bins": 80, "smoothing": 1.0, "label": "isotonic"},
        {"estimator": "logreg", "num_bins": 80, "smoothing": 1.0, "label": "logreg"},
    ]

    all_summary_rows = []
    all_diag_rows = []

    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n--- Config: {label} ---")
        res_df, diag_df = run_one_config(
            df, seeds, alphas,
            clip_phi=1_000_000.0, min_fit=50, **cfg,
        )
        cfg["label"] = label  # restore

        # Summarize at α=0.10
        for strategy in ["pertemplate", "global"]:
            sub = res_df[(res_df["strategy"] == strategy) & (res_df["alpha"] == 0.10)]
            if len(sub) == 0:
                continue
            all_summary_rows.append({
                "config": label,
                "strategy": strategy,
                "rejections": sub["rejections"].mean(),
                "fdp": sub["fdp"].mean(),
                "recall": sub["recall"].mean(),
                "mean_family_fdp": sub["mean_family_fdp"].mean(),
            })

        # φ_max diagnostic
        if len(diag_df) > 0:
            for tmpl, grp in diag_df[diag_df["own_model"]].groupby("template"):
                all_diag_rows.append({
                    "config": label,
                    "template": tmpl,
                    "phi_max_mean": grp["phi_max"].mean(),
                })

    summary_df = pd.DataFrame(all_summary_rows)
    diag_summary = pd.DataFrame(all_diag_rows)

    print("\n\n========= RESULTS at α=0.10 (mean over 20 seeds) =========")
    print(summary_df.to_string(index=False))

    print("\n========= PER-TEMPLATE φ_max (own models only) =========")
    if len(diag_summary) > 0:
        pivot = diag_summary.pivot(index="template", columns="config", values="phi_max_mean")
        print(pivot.to_string())

    summary_df.to_csv(args.out_dir / "estimator_sweep_results.csv", index=False)
    diag_summary.to_csv(args.out_dir / "estimator_sweep_phimax.csv", index=False)
    print(f"\nWrote to {args.out_dir}")


if __name__ == "__main__":
    main()
