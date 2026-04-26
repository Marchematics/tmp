#!/usr/bin/env python3
"""Controlled contamination sweep: inject η fraction of mislabeled nulls into
a clean candidate table, run the formal betting pipeline for each η, and
evaluate against original (clean) ground truth.

This validates Theorem 2 (contamination saturation) and Corollary 1
(power-collapse phase transition) in a controlled setting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_hallucination_fdr_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ovd_hallucination_fdr.evalues import e_bh  # noqa: E402
from ovd_hallucination_fdr.io import ensure_dir, write_json  # noqa: E402

# Re-use pipeline utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_formal_betting_pipeline import (  # noqa: E402
    DensityRatioModel,
    compute_self_normalized_evalues,
    fit_density_ratio_model,
    make_quantile_bins,
    split_families,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controlled contamination sweep")
    p.add_argument("--candidate-table", type=Path, required=True,
                    help="Path to clean (exhaustive-annotation) candidate_table.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--etas", type=str,
                    default="0.0,0.05,0.10,0.15,0.20,0.30,0.40,0.50",
                    help="Comma-separated contamination fractions")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                    help="Seeds for both contamination randomness and split randomness")
    p.add_argument("--split-ratios", type=str, default="0.6,0.2,0.2")
    p.add_argument("--estimator", type=str, default="hist")
    p.add_argument("--num-bins", type=int, default=80)
    p.add_argument("--smoothing", type=float, default=0.0)
    p.add_argument("--clip-phi", type=float, default=1_000_000.0)
    return p.parse_args()


def contaminate_table(
    df: pd.DataFrame,
    eta: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flip eta fraction of present families to 'absent', simulating missing annotations.

    Returns (contaminated_df, original_labels_df, eta_actual).
    original_labels_df has columns [family_id, candidate_idx, orig_is_tp, orig_is_null].
    eta_actual = fraction of null pool that are truly TP (for comparing with Theorem 2).
    """
    # Save original labels
    orig = df[["family_id", "candidate_idx", "is_tp", "is_null"]].copy()
    orig.columns = ["family_id", "candidate_idx", "orig_is_tp", "orig_is_null"]

    if eta <= 0.0:
        return df.copy(), orig, 0.0

    # Find present families (those with is_prompt_absent=False)
    present_families = df.loc[~df["is_prompt_absent"], "family_id"].unique()
    n_flip = max(1, int(round(eta * len(present_families))))
    flip_families = set(rng.choice(present_families, size=n_flip, replace=False))

    contaminated = df.copy()
    mask = contaminated["family_id"].isin(flip_families)

    # Count true TPs that will be mislabeled as null
    n_tp_flipped = int(df.loc[mask, "is_tp"].sum())
    # After contamination, total null candidates = original nulls + all flipped candidates
    n_null_after = int(df["is_null"].sum()) + int(mask.sum())
    # η_actual = fraction of null pool that are truly TP
    eta_actual = n_tp_flipped / max(n_null_after, 1)

    contaminated.loc[mask, "is_prompt_absent"] = True
    contaminated.loc[mask, "is_tp"] = False
    contaminated.loc[mask, "is_null"] = True
    contaminated.loc[mask, "gt_count"] = 0

    return contaminated, orig, eta_actual


def run_single(
    df: pd.DataFrame,
    orig_labels: pd.DataFrame,
    *,
    seed: int,
    alpha: float,
    split_ratios: tuple[float, float, float],
    estimator: str,
    num_bins: int,
    smoothing: float,
    clip_phi: float,
) -> dict:
    """Run formal pipeline on contaminated table, evaluate against original labels."""
    split = split_families(df, seed=seed, ratios=split_ratios)
    df = df.copy()
    df["split"] = split

    fit_df = df[df["split"] == "fit"].copy()
    cal_df = df[df["split"] == "cal"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Fit density ratio on contaminated data
    model, bin_table, ratio_summary = fit_density_ratio_model(
        fit_df,
        estimator=estimator,
        num_bins=num_bins,
        smoothing=smoothing,
        clip_phi=clip_phi,
        max_logreg_iter=1000,
    )

    phi_max = ratio_summary.get("max_phi", 0.0)

    # Calibration
    cal_null_scores = cal_df.loc[cal_df["is_null"], "score"].to_numpy(dtype=float)
    if cal_null_scores.size == 0:
        return {"phi_max": phi_max, "fdp_contaminated": 0, "fdp_true": 0,
                "recall_true": 0, "rejections": 0, "error": "no_cal_null"}

    cal_null_phi = model.phi(cal_null_scores)
    sum_phi_cal = float(np.sum(cal_null_phi))
    n_cal = int(cal_null_phi.size)

    if sum_phi_cal <= 0:
        return {"phi_max": phi_max, "fdp_contaminated": 0, "fdp_true": 0,
                "recall_true": 0, "rejections": 0, "error": "zero_phi_sum"}

    # Test e-values
    test_scores = test_df["score"].to_numpy(dtype=float)
    test_phi = model.phi(test_scores)
    test_df = test_df.copy()
    test_df["betting_evalue"] = compute_self_normalized_evalues(test_phi, sum_phi_cal, n_cal)

    # e-BH per family, collect rejections
    total_rejections = 0
    total_fp_contaminated = 0
    total_fp_true = 0
    total_tp_true = 0
    rejected_indices = []

    for family_id, group in test_df.groupby("family_id", sort=False):
        rejected_pos = e_bh(group["betting_evalue"].to_numpy(dtype=float), alpha)
        if not rejected_pos:
            continue
        selected = group.iloc[rejected_pos]
        total_rejections += len(selected)
        # Contaminated metric (what pipeline sees)
        total_fp_contaminated += int(selected["is_null"].sum())
        rejected_indices.extend(selected.index.tolist())

    # Now evaluate against ORIGINAL labels
    if rejected_indices:
        rejected_df = test_df.loc[rejected_indices].copy()
        # Merge with original labels
        rejected_merged = rejected_df.merge(
            orig_labels, on=["family_id", "candidate_idx"], how="left"
        )
        total_fp_true = int(rejected_merged["orig_is_null"].sum())
        total_tp_true = int(rejected_merged["orig_is_tp"].sum())

    # Total TP in test set (using original labels)
    test_merged = test_df.merge(orig_labels, on=["family_id", "candidate_idx"], how="left")
    total_tp_in_test = int(test_merged["orig_is_tp"].sum())

    # LOO validity
    loo_evalues = n_cal * cal_null_phi / max(sum_phi_cal, 1e-12)
    mean_loo = float(np.mean(loo_evalues))

    return {
        "phi_max": phi_max,
        "n_cal_null": n_cal,
        "sum_phi_cal": sum_phi_cal,
        "rejections": total_rejections,
        "fp_contaminated": total_fp_contaminated,
        "fp_true": total_fp_true,
        "tp_true": total_tp_true,
        "fdp_contaminated": total_fp_contaminated / max(total_rejections, 1),
        "fdp_true": total_fp_true / max(total_rejections, 1),
        "recall_true": total_tp_true / max(total_tp_in_test, 1),
        "total_tp_in_test": total_tp_in_test,
        "mean_loo_evalue": mean_loo,
    }


def theoretical_phi_max_ceiling(eta: float) -> float:
    """Theorem 2: phi_max <= 1/eta under contamination."""
    if eta <= 0:
        return float("inf")
    return 1.0 / eta


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    etas = [float(x.strip()) for x in args.etas.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    ratios_parts = [float(x.strip()) for x in args.split_ratios.split(",")]
    total = sum(ratios_parts)
    split_ratios = tuple(x / total for x in ratios_parts)

    df_clean = pd.read_csv(args.candidate_table)
    for col in ["is_tp", "is_null", "is_prompt_absent"]:
        if df_clean[col].dtype != bool:
            df_clean[col] = df_clean[col].astype(str).str.lower().eq("true")

    n_present = int((~df_clean["is_prompt_absent"]).groupby(df_clean["family_id"]).first().sum())
    print(f"Clean table: {len(df_clean)} candidates, "
          f"{df_clean['family_id'].nunique()} families, "
          f"{n_present} present families")

    all_rows = []

    for eta in etas:
        print(f"\n--- eta = {eta:.2f} ---")
        for seed in seeds:
            rng = np.random.default_rng(seed + 42)
            contaminated_df, orig_labels, eta_actual = contaminate_table(df_clean, eta, rng)

            result = run_single(
                contaminated_df,
                orig_labels,
                seed=seed,
                alpha=args.alpha,
                split_ratios=split_ratios,
                estimator=args.estimator,
                num_bins=args.num_bins,
                smoothing=args.smoothing,
                clip_phi=args.clip_phi,
            )
            row = {"eta": eta, "eta_actual": eta_actual, "seed": seed, "alpha": args.alpha, **result}
            all_rows.append(row)
            print(f"  seed={seed}: eta_actual={eta_actual:.4f}, phi_max={result['phi_max']:.1f}, "
                  f"rej={result['rejections']}, "
                  f"fdp_true={result['fdp_true']:.3f}, "
                  f"recall={result['recall_true']:.3f}")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(args.out_dir / "contamination_sweep_raw.csv", index=False)

    # Aggregate over seeds
    agg = results_df.groupby("eta").agg(
        eta_actual_mean=("eta_actual", "mean"),
        eta_actual_std=("eta_actual", "std"),
        phi_max_mean=("phi_max", "mean"),
        phi_max_std=("phi_max", "std"),
        phi_max_median=("phi_max", "median"),
        fdp_true_mean=("fdp_true", "mean"),
        fdp_true_std=("fdp_true", "std"),
        fdp_contaminated_mean=("fdp_contaminated", "mean"),
        recall_mean=("recall_true", "mean"),
        recall_std=("recall_true", "std"),
        rejections_mean=("rejections", "mean"),
        rejections_std=("rejections", "std"),
        mean_loo_mean=("mean_loo_evalue", "mean"),
    ).reset_index()
    agg["theoretical_phi_ceiling"] = agg["eta_actual_mean"].apply(theoretical_phi_max_ceiling)
    agg.to_csv(args.out_dir / "contamination_sweep_summary.csv", index=False)
    print("\n=== Summary ===")
    print(agg.to_string(index=False))

    # --- Generate plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Use eta_actual (true contamination fraction of null pool) for x-axis
    eta_vals = agg["eta_actual_mean"].values

    # Panel (a): phi_max vs eta_actual with theoretical ceiling
    ax = axes[0]
    ax.plot(eta_vals, agg["phi_max_mean"], "o-",
            color="#2166ac", label=r"Empirical $\phi_{\max}$")
    # Theoretical ceiling (only for eta > 0)
    eta_pos = eta_vals[eta_vals > 0]
    if len(eta_pos) > 0:
        eta_theory = np.linspace(min(eta_pos) * 0.5, max(eta_pos) * 1.2, 200)
        ax.plot(eta_theory, 1.0 / eta_theory, "--", color="#b2182b",
                label=r"Theorem 2: $1/\eta$", linewidth=2)
    ax.set_xlabel(r"Null-pool contamination $\eta_{\mathrm{actual}}$", fontsize=12)
    ax.set_ylabel(r"$\phi_{\max}$", fontsize=12)
    ax.set_title(r"(a) Density-ratio peak vs contamination", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    # Panel (b): FDP vs eta_actual with alpha line
    ax = axes[1]
    ax.plot(eta_vals, agg["fdp_true_mean"], "D-",
            color="#b2182b", label="True FDP")
    ax.axhline(y=args.alpha, color="gray", linestyle="--", linewidth=1.5,
               label=rf"$\alpha = {args.alpha}$")
    ax.set_xlabel(r"Null-pool contamination $\eta_{\mathrm{actual}}$", fontsize=12)
    ax.set_ylabel("FDP (true labels)", fontsize=12)
    ax.set_title("(b) FDP vs contamination", fontsize=12)
    ax.set_ylim(-0.02, min(1.0, max(agg["fdp_true_mean"]) * 1.5 + 0.05))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    # Panel (c): Recall vs eta_actual
    ax = axes[2]
    ax.plot(eta_vals, agg["recall_mean"], "o-",
            color="#4daf4a", label="Recall (true labels)")
    ax.set_xlabel(r"Null-pool contamination $\eta_{\mathrm{actual}}$", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("(c) Power collapse", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out_dir / "contamination_sweep.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / "contamination_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlots saved to {args.out_dir}/contamination_sweep.{{pdf,png}}")

    # Save config
    write_json(args.out_dir / "config.json", {
        "candidate_table": str(args.candidate_table),
        "etas": etas,
        "alpha": args.alpha,
        "seeds": seeds,
        "estimator": args.estimator,
        "num_bins": args.num_bins,
        "smoothing": args.smoothing,
        "n_present_families": n_present,
    })


if __name__ == "__main__":
    main()
