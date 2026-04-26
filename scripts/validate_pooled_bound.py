#!/usr/bin/env python3
"""Validate a finite-sample pooled-FDP concentration bound.

The primary input is one or more ``family_strata_results.csv`` files produced
by ``run_formal_betting_pipeline.py``.  When the corresponding
``seed_*/test_candidates_with_evalues.csv`` files are present next to the
strata table, the script uses them to compute exact per-seed family-size
statistics.  Otherwise it falls back to stratum labels when possible.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FamilyStats:
    """Family-size statistics for one random split."""

    num_families: int
    k_min: float
    k_max: float
    k_sum: float
    k_sum_sq: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap empirical coverage of the pooled-FDP bound from "
            "family_strata_results.csv."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Path(s) to family_strata_results.csv.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="Nominal FDP level.")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Tail probability in the theory bound.",
    )
    parser.add_argument(
        "--method",
        default="betting",
        help="Method column value to keep.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("pooled_bound_validation"),
        help="Directory for the CSV table and plot.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10_000,
        help="Bootstrap replicates for coverage-rate confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--dep-factor",
        type=float,
        default=1.0,
        help=(
            "Multiplicative dependence factor chi. Use 1 for independent "
            "families; use a dependency-graph coloring number for weak "
            "dependence diagnostics."
        ),
    )
    parser.add_argument(
        "--rho-floor",
        type=float,
        default=1e-12,
        help="Numerical floor for observed rejection mass rho_R.",
    )
    parser.add_argument(
        "--kmax-for-open-stratum",
        type=float,
        default=None,
        help=(
            "Fallback upper bound for open strata such as K>30 when "
            "per-seed candidate files are unavailable."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write CSV outputs but skip the PNG figure.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: Iterable[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_seed_pooled_rows(path: Path, alpha: float, method: str) -> pd.DataFrame:
    strata = pd.read_csv(path)
    require_columns(
        strata,
        ["alpha", "seed", "method", "total_rejections", "fp"],
        path,
    )
    sub = strata[
        np.isclose(strata["alpha"].astype(float), alpha)
        & (strata["method"].astype(str) == method)
    ].copy()
    if sub.empty:
        raise ValueError(f"No rows found in {path} for alpha={alpha} method={method!r}.")

    agg_spec: dict[str, tuple[str, str]] = {
        "total_rejections": ("total_rejections", "sum"),
        "fp": ("fp", "sum"),
    }
    if "tp" in sub.columns:
        agg_spec["tp"] = ("tp", "sum")
    if "num_test_families" in sub.columns:
        agg_spec["num_test_families_from_strata"] = ("num_test_families", "sum")
    if "num_rejected_families" in sub.columns:
        agg_spec["num_rejected_families"] = ("num_rejected_families", "sum")
    if "total_pruned" in sub.columns:
        agg_spec["total_pruned"] = ("total_pruned", "sum")

    grouped = sub.groupby("seed", as_index=False).agg(**agg_spec)
    grouped["fdp_pooled"] = grouped["fp"] / grouped["total_rejections"].clip(lower=1)
    grouped["alpha"] = alpha
    grouped["method"] = method
    grouped["input"] = str(path)
    grouped["dataset"] = path.parent.name

    if "mean_family_fdp" in sub.columns and "num_test_families" in sub.columns:
        weighted = (
            sub.assign(_weighted_fdp=sub["mean_family_fdp"] * sub["num_test_families"])
            .groupby("seed")
            .agg(_weighted_fdp=("_weighted_fdp", "sum"), _families=("num_test_families", "sum"))
            .reset_index()
        )
        weighted["mean_family_fdp"] = weighted["_weighted_fdp"] / weighted["_families"].clip(lower=1)
        grouped = grouped.merge(weighted[["seed", "mean_family_fdp"]], on="seed", how="left")

    return grouped


def candidate_family_stats(root: Path, seed: int) -> FamilyStats | None:
    candidates = root / f"seed_{seed}" / "test_candidates_with_evalues.csv"
    if not candidates.exists():
        return None
    try:
        fam = pd.read_csv(candidates, usecols=["family_id", "family_size"])
    except ValueError:
        return None
    fam = fam.drop_duplicates("family_id")
    if fam.empty:
        return None
    sizes = fam["family_size"].astype(float).to_numpy()
    return FamilyStats(
        num_families=int(sizes.size),
        k_min=float(np.min(sizes)),
        k_max=float(np.max(sizes)),
        k_sum=float(np.sum(sizes)),
        k_sum_sq=float(np.sum(sizes * sizes)),
        source="candidate_files",
    )


def stratum_bounds(label: str, fallback_open_kmax: float | None) -> tuple[float, float] | None:
    label = str(label).strip()
    match = re.fullmatch(r"K<=([0-9.]+)", label)
    if match:
        return 1.0, float(match.group(1))
    match = re.fullmatch(r"([0-9.]+)<K<=([0-9.]+)", label)
    if match:
        lo = math.floor(float(match.group(1))) + 1.0
        return lo, float(match.group(2))
    match = re.fullmatch(r"K>([0-9.]+)", label)
    if match and fallback_open_kmax is not None:
        lo = math.floor(float(match.group(1))) + 1.0
        return lo, float(fallback_open_kmax)
    return None


def fallback_family_stats(
    strata_path: Path,
    seed: int,
    alpha: float,
    method: str,
    fallback_open_kmax: float | None,
) -> FamilyStats | None:
    strata = pd.read_csv(strata_path)
    if "stratum" not in strata.columns or "num_test_families" not in strata.columns:
        return None
    sub = strata[
        np.isclose(strata["alpha"].astype(float), alpha)
        & (strata["method"].astype(str) == method)
        & (strata["seed"].astype(int) == int(seed))
    ].copy()
    if sub.empty:
        return None

    total_families = 0
    k_min = math.inf
    k_max = 0.0
    k_sum = 0.0
    k_sum_sq = 0.0
    for _, row in sub.iterrows():
        bounds = stratum_bounds(str(row["stratum"]), fallback_open_kmax)
        if bounds is None:
            return None
        lo, hi = bounds
        n = int(row["num_test_families"])
        midpoint = 0.5 * (lo + hi)
        total_families += n
        k_min = min(k_min, lo)
        k_max = max(k_max, hi)
        k_sum += n * midpoint
        k_sum_sq += n * midpoint * midpoint

    if total_families <= 0 or not math.isfinite(k_min):
        return None
    return FamilyStats(
        num_families=total_families,
        k_min=float(k_min),
        k_max=float(k_max),
        k_sum=float(k_sum),
        k_sum_sq=float(k_sum_sq),
        source="stratum_midpoint_fallback",
    )


def add_bound_columns(
    rows: pd.DataFrame,
    strata_path: Path,
    delta: float,
    dep_factor: float,
    rho_floor: float,
    fallback_open_kmax: float | None,
) -> pd.DataFrame:
    enriched = rows.copy()
    stats_by_seed: dict[int, FamilyStats] = {}
    for seed in enriched["seed"].astype(int).tolist():
        stats = candidate_family_stats(strata_path.parent, seed)
        if stats is None:
            stats = fallback_family_stats(
                strata_path,
                seed,
                float(enriched["alpha"].iloc[0]),
                str(enriched["method"].iloc[0]),
                fallback_open_kmax,
            )
        if stats is not None:
            stats_by_seed[seed] = stats

    stat_rows = []
    for _, row in enriched.iterrows():
        seed = int(row["seed"])
        stats = stats_by_seed.get(seed)
        if stats is None:
            stat_rows.append(
                {
                    "num_families": np.nan,
                    "k_min": np.nan,
                    "k_max": np.nan,
                    "k_sum": np.nan,
                    "k_sum_sq": np.nan,
                    "family_stats_source": "missing",
                    "rho_R": np.nan,
                    "epsilon_theory": np.nan,
                    "epsilon_sequence": np.nan,
                    "covered_theory": np.nan,
                    "covered_sequence": np.nan,
                }
            )
            continue

        total_rejections = float(row["total_rejections"])
        rho = total_rejections / stats.k_sum if stats.k_sum > 0 else 0.0
        rho_eff = max(rho, rho_floor)
        log_term = math.log(1.0 / delta)
        epsilon_theory = (
            (stats.k_max / stats.k_min)
            * (1.0 / rho_eff)
            * math.sqrt(2.0 * dep_factor * log_term / stats.num_families)
        )
        epsilon_sequence = (
            (1.0 / rho_eff)
            * math.sqrt(2.0 * dep_factor * log_term * stats.k_sum_sq / (stats.k_sum * stats.k_sum))
        )
        threshold_theory = float(row["alpha"]) + epsilon_theory
        threshold_sequence = float(row["alpha"]) + epsilon_sequence
        fdp = float(row["fdp_pooled"])
        stat_rows.append(
            {
                "num_families": stats.num_families,
                "k_min": stats.k_min,
                "k_max": stats.k_max,
                "k_sum": stats.k_sum,
                "k_sum_sq": stats.k_sum_sq,
                "family_stats_source": stats.source,
                "rho_R": rho,
                "epsilon_theory": epsilon_theory,
                "epsilon_sequence": epsilon_sequence,
                "covered_theory": bool(fdp <= threshold_theory),
                "covered_sequence": bool(fdp <= threshold_sequence),
            }
        )

    return pd.concat([enriched.reset_index(drop=True), pd.DataFrame(stat_rows)], axis=1)


def bootstrap_rate(values: pd.Series, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    clean = values.dropna().astype(float).to_numpy()
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    observed = float(np.mean(clean))
    if n_bootstrap <= 0:
        return observed, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(n_bootstrap, clean.size), replace=True)
    boot = draws.mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, float(lo), float(hi)


def summarize(enriched: pd.DataFrame, delta: float, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for dataset, group in enriched.groupby("dataset", sort=False):
        cov, cov_lo, cov_hi = bootstrap_rate(group["covered_theory"], n_bootstrap, seed)
        cov_seq, cov_seq_lo, cov_seq_hi = bootstrap_rate(group["covered_sequence"], n_bootstrap, seed + 17)
        raw_pass, raw_lo, raw_hi = bootstrap_rate(
            (group["fdp_pooled"] <= group["alpha"]).astype(float),
            n_bootstrap,
            seed + 31,
        )
        rows.append(
            {
                "dataset": dataset,
                "input": group["input"].iloc[0],
                "alpha": float(group["alpha"].iloc[0]),
                "delta": delta,
                "target_coverage": 1.0 - delta,
                "num_splits": int(len(group)),
                "mean_num_families": float(group["num_families"].mean()),
                "mean_k_min": float(group["k_min"].mean()),
                "mean_k_max": float(group["k_max"].mean()),
                "mean_k_sum": float(group["k_sum"].mean()),
                "mean_rho_R": float(group["rho_R"].mean()),
                "mean_rejections": float(group["total_rejections"].mean()),
                "mean_pooled_fdp": float(group["fdp_pooled"].mean()),
                "sd_pooled_fdp": float(group["fdp_pooled"].std(ddof=1)),
                "raw_pass_rate": raw_pass,
                "raw_pass_boot_lo": raw_lo,
                "raw_pass_boot_hi": raw_hi,
                "mean_epsilon_theory": float(group["epsilon_theory"].mean()),
                "coverage_theory": cov,
                "coverage_theory_boot_lo": cov_lo,
                "coverage_theory_boot_hi": cov_hi,
                "mean_epsilon_sequence": float(group["epsilon_sequence"].mean()),
                "coverage_sequence": cov_seq,
                "coverage_sequence_boot_lo": cov_seq_lo,
                "coverage_sequence_boot_hi": cov_seq_hi,
                "family_stats_source": ",".join(sorted(set(group["family_stats_source"].astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def write_plot(summary: pd.DataFrame, enriched: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    x = np.arange(len(summary))
    labels = summary["dataset"].tolist()
    axes[0].bar(x - 0.2, summary["raw_pass_rate"], width=0.4, label="FDP <= alpha")
    axes[0].bar(x + 0.2, summary["coverage_theory"], width=0.4, label="FDP <= alpha + eps")
    axes[0].axhline(summary["target_coverage"].iloc[0], color="black", linestyle="--", linewidth=1.0, label="1 - delta")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("coverage rate")
    axes[0].set_title("Bootstrap coverage")
    axes[0].legend(fontsize=8)

    for i, dataset in enumerate(labels):
        group = enriched[enriched["dataset"] == dataset]
        jitter = np.linspace(-0.12, 0.12, max(len(group), 2))[: len(group)]
        axes[1].scatter(np.full(len(group), i) + jitter, group["fdp_pooled"], s=22, alpha=0.75)
        axes[1].hlines(group["alpha"].iloc[0], i - 0.25, i + 0.25, color="black", linewidth=1.0)
        threshold = group["alpha"].iloc[0] + min(float(group["epsilon_theory"].mean()), 1.0)
        axes[1].hlines(threshold, i - 0.25, i + 0.25, color="tab:red", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel("pooled FDP")
    axes[1].set_title("Per-split pooled FDP")
    axes[1].set_ylim(bottom=0)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.delta < 1.0):
        raise ValueError("--delta must lie in (0, 1).")
    if args.dep_factor < 1.0:
        raise ValueError("--dep-factor should be at least 1.")

    all_rows = []
    for input_path in args.inputs:
        input_path = input_path.resolve()
        rows = load_seed_pooled_rows(input_path, args.alpha, args.method)
        enriched = add_bound_columns(
            rows,
            input_path,
            args.delta,
            args.dep_factor,
            args.rho_floor,
            args.kmax_for_open_stratum,
        )
        all_rows.append(enriched)

    all_enriched = pd.concat(all_rows, ignore_index=True)
    summary = summarize(all_enriched, args.delta, args.bootstrap, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "pooled_bound_seed_details.csv"
    summary_path = args.out_dir / "pooled_bound_validation.csv"
    all_enriched.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    if not args.no_plot:
        write_plot(summary, all_enriched, args.out_dir / "pooled_bound_validation.png")

    display_cols = [
        "dataset",
        "num_splits",
        "mean_num_families",
        "mean_k_max",
        "mean_rho_R",
        "mean_pooled_fdp",
        "raw_pass_rate",
        "mean_epsilon_theory",
        "coverage_theory",
        "target_coverage",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {detail_path}")
    if not args.no_plot:
        print(f"Wrote {args.out_dir / 'pooled_bound_validation.png'}")


if __name__ == "__main__":
    main()
