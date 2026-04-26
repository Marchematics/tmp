#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_hallucination_fdr_matplotlib")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_formal_betting_pipeline import fit_density_ratio_model, split_families  # noqa: E402


DEFAULT_DATASETS = {
    "coco": PROJECT_ROOT / "outputs" / "coco_groundingdino_gonogo_1000_mix_analysis" / "candidate_table.csv",
    "voc": PROJECT_ROOT / "outputs" / "voc_1000_gdino_analysis" / "candidate_table.csv",
    "lvis": PROJECT_ROOT
    / "outputs"
    / "lvis_groundingdino_1000_presentall_absentraremix_analysis"
    / "candidate_table.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B3 robustness diagnostics: cross-family LOO KS/MMD and LOO boxplots.")
    parser.add_argument(
        "--cross-family-table",
        type=Path,
        default=DEFAULT_DATASETS["coco"],
        help="Candidate table used for pairwise cross-family LOO KS/MMD.",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "robustness_analysis")
    parser.add_argument("--cross-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--loo-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--num-bins", type=int, default=80)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--clip-phi", type=float, default=1_000_000.0)
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_candidate_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["is_tp", "is_null", "is_prompt_absent"]:
        if col in df.columns and df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def ks_2samp_stat_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    values = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, values, side="right") / max(len(x), 1)
    cdf_y = np.searchsorted(y, values, side="right") / max(len(y), 1)
    stat = float(np.max(np.abs(cdf_x - cdf_y))) if values.size else 0.0
    n_eff = len(x) * len(y) / max(len(x) + len(y), 1)
    # Asymptotic two-sample KS tail; conservative enough after Bonferroni and
    # much faster than exact p-values over hundreds of thousands of pairs.
    pvalue = min(1.0, 2.0 * math.exp(-2.0 * n_eff * stat * stat))
    return stat, pvalue


def median_bandwidth(values: np.ndarray, *, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 1.0
    rng = np.random.default_rng(seed + 991)
    sample = values
    if values.size > 2000:
        sample = values[rng.choice(values.size, size=2000, replace=False)]
    diffs = np.abs(sample[:, None] - sample[None, :])
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(max(np.median(diffs), 1e-6))


def rbf_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    kxx = np.exp(-gamma * (x - x.T) ** 2).mean()
    kyy = np.exp(-gamma * (y - y.T) ** 2).mean()
    kxy = np.exp(-gamma * (x - y.T) ** 2).mean()
    return float(max(kxx + kyy - 2.0 * kxy, 0.0))


def cal_null_loo_by_family(
    df: pd.DataFrame,
    *,
    seed: int,
    num_bins: int,
    smoothing: float,
    clip_phi: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    split = split_families(df, seed=seed, ratios=(0.6, 0.2, 0.2))
    df_seed = df.assign(split=split)
    fit_df = df_seed[df_seed["split"] == "fit"].copy()
    cal_df = df_seed[df_seed["split"] == "cal"].copy()
    model, _, ratio_summary = fit_density_ratio_model(
        fit_df,
        estimator="hist",
        num_bins=num_bins,
        smoothing=smoothing,
        clip_phi=clip_phi,
        max_logreg_iter=1000,
    )
    cal_null = cal_df[cal_df["is_null"]].copy()
    if cal_null.empty:
        raise ValueError(f"No calibration null candidates for seed {seed}")
    phi = model.phi(cal_null["score"].to_numpy(dtype=float))
    sum_phi = float(phi.sum())
    n_cal = int(phi.size)
    cal_null["phi"] = phi
    cal_null["loo_evalue"] = n_cal * phi / max(sum_phi, 1e-12)
    meta = {
        "seed": seed,
        "num_cal_null": n_cal,
        "sum_phi_cal": sum_phi,
        "mean_loo": float(cal_null["loo_evalue"].mean()),
        "max_phi": float(ratio_summary.get("max_phi", np.nan)),
    }
    return cal_null, meta


def compute_cross_family_ks_mmd(
    df: pd.DataFrame,
    *,
    seeds: list[int],
    num_bins: int,
    smoothing: float,
    clip_phi: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    seed_meta: list[dict[str, float | int]] = []
    for seed in seeds:
        cal_null, meta = cal_null_loo_by_family(
            df,
            seed=seed,
            num_bins=num_bins,
            smoothing=smoothing,
            clip_phi=clip_phi,
        )
        seed_meta.append(meta)
        groups = {
            str(fam): group["loo_evalue"].to_numpy(dtype=float)
            for fam, group in cal_null.groupby("family_id", sort=False)
            if len(group) > 0
        }
        bandwidth = median_bandwidth(cal_null["loo_evalue"].to_numpy(dtype=float), seed=seed)
        for left, right in combinations(groups, 2):
            x = groups[left]
            y = groups[right]
            ks_stat, ks_pvalue = ks_2samp_stat_pvalue(x, y)
            rows.append(
                {
                    "seed": seed,
                    "family_left": left,
                    "family_right": right,
                    "n_left": int(x.size),
                    "n_right": int(y.size),
                    "mean_left": float(np.mean(x)),
                    "mean_right": float(np.mean(y)),
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "mmd2_rbf": rbf_mmd2(x, y, bandwidth),
                    "rbf_bandwidth": bandwidth,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        bonf = max(len(out), 1)
        out["ks_pvalue_bonferroni"] = np.minimum(out["ks_pvalue"] * bonf, 1.0)
        out["ks_reject_bonferroni_0p05"] = out["ks_pvalue_bonferroni"] <= 0.05
    return out, pd.DataFrame(seed_meta)


def plot_cross_family(df: pd.DataFrame, meta_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].hist(df["ks_stat"], bins=40, color="#4c78a8", alpha=0.85)
    reject_rate = float(df["ks_reject_bonferroni_0p05"].mean()) if "ks_reject_bonferroni_0p05" in df else 0.0
    axes[0].set_title(f"Two-sample KS, Bonferroni reject={reject_rate:.3f}")
    axes[0].set_xlabel("KS statistic")
    axes[0].set_ylabel("Family-pair count")
    axes[0].grid(alpha=0.25)

    axes[1].hist(df["mmd2_rbf"], bins=40, color="#59a14f", alpha=0.85)
    axes[1].set_title(f"RBF MMD$^2$, LOO mean={meta_df['mean_loo'].mean():.3f}")
    axes[1].set_xlabel("MMD$^2$")
    axes[1].set_ylabel("Family-pair count")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def compute_loo_distribution(
    datasets: dict[str, Path],
    *,
    seeds: list[int],
    num_bins: int,
    smoothing: float,
    clip_phi: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dataset, path in datasets.items():
        if not path.exists():
            continue
        df = load_candidate_table(path)
        for seed in seeds:
            cal_null, meta = cal_null_loo_by_family(
                df,
                seed=seed,
                num_bins=num_bins,
                smoothing=smoothing,
                clip_phi=clip_phi,
            )
            family_stats = (
                cal_null.groupby("family_id", sort=False)["loo_evalue"]
                .agg(["mean", "std", "count", "max"])
                .reset_index()
            )
            for row in family_stats.to_dict("records"):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "family_id": row["family_id"],
                        "mean_loo_evalue": float(row["mean"]),
                        "std_loo_evalue": float(row["std"]) if not pd.isna(row["std"]) else 0.0,
                        "num_cal_null": int(row["count"]),
                        "max_loo_evalue": float(row["max"]),
                        "global_mean_loo": float(meta["mean_loo"]),
                        "max_phi": float(meta["max_phi"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_loo_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    labels = []
    values = []
    for dataset, group in df.groupby("dataset", sort=False):
        labels.append(str(dataset).upper())
        values.append(group["mean_loo_evalue"].to_numpy(dtype=float))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label="LOO mean = 1")
    ax.set_ylabel("Per-family mean LOO e-value")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cross_df = load_candidate_table(args.cross_family_table)
    cross_rows, cross_meta = compute_cross_family_ks_mmd(
        cross_df,
        seeds=parse_ints(args.cross_seeds),
        num_bins=args.num_bins,
        smoothing=args.smoothing,
        clip_phi=args.clip_phi,
    )
    cross_rows.to_csv(args.out_dir / "cross_family_ks_mmd.csv", index=False)
    cross_meta.to_csv(args.out_dir / "cross_family_ks_mmd_seed_summary.csv", index=False)
    plot_cross_family(cross_rows, cross_meta, args.out_dir / "cross_family_ks_mmd.pdf")

    loo_df = compute_loo_distribution(
        DEFAULT_DATASETS,
        seeds=parse_ints(args.loo_seeds),
        num_bins=args.num_bins,
        smoothing=args.smoothing,
        clip_phi=args.clip_phi,
    )
    loo_df.to_csv(args.out_dir / "loo_per_family_distribution.csv", index=False)
    plot_loo_distribution(loo_df, args.out_dir / "loo_per_family_distribution.pdf")

    summary = {
        "cross_family_table": str(args.cross_family_table),
        "cross_family_rows": int(len(cross_rows)),
        "cross_family_bonferroni_reject_rate_0p05": float(cross_rows["ks_reject_bonferroni_0p05"].mean())
        if not cross_rows.empty
        else 0.0,
        "loo_distribution_rows": int(len(loo_df)),
        "loo_dataset_means": loo_df.groupby("dataset")["mean_loo_evalue"].mean().to_dict(),
    }
    (args.out_dir / "robustness_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
