#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_hallucination_fdr_matplotlib")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ovd_hallucination_fdr.evalues import e_bh  # noqa: E402
from ovd_hallucination_fdr.io import ensure_dir, write_json  # noqa: E402


DEFAULT_ALPHAS = "0.01,0.02,0.05,0.1,0.15,0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline oracle betting-e-value pilot. Fits a null-vs-TP score density "
            "ratio on one family split, evaluates e-BH on the held-out families."
        )
    )
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=str, default=DEFAULT_ALPHAS)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--num-bins", type=int, default=80)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--clip-evalue", type=float, default=1_000_000.0)
    parser.add_argument("--seed-salt", type=str, default="betting_oracle_v1")
    return parser.parse_args()


def stable_unit_interval(value: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}::{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def parse_alphas(text: str) -> list[float]:
    alphas = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not alphas:
        raise ValueError("At least one alpha is required")
    for alpha in alphas:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
    return sorted(set(alphas))


def make_quantile_bins(scores: np.ndarray, num_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(scores, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        lo = float(np.min(scores))
        hi = float(np.max(scores))
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        edges = np.linspace(lo, hi, min(num_bins, 10) + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def fit_density_ratio(
    fit_df: pd.DataFrame,
    *,
    num_bins: int,
    smoothing: float,
    clip_evalue: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float]]:
    null_scores = fit_df.loc[fit_df["is_null"], "score"].to_numpy(dtype=float)
    tp_scores = fit_df.loc[fit_df["is_tp"], "score"].to_numpy(dtype=float)
    if len(null_scores) == 0 or len(tp_scores) == 0:
        raise ValueError("Fit split must contain both null and true-positive scores")

    edges = make_quantile_bins(fit_df["score"].to_numpy(dtype=float), num_bins)
    null_counts, _ = np.histogram(null_scores, bins=edges)
    tp_counts, _ = np.histogram(tp_scores, bins=edges)
    bins = len(edges) - 1

    p0 = (null_counts.astype(float) + smoothing) / (len(null_scores) + smoothing * bins)
    p1 = (tp_counts.astype(float) + smoothing) / (len(tp_scores) + smoothing * bins)
    raw_ratio = p1 / p0
    clipped_ratio = np.minimum(raw_ratio, clip_evalue)

    # Clipping can only reduce E_0[e]. Keep the explicit number for reporting.
    fit_null_mean = float(np.sum(p0 * clipped_ratio))

    finite_left = edges[:-1].copy()
    finite_right = edges[1:].copy()
    finite_left[0] = float(np.min(fit_df["score"]))
    finite_right[-1] = float(np.max(fit_df["score"]))
    bin_table = pd.DataFrame(
        {
            "bin": np.arange(bins),
            "left": finite_left,
            "right": finite_right,
            "null_count": null_counts,
            "tp_count": tp_counts,
            "p0": p0,
            "p1": p1,
            "raw_evalue": raw_ratio,
            "evalue": clipped_ratio,
        }
    )
    summary = {
        "num_fit_null": int(len(null_scores)),
        "num_fit_tp": int(len(tp_scores)),
        "num_bins_realized": int(bins),
        "max_evalue": float(np.max(clipped_ratio)),
        "fit_null_evalue_mean": fit_null_mean,
    }
    return edges, clipped_ratio, bin_table, summary


def assign_evalues(scores: np.ndarray, edges: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    bin_ids = np.searchsorted(edges, scores, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, len(ratio) - 1)
    return ratio[bin_ids]


def run_ebh_curve(test_df: pd.DataFrame, alphas: list[float]) -> pd.DataFrame:
    grouped = [(family_id, group.copy()) for family_id, group in test_df.groupby("family_id", sort=False)]
    total_tp = int(test_df["is_tp"].sum())
    rows: list[dict[str, float | int]] = []
    for alpha in alphas:
        total_rejections = 0
        total_fp = 0
        total_tp_rej = 0
        family_fdps: list[float] = []
        rejected_families = 0
        for _, group in grouped:
            rejected_pos = e_bh(group["betting_evalue"].to_numpy(dtype=float), alpha)
            if not rejected_pos:
                family_fdps.append(0.0)
                continue
            selected = group.iloc[rejected_pos]
            fp = int(selected["is_null"].sum())
            tp = int(selected["is_tp"].sum())
            total_rejections += len(selected)
            total_fp += fp
            total_tp_rej += tp
            family_fdps.append(fp / max(len(selected), 1))
            rejected_families += 1

        rows.append(
            {
                "alpha": alpha,
                "num_test_families": len(grouped),
                "num_rejected_families": rejected_families,
                "total_rejections": total_rejections,
                "fp": total_fp,
                "tp": total_tp_rej,
                "fdp": total_fp / max(total_rejections, 1),
                "recall": total_tp_rej / max(total_tp, 1),
                "mean_family_fdp": float(np.mean(family_fdps)) if family_fdps else 0.0,
            }
        )
    return pd.DataFrame(rows)


def family_size_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    sizes = df.groupby("family_id").size().astype(int)
    quantiles = sizes.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    summary = {
        "num_families": int(len(sizes)),
        "mean": float(sizes.mean()),
        "std": float(sizes.std(ddof=0)),
        "min": int(sizes.min()),
        "p10": float(quantiles.loc[0.1]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.9]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": int(sizes.max()),
        "num_gt_30": int((sizes > 30).sum()),
    }
    table = sizes.rename("family_size").reset_index()
    return table, summary


def prompt_null_scale(df: pd.DataFrame) -> pd.DataFrame:
    null_df = df[df["is_null"]].copy()
    grouped = null_df.groupby("prompt")["score"]
    table = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
    q = grouped.quantile([0.5, 0.9, 0.99]).unstack()
    q.columns = ["q50", "q90", "q99"]
    table = table.merge(q.reset_index(), on="prompt", how="left")
    table = table.sort_values(["mean", "count"], ascending=[False, False])
    return table


def plot_curve(curve: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(curve["alpha"], curve["total_rejections"], marker="o", color="#2166ac", label="Rejections")
    ax1.set_xlabel("e-BH alpha")
    ax1.set_ylabel("Total rejections", color="#2166ac")
    ax1.tick_params(axis="y", labelcolor="#2166ac")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(curve["alpha"], curve["fdp"], marker="s", color="#b2182b", label="FDP")
    ax2.plot(curve["alpha"], curve["recall"], marker="^", color="#1b7837", label="Recall")
    ax2.set_ylabel("FDP / recall")
    ax2.set_ylim(-0.03, 1.03)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ratio(bin_table: pd.DataFrame, out_path: Path) -> None:
    finite = bin_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["left", "right"]).copy()
    finite["mid"] = 0.5 * (finite["left"] + finite["right"])
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(finite["mid"], finite["evalue"], color="#2166ac")
    ax.set_yscale("log")
    ax.set_xlabel("Score bin midpoint")
    ax.set_ylabel("Betting e-value, log scale")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_family_sizes(family_sizes: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(family_sizes["family_size"], bins=50, color="#2166ac", alpha=0.8)
    ax.set_xlabel("Candidate count per image-query family")
    ax.set_ylabel("Family count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    alphas = parse_alphas(args.alphas)

    df = pd.read_csv(args.candidate_table)
    for col in ["is_tp", "is_null", "is_prompt_absent"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().eq("true")

    family_keys = df["family_id"].drop_duplicates()
    fit_families = {
        family_id
        for family_id in family_keys
        if stable_unit_interval(str(family_id), args.seed_salt) < args.fit_fraction
    }
    df["oracle_split"] = np.where(df["family_id"].isin(fit_families), "fit", "test")
    fit_df = df[df["oracle_split"] == "fit"].copy()
    test_df = df[df["oracle_split"] == "test"].copy()

    edges, ratio, bin_table, ratio_summary = fit_density_ratio(
        fit_df,
        num_bins=args.num_bins,
        smoothing=args.smoothing,
        clip_evalue=args.clip_evalue,
    )
    test_df["betting_evalue"] = assign_evalues(test_df["score"].to_numpy(dtype=float), edges, ratio)

    curve = run_ebh_curve(test_df, alphas)
    family_sizes, family_summary = family_size_summary(df)
    prompt_scale = prompt_null_scale(df)

    summary = {
        "candidate_table": str(args.candidate_table),
        "fit_fraction": args.fit_fraction,
        "num_total_candidates": int(len(df)),
        "num_fit_candidates": int(len(fit_df)),
        "num_test_candidates": int(len(test_df)),
        "num_total_families": int(df["family_id"].nunique()),
        "num_fit_families": int(fit_df["family_id"].nunique()),
        "num_test_families": int(test_df["family_id"].nunique()),
        "num_test_tp": int(test_df["is_tp"].sum()),
        "num_test_null": int(test_df["is_null"].sum()),
        "density_ratio": ratio_summary,
        "family_size": family_summary,
        "best_alpha_by_rejections_at_fdp_le_alpha": None,
    }
    valid = curve[curve["fdp"] <= curve["alpha"]]
    if len(valid) > 0:
        best = valid.sort_values(["total_rejections", "alpha"], ascending=[False, True]).iloc[0]
        summary["best_alpha_by_rejections_at_fdp_le_alpha"] = {
            "alpha": float(best["alpha"]),
            "total_rejections": int(best["total_rejections"]),
            "fdp": float(best["fdp"]),
            "recall": float(best["recall"]),
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
        }

    write_json(args.out_dir / "summary.json", summary)
    bin_table.to_csv(args.out_dir / "density_ratio_bins.csv", index=False)
    test_df.to_csv(args.out_dir / "test_candidates_with_betting_evalues.csv", index=False)
    curve.to_csv(args.out_dir / "ebh_oracle_curve.csv", index=False)
    family_sizes.to_csv(args.out_dir / "family_sizes.csv", index=False)
    prompt_scale.to_csv(args.out_dir / "prompt_null_score_scale.csv", index=False)
    plot_curve(curve, args.out_dir / "ebh_oracle_curve.png")
    plot_ratio(bin_table, args.out_dir / "density_ratio_evalues.png")
    plot_family_sizes(family_sizes, args.out_dir / "family_size_hist.png")

    print(f"Wrote betting oracle analysis to {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
