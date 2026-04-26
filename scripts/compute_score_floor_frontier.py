#!/usr/bin/env python3
"""EXP2: Score-floor frontier — theory-predicted vs actual recall curve.

For each score floor s_floor, computes:
  1. Theoretical ceiling: count TPs with score >= s_floor AND phi >= T_1(K_eff)
     where K_eff = family size after pruning candidates below s_floor.
  2. Actual e-BH recall from the pipeline output.

Produces a figure: "s_floor vs recall" with theory-predicted ceiling and actual e-BH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_hallucination_fdr_matplotlib")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ovd_hallucination_fdr.evalues import e_bh  # noqa: E402


def reconstruct_phi(scores: np.ndarray, bins_df: pd.DataFrame) -> np.ndarray:
    edges = np.concatenate([bins_df["left"].values, [bins_df["right"].values[-1]]])
    phi_vals = bins_df["phi"].values
    bin_ids = np.searchsorted(edges, scores, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, len(phi_vals) - 1)
    return phi_vals[bin_ids]


def activation_threshold(K: int, C_m: float, m: int, alpha: float, r: int = 1) -> float:
    denom = alpha * r * (m + 1) - K
    if denom <= 0:
        return float("inf")
    return K * C_m / denom


def compute_self_normalized_evalues(phi_vals, sum_phi_cal, n_cal):
    denom = phi_vals + sum_phi_cal
    out = np.zeros_like(phi_vals, dtype=float)
    valid = denom > 0
    out[valid] = (n_cal + 1) * phi_vals[valid] / denom[valid]
    return out


def process_seed(pipeline_dir: Path, seed: int, alpha: float,
                 seed_meta: dict, score_floors: list[float],
                 original_total_tp: int | None = None) -> list[dict]:
    seed_dir = pipeline_dir / f"seed_{seed}"
    bins_df = pd.read_csv(seed_dir / "density_ratio_bins.csv")
    test_df = pd.read_csv(seed_dir / "test_candidates_with_evalues.csv")

    for col in ["is_tp", "is_null"]:
        if test_df[col].dtype != bool:
            test_df[col] = test_df[col].astype(str).str.lower().eq("true")

    C_m = seed_meta["sum_phi_cal"]
    m = seed_meta["num_cal_null"]

    scores = test_df["score"].to_numpy(dtype=float)
    phi = reconstruct_phi(scores, bins_df)
    test_df["phi"] = phi

    # Use original total TP count if provided (for pre-filter comparison)
    total_tp_all = original_total_tp if original_total_tp else int(test_df["is_tp"].sum())

    # Compute e-values (same as pipeline)
    sum_phi_cal = C_m
    n_cal = m
    test_df["evalue"] = compute_self_normalized_evalues(phi, sum_phi_cal, n_cal)

    rows = []
    for sf in score_floors:
        # --- Theory: T1-ceiling after floor pruning ---
        n_tp_reachable = 0
        n_tp_surviving = 0
        # --- Actual: e-BH after floor pruning ---
        n_tp_rejected = 0
        n_fp_rejected = 0
        n_total_rejected = 0

        for fam_id, group in test_df.groupby("family_id", sort=False):
            if sf > 0:
                group = group[group["score"] >= sf]
            if group.empty:
                continue

            K_eff = len(group)
            is_tp = group["is_tp"].to_numpy(dtype=bool)
            phi_fam = group["phi"].to_numpy(dtype=float)

            # Theory ceiling: T_1 for effective K
            T1 = activation_threshold(K_eff, C_m, m, alpha, 1)
            n_tp_surviving += int(is_tp.sum())
            n_tp_reachable += int((is_tp & (phi_fam >= T1)).sum())

            # Actual e-BH
            evalues = group["evalue"].to_numpy(dtype=float)
            rejected_pos = e_bh(evalues, alpha)
            if rejected_pos:
                selected = group.iloc[rejected_pos]
                n_tp_rejected += int(selected["is_tp"].sum())
                n_fp_rejected += int(selected["is_null"].sum())
                n_total_rejected += len(selected)

        recall_ceiling = n_tp_reachable / max(total_tp_all, 1)
        recall_actual = n_tp_rejected / max(total_tp_all, 1)
        fdp = n_fp_rejected / max(n_total_rejected, 1)

        rows.append({
            "seed": seed,
            "score_floor": sf,
            "n_tp_surviving": n_tp_surviving,
            "n_tp_reachable": n_tp_reachable,
            "n_tp_rejected": n_tp_rejected,
            "n_total_rejected": n_total_rejected,
            "recall_ceiling": recall_ceiling,
            "recall_actual": recall_actual,
            "fdp": fdp,
            "efficiency": recall_actual / max(recall_ceiling, 1e-12),
            "total_tp_all": total_tp_all,
        })

    return rows


def plot_frontier(df: pd.DataFrame, out_path: Path, alpha: float, title: str) -> None:
    mean = df.groupby("score_floor").agg(
        recall_ceiling=("recall_ceiling", "mean"),
        recall_actual=("recall_actual", "mean"),
        recall_ceiling_std=("recall_ceiling", "std"),
        recall_actual_std=("recall_actual", "std"),
        efficiency=("efficiency", "mean"),
        fdp=("fdp", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: recall curves
    ax = axes[0]
    sf = mean["score_floor"]
    ax.plot(sf, mean["recall_ceiling"] * 100, "s--", color="#2ca02c", label="T₁ ceiling (theory)", markersize=5)
    ax.fill_between(sf,
                    (mean["recall_ceiling"] - mean["recall_ceiling_std"]) * 100,
                    (mean["recall_ceiling"] + mean["recall_ceiling_std"]) * 100,
                    alpha=0.15, color="#2ca02c")
    ax.plot(sf, mean["recall_actual"] * 100, "o-", color="#1f77b4", label="e-BH recall (actual)", markersize=5)
    ax.fill_between(sf,
                    (mean["recall_actual"] - mean["recall_actual_std"]) * 100,
                    (mean["recall_actual"] + mean["recall_actual_std"]) * 100,
                    alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Score floor $s_{\\mathrm{floor}}$")
    ax.set_ylabel("Recall (%)")
    ax.set_title(f"{title}\n(α = {alpha})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    # Right: efficiency
    ax2 = axes[1]
    ax2.plot(sf, mean["efficiency"] * 100, "D-", color="#d62728", markersize=5)
    ax2.set_xlabel("Score floor $s_{\\mathrm{floor}}$")
    ax2.set_ylabel("Efficiency = actual / ceiling (%)")
    ax2.set_title("Frontier efficiency")
    ax2.grid(alpha=0.25)
    eff_max = mean["efficiency"].max() * 100
    ax2.set_ylim(0, max(110, eff_max * 1.15))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compute score-floor frontier curves")
    p.add_argument("--pipeline-dir", type=Path, required=True)
    p.add_argument("--label", type=str, default="")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seeds", type=str, default="0-19")
    p.add_argument("--score-floors", type=str,
                   default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    score_floors = [float(x) for x in args.score_floors.split(",")]

    with open(args.pipeline_dir / "seed_summaries.json") as f:
        seed_summaries = json.load(f)

    out_dir = args.out_dir or args.pipeline_dir / "score_floor_frontier"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in seeds:
        meta = seed_summaries[str(seed)]
        rows = process_seed(args.pipeline_dir, seed, args.alpha, meta, score_floors)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "score_floor_frontier.csv", index=False)

    # Print summary
    mean = df.groupby("score_floor").agg(
        recall_ceiling=("recall_ceiling", "mean"),
        recall_actual=("recall_actual", "mean"),
        efficiency=("efficiency", "mean"),
        fdp=("fdp", "mean"),
    ).reset_index()

    print(f"\n{'s_floor':>8} | {'ceiling':>8} | {'actual':>8} | {'efficiency':>10} | {'fdp':>8}")
    print("-" * 55)
    for _, row in mean.iterrows():
        print(f"{row['score_floor']:>8.2f} | {row['recall_ceiling']:>7.1%} | "
              f"{row['recall_actual']:>7.1%} | {row['efficiency']:>9.1%} | {row['fdp']:>7.1%}")

    title = args.label or args.pipeline_dir.name
    plot_frontier(df, out_dir / "score_floor_frontier.png", args.alpha, title)
    plot_frontier(df, out_dir / "score_floor_frontier.pdf", args.alpha, title)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
