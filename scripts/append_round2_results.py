#!/usr/bin/env python3
"""Append Round 2 sprint results to paper_ground_truth_table_2026-04-14.csv.

Appends:
  - adaptive_floor_results.csv → "score_floor" table, methods adaptive_fit / fixed_040 / no_floor
  - score_quantile_table.csv   → "conditional_validity" table, method score_quantile_<bin>
  - object_size_table.csv      → "conditional_validity" table, method object_size_<bin>
  - per_category_table.csv     → "conditional_validity" table, method per_category_<cat>
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
GT_PATH = OUTPUTS / "paper_ground_truth_table_2026-04-14.csv"

GT_COLS = [
    "table", "dataset", "detector", "alpha", "method",
    "fdp_pooled_mean", "fdp_pooled_std", "fdp_family_mean", "recall_mean",
    "rejections_mean", "phi_max_median", "loo_mean", "n_seeds", "source",
    "pooled_bound_eps", "pooled_bound_delta", "pooled_bound_covered",
    "pooled_bound_eps_stratified",
]


def from_adaptive_floor() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "adaptive_floor_results.csv")
    # Use rho=0.85 as primary mode for headline
    df = df[df["rho"] == 0.85].copy()
    df["table"] = "score_floor"
    df["method"] = df["mode"].map({
        "adaptive_fit": "adaptive_fit_rho085",
        "fixed_0.40":   "fixed_floor_040",
        "no_floor":     "no_floor",
    })
    out = pd.DataFrame({
        "table": df["table"],
        "dataset": df["dataset"],
        "detector": df["detector"],
        "alpha": df["alpha"],
        "method": df["method"],
        "fdp_pooled_mean": df["pooled_fdp_mean"],
        "fdp_pooled_std": df["pooled_fdp_std"],
        "fdp_family_mean": df["per_family_fdp_mean"],
        "recall_mean": df["recall_mean"],
        "rejections_mean": df["rejections_mean"],
        "phi_max_median": "",
        "loo_mean": "",
        "n_seeds": 20,
        "source": "outputs/adaptive_floor_results.csv",
        "pooled_bound_eps": "",
        "pooled_bound_delta": "",
        "pooled_bound_covered": "",
        "pooled_bound_eps_stratified": "",
    })
    return out


def from_stratified(path: Path, table_label: str, prefix: str,
                     has_per_family: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "table": "conditional_validity",
        "dataset": df["dataset"],
        "detector": df["detector"],
        "alpha": df["alpha"],
        "method": prefix + "_" + df["bin"].astype(str),
        "fdp_pooled_mean": df["pooled_fdp"],
        "fdp_pooled_std": "",
        "fdp_family_mean": df["per_family_fdp"] if has_per_family else "",
        "recall_mean": df["recall"],
        "rejections_mean": df["n_rejections"],
        "phi_max_median": "",
        "loo_mean": df["loo_mean"],
        "n_seeds": 20,
        "source": f"outputs/{path.name}",
        "pooled_bound_eps": "",
        "pooled_bound_delta": "",
        "pooled_bound_covered": df["pass_rate"],
        "pooled_bound_eps_stratified": "",
    })
    return out


def main() -> None:
    gt = pd.read_csv(GT_PATH)
    print(f"Loaded {len(gt)} existing rows")

    parts = [gt]

    # B1
    b1 = from_adaptive_floor()
    print(f"  + {len(b1)} rows from adaptive_floor")
    parts.append(b1)

    # A1
    sq_path = OUTPUTS / "score_quantile_table.csv"
    if sq_path.exists():
        a1 = from_stratified(sq_path, "conditional_validity", "score_quantile", has_per_family=False)
        print(f"  + {len(a1)} rows from score_quantile")
        parts.append(a1)

    # A2
    os_path = OUTPUTS / "object_size_table.csv"
    if os_path.exists():
        a2 = from_stratified(os_path, "conditional_validity", "object_size", has_per_family=False)
        print(f"  + {len(a2)} rows from object_size")
        parts.append(a2)

    # A3
    pc_path = OUTPUTS / "per_category_table.csv"
    if pc_path.exists():
        a3 = from_stratified(pc_path, "conditional_validity", "per_category", has_per_family=True)
        print(f"  + {len(a3)} rows from per_category")
        parts.append(a3)

    out = pd.concat(parts, ignore_index=True)
    out = out[GT_COLS]  # ensure column order
    out.to_csv(GT_PATH, index=False)
    print(f"Wrote {len(out)} total rows to {GT_PATH}")


if __name__ == "__main__":
    main()
