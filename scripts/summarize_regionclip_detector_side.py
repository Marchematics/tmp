#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = PROJECT / "outputs" / "paper_ground_truth_table_2026-04-14.csv"
DEFAULT_ANALYSIS = PROJECT / "outputs" / "coco_regionclip_1000_20seed_analysis"
DEFAULT_FORMAL = PROJECT / "outputs" / "coco_regionclip_1000_20seed_formal_hist"
DEFAULT_RAW_JSONL = PROJECT / "outputs" / "coco_regionclip_1000_mix.jsonl"
DEFAULT_SIDE_CSV = PROJECT / "outputs" / "regionclip_detector_side_comparison.csv"
ALPHAS = [0.05, 0.10, 0.15, 0.20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append RegionCLIP detector-side comparison rows.")
    parser.add_argument("--main-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--formal-dir", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW_JSONL)
    parser.add_argument("--side-csv", type=Path, default=DEFAULT_SIDE_CSV)
    parser.add_argument("--vanilla-threshold", type=float, default=0.5)
    return parser.parse_args()


def rel_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT))
    except ValueError:
        return str(path)


def fmt_float(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def vanilla_metrics(candidate_table: Path, threshold: float) -> dict[str, float]:
    df = pd.read_csv(candidate_table)
    selected = df[df["score"] >= threshold]
    total_rejections = int(len(selected))
    fp = int(selected["is_null"].astype(bool).sum()) if total_rejections else 0
    tp = int(selected["is_tp"].astype(bool).sum()) if total_rejections else 0
    total_tp = int(df["is_tp"].astype(bool).sum())

    family_fdps: list[float] = []
    for _, group in df.groupby("family_id", sort=False):
        group_selected = group[group["score"] >= threshold]
        if group_selected.empty:
            family_fdps.append(0.0)
            continue
        family_fdps.append(float(group_selected["is_null"].astype(bool).sum() / len(group_selected)))

    return {
        "fdp": fp / max(total_rejections, 1),
        "recall": tp / max(total_tp, 1),
        "total_rejections": float(total_rejections),
        "mean_family_fdp": float(np.mean(family_fdps)) if family_fdps else 0.0,
        "fp": float(fp),
        "tp": float(tp),
        "total_tp": float(total_tp),
        "num_families": float(df["family_id"].nunique()),
    }


def wrapper_rows(formal_dir: Path) -> tuple[dict[float, dict[str, float]], float, float]:
    mean = pd.read_csv(formal_dir / "mean_results_by_alpha.csv")
    std = pd.read_csv(formal_dir / "std_results_by_alpha.csv")
    seed = pd.read_csv(formal_dir / "seed_alpha_results.csv")
    loo = pd.read_csv(formal_dir / "loo_validity.csv")
    summaries = json.loads((formal_dir / "seed_summaries.json").read_text())

    phi_max = [float(row["density_ratio"]["max_phi"]) for row in summaries.values()]
    phi_max_median = float(np.median(phi_max)) if phi_max else float("nan")
    loo_mean = float(loo["mean_loo_evalue"].mean()) if not loo.empty else float("nan")

    by_alpha: dict[float, dict[str, float]] = {}
    for alpha in ALPHAS:
        m = mean[(mean["method"] == "betting") & (np.isclose(mean["alpha"], alpha))]
        s = std[(std["method"] == "betting") & (np.isclose(std["alpha"], alpha))]
        g = seed[(seed["method"] == "betting") & (np.isclose(seed["alpha"], alpha))]
        if m.empty or s.empty or g.empty:
            raise RuntimeError(f"missing betting rows for alpha={alpha}")
        by_alpha[alpha] = {
            "fdp": float(m.iloc[0]["fdp"]),
            "fdp_std": float(s.iloc[0]["fdp"]),
            "recall": float(m.iloc[0]["recall"]),
            "rejections": float(m.iloc[0]["total_rejections"]),
            "mean_family_fdp": float(g["mean_family_fdp"].mean()),
        }
    return by_alpha, phi_max_median, loo_mean


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    candidate_table = args.analysis_dir / "candidate_table.csv"
    vanilla = vanilla_metrics(candidate_table, args.vanilla_threshold)
    wrapped, phi_max_median, loo_mean = wrapper_rows(args.formal_dir)

    rows: list[dict[str, str]] = []
    raw_source = rel_project(args.raw_jsonl)
    formal_source = rel_project(args.formal_dir)

    for alpha in ALPHAS:
        rows.append(
            {
                "table": "detector_side_comparison",
                "dataset": "COCO val",
                "detector": "RegionCLIP",
                "alpha": fmt_float(alpha),
                "method": "regionclip_vanilla",
                "fdp_pooled_mean": fmt_float(vanilla["fdp"]),
                "fdp_pooled_std": "0",
                "fdp_family_mean": fmt_float(vanilla["mean_family_fdp"]),
                "recall_mean": fmt_float(vanilla["recall"]),
                "rejections_mean": fmt_float(vanilla["total_rejections"]),
                "phi_max_median": "",
                "loo_mean": "",
                "n_seeds": "1",
                "source": raw_source,
                "pooled_bound_eps": "",
                "pooled_bound_delta": "",
                "pooled_bound_covered": "",
                "pooled_bound_eps_stratified": "",
            }
        )

    for alpha in ALPHAS:
        row = wrapped[alpha]
        rows.append(
            {
                "table": "detector_side_comparison",
                "dataset": "COCO val",
                "detector": "RegionCLIP",
                "alpha": fmt_float(alpha),
                "method": "regionclip_plus_wrapper",
                "fdp_pooled_mean": fmt_float(row["fdp"]),
                "fdp_pooled_std": fmt_float(row["fdp_std"]),
                "fdp_family_mean": fmt_float(row["mean_family_fdp"]),
                "recall_mean": fmt_float(row["recall"]),
                "rejections_mean": fmt_float(row["rejections"]),
                "phi_max_median": fmt_float(phi_max_median),
                "loo_mean": fmt_float(loo_mean),
                "n_seeds": "20",
                "source": formal_source,
                "pooled_bound_eps": "",
                "pooled_bound_delta": "",
                "pooled_bound_covered": "",
                "pooled_bound_eps_stratified": "",
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    table = pd.read_csv(args.main_table, dtype=str, keep_default_na=False)
    cols = list(table.columns)
    missing = sorted(set(cols) - set(rows[0]))
    if missing:
        raise RuntimeError(f"new rows missing columns: {missing}")

    mask = (
        (table["table"] == "detector_side_comparison")
        & (table["detector"] == "RegionCLIP")
        & table["method"].isin(["regionclip_vanilla", "regionclip_plus_wrapper"])
    )
    preserve_cols = [
        "pooled_bound_eps",
        "pooled_bound_delta",
        "pooled_bound_covered",
        "pooled_bound_eps_stratified",
    ]
    existing = table.loc[mask].copy()
    if not existing.empty:
        by_key = {
            (r["method"], r["alpha"]): r
            for r in existing.to_dict(orient="records")
        }
        for row in rows:
            old = by_key.get((row["method"], row["alpha"]))
            if old is None:
                continue
            for col in preserve_cols:
                if col in old and old[col] != "":
                    row[col] = old[col]

    table = table.loc[~mask].copy()
    out_df = pd.concat([table, pd.DataFrame(rows, columns=cols)], ignore_index=True)
    out_df.to_csv(args.main_table, index=False)

    with args.side_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} RegionCLIP rows to {args.main_table}")
    print(f"side comparison copy: {args.side_csv}")


if __name__ == "__main__":
    main()
