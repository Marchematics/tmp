#!/usr/bin/env python3
"""Task-aligned CRC-style score-threshold baseline.

This script uses the same family split protocol as the formal betting pipeline,
calibrates one global detector-score gate on calibration families, and evaluates
the resulting rejected boxes on held-out test families. It is a comparator for
conformal risk-control style thresholding, not a replacement for family-level
e-BH: the default loss is the per-family FDP induced by a score threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ALPHAS = "0.05,0.1,0.15,0.2"
DEFAULT_SEEDS = "0,1,2,3,4"
DEFAULT_SPLIT = "0.6,0.2,0.2"


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_alphas(text: str) -> list[float]:
    alphas = parse_floats(text)
    if not alphas:
        raise ValueError("At least one alpha is required")
    for alpha in alphas:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
    return sorted(set(alphas))


def parse_split(text: str) -> tuple[float, float, float]:
    parts = parse_floats(text)
    if len(parts) != 3:
        raise ValueError("split-ratios must have three comma-separated values")
    total = sum(parts)
    if total <= 0:
        raise ValueError("split-ratios must be positive")
    fit, cal, test = [part / total for part in parts]
    return fit, cal, test


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def split_families(df: pd.DataFrame, *, seed: int, ratios: tuple[float, float, float]) -> pd.Series:
    rng = np.random.default_rng(seed)
    families = df["family_id"].drop_duplicates().to_numpy()
    rng.shuffle(families)
    n_total = len(families)
    n_fit = int(round(ratios[0] * n_total))
    n_cal = int(round(ratios[1] * n_total))
    n_fit = max(1, min(n_fit, n_total - 2))
    n_cal = max(1, min(n_cal, n_total - n_fit - 1))
    fit_set = set(families[:n_fit])
    cal_set = set(families[n_fit : n_fit + n_cal])
    split = np.where(df["family_id"].isin(fit_set), "fit", "test")
    split = np.where(df["family_id"].isin(cal_set), "cal", split)
    return pd.Series(split, index=df.index)


def threshold_grid(scores: np.ndarray, max_thresholds: int) -> np.ndarray:
    unique_scores = np.unique(np.asarray(scores, dtype=float))
    unique_scores = unique_scores[np.isfinite(unique_scores)]
    if unique_scores.size == 0:
        return np.asarray([], dtype=float)
    unique_scores = np.sort(unique_scores)[::-1]
    if unique_scores.size <= max_thresholds:
        return unique_scores
    idx = np.unique(np.linspace(0, unique_scores.size - 1, max_thresholds).round().astype(int))
    return unique_scores[idx]


def family_fdp_loss(df: pd.DataFrame, threshold: float) -> tuple[float, int, int, int]:
    losses: list[float] = []
    total_selected = 0
    total_fp = 0
    total_tp = 0
    for _, group in df.groupby("family_id", sort=False):
        selected = group[group["score"] >= threshold]
        if selected.empty:
            losses.append(0.0)
            continue
        fp = int(selected["is_null"].sum())
        tp = int(selected["is_tp"].sum())
        total_selected += int(len(selected))
        total_fp += fp
        total_tp += tp
        losses.append(fp / max(len(selected), 1))
    return float(np.mean(losses)) if losses else 0.0, total_selected, total_fp, total_tp


def crc_upper_bound(mean_loss: float, n_families: int) -> float:
    if n_families <= 0:
        return 1.0
    return float((n_families * mean_loss + 1.0) / (n_families + 1.0))


def calibrate_family_threshold(
    cal_df: pd.DataFrame,
    *,
    alpha: float,
    max_thresholds: int,
    min_cal_rejections: int,
) -> dict[str, float | int]:
    n_families = int(cal_df["family_id"].nunique())
    best: dict[str, float | int] | None = None
    if cal_df.empty or n_families == 0:
        return {
            "threshold": float("inf"),
            "cal_loss": 0.0,
            "cal_bound": 1.0,
            "cal_rejections": 0,
            "cal_fp": 0,
            "cal_tp": 0,
        }

    scores = cal_df["score"].to_numpy(dtype=float)
    is_null = cal_df["is_null"].to_numpy(dtype=bool)
    family_codes, _ = pd.factorize(cal_df["family_id"], sort=False)
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_null = is_null[order]
    sorted_family = family_codes[order]

    family_selected = np.zeros(n_families, dtype=np.int64)
    family_fp = np.zeros(n_families, dtype=np.int64)
    position = 0
    total_selected = 0
    total_fp = 0

    for threshold in threshold_grid(scores, max_thresholds):
        threshold = float(threshold)
        while position < sorted_scores.size and float(sorted_scores[position]) >= threshold:
            family_idx = int(sorted_family[position])
            family_selected[family_idx] += 1
            if bool(sorted_null[position]):
                family_fp[family_idx] += 1
                total_fp += 1
            total_selected += 1
            position += 1
        if total_selected <= 0:
            continue
        family_losses = np.divide(
            family_fp,
            family_selected,
            out=np.zeros(n_families, dtype=float),
            where=family_selected > 0,
        )
        mean_loss = float(np.mean(family_losses))
        fp = int(total_fp)
        tp = int(total_selected - total_fp)
        if total_selected < min_cal_rejections:
            continue
        upper = crc_upper_bound(mean_loss, n_families)
        if upper > alpha:
            continue
        candidate = {
            "threshold": float(threshold),
            "cal_loss": float(mean_loss),
            "cal_bound": float(upper),
            "cal_rejections": int(total_selected),
            "cal_fp": int(fp),
            "cal_tp": int(tp),
        }
        if best is None or int(candidate["cal_rejections"]) > int(best["cal_rejections"]):
            best = candidate
    if best is not None:
        return best
    return {
        "threshold": float("inf"),
        "cal_loss": 0.0,
        "cal_bound": 1.0 / max(n_families + 1, 1),
        "cal_rejections": 0,
        "cal_fp": 0,
        "cal_tp": 0,
    }


def calibrate_pooled_threshold(
    cal_df: pd.DataFrame,
    *,
    alpha: float,
    max_thresholds: int,
    min_cal_rejections: int,
    plus_one: bool,
) -> dict[str, float | int]:
    if cal_df.empty:
        return {
            "threshold": float("inf"),
            "cal_loss": 0.0,
            "cal_bound": 1.0,
            "cal_rejections": 0,
            "cal_fp": 0,
            "cal_tp": 0,
        }

    scores = cal_df["score"].to_numpy(dtype=float)
    is_null = cal_df["is_null"].to_numpy(dtype=bool)
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_null = is_null[order]

    best: dict[str, float | int] | None = None
    position = 0
    total_selected = 0
    total_fp = 0

    for threshold in threshold_grid(scores, max_thresholds):
        threshold = float(threshold)
        while position < sorted_scores.size and float(sorted_scores[position]) >= threshold:
            if bool(sorted_null[position]):
                total_fp += 1
            total_selected += 1
            position += 1
        if total_selected < min_cal_rejections:
            continue
        cal_fdp = total_fp / max(total_selected, 1)
        if plus_one:
            bound = (total_fp + 1.0) / (total_selected + 1.0)
        else:
            bound = cal_fdp
        if bound > alpha:
            continue
        candidate = {
            "threshold": float(threshold),
            "cal_loss": float(cal_fdp),
            "cal_bound": float(bound),
            "cal_rejections": int(total_selected),
            "cal_fp": int(total_fp),
            "cal_tp": int(total_selected - total_fp),
        }
        if best is None or int(candidate["cal_rejections"]) > int(best["cal_rejections"]):
            best = candidate
    if best is not None:
        return best
    return {
        "threshold": float("inf"),
        "cal_loss": 0.0,
        "cal_bound": 0.0,
        "cal_rejections": 0,
        "cal_fp": 0,
        "cal_tp": 0,
    }


def calibrate_threshold(
    cal_df: pd.DataFrame,
    *,
    alpha: float,
    max_thresholds: int,
    min_cal_rejections: int,
    risk: str,
) -> dict[str, float | int]:
    if risk == "family_crc":
        return calibrate_family_threshold(
            cal_df,
            alpha=alpha,
            max_thresholds=max_thresholds,
            min_cal_rejections=min_cal_rejections,
        )
    if risk == "pooled_emp":
        return calibrate_pooled_threshold(
            cal_df,
            alpha=alpha,
            max_thresholds=max_thresholds,
            min_cal_rejections=min_cal_rejections,
            plus_one=False,
        )
    if risk == "pooled_plus1":
        return calibrate_pooled_threshold(
            cal_df,
            alpha=alpha,
            max_thresholds=max_thresholds,
            min_cal_rejections=min_cal_rejections,
            plus_one=True,
        )
    raise ValueError(f"Unknown risk mode: {risk}")


def evaluate_threshold(test_df: pd.DataFrame, threshold: float, alpha: float) -> dict[str, float | int]:
    total_tp = int(test_df["is_tp"].sum())
    total_rejections = 0
    total_fp = 0
    total_tp_rej = 0
    rejected_families = 0
    family_fdps: list[float] = []
    for _, group in test_df.groupby("family_id", sort=False):
        selected = group[group["score"] >= threshold]
        if selected.empty:
            family_fdps.append(0.0)
            continue
        fp = int(selected["is_null"].sum())
        tp = int(selected["is_tp"].sum())
        total_rejections += int(len(selected))
        total_fp += fp
        total_tp_rej += tp
        rejected_families += 1
        family_fdps.append(fp / max(len(selected), 1))
    return {
        "alpha": alpha,
        "num_test_families": int(test_df["family_id"].nunique()),
        "num_rejected_families": rejected_families,
        "total_rejections": total_rejections,
        "fp": total_fp,
        "tp": total_tp_rej,
        "fdp": total_fp / max(total_rejections, 1),
        "recall": total_tp_rej / max(total_tp, 1),
        "mean_family_fdp": float(np.mean(family_fdps)) if family_fdps else 0.0,
    }


def run_baseline(
    df: pd.DataFrame,
    *,
    alphas: list[float],
    seeds: list[int],
    split_ratios: tuple[float, float, float],
    max_thresholds: int,
    min_cal_rejections: int,
    risk: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        split = split_families(df, seed=seed, ratios=split_ratios)
        cal_df = df[split == "cal"].copy()
        test_df = df[split == "test"].copy()
        for alpha in alphas:
            cal_info = calibrate_threshold(
                cal_df,
                alpha=alpha,
                max_thresholds=max_thresholds,
                min_cal_rejections=min_cal_rejections,
                risk=risk,
            )
            row = evaluate_threshold(test_df, float(cal_info["threshold"]), alpha)
            row.update(cal_info)
            row.update({"seed": seed, "method": f"scoregate_{risk}"})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=str, default=DEFAULT_ALPHAS)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--split-ratios", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max-thresholds", type=int, default=2000)
    parser.add_argument("--min-cal-rejections", type=int, default=1)
    parser.add_argument(
        "--risk",
        type=str,
        choices=["family_crc", "pooled_emp", "pooled_plus1"],
        default="family_crc",
        help=(
            "Calibration risk target. family_crc controls a CRC-style upper "
            "bound on mean family FDP; pooled_* controls pooled calibration FDP."
        ),
    )
    args = parser.parse_args()

    alphas = parse_alphas(args.alphas)
    seeds = parse_ints(args.seeds)
    split_ratios = parse_split(args.split_ratios)
    if args.max_thresholds <= 0:
        raise ValueError("--max-thresholds must be positive")

    df = pd.read_csv(args.candidate_table)
    required = {"family_id", "score", "is_tp"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Candidate table missing columns: {missing}")
    df = df.copy()
    df["is_tp"] = normalize_bool(df["is_tp"])
    if "is_null" in df.columns:
        df["is_null"] = normalize_bool(df["is_null"])
    else:
        df["is_null"] = ~df["is_tp"]
    df["score"] = df["score"].astype(float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_results = run_baseline(
        df,
        alphas=alphas,
        seeds=seeds,
        split_ratios=split_ratios,
        max_thresholds=int(args.max_thresholds),
        min_cal_rejections=int(args.min_cal_rejections),
        risk=str(args.risk),
    )
    seed_results.to_csv(args.out_dir / "seed_alpha_results.csv", index=False)

    metric_cols = ["total_rejections", "fdp", "recall", "mean_family_fdp", "cal_bound", "cal_loss"]
    mean_results = seed_results.groupby(["method", "alpha"], as_index=False)[metric_cols].mean()
    std_results = seed_results.groupby(["method", "alpha"], as_index=False)[metric_cols].std()
    mean_results.to_csv(args.out_dir / "mean_results_by_alpha.csv", index=False)
    std_results.to_csv(args.out_dir / "std_results_by_alpha.csv", index=False)

    config = {
        "candidate_table": str(args.candidate_table),
        "alphas": alphas,
        "seeds": seeds,
        "split_ratios": split_ratios,
        "max_thresholds": int(args.max_thresholds),
        "min_cal_rejections": int(args.min_cal_rejections),
        "risk": str(args.risk),
        "loss": (
            "per-family FDP with CRC-style bounded-risk upper correction"
            if args.risk == "family_crc"
            else "pooled calibration FDP"
        ),
    }
    with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    alpha_010 = seed_results[seed_results["alpha"] == 0.10]
    if not alpha_010.empty:
        print("=== CRC-style score gate at alpha=0.10 ===")
        print(f"Rejections: {alpha_010['total_rejections'].mean():.1f}")
        print(f"Pooled FDP: {100.0 * alpha_010['fdp'].mean():.2f}%")
        print(f"Family FDP: {100.0 * alpha_010['mean_family_fdp'].mean():.2f}%")
        print(f"Recall:     {100.0 * alpha_010['recall'].mean():.2f}%")


if __name__ == "__main__":
    main()
