#!/usr/bin/env python3
"""Exploratory free-form mitigation sweep.

This script keeps the detector and the score-only density-ratio estimator fixed,
then tests observable gates before self-normalized e-BH:

* score-tail gates: keep candidates with score >= tau and normalize on cal nulls
  in the same tail;
* family-size gates: keep families with K <= k_max and normalize on cal nulls
  from calibration families satisfying the same gate;
* optional template gates.

The reported "full_recall" uses all test true positives as denominator, while
"gated_recall" uses only true positives left after the gate. The sweep is meant
as a diagnostic for whether any practically defensible supported slice exists.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_formal_betting_pipeline import (  # noqa: E402
    compute_self_normalized_evalues,
    e_bh,
    fit_density_ratio_model,
    split_families,
)


DEFAULT_SEEDS = ",".join(str(i) for i in range(20))
DEFAULT_ALPHAS = "0.05,0.10,0.15,0.20"
DEFAULT_SPLIT = "0.6,0.2,0.2"
DEFAULT_SCORE_THRESHOLDS = (
    "none,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,"
    "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"
)
DEFAULT_K_MAX = "all,3,5,10,20,30"


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_split(text: str) -> tuple[float, float, float]:
    vals = parse_floats(text)
    if len(vals) != 3:
        raise ValueError("--split-ratios must contain three comma-separated values")
    total = sum(vals)
    if total <= 0:
        raise ValueError("--split-ratios must be positive")
    return vals[0] / total, vals[1] / total, vals[2] / total


def parse_score_thresholds(text: str) -> list[float | None]:
    out: list[float | None] = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"none", "all", "nan"}:
            out.append(None)
        else:
            out.append(float(item))
    return out


def parse_k_max(text: str) -> list[int | None]:
    out: list[int | None] = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"all", "none", "nan"}:
            out.append(None)
        else:
            out.append(int(item))
    return out


def extract_template(prompt: str) -> str:
    patterns = [
        (r"^a visible (.+) in the scene$", "a visible {cat} in the scene"),
        (r"^a small (.+) near the center$", "a small {cat} near the center"),
        (r"^a photo of a (.+)$", "a photo of a {cat}"),
        (r"^a (.+) in the scene$", "a {cat} in the scene"),
        (r"^a (.+) object$", "a {cat} object"),
        (r"^the (.+)$", "the {cat}"),
    ]
    for pattern, template in patterns:
        if re.match(pattern, str(prompt)):
            return template
    return "other"


def ebh_totals(test_df: pd.DataFrame, alpha: float) -> dict[str, float | int]:
    total_rejections = 0
    total_fp = 0
    total_tp = 0
    rejected_families = 0
    family_fdps: list[float] = []

    for _, group in test_df.groupby("family_id", sort=False):
        rejected_pos = e_bh(group["mitigated_evalue"].to_numpy(dtype=float), alpha)
        if not rejected_pos:
            family_fdps.append(0.0)
            continue
        selected = group.iloc[rejected_pos]
        fp = int(selected["is_null"].sum())
        tp = int(selected["is_tp"].sum())
        total_rejections += int(len(selected))
        total_fp += fp
        total_tp += tp
        rejected_families += 1
        family_fdps.append(fp / max(int(len(selected)), 1))

    return {
        "num_rejected_families": rejected_families,
        "total_rejections": total_rejections,
        "fp": total_fp,
        "tp": total_tp,
        "fdp": total_fp / max(total_rejections, 1),
        "mean_family_fdp": float(np.mean(family_fdps)) if family_fdps else 0.0,
    }


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby("family_id").size().rename("family_size")
    return df.merge(sizes, left_on="family_id", right_index=True, how="left")


def apply_gate(
    df: pd.DataFrame,
    *,
    score_threshold: float | None,
    k_max: int | None,
    template: str | None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if score_threshold is not None:
        mask &= df["score"] >= score_threshold
    if k_max is not None:
        mask &= df["family_size"] <= k_max
    if template is not None:
        mask &= df["template"] == template
    return df[mask].copy()


def frontier_stats(
    test_eval: pd.DataFrame,
    *,
    alpha: float,
    n_cal: int,
    sum_phi_cal: float,
    phi_max: float,
) -> dict[str, float]:
    if test_eval.empty:
        return {
            "frontier_supported_family_frac": 0.0,
            "frontier_T1_median": float("nan"),
            "frontier_T1_mean": float("nan"),
        }
    family_k = test_eval.groupby("family_id").size().to_numpy(dtype=float)
    denom = alpha * (n_cal + 1) - family_k
    t1 = np.full_like(family_k, np.inf, dtype=float)
    valid = denom > 0
    t1[valid] = family_k[valid] * sum_phi_cal / denom[valid]
    supported = phi_max >= t1
    finite_t1 = t1[np.isfinite(t1)]
    return {
        "frontier_supported_family_frac": float(np.mean(supported)),
        "frontier_T1_median": float(np.median(finite_t1)) if finite_t1.size else float("inf"),
        "frontier_T1_mean": float(np.mean(finite_t1)) if finite_t1.size else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--alphas", type=str, default=DEFAULT_ALPHAS)
    parser.add_argument("--split-ratios", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--score-thresholds", type=str, default=DEFAULT_SCORE_THRESHOLDS)
    parser.add_argument("--k-max", type=str, default=DEFAULT_K_MAX)
    parser.add_argument("--include-template-gates", action="store_true")
    parser.add_argument("--min-cal-null", type=int, default=50)
    parser.add_argument("--estimator", choices=["hist", "kde", "logreg", "isotonic"], default="hist")
    parser.add_argument("--num-bins", type=int, default=80)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--clip-phi", type=float, default=1_000_000.0)
    parser.add_argument("--max-logreg-iter", type=int, default=1000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    alphas = sorted(set(parse_floats(args.alphas)))
    split_ratios = parse_split(args.split_ratios)
    score_thresholds = parse_score_thresholds(args.score_thresholds)
    k_max_values = parse_k_max(args.k_max)

    df = pd.read_csv(args.candidate_table)
    for col in ["is_tp", "is_null", "is_prompt_absent"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().eq("true")
    df = add_family_size(df)
    df["template"] = df["prompt"].apply(extract_template)

    template_values: list[str | None] = [None]
    if args.include_template_gates:
        template_values.extend(sorted(df["template"].dropna().unique().tolist()))

    rows: list[dict[str, float | int | str]] = []
    seed_meta: dict[str, dict[str, float | int]] = {}

    for seed in seeds:
        split = split_families(df, seed=seed, ratios=split_ratios)
        df_seed = df.copy()
        df_seed["split"] = split

        fit_df = df_seed[df_seed["split"] == "fit"].copy()
        cal_df = df_seed[df_seed["split"] == "cal"].copy()
        test_df = df_seed[df_seed["split"] == "test"].copy()
        total_test_tp = int(test_df["is_tp"].sum())

        model, _, ratio_summary = fit_density_ratio_model(
            fit_df,
            estimator=args.estimator,
            num_bins=args.num_bins,
            smoothing=args.smoothing,
            clip_phi=args.clip_phi,
            max_logreg_iter=args.max_logreg_iter,
        )
        phi_max = float(ratio_summary.get("max_phi", np.nan))

        seed_meta[str(seed)] = {
            "num_fit_families": int(fit_df["family_id"].nunique()),
            "num_cal_families": int(cal_df["family_id"].nunique()),
            "num_test_families": int(test_df["family_id"].nunique()),
            "total_test_tp": total_test_tp,
            "phi_max": phi_max,
        }

        for score_threshold in score_thresholds:
            for k_max in k_max_values:
                for template in template_values:
                    cal_gate = apply_gate(
                        cal_df[cal_df["is_null"]].copy(),
                        score_threshold=score_threshold,
                        k_max=k_max,
                        template=template,
                    )
                    if len(cal_gate) < args.min_cal_null:
                        continue
                    cal_phi = model.phi(cal_gate["score"].to_numpy(dtype=float))
                    sum_phi_cal = float(np.sum(cal_phi))
                    n_cal = int(len(cal_phi))
                    if sum_phi_cal <= 0:
                        continue

                    test_eval = apply_gate(
                        test_df,
                        score_threshold=score_threshold,
                        k_max=k_max,
                        template=template,
                    )
                    if test_eval.empty:
                        continue

                    test_phi = model.phi(test_eval["score"].to_numpy(dtype=float))
                    test_eval["mitigated_evalue"] = compute_self_normalized_evalues(
                        test_phi,
                        sum_phi_cal,
                        n_cal,
                    )

                    gated_tp = int(test_eval["is_tp"].sum())
                    for alpha in alphas:
                        totals = ebh_totals(test_eval, alpha)
                        front = frontier_stats(
                            test_eval,
                            alpha=alpha,
                            n_cal=n_cal,
                            sum_phi_cal=sum_phi_cal,
                            phi_max=phi_max,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "alpha": alpha,
                                "score_threshold": "none" if score_threshold is None else score_threshold,
                                "k_max": "all" if k_max is None else k_max,
                                "template": "all" if template is None else template,
                                "cal_null": n_cal,
                                "sum_phi_cal": sum_phi_cal,
                                "phi_max": phi_max,
                                "num_test_families": int(test_eval["family_id"].nunique()),
                                "num_test_candidates": int(len(test_eval)),
                                "gated_tp_total": gated_tp,
                                "full_tp_total": total_test_tp,
                                **totals,
                                "gated_recall": int(totals["tp"]) / max(gated_tp, 1),
                                "full_recall": int(totals["tp"]) / max(total_test_tp, 1),
                                **front,
                            }
                        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.out_dir / "mitigation_seed_results.csv", index=False)

    group_cols = ["alpha", "score_threshold", "k_max", "template"]
    metric_cols = [
        "cal_null",
        "sum_phi_cal",
        "phi_max",
        "num_test_families",
        "num_test_candidates",
        "gated_tp_total",
        "total_rejections",
        "fp",
        "tp",
        "fdp",
        "gated_recall",
        "full_recall",
        "frontier_supported_family_frac",
        "frontier_T1_median",
    ]
    mean_df = result_df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    std_df = result_df.groupby(group_cols, dropna=False)[metric_cols].std().reset_index()
    mean_df.to_csv(args.out_dir / "mitigation_mean_results.csv", index=False)
    std_df.to_csv(args.out_dir / "mitigation_std_results.csv", index=False)

    alpha_010 = mean_df[np.isclose(mean_df["alpha"], 0.10)].copy()
    controlled = alpha_010[alpha_010["fdp"] <= 0.10].copy()
    controlled = controlled.sort_values(
        ["full_recall", "total_rejections", "gated_recall"],
        ascending=[False, False, False],
    )
    controlled.to_csv(args.out_dir / "controlled_alpha010_candidates.csv", index=False)

    best = controlled.head(20).to_dict("records")
    summary = {
        "description": "Free-form mitigation sweep with score-tail and family-size gates.",
        "candidate_table": str(args.candidate_table),
        "num_seed_rows": int(len(result_df)),
        "num_mean_rows": int(len(mean_df)),
        "min_cal_null": int(args.min_cal_null),
        "seed_meta": seed_meta,
        "best_alpha010_controlled": best,
    }
    with open(args.out_dir / "mitigation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Best alpha=0.10 controlled gates (mean FDP <= 0.10) ===")
    if controlled.empty:
        print("No controlled gates found.")
    else:
        cols = [
            "score_threshold",
            "k_max",
            "template",
            "cal_null",
            "total_rejections",
            "fdp",
            "gated_recall",
            "full_recall",
            "frontier_supported_family_frac",
            "frontier_T1_median",
        ]
        print(controlled[cols].head(20).to_string(index=False))
    print(f"\nWrote sweep to {args.out_dir}")


if __name__ == "__main__":
    main()
