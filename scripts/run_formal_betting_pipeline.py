#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_hallucination_fdr_matplotlib")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ovd_hallucination_fdr.evalues import bh, conformal_pvalues, e_bh  # noqa: E402
from ovd_hallucination_fdr.io import ensure_dir, write_json  # noqa: E402


DEFAULT_ALPHAS = "0.05,0.1,0.15,0.2"
DEFAULT_SEEDS = "0,1,2,3,4"
DEFAULT_SPLIT = "0.6,0.2,0.2"
DEFAULT_CAL_SIZES = "500,1000,2000,5000,all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Formal three-split pipeline with self-normalized conformal betting e-values. "
            "Splits by family_id, fits a density ratio on fit split, uses null-only "
            "calibration for self-normalized e-values, then applies per-family e-BH."
        )
    )
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=str, default=DEFAULT_ALPHAS)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--split-ratios", type=str, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--estimator",
        type=str,
        choices=["hist", "kde", "logreg", "isotonic"],
        default="hist",
    )
    parser.add_argument("--num-bins", type=int, default=80)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--clip-phi", type=float, default=1_000_000.0)
    parser.add_argument(
        "--phi-laplace-eps",
        type=float,
        default=0.0,
        help=(
            "Optional Laplace floor for histogram density-ratio bins before clipping. "
            "Default 0 preserves the vanilla estimator; eps=1e-3 regularizes zero-null "
            "bins that would otherwise hit --clip-phi."
        ),
    )
    parser.add_argument("--cal-null-sizes", type=str, default=DEFAULT_CAL_SIZES)
    parser.add_argument("--max-logreg-iter", type=int, default=1000)
    parser.add_argument(
        "--mondrian",
        type=str,
        choices=["none", "prompt", "absent_status", "prompt_cluster"],
        default="none",
        help="Apply group-conditional (Mondrian) self-normalized calibration.",
    )
    parser.add_argument(
        "--mondrian-min-null",
        type=int,
        default=200,
        help="Minimum cal null count required for a Mondrian group; otherwise fallback to global.",
    )
    parser.add_argument(
        "--prompt-cluster-k",
        type=int,
        default=5,
        help="Number of prompt clusters for mondrian=prompt_cluster.",
    )
    parser.add_argument(
        "--score-floor",
        type=float,
        default=0.0,
        help="Drop test candidates with score < this threshold before e-BH. "
        "Reduces effective family size K, lowering the activation threshold T_1. "
        "Cal pool and phi estimation are unaffected.",
    )
    parser.add_argument("--seed-salt", type=str, default="formal_betting_v1")
    return parser.parse_args()


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


def parse_cal_sizes(text: str) -> list[int | None]:
    sizes: list[int | None] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "all":
            sizes.append(None)
        else:
            sizes.append(int(token))
    if not sizes:
        sizes.append(None)
    return sizes


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


@dataclass
class DensityRatioModel:
    name: str
    clip_phi: float
    edges: np.ndarray | None = None
    ratio: np.ndarray | None = None
    kde0: object | None = None
    kde1: object | None = None
    logreg: object | None = None
    isotonic: object | None = None
    prior_ratio: float | None = None

    def phi(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.name == "hist":
            if self.edges is None or self.ratio is None:
                raise ValueError("Histogram model not fit")
            bin_ids = np.searchsorted(self.edges, scores, side="right") - 1
            bin_ids = np.clip(bin_ids, 0, len(self.ratio) - 1)
            out = self.ratio[bin_ids]
        elif self.name == "kde":
            if self.kde0 is None or self.kde1 is None:
                raise ValueError("KDE model not fit")
            dens0 = np.maximum(self.kde0(scores), 1e-12)
            dens1 = np.maximum(self.kde1(scores), 1e-12)
            out = dens1 / dens0
        elif self.name == "logreg":
            if self.logreg is None or self.prior_ratio is None:
                raise ValueError("Logreg model not fit")
            probs = self.logreg.predict_proba(scores.reshape(-1, 1))[:, 1]
            probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
            odds = probs / (1.0 - probs)
            out = odds * self.prior_ratio
        elif self.name == "isotonic":
            if self.isotonic is None or self.prior_ratio is None:
                raise ValueError("Isotonic model not fit")
            probs = self.isotonic.predict(scores)
            probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
            odds = probs / (1.0 - probs)
            out = odds * self.prior_ratio
        else:
            raise ValueError(f"Unknown estimator: {self.name}")
        if self.clip_phi is not None and np.isfinite(self.clip_phi):
            out = np.minimum(out, self.clip_phi)
        return out


def fit_density_ratio_model(
    fit_df: pd.DataFrame,
    *,
    estimator: str,
    num_bins: int,
    smoothing: float,
    clip_phi: float,
    max_logreg_iter: int,
    phi_laplace_eps: float = 0.0,
) -> tuple[DensityRatioModel, pd.DataFrame, dict[str, float]]:
    null_scores = fit_df.loc[fit_df["is_null"], "score"].to_numpy(dtype=float)
    tp_scores = fit_df.loc[fit_df["is_tp"], "score"].to_numpy(dtype=float)
    if len(null_scores) == 0 or len(tp_scores) == 0:
        raise ValueError("Fit split must contain both null and true-positive scores")

    summary: dict[str, float] = {
        "num_fit_null": int(len(null_scores)),
        "num_fit_tp": int(len(tp_scores)),
    }

    if estimator == "hist":
        edges = make_quantile_bins(fit_df["score"].to_numpy(dtype=float), num_bins)
        null_counts, _ = np.histogram(null_scores, bins=edges)
        tp_counts, _ = np.histogram(tp_scores, bins=edges)
        bins = len(edges) - 1
        base_p0 = (null_counts.astype(float) + smoothing) / (len(null_scores) + smoothing * bins)
        base_p1 = (tp_counts.astype(float) + smoothing) / (len(tp_scores) + smoothing * bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            vanilla_raw_ratio = base_p1 / base_p0
        null_floor = float(smoothing)
        tp_floor = float(smoothing)
        if phi_laplace_eps > 0:
            if not np.isfinite(clip_phi) or clip_phi <= 0:
                raise ValueError("--phi-laplace-eps requires a positive finite --clip-phi")
            # Only adds mass to bins that vanilla would leave at zero.  With
            # smoothing=0 this is a Laplace fallback for clipped sparse bins;
            # with smoothing>0 it is a no-op up to the requested tiny epsilon.
            null_floor = max(null_floor, float(phi_laplace_eps))
            tp_floor = max(tp_floor, float(phi_laplace_eps))
        p0 = (null_counts.astype(float) + null_floor) / (len(null_scores) + null_floor * bins)
        p1 = (tp_counts.astype(float) + tp_floor) / (len(tp_scores) + tp_floor * bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p1 / p0
        raw_ratio = ratio.copy()
        if phi_laplace_eps > 0 and np.isfinite(clip_phi):
            clipped = ~np.isfinite(raw_ratio) | (raw_ratio > clip_phi)
            clipped_raw = np.nan_to_num(raw_ratio[clipped], nan=clip_phi, posinf=clip_phi, neginf=0.0)
            ratio[clipped] = (np.minimum(clipped_raw, clip_phi) + phi_laplace_eps) / (
                1.0 + phi_laplace_eps / clip_phi
            )
        if np.isfinite(clip_phi):
            ratio = np.minimum(ratio, clip_phi)
        model = DensityRatioModel(name="hist", clip_phi=clip_phi, edges=edges, ratio=ratio)
        fit_null_mean = float(np.sum(p0 * ratio))
        summary.update(
            {
                "num_bins_realized": int(bins),
                "max_phi": float(np.max(ratio)),
                "raw_max_phi": float(np.nanmax(vanilla_raw_ratio)),
                "regularized_raw_max_phi": float(np.nanmax(raw_ratio)),
                "num_phi_clipped_bins": int(
                    np.sum(~np.isfinite(vanilla_raw_ratio) | (vanilla_raw_ratio > clip_phi))
                )
                if np.isfinite(clip_phi)
                else 0,
                "phi_laplace_eps": float(phi_laplace_eps),
                "fit_null_phi_mean": fit_null_mean,
            }
        )
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
                "raw_phi": vanilla_raw_ratio,
                "regularized_raw_phi": raw_ratio,
                "phi": ratio,
            }
        )
        return model, bin_table, summary

    if estimator == "kde":
        from scipy.stats import gaussian_kde

        kde0 = gaussian_kde(null_scores, bw_method="silverman")
        kde1 = gaussian_kde(tp_scores, bw_method="silverman")
        model = DensityRatioModel(name="kde", clip_phi=clip_phi, kde0=kde0, kde1=kde1)
        fit_null_phi_mean = float(np.mean(model.phi(null_scores)))
        summary.update(
            {
                "num_bins_realized": float("nan"),
                "max_phi": float(np.max(model.phi(fit_df["score"].to_numpy(dtype=float)))),
                "fit_null_phi_mean": fit_null_phi_mean,
            }
        )
        return model, pd.DataFrame(), summary

    if estimator == "logreg":
        from sklearn.linear_model import LogisticRegression

        scores = fit_df["score"].to_numpy(dtype=float)
        labels = fit_df["is_tp"].to_numpy(dtype=int)
        clf = LogisticRegression(max_iter=max_logreg_iter, solver="lbfgs")
        clf.fit(scores.reshape(-1, 1), labels)
        n0 = len(null_scores)
        n1 = len(tp_scores)
        prior_ratio = n0 / max(n1, 1)
        model = DensityRatioModel(name="logreg", clip_phi=clip_phi, logreg=clf, prior_ratio=prior_ratio)
        fit_null_phi_mean = float(np.mean(model.phi(null_scores)))
        summary.update(
            {
                "num_bins_realized": float("nan"),
                "max_phi": float(np.max(model.phi(scores))),
                "fit_null_phi_mean": fit_null_phi_mean,
            }
        )
        return model, pd.DataFrame(), summary

    if estimator == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        scores = fit_df["score"].to_numpy(dtype=float)
        labels = fit_df["is_tp"].to_numpy(dtype=int)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores, labels)
        n0 = len(null_scores)
        n1 = len(tp_scores)
        prior_ratio = n0 / max(n1, 1)
        model = DensityRatioModel(name="isotonic", clip_phi=clip_phi, isotonic=iso, prior_ratio=prior_ratio)
        fit_null_phi_mean = float(np.mean(model.phi(null_scores)))
        summary.update(
            {
                "num_bins_realized": float("nan"),
                "max_phi": float(np.max(model.phi(scores))),
                "fit_null_phi_mean": fit_null_phi_mean,
            }
        )
        return model, pd.DataFrame(), summary

    raise ValueError(f"Unknown estimator: {estimator}")


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


def compute_self_normalized_evalues(phi_vals: np.ndarray, sum_phi_cal: float, n_cal: int) -> np.ndarray:
    denom = phi_vals + sum_phi_cal
    out = np.zeros_like(phi_vals, dtype=float)
    valid = denom > 0
    out[valid] = (n_cal + 1) * phi_vals[valid] / denom[valid]
    return out


def compute_mondrian_evalues(
    test_df: pd.DataFrame,
    *,
    phi_scores: np.ndarray,
    cal_df: pd.DataFrame,
    model: DensityRatioModel,
    group_col: str,
    min_null: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    cal_null = cal_df[cal_df["is_null"]].copy()
    cal_null_phi = model.phi(cal_null["score"].to_numpy(dtype=float))
    cal_null = cal_null.assign(phi=cal_null_phi)

    group_stats = (
        cal_null.groupby(group_col, sort=False)["phi"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "sum_phi", "count": "n_null"})
    )
    global_sum = float(cal_null_phi.sum())
    global_n = int(cal_null_phi.size)
    if global_sum <= 0 or global_n <= 0:
        raise ValueError("Global cal null pool is empty for Mondrian calibration")

    sums = group_stats["sum_phi"].to_dict()
    counts = group_stats["n_null"].to_dict()

    group_keys = test_df[group_col].to_numpy()
    sum_phi = np.array([sums.get(k, global_sum) for k in group_keys], dtype=float)
    n_cal = np.array([counts.get(k, global_n) for k in group_keys], dtype=int)
    fallback = (n_cal < min_null)
    sum_phi[fallback] = global_sum
    n_cal[fallback] = global_n

    denom = phi_scores + sum_phi
    out = np.zeros_like(phi_scores, dtype=float)
    valid = denom > 0
    out[valid] = (n_cal[valid] + 1) * phi_scores[valid] / denom[valid]

    group_table = (
        group_stats.reset_index()
        .assign(min_null=min_null)
        .rename(columns={group_col: "mondrian_group"})
        .sort_values(["n_null", "sum_phi"], ascending=[False, False])
    )
    return out, group_table


def assign_prompt_clusters(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    k: int,
    min_null: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal_null = cal_df[cal_df["is_null"]]
    if cal_null.empty:
        cal_df = cal_df.assign(mondrian_group="global")
        test_df = test_df.assign(mondrian_group="global")
        return cal_df, test_df

    stats = (
        cal_null.groupby("prompt")
        .agg(mean_score=("score", "mean"), n_null=("score", "size"))
        .reset_index()
    )
    if stats.empty:
        cal_df = cal_df.assign(mondrian_group="global")
        test_df = test_df.assign(mondrian_group="global")
        return cal_df, test_df

    stats.loc[stats["n_null"] < min_null, "mean_score"] = np.nan
    valid = stats["mean_score"].notna()
    stats["cluster"] = "global"
    if valid.any():
        k_use = min(int(k), int(valid.sum()))
        if k_use <= 1:
            stats.loc[valid, "cluster"] = "cluster_0"
        else:
            stats.loc[valid, "cluster"] = pd.qcut(
                stats.loc[valid, "mean_score"],
                q=k_use,
                duplicates="drop",
            ).astype(str)
    prompt_to_cluster = dict(zip(stats["prompt"], stats["cluster"]))

    cal_df = cal_df.assign(
        mondrian_group=cal_df["prompt"].map(prompt_to_cluster).fillna("global")
    )
    test_df = test_df.assign(
        mondrian_group=test_df["prompt"].map(prompt_to_cluster).fillna("global")
    )
    return cal_df, test_df


def ebh_summary(
    test_df: pd.DataFrame,
    *,
    evalue_col: str,
    alpha: float,
    score_floor: float = 0.0,
) -> dict[str, float | int]:
    grouped = [(family_id, group.copy()) for family_id, group in test_df.groupby("family_id", sort=False)]
    # Recall denominator: all TPs in original test set (before floor pruning)
    total_tp = int(test_df["is_tp"].sum())
    total_rejections = 0
    total_fp = 0
    total_tp_rej = 0
    total_pruned = 0
    family_fdps: list[float] = []
    rejected_families = 0
    for _, group in grouped:
        if score_floor > 0:
            before = len(group)
            group = group[group["score"] >= score_floor]
            total_pruned += before - len(group)
        if group.empty:
            family_fdps.append(0.0)
            continue
        rejected_pos = e_bh(group[evalue_col].to_numpy(dtype=float), alpha)
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
    return {
        "alpha": alpha,
        "num_test_families": len(grouped),
        "num_rejected_families": rejected_families,
        "total_rejections": total_rejections,
        "fp": total_fp,
        "tp": total_tp_rej,
        "fdp": total_fp / max(total_rejections, 1),
        "recall": total_tp_rej / max(total_tp, 1),
        "mean_family_fdp": float(np.mean(family_fdps)) if family_fdps else 0.0,
        "score_floor": score_floor,
        "total_pruned": total_pruned,
    }


def bh_summary(
    test_df: pd.DataFrame,
    *,
    pvalue_col: str,
    alpha: float,
    score_floor: float = 0.0,
) -> dict[str, float | int]:
    grouped = [(family_id, group.copy()) for family_id, group in test_df.groupby("family_id", sort=False)]
    total_tp = int(test_df["is_tp"].sum())
    total_rejections = 0
    total_fp = 0
    total_tp_rej = 0
    family_fdps: list[float] = []
    rejected_families = 0
    for _, group in grouped:
        if score_floor > 0:
            group = group[group["score"] >= score_floor]
        if group.empty:
            family_fdps.append(0.0)
            continue
        rejected_pos = bh(group[pvalue_col].to_numpy(dtype=float), alpha)
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
    return {
        "alpha": alpha,
        "num_test_families": len(grouped),
        "num_rejected_families": rejected_families,
        "total_rejections": total_rejections,
        "fp": total_fp,
        "tp": total_tp_rej,
        "fdp": total_fp / max(total_rejections, 1),
        "recall": total_tp_rej / max(total_tp, 1),
        "mean_family_fdp": float(np.mean(family_fdps)) if family_fdps else 0.0,
        "score_floor": score_floor,
    }


def oracle_score_threshold_curve(test_df: pd.DataFrame, alphas: list[float]) -> pd.DataFrame:
    df = test_df.sort_values("score", ascending=False).reset_index(drop=True)
    fp = df["is_null"].to_numpy(dtype=int)
    tp = df["is_tp"].to_numpy(dtype=int)
    cum_fp = np.cumsum(fp)
    cum_tp = np.cumsum(tp)
    cum_total = np.arange(1, len(df) + 1)
    fdp = cum_fp / np.maximum(cum_total, 1)
    recall = cum_tp / max(int(tp.sum()), 1)
    rows: list[dict[str, float | int]] = []
    for alpha in alphas:
        eligible = np.where(fdp <= alpha)[0]
        if eligible.size == 0:
            rows.append(
                {
                    "alpha": alpha,
                    "total_rejections": 0,
                    "fp": 0,
                    "tp": 0,
                    "fdp": 0.0,
                    "recall": 0.0,
                    "threshold": float("nan"),
                }
            )
            continue
        idx = int(eligible[-1])
        rows.append(
            {
                "alpha": alpha,
                "total_rejections": int(cum_total[idx]),
                "fp": int(cum_fp[idx]),
                "tp": int(cum_tp[idx]),
                "fdp": float(fdp[idx]),
                "recall": float(recall[idx]),
                "threshold": float(df.loc[idx, "score"]),
            }
        )
    return pd.DataFrame(rows)


def build_binary_evalues(scores: np.ndarray, cal_null_scores: np.ndarray, alpha: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    m = int(cal_null_scores.size)
    if m == 0:
        raise ValueError("No calibration null scores for binary e-values")
    q = int(np.ceil((1.0 - alpha) * (m + 1)))
    denom = int(np.ceil(alpha * (m + 1)))
    if denom <= 0:
        raise ValueError("Invalid alpha/calibration combination")
    if q > m:
        return np.zeros_like(scores, dtype=float)
    threshold = float(np.sort(cal_null_scores)[q - 1])
    scale = float((m + 1) / denom)
    return np.where(scores >= threshold, scale, 0.0).astype(float)


def family_size_strata(df: pd.DataFrame) -> pd.Series:
    sizes = df.groupby("family_id").size()
    bins = pd.cut(
        sizes,
        bins=[-np.inf, 5, 30, np.inf],
        labels=["K<=5", "5<K<=30", "K>30"],
    )
    return sizes.to_frame("family_size").assign(stratum=bins).reset_index()


def family_size_strata_fine(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby("family_id").size()
    bins = pd.cut(
        sizes,
        bins=[-np.inf, 2, 5, 10, 20, 50, np.inf],
        labels=["K<=2", "3<=K<=5", "6<=K<=10", "11<=K<=20", "21<=K<=50", "K>50"],
    )
    return sizes.to_frame("family_size").assign(stratum_fine=bins).reset_index()


def plot_method_curve(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method, group in summary_df.groupby("method"):
        ax.plot(group["alpha"], group["recall"], marker="o", label=f"{method} recall")
        ax.plot(group["alpha"], group["fdp"], marker="s", linestyle="--", label=f"{method} FDP")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Recall / FDP")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cal_size_sweep(sweep_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for size, group in sweep_df.groupby("cal_null_size"):
        ax.plot(group["alpha"], group["recall"], marker="o", label=f"n={size}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Recall (betting e-BH)")
    ax.set_title("Calibration null size sweep")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    total_start = time.perf_counter()
    alphas = parse_alphas(args.alphas)
    seeds = parse_ints(args.seeds)
    split_ratios = parse_split(args.split_ratios)
    cal_sizes = parse_cal_sizes(args.cal_null_sizes)

    df = pd.read_csv(args.candidate_table)
    for col in ["is_tp", "is_null", "is_prompt_absent"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().eq("true")

    all_seed_rows: list[dict[str, float | int | str]] = []
    all_oracle_rows: list[dict[str, float | int | str]] = []
    all_strata_rows: list[dict[str, float | int | str]] = []
    all_strata_fine_rows: list[dict[str, float | int | str]] = []
    all_cal_sweep_rows: list[dict[str, float | int | str]] = []
    all_loocv_rows: list[dict[str, float | int | str]] = []
    per_seed_summary: dict[str, dict[str, float | int | str]] = {}

    seed_wall_seconds: dict[str, float] = {}
    for seed in seeds:
        seed_start = time.perf_counter()
        seed_key = str(seed)
        split = split_families(df, seed=seed, ratios=split_ratios)
        df_seed = df.copy()
        df_seed["split"] = split

        fit_df = df_seed[df_seed["split"] == "fit"].copy()
        cal_df = df_seed[df_seed["split"] == "cal"].copy()
        test_df = df_seed[df_seed["split"] == "test"].copy()

        model, bin_table, ratio_summary = fit_density_ratio_model(
            fit_df,
            estimator=args.estimator,
            num_bins=args.num_bins,
            smoothing=args.smoothing,
            clip_phi=args.clip_phi,
            max_logreg_iter=args.max_logreg_iter,
            phi_laplace_eps=args.phi_laplace_eps,
        )

        cal_null_scores = cal_df.loc[cal_df["is_null"], "score"].to_numpy(dtype=float)
        if cal_null_scores.size == 0:
            raise ValueError("Calibration split must contain null candidates")
        cal_null_phi = model.phi(cal_null_scores)
        sum_phi_cal = float(np.sum(cal_null_phi))
        n_cal = int(cal_null_phi.size)
        if sum_phi_cal <= 0:
            raise ValueError("Calibration null phi sum is non-positive; check estimator")

        test_scores = test_df["score"].to_numpy(dtype=float)
        test_phi = model.phi(test_scores)
        if args.mondrian in {"prompt", "absent_status", "prompt_cluster"}:
            if args.mondrian == "prompt":
                group_col = "prompt"
            elif args.mondrian == "absent_status":
                group_col = "is_prompt_absent"
            else:
                cal_df, test_df = assign_prompt_clusters(
                    cal_df,
                    test_df,
                    k=args.prompt_cluster_k,
                    min_null=args.mondrian_min_null,
                )
                group_col = "mondrian_group"
            betting_evalues, group_table = compute_mondrian_evalues(
                test_df,
                phi_scores=test_phi,
                cal_df=cal_df,
                model=model,
                group_col=group_col,
                min_null=args.mondrian_min_null,
            )
            test_df["betting_evalue"] = betting_evalues
        else:
            test_df["betting_evalue"] = compute_self_normalized_evalues(test_phi, sum_phi_cal, n_cal)

        loo_evalues = n_cal * cal_null_phi / max(sum_phi_cal, 1e-12)
        all_loocv_rows.append(
            {
                "seed": seed,
                "mean_loo_evalue": float(np.mean(loo_evalues)),
                "std_loo_evalue": float(np.std(loo_evalues)),
                "max_loo_evalue": float(np.max(loo_evalues)),
                "p95_loo_evalue": float(np.quantile(loo_evalues, 0.95)),
                "p99_loo_evalue": float(np.quantile(loo_evalues, 0.99)),
                "mondrian": args.mondrian,
            }
        )

        oracle_curve = oracle_score_threshold_curve(test_df, alphas)
        oracle_curve["seed"] = seed
        oracle_curve["method"] = "score_oracle"
        all_oracle_rows.extend(oracle_curve.to_dict("records"))

        sf = args.score_floor
        for alpha in alphas:
            row = ebh_summary(test_df, evalue_col="betting_evalue", alpha=alpha, score_floor=sf)
            row.update({"seed": seed, "method": "betting"})
            all_seed_rows.append(row)

        for alpha in alphas:
            test_df["binary_evalue"] = build_binary_evalues(test_scores, cal_null_scores, alpha)
            row = ebh_summary(test_df, evalue_col="binary_evalue", alpha=alpha, score_floor=sf)
            row.update({"seed": seed, "method": "binary"})
            all_seed_rows.append(row)

        test_df["conformal_pvalue"] = conformal_pvalues(test_scores, cal_null_scores)
        for alpha in alphas:
            row = bh_summary(test_df, pvalue_col="conformal_pvalue", alpha=alpha, score_floor=sf)
            row.update({"seed": seed, "method": "pvalue_bh"})
            all_seed_rows.append(row)

        strata = family_size_strata(test_df)
        test_df = test_df.merge(strata, on="family_id", how="left")
        for alpha in alphas:
            for stratum, group in test_df.groupby("stratum", sort=False, observed=False):
                row = ebh_summary(group, evalue_col="betting_evalue", alpha=alpha, score_floor=sf)
                row.update({"seed": seed, "method": "betting", "stratum": str(stratum)})
                all_strata_rows.append(row)

        strata_fine = family_size_strata_fine(test_df)
        test_df = test_df.merge(strata_fine[["family_id", "stratum_fine"]], on="family_id", how="left")
        for alpha in alphas:
            for stratum, group in test_df.groupby("stratum_fine", sort=False, observed=False):
                row = ebh_summary(group, evalue_col="betting_evalue", alpha=alpha, score_floor=sf)
                row.update({"seed": seed, "method": "betting", "stratum_fine": str(stratum)})
                all_strata_fine_rows.append(row)

        rng = np.random.default_rng(seed + 13)
        for size in cal_sizes:
            if size is None or size >= n_cal:
                idx = np.arange(n_cal)
                size_label = "all"
            else:
                idx = rng.choice(n_cal, size=size, replace=False)
                size_label = str(size)
            subset_sum = float(np.sum(cal_null_phi[idx]))
            subset_n = int(idx.size)
            if subset_sum <= 0:
                continue
            sweep_evalues = compute_self_normalized_evalues(test_phi, subset_sum, subset_n)
            test_df["betting_evalue_sweep"] = sweep_evalues
            for alpha in alphas:
                row = ebh_summary(test_df, evalue_col="betting_evalue_sweep", alpha=alpha, score_floor=sf)
                row.update(
                    {
                        "seed": seed,
                        "method": "betting",
                        "cal_null_size": size_label,
                        "n_cal": subset_n,
                    }
                )
                all_cal_sweep_rows.append(row)

        per_seed_summary[seed_key] = {
            "seed": seed,
            "num_fit_families": int(fit_df["family_id"].nunique()),
            "num_cal_families": int(cal_df["family_id"].nunique()),
            "num_test_families": int(test_df["family_id"].nunique()),
            "num_fit_candidates": int(len(fit_df)),
            "num_cal_candidates": int(len(cal_df)),
            "num_test_candidates": int(len(test_df)),
            "num_cal_null": int(n_cal),
            "sum_phi_cal": float(sum_phi_cal),
            "density_ratio": ratio_summary,
        }

        seed_dir = args.out_dir / f"seed_{seed}"
        ensure_dir(seed_dir)
        if not bin_table.empty:
            bin_table.to_csv(seed_dir / "density_ratio_bins.csv", index=False)
        test_df.to_csv(seed_dir / "test_candidates_with_evalues.csv", index=False)
        oracle_curve.to_csv(seed_dir / "score_oracle_curve.csv", index=False)
        pd.DataFrame([per_seed_summary[seed_key]]).to_csv(seed_dir / "seed_summary.csv", index=False)
        if args.mondrian in {"prompt", "absent_status"}:
            group_table.to_csv(seed_dir / f"mondrian_{args.mondrian}_groups.csv", index=False)
        seed_wall_seconds[seed_key] = float(time.perf_counter() - seed_start)

    seed_results = pd.DataFrame(all_seed_rows)
    oracle_results = pd.DataFrame(all_oracle_rows)
    strata_results = pd.DataFrame(all_strata_rows)
    strata_fine_results = pd.DataFrame(all_strata_fine_rows)
    cal_sweep_results = pd.DataFrame(all_cal_sweep_rows)
    loo_results = pd.DataFrame(all_loocv_rows)

    seed_results.to_csv(args.out_dir / "seed_alpha_results.csv", index=False)
    oracle_results.to_csv(args.out_dir / "score_oracle_results.csv", index=False)
    strata_results.to_csv(args.out_dir / "family_strata_results.csv", index=False)
    strata_fine_results.to_csv(args.out_dir / "family_strata_fine_results.csv", index=False)
    cal_sweep_results.to_csv(args.out_dir / "cal_null_size_sweep.csv", index=False)
    loo_results.to_csv(args.out_dir / "loo_validity.csv", index=False)
    write_json(args.out_dir / "seed_summaries.json", per_seed_summary)

    method_mean = (
        seed_results.groupby(["method", "alpha"])
        .agg(
            total_rejections=("total_rejections", "mean"),
            fdp=("fdp", "mean"),
            recall=("recall", "mean"),
        )
        .reset_index()
    )
    method_std = (
        seed_results.groupby(["method", "alpha"])
        .agg(
            total_rejections=("total_rejections", "std"),
            fdp=("fdp", "std"),
            recall=("recall", "std"),
        )
        .reset_index()
    )
    method_mean.to_csv(args.out_dir / "mean_results_by_alpha.csv", index=False)
    method_std.to_csv(args.out_dir / "std_results_by_alpha.csv", index=False)

    oracle_mean = (
        oracle_results.groupby(["method", "alpha"])
        .agg(
            total_rejections=("total_rejections", "mean"),
            fdp=("fdp", "mean"),
            recall=("recall", "mean"),
            threshold=("threshold", "mean"),
        )
        .reset_index()
    )
    oracle_mean.to_csv(args.out_dir / "score_oracle_mean_by_alpha.csv", index=False)

    plot_method_curve(method_mean, args.out_dir / "method_alpha_curve.png", "Betting vs Binary (mean over seeds)")
    if not cal_sweep_results.empty:
        sweep_mean = (
            cal_sweep_results.groupby(["cal_null_size", "alpha"])
            .agg(recall=("recall", "mean"))
            .reset_index()
        )
        plot_cal_size_sweep(sweep_mean, args.out_dir / "cal_null_size_sweep.png")

    summary = {
        "candidate_table": str(args.candidate_table),
        "estimator": args.estimator,
        "alphas": alphas,
        "seeds": seeds,
        "split_ratios": split_ratios,
        "mondrian": args.mondrian,
        "mondrian_min_null": args.mondrian_min_null,
        "score_floor": args.score_floor,
        "phi_laplace_eps": args.phi_laplace_eps,
        "num_total_candidates": int(len(df)),
        "num_total_families": int(df["family_id"].nunique()),
        "per_seed_summary_path": str(args.out_dir / "seed_summaries.json"),
        "seed_wall_seconds": seed_wall_seconds,
        "total_wall_seconds": float(time.perf_counter() - total_start),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(f"Wrote formal pipeline outputs to {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
