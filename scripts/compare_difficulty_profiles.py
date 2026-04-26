#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_hallucination_fdr_matplotlib")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate-level difficulty profiles between two OVD runs. "
            "Produces TP/null score diagnostics, family-size diagnostics, and a "
            "compact summary json/csv for paper figures."
        )
    )
    parser.add_argument("--left-table", type=Path, required=True)
    parser.add_argument("--right-table", type=Path, required=True)
    parser.add_argument("--left-label", type=str, default="left")
    parser.add_argument("--right-label", type=str, default="right")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_candidate_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["family_id", "image_id", "score", "is_tp", "is_null"],
    )
    for col in ["is_tp", "is_null"]:
        df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def quantile_summary(scores: pd.Series) -> dict[str, float]:
    return {
        "count": float(scores.size),
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "q75": float(scores.quantile(0.75)),
        "q90": float(scores.quantile(0.90)),
        "q95": float(scores.quantile(0.95)),
        "q99": float(scores.quantile(0.99)),
    }


def family_summary(df: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    sizes = df.groupby("family_id").size()
    return sizes, {
        "count": float(sizes.size),
        "mean": float(sizes.mean()),
        "median": float(sizes.median()),
        "q75": float(sizes.quantile(0.75)),
        "q90": float(sizes.quantile(0.90)),
        "q95": float(sizes.quantile(0.95)),
        "q99": float(sizes.quantile(0.99)),
        "max": float(sizes.max()),
    }


def empirical_auc(df: pd.DataFrame) -> float:
    tp_scores = df.loc[df["is_tp"], "score"].to_numpy(dtype=float)
    null_scores = df.loc[df["is_null"], "score"].to_numpy(dtype=float)
    if tp_scores.size == 0 or null_scores.size == 0:
        return float("nan")
    comparisons = (tp_scores[:, None] > null_scores[None, :]).mean()
    ties = (tp_scores[:, None] == null_scores[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def build_dataset_summary(df: pd.DataFrame, label: str) -> tuple[dict[str, object], pd.Series]:
    tp_scores = df.loc[df["is_tp"], "score"]
    null_scores = df.loc[df["is_null"], "score"]
    family_sizes, fam_summary = family_summary(df)
    summary = {
        "label": label,
        "num_candidates": int(len(df)),
        "num_images": int(df["image_id"].nunique()),
        "num_families": int(df["family_id"].nunique()),
        "num_tp": int(df["is_tp"].sum()),
        "num_null": int(df["is_null"].sum()),
        "score_auc_tp_vs_null": empirical_auc(df),
        "tp_score": quantile_summary(tp_scores),
        "null_score": quantile_summary(null_scores),
        "family_size": fam_summary,
    }
    return summary, family_sizes


def plot_profiles(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_label: str,
    right_label: str,
    out_path: Path,
) -> None:
    colors = {left_label: "#1f77b4", right_label: "#d62728"}

    left_tp = left_df.loc[left_df["is_tp"], "score"].to_numpy(dtype=float)
    right_tp = right_df.loc[right_df["is_tp"], "score"].to_numpy(dtype=float)
    left_null = left_df.loc[left_df["is_null"], "score"].to_numpy(dtype=float)
    right_null = right_df.loc[right_df["is_null"], "score"].to_numpy(dtype=float)
    left_fam = left_df.groupby("family_id").size().to_numpy(dtype=float)
    right_fam = right_df.groupby("family_id").size().to_numpy(dtype=float)

    score_bins = np.linspace(0.0, 1.0, 41)
    fam_bins = np.arange(1, max(int(left_fam.max()), int(right_fam.max())) + 3) - 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    axes[0].hist(left_tp, bins=score_bins, density=True, alpha=0.45, color=colors[left_label], label=left_label)
    axes[0].hist(right_tp, bins=score_bins, density=True, alpha=0.45, color=colors[right_label], label=right_label)
    axes[0].set_title("TP Score Distribution")
    axes[0].set_xlabel("score")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].hist(left_null, bins=score_bins, density=True, alpha=0.45, color=colors[left_label], label=left_label)
    axes[1].hist(right_null, bins=score_bins, density=True, alpha=0.45, color=colors[right_label], label=right_label)
    axes[1].set_title("Null Score Distribution")
    axes[1].set_xlabel("score")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].hist(left_fam, bins=fam_bins, density=True, alpha=0.45, color=colors[left_label], label=left_label)
    axes[2].hist(right_fam, bins=fam_bins, density=True, alpha=0.45, color=colors[right_label], label=right_label)
    axes[2].set_title("Per-Family Candidate Count")
    axes[2].set_xlabel("family size K")
    axes[2].set_ylabel("density")
    axes[2].set_xlim(0, min(max(left_fam.max(), right_fam.max()), 120))
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def flatten_summary(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    label = str(summary["label"])
    for block in ["tp_score", "null_score", "family_size"]:
        block_data = summary[block]
        assert isinstance(block_data, dict)
        for key, value in block_data.items():
            rows.append({"dataset": label, "block": block, "metric": key, "value": value})
    rows.append({"dataset": label, "block": "global", "metric": "score_auc_tp_vs_null", "value": summary["score_auc_tp_vs_null"]})
    rows.append({"dataset": label, "block": "global", "metric": "num_candidates", "value": summary["num_candidates"]})
    rows.append({"dataset": label, "block": "global", "metric": "num_images", "value": summary["num_images"]})
    rows.append({"dataset": label, "block": "global", "metric": "num_families", "value": summary["num_families"]})
    rows.append({"dataset": label, "block": "global", "metric": "num_tp", "value": summary["num_tp"]})
    rows.append({"dataset": label, "block": "global", "metric": "num_null", "value": summary["num_null"]})
    return rows


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    left_df = load_candidate_table(args.left_table)
    right_df = load_candidate_table(args.right_table)

    left_summary, _ = build_dataset_summary(left_df, args.left_label)
    right_summary, _ = build_dataset_summary(right_df, args.right_label)

    plot_profiles(
        left_df,
        right_df,
        args.left_label,
        args.right_label,
        args.out_dir / "difficulty_profiles.png",
    )

    summary = {"left": left_summary, "right": right_summary}
    (args.out_dir / "difficulty_summary.json").write_text(json.dumps(summary, indent=2))

    rows = flatten_summary(left_summary) + flatten_summary(right_summary)
    pd.DataFrame(rows).to_csv(args.out_dir / "difficulty_summary_long.csv", index=False)

    print(f"Wrote difficulty comparison outputs to {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
