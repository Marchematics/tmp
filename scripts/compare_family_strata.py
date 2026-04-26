#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_hallucination_fdr_matplotlib")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare family-size-stratified performance across two formal pipeline "
            "outputs and produce a paper-ready summary plot."
        )
    )
    parser.add_argument("--left-csv", type=Path, required=True)
    parser.add_argument("--right-csv", type=Path, required=True)
    parser.add_argument("--left-label", type=str, default="left")
    parser.add_argument("--right-label", type=str, default="right")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_strata(path: Path, label: str, alpha: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["method"] == "betting") & (df["alpha"].round(6) == round(alpha, 6))].copy()
    if df.empty:
        raise ValueError(f"No betting rows found for alpha={alpha} in {path}")
    df["dataset"] = label
    df["rejection_rate"] = df["num_rejected_families"] / df["num_test_families"].clip(lower=1)
    return df


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    left = load_strata(args.left_csv, args.left_label, args.alpha)
    right = load_strata(args.right_csv, args.right_label, args.alpha)
    both = pd.concat([left, right], ignore_index=True)

    order = ["K<=5", "5<K<=30", "K>30"]
    summary = (
        both.groupby(["dataset", "stratum"], as_index=False)
        .agg(
            num_test_families=("num_test_families", "mean"),
            rejection_rate=("rejection_rate", "mean"),
            recall=("recall", "mean"),
            fdp=("fdp", "mean"),
            total_rejections=("total_rejections", "mean"),
        )
    )
    summary["stratum"] = pd.Categorical(summary["stratum"], categories=order, ordered=True)
    summary = summary.sort_values(["stratum", "dataset"]).reset_index(drop=True)

    summary.to_csv(args.out_dir / "family_strata_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {args.left_label: "#1f77b4", args.right_label: "#d62728"}

    x = list(range(len(order)))
    width = 0.36
    for i, label in enumerate([args.left_label, args.right_label]):
        subset = summary[summary["dataset"] == label].set_index("stratum").reindex(order).reset_index()
        offset = (-width / 2) if i == 0 else (width / 2)
        xpos = [v + offset for v in x]
        axes[0].bar(xpos, subset["rejection_rate"], width=width, label=label, color=colors[label], alpha=0.85)
        axes[1].bar(xpos, subset["num_test_families"], width=width, label=label, color=colors[label], alpha=0.85)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order)
    axes[0].set_ylabel("Mean rejection rate")
    axes[0].set_title(f"Betting e-BH at alpha={args.alpha:.2f}")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend(frameon=False)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_ylabel("Mean test families")
    axes[1].set_title("Family-size mass")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.out_dir / "family_strata_comparison.png", dpi=180)
    plt.close(fig)

    summary_payload = {
        "alpha": args.alpha,
        "left_label": args.left_label,
        "right_label": args.right_label,
        "rows": summary.to_dict("records"),
    }
    (args.out_dir / "family_strata_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(f"Wrote family-size comparison outputs to {args.out_dir}")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
