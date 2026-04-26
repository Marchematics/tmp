#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_hallucination_fdr_matplotlib")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ovd_hallucination_fdr.analysis import (  # noqa: E402
    build_evalue_tables,
    fdp_recall_curve,
    flatten_candidate_records,
    top_fraction_fdp,
)
from ovd_hallucination_fdr.io import ensure_dir, read_jsonl, write_csv, write_json  # noqa: E402


CANDIDATE_FIELDS = [
    "family_id",
    "dataset",
    "image_id",
    "file_name",
    "category_id",
    "category_name",
    "prompt",
    "is_prompt_absent",
    "gt_count",
    "candidate_idx",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "max_iou",
    "is_tp",
    "is_null",
]

CURVE_FIELDS = ["selected", "selected_fraction", "threshold", "fp", "tp", "fdp", "recall"]
EVALUE_FIELDS = CANDIDATE_FIELDS + ["evalue", "split"]
FAMILY_FIELDS = [
    "family_id",
    "dataset",
    "image_id",
    "category_id",
    "category_name",
    "prompt",
    "is_prompt_absent",
    "num_candidates",
    "num_rejected",
    "fp",
    "tp",
    "fdp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze OVD hallucination candidate JSONL.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cal-fraction", type=float, default=0.5)
    parser.add_argument("--curve-grid-size", type=int, default=100)
    return parser.parse_args()


def plot_fdp_recall(curve: list[dict[str, Any]], out_path: Path) -> None:
    if not curve:
        return
    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    xs = [row["selected_fraction"] for row in curve]
    ax1.plot(xs, [row["fdp"] for row in curve], color="#b2182b", label="FDP")
    ax1.set_xlabel("Selected top score fraction")
    ax1.set_ylabel("FDP", color="#b2182b")
    ax1.tick_params(axis="y", labelcolor="#b2182b")
    ax1.set_ylim(-0.03, 1.03)

    ax2 = ax1.twinx()
    ax2.plot(xs, [row["recall"] for row in curve], color="#2166ac", label="Recall")
    ax2.set_ylabel("Recall", color="#2166ac")
    ax2.tick_params(axis="y", labelcolor="#2166ac")
    ax2.set_ylim(-0.03, 1.03)

    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_evalue_hist(evalue_rows: list[dict[str, Any]], out_path: Path) -> None:
    if not evalue_rows:
        return
    null_values = [row["evalue"] for row in evalue_rows if row["is_null"]]
    alt_values = [row["evalue"] for row in evalue_rows if row["is_tp"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bins = sorted(set(null_values + alt_values))
    if len(bins) <= 2:
        bins = max(5, len(bins) + 2)
    ax.hist(null_values, bins=bins, alpha=0.65, label="Null", color="#b2182b")
    if alt_values:
        ax.hist(alt_values, bins=bins, alpha=0.65, label="TP", color="#2166ac")
    ax.set_xlabel("Conformal binary e-value")
    ax.set_ylabel("Candidate count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    family_records = read_jsonl(args.input_jsonl)
    candidate_rows = flatten_candidate_records(family_records)
    absent_rows = [row for row in candidate_rows if row["is_prompt_absent"]]

    curve_all = fdp_recall_curve(candidate_rows, grid_size=args.curve_grid_size)
    curve_absent = fdp_recall_curve(absent_rows, grid_size=args.curve_grid_size)
    evalue_rows, family_rows, evalue_summary = build_evalue_tables(
        candidate_rows,
        alpha=args.alpha,
        cal_fraction=args.cal_fraction,
    )

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "alpha": args.alpha,
        "cal_fraction": args.cal_fraction,
        "num_families": len(family_records),
        "num_candidates": len(candidate_rows),
        "num_absent_candidates": len(absent_rows),
        "num_true_positive_candidates": int(sum(1 for row in candidate_rows if row["is_tp"])),
        "num_null_candidates": int(sum(1 for row in candidate_rows if row["is_null"])),
        "top10pct_fdp": top_fraction_fdp(candidate_rows, 0.10),
        "absent_top10pct_fdp": top_fraction_fdp(absent_rows, 0.10),
        "evalue": evalue_summary,
    }

    write_json(args.out_dir / "summary.json", summary)
    write_csv(args.out_dir / "candidate_table.csv", candidate_rows, CANDIDATE_FIELDS)
    write_csv(args.out_dir / "fdp_recall_curve.csv", curve_all, CURVE_FIELDS)
    write_csv(args.out_dir / "fdp_recall_curve_absent.csv", curve_absent, CURVE_FIELDS)
    write_csv(args.out_dir / "evalue_table.csv", evalue_rows, EVALUE_FIELDS)
    write_csv(args.out_dir / "e_bh_family_summary.csv", family_rows, FAMILY_FIELDS)
    plot_fdp_recall(curve_all, args.out_dir / "fdp_recall_curve.png")
    plot_fdp_recall(curve_absent, args.out_dir / "fdp_recall_curve_absent.png")
    plot_evalue_hist(evalue_rows, args.out_dir / "evalue_hist.png")

    print(f"Wrote analysis to {args.out_dir}")
    print(f"Candidates: {summary['num_candidates']} | absent candidates: {summary['num_absent_candidates']}")
    print(f"Top-10% FDP: {summary['top10pct_fdp']}")
    print(f"Absent top-10% FDP: {summary['absent_top10pct_fdp']}")
    print(f"Test null e-value mean: {evalue_summary['test_null_evalue_mean']}")


if __name__ == "__main__":
    main()
