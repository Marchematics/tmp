#!/usr/bin/env python3
"""Aggregate estimator ablation: hist, KDE, logistic, isotonic + bin sweep.

Reads existing formal_<estimator>/mean_results_by_alpha.csv from
coco_groundingdino_gonogo_1000_mix_formal_*/ directories.
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "estimator_ablation_summary.csv"

ESTIMATORS = [
    ("hist (80-bin, default)", "coco_groundingdino_gonogo_1000_mix_formal_hist"),
    ("isotonic",                "coco_groundingdino_gonogo_1000_mix_formal_isotonic"),
    ("KDE",                     "coco_groundingdino_gonogo_1000_mix_formal_kde"),
    ("logistic",                "coco_groundingdino_gonogo_1000_mix_formal_logreg"),
    ("hist + smooth=0.5",       "coco_groundingdino_gonogo_1000_mix_formal_hist_smooth0p5"),
    ("hist + smooth=1.0",       "coco_groundingdino_gonogo_1000_mix_formal_hist_smooth1"),
    ("hist + smooth=2.0",       "coco_groundingdino_gonogo_1000_mix_formal_hist_smooth2"),
    ("hist + smooth=5.0",       "coco_groundingdino_gonogo_1000_mix_formal_hist_smooth5"),
]


def main() -> None:
    rows = []
    for name, dirname in ESTIMATORS:
        path = PROJECT_ROOT / "outputs" / dirname / "mean_results_by_alpha.csv"
        if not path.exists():
            print(f"[skip] {path}")
            continue
        df = pd.read_csv(path)
        df = df[df["method"] == "betting"]
        for _, r in df.iterrows():
            rows.append({
                "estimator": name,
                "alpha": float(r["alpha"]),
                "rejections": float(r["total_rejections"]),
                "fdp_pooled": float(r["fdp"]),
                "recall": float(r["recall"]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} rows to {OUT}\n")
    print("=== α=0.10 summary (COCO/GDino 1k, 5 seeds) ===")
    sub = out[out["alpha"] == 0.10].sort_values("recall", ascending=False)
    print(sub[["estimator", "rejections", "fdp_pooled", "recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
