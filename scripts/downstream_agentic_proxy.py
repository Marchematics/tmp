"""Lightweight downstream agentic proxy for OVD wrapper vs naive thresholding.

Two zero-cost tasks derived from existing seed_*/test_candidates_with_evalues.csv:

  Task A (presence/absence):
      "Is there a {prompt} in this image?"
      GT: NOT is_prompt_absent
      Naive: any candidate with score > tau -> YES
      Wrapper: any candidate in e-BH reject set at alpha -> YES

  Task B (count):
      "How many {prompt} are there?"
      GT: number of DISTINCT TP clusters (object-level) in the family
      Naive: count of candidates with score > tau
      Wrapper: count of clusters rejected by nms-aware calibrated e-BH at alpha

We aggregate over COCO/GDino and COCO/RegionCLIP seed_0 (single seed, ~500 families
each -> >> the "20 image" floor). Metric: balanced accuracy for Task A, mean absolute
count error for Task B. The claim is that the wrapper provides one FDR-calibrated
operating point that dominates *every* fixed naive threshold simultaneously.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ovd_hallucination_fdr.evalues import e_bh  # noqa: E402
from ovd_hallucination_fdr.matching import nms_clusters_xyxy  # noqa: E402


SOURCES = [
    ("COCO/GDino",
     "outputs/coco_groundingdino_gonogo_1000_mix_formal_hist/seed_0/test_candidates_with_evalues.csv"),
    ("COCO/RegionCLIP",
     "outputs/coco_regionclip_1000_20seed_formal_hist/seed_0/test_candidates_with_evalues.csv"),
]
NAIVE_TAUS = [0.10, 0.20, 0.30, 0.40, 0.50]
WRAPPER_ALPHAS = [0.05, 0.10, 0.20]
IOU_THRESH = 0.50


def task_A_presence(df: pd.DataFrame, tau: float | None, alpha: float | None) -> dict:
    """Binary decision per family."""
    tp = fp = tn = fn = 0
    for fid, g in df.groupby("family_id"):
        gt_yes = not bool(g["is_prompt_absent"].iloc[0])
        if tau is not None:
            pred_yes = bool((g["score"] > tau).any())
        else:
            evs = g["betting_evalue"].to_numpy(dtype=np.float64)
            rej = e_bh(evs, alpha)
            pred_yes = len(rej) > 0
        if gt_yes and pred_yes:
            tp += 1
        elif gt_yes and not pred_yes:
            fn += 1
        elif not gt_yes and pred_yes:
            fp += 1
        else:
            tn += 1
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return dict(
        sensitivity=sens, specificity=spec,
        balanced_acc=(sens + spec) / 2,
        accuracy=(tp + tn) / max(tp + tn + fp + fn, 1),
        tp=tp, fp=fp, tn=tn, fn=fn,
        false_alarm_rate=fp / max(tp + fn + fp + tn, 1),
    )


def task_B_count(df: pd.DataFrame, tau: float | None, alpha: float | None) -> dict:
    """Object-count error per family."""
    abs_errs = []
    zero_errs = []  # absent-family error (gt_count=0)
    present_errs = []
    for fid, g in df.groupby("family_id"):
        absent = bool(g["is_prompt_absent"].iloc[0])
        # GT count = distinct TP-clusters on the NMS graph (object-level)
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
        is_tp = g["is_tp"].to_numpy(dtype=bool)
        clusters = nms_clusters_xyxy(boxes, IOU_THRESH)
        if absent:
            gt_count = 0
        else:
            gt_count = sum(1 for m in clusters if is_tp[m].any())
        if tau is not None:
            pred_idx = np.where(g["score"].to_numpy() > tau)[0]
            pred_count = len({
                ci for ci, m in enumerate(clusters)
                if len(set(m) & set(pred_idx.tolist())) > 0
            })
        else:
            evs = g["betting_evalue"].to_numpy(dtype=np.float64)
            rej = e_bh(evs, alpha)
            pred_count = len({
                ci for ci, m in enumerate(clusters)
                if len(set(m) & set(rej)) > 0
            })
        err = abs(pred_count - gt_count)
        abs_errs.append(err)
        if absent:
            zero_errs.append(err)
        else:
            present_errs.append(err)
    return dict(
        mae=float(np.mean(abs_errs)),
        mae_absent=float(np.mean(zero_errs)) if zero_errs else 0.0,
        mae_present=float(np.mean(present_errs)) if present_errs else 0.0,
        n_families=len(abs_errs),
    )


def main() -> None:
    rows = []
    for tag, rel in SOURCES:
        path = PROJECT / rel
        if not path.exists():
            print(f"[skip] {path}")
            continue
        df = pd.read_csv(path)
        n_fams = df["family_id"].nunique()
        print(f"\n=== {tag}: {n_fams} families ===")
        # Task A - naive
        for tau in NAIVE_TAUS:
            r = task_A_presence(df, tau, None)
            b = task_B_count(df, tau, None)
            rows.append(dict(
                source=tag, task="decision_bal_acc", method=f"naive_tau={tau:.2f}",
                bal_acc=r["balanced_acc"], acc=r["accuracy"],
                far=r["false_alarm_rate"], mae=b["mae"],
                mae_absent=b["mae_absent"], mae_present=b["mae_present"],
            ))
        # Task A - wrapper
        for alpha in WRAPPER_ALPHAS:
            r = task_A_presence(df, None, alpha)
            b = task_B_count(df, None, alpha)
            rows.append(dict(
                source=tag, task="decision_bal_acc",
                method=f"wrapper_eBH_alpha={alpha:.2f}",
                bal_acc=r["balanced_acc"], acc=r["accuracy"],
                far=r["false_alarm_rate"], mae=b["mae"],
                mae_absent=b["mae_absent"], mae_present=b["mae_present"],
            ))
    out_df = pd.DataFrame(rows)
    out_path = PROJECT / "outputs" / "downstream_agentic_proxy"
    out_path.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path / "results.csv", index=False)
    print("\n=== Balanced accuracy / FAR / count-MAE ===")
    print(out_df.round(4).to_string(index=False))
    print(f"\nwritten: {out_path/'results.csv'}")


if __name__ == "__main__":
    main()
