"""NMS-aware cluster e-BH evaluation (post-hoc over existing seed dirs).

Implements `docs/theory_notes/nms_aware_evalue.md` Theorem 1 (predictable
convex form, uniform weights) as a pure post-processing layer on top of
existing `seed_*/test_candidates_with_evalues.csv` outputs. Nothing in the
main formal betting pipeline changes — we just re-aggregate the per-candidate
e-values into cluster e-values and re-run e-BH.

Usage:
    python scripts/run_nms_aware_evaluation.py \
        --source-dir outputs/coco_gdino_1000_20seed_formal_hist \
        --out-dir outputs/coco_gdino_1000_nmsaware_20seed \
        --iou-thresh 0.7 \
        --alphas 0.05,0.10,0.15,0.20

Outputs:
    <out-dir>/summary.csv          # per-(alpha, seed) NMS-aware metrics
    <out-dir>/aggregate.csv        # per-alpha mean/std (20-seed)
    <out-dir>/baseline_delta.csv   # side-by-side vs baseline e-BH
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ovd_hallucination_fdr.evalues import e_bh, nms_aware_e_bh  # noqa: E402
from ovd_hallucination_fdr.matching import nms_clusters_xyxy  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True,
                    help="Existing formal_hist output dir with seed_*/test_candidates_with_evalues.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iou-thresh", type=float, default=0.7)
    ap.add_argument("--alphas", type=str, default="0.05,0.10,0.15,0.20")
    ap.add_argument("--evalue-col", type=str, default="betting_evalue")
    ap.add_argument("--seeds-limit", type=int, default=None,
                    help="If set, only process this many seeds (for smoke tests).")
    return ap.parse_args()


def family_clusters(
    group: pd.DataFrame, iou_thresh: float
) -> tuple[list[list[int]], np.ndarray]:
    """Return (clusters in positional-idx space, positional order).

    The positional order reindexes the family members 0..K-1; mapping back
    to original dataframe indices uses group.index[positional_idx].
    """
    boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    clusters = nms_clusters_xyxy(boxes, iou_thresh)
    return clusters, group.index.to_numpy()


def evaluate_family(
    group: pd.DataFrame,
    evalue_col: str,
    alpha: float,
    iou_thresh: float,
) -> dict:
    """Run NMS-aware cluster e-BH within one family.

    Returns a dict with counts (rejected clusters, fp clusters, tp clusters,
    sum of is_tp among rejected clusters' members — optimistic recall count,
    and cluster-level TP count — conservative recall count).
    """
    K = len(group)
    if K == 0:
        return dict(
            K=0, M=0, rej=0, fp=0, tp=0,
            tp_members=0, cluster_sizes=[],
        )
    evs = group[evalue_col].to_numpy(dtype=np.float64)
    boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    clusters = nms_clusters_xyxy(boxes, iou_thresh)
    M = len(clusters)
    # Build per-cluster TP/null flags
    is_tp_vec = group["is_tp"].to_numpy(dtype=bool)
    is_null_vec = group["is_null"].to_numpy(dtype=bool)
    rej_clusters, _ = nms_aware_e_bh(evs, clusters, alpha)
    fp = 0
    tp = 0
    tp_members = 0
    for c_idx in rej_clusters:
        members = clusters[c_idx]
        any_tp = bool(is_tp_vec[members].any())
        all_null = bool(is_null_vec[members].all())
        if any_tp:
            tp += 1
            tp_members += int(is_tp_vec[members].sum())
        elif all_null:
            fp += 1
        # else: ambiguous (no tp but not all null) — skip
    return dict(
        K=K,
        M=M,
        rej=len(rej_clusters),
        fp=fp,
        tp=tp,
        tp_members=tp_members,
        cluster_sizes=[len(c) for c in clusters],
    )


def evaluate_baseline_family(
    group: pd.DataFrame, evalue_col: str, alpha: float, iou_thresh: float
) -> dict:
    """Replicate baseline e-BH on the same family for side-by-side comparison.

    Also computes an apples-to-apples cluster-level hit count: we build the
    SAME NMS cluster graph on the family, then count how many clusters the
    baseline's rejected candidates collectively cover (a "cluster hit" = any
    candidate in that cluster was rejected). This makes recall comparable
    between baseline (candidate-level) and nms-aware (cluster-level).
    """
    K = len(group)
    if K == 0:
        return dict(rej=0, fp=0, tp=0, cluster_hit_tp=0, cluster_hit_fp=0)
    evs = group[evalue_col].to_numpy(dtype=np.float64)
    is_tp_vec = group["is_tp"].to_numpy(dtype=bool)
    is_null_vec = group["is_null"].to_numpy(dtype=bool)
    rej = e_bh(evs, alpha)
    if not rej:
        return dict(rej=0, fp=0, tp=0, cluster_hit_tp=0, cluster_hit_fp=0)
    sel_tp = int(is_tp_vec[rej].sum())
    sel_null = int(is_null_vec[rej].sum())
    # Apples-to-apples cluster hits on the SAME NMS graph
    boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    clusters = nms_clusters_xyxy(boxes, iou_thresh)
    # Map positional index -> cluster_idx
    cand2cluster = np.full(K, -1, dtype=np.int32)
    for c_idx, members in enumerate(clusters):
        for m in members:
            cand2cluster[m] = c_idx
    hit_clusters = set(int(cand2cluster[i]) for i in rej)
    cluster_hit_tp = 0
    cluster_hit_fp = 0
    for c_idx in hit_clusters:
        members = clusters[c_idx]
        any_tp = bool(is_tp_vec[members].any())
        all_null = bool(is_null_vec[members].all())
        if any_tp:
            cluster_hit_tp += 1
        elif all_null:
            cluster_hit_fp += 1
    return dict(
        rej=len(rej), fp=sel_null, tp=sel_tp,
        cluster_hit_tp=cluster_hit_tp, cluster_hit_fp=cluster_hit_fp,
    )


def process_seed_dir(
    seed_dir: Path,
    evalue_col: str,
    alphas: list[float],
    iou_thresh: float,
) -> list[dict]:
    cand_path = seed_dir / "test_candidates_with_evalues.csv"
    if not cand_path.exists():
        return []
    df = pd.read_csv(cand_path)
    need_cols = {"family_id", "x1", "y1", "x2", "y2", "is_tp", "is_null", evalue_col}
    missing = need_cols - set(df.columns)
    if missing:
        print(f"  [skip] {seed_dir.name}: missing {missing}")
        return []
    total_tp_candidates = int(df["is_tp"].sum())
    # Compute total distinct "TP clusters" across the whole test set under the SAME NMS graph
    # (a cluster containing at least one TP candidate). Used as a cluster-level recall denominator.
    total_tp_clusters = 0
    for _fid, _g in df.groupby("family_id", sort=False):
        if len(_g) == 0:
            continue
        _boxes = _g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
        _is_tp = _g["is_tp"].to_numpy(dtype=bool)
        _clusters = nms_clusters_xyxy(_boxes, iou_thresh)
        for _members in _clusters:
            if _is_tp[_members].any():
                total_tp_clusters += 1
    rows = []
    for alpha in alphas:
        # NMS-aware
        fam_stats = []
        tot_rej = tot_fp = tot_tp = tot_tp_members = 0
        tot_K = tot_M = 0
        cluster_sizes_all: list[int] = []
        family_fdps_nms: list[float] = []
        # Baseline
        base_rej = base_fp = base_tp = 0
        base_cluster_hit_tp = base_cluster_hit_fp = 0
        family_fdps_base: list[float] = []

        for fid, g in df.groupby("family_id", sort=False):
            s = evaluate_family(g, evalue_col, alpha, iou_thresh)
            fam_stats.append(s)
            tot_rej += s["rej"]
            tot_fp += s["fp"]
            tot_tp += s["tp"]
            tot_tp_members += s["tp_members"]
            tot_K += s["K"]
            tot_M += s["M"]
            cluster_sizes_all.extend(s["cluster_sizes"])
            if s["rej"] > 0:
                family_fdps_nms.append(s["fp"] / s["rej"])
            else:
                family_fdps_nms.append(0.0)

            b = evaluate_baseline_family(g, evalue_col, alpha, iou_thresh)
            base_rej += b["rej"]
            base_fp += b["fp"]
            base_tp += b["tp"]
            base_cluster_hit_tp += b["cluster_hit_tp"]
            base_cluster_hit_fp += b["cluster_hit_fp"]
            if b["rej"] > 0:
                family_fdps_base.append(b["fp"] / b["rej"])
            else:
                family_fdps_base.append(0.0)

        rows.append(dict(
            seed=int(seed_dir.name.split("_")[-1]) if seed_dir.name.startswith("seed_") else -1,
            alpha=alpha,
            iou_thresh=iou_thresh,
            n_candidates=tot_K,
            n_clusters=tot_M,
            compression_ratio=(tot_M / tot_K) if tot_K else 0.0,
            mean_cluster_size=float(np.mean(cluster_sizes_all)) if cluster_sizes_all else 0.0,
            max_cluster_size=int(max(cluster_sizes_all)) if cluster_sizes_all else 0,
            nms_rej=tot_rej,
            nms_fp=tot_fp,
            nms_tp=tot_tp,
            nms_tp_members=tot_tp_members,
            nms_fdp_pooled=tot_fp / max(tot_rej, 1),
            nms_family_fdp_mean=float(np.mean(family_fdps_nms)) if family_fdps_nms else 0.0,
            # Conservative recall: rejected clusters containing a TP / total TP candidates
            # Optimistic recall: sum is_tp in rejected clusters / total TP candidates
            nms_recall_cluster=tot_tp / max(total_tp_candidates, 1),
            nms_recall_members=tot_tp_members / max(total_tp_candidates, 1),
            base_rej=base_rej,
            base_fp=base_fp,
            base_tp=base_tp,
            base_fdp_pooled=base_fp / max(base_rej, 1),
            base_family_fdp_mean=float(np.mean(family_fdps_base)) if family_fdps_base else 0.0,
            base_recall=base_tp / max(total_tp_candidates, 1),
            # APPLES-TO-APPLES object-level metrics: both methods project onto
            # the same NMS cluster graph, then count distinct clusters containing
            # at least one TP (a proxy for distinct underlying GT objects).
            total_tp_clusters=total_tp_clusters,  # computed per-seed outside loop
            nms_hit_tp_clusters=tot_tp,
            base_hit_tp_clusters=base_cluster_hit_tp,
            nms_recall_object=tot_tp / max(total_tp_clusters, 1),
            base_recall_object=base_cluster_hit_tp / max(total_tp_clusters, 1),
        ))
    return rows


def main():
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(d for d in args.source_dir.iterdir()
                       if d.is_dir() and d.name.startswith("seed_"))
    if args.seeds_limit:
        seed_dirs = seed_dirs[: args.seeds_limit]
    print(f"processing {len(seed_dirs)} seed dirs from {args.source_dir}")

    all_rows = []
    for sd in seed_dirs:
        r = process_seed_dir(sd, args.evalue_col, alphas, args.iou_thresh)
        print(f"  {sd.name}: {len(r)} alpha rows")
        all_rows.extend(r)

    if not all_rows:
        print("no rows produced")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_dir / "summary.csv", index=False)
    print(f"written: {args.out_dir/'summary.csv'}")

    # Aggregate per alpha
    agg = df.groupby("alpha").agg(
        n_seeds=("seed", "count"),
        compression_ratio_mean=("compression_ratio", "mean"),
        mean_cluster_size_mean=("mean_cluster_size", "mean"),
        max_cluster_size_max=("max_cluster_size", "max"),
        nms_fdp_pooled_mean=("nms_fdp_pooled", "mean"),
        nms_fdp_pooled_std=("nms_fdp_pooled", "std"),
        nms_family_fdp_mean=("nms_family_fdp_mean", "mean"),
        nms_rej_mean=("nms_rej", "mean"),
        nms_recall_cluster_mean=("nms_recall_cluster", "mean"),
        nms_recall_members_mean=("nms_recall_members", "mean"),
        nms_recall_object_mean=("nms_recall_object", "mean"),
        base_fdp_pooled_mean=("base_fdp_pooled", "mean"),
        base_family_fdp_mean=("base_family_fdp_mean", "mean"),
        base_rej_mean=("base_rej", "mean"),
        base_recall_mean=("base_recall", "mean"),
        base_recall_object_mean=("base_recall_object", "mean"),
    ).reset_index()
    agg["rej_ratio_nms_over_base"] = (
        agg["nms_rej_mean"] / agg["base_rej_mean"].replace(0, np.nan)
    ).round(4)
    agg["recall_ratio_nms_members_over_base"] = (
        agg["nms_recall_members_mean"] / agg["base_recall_mean"].replace(0, np.nan)
    ).round(4)
    # The honest apples-to-apples recall ratio on the shared cluster graph:
    agg["recall_ratio_object"] = (
        agg["nms_recall_object_mean"] / agg["base_recall_object_mean"].replace(0, np.nan)
    ).round(4)
    agg.to_csv(args.out_dir / "aggregate.csv", index=False)
    print(f"written: {args.out_dir/'aggregate.csv'}")
    print("\n=== aggregate ===")
    cols_show = [
        "alpha", "n_seeds",
        "compression_ratio_mean", "mean_cluster_size_mean", "max_cluster_size_max",
        "nms_fdp_pooled_mean", "base_fdp_pooled_mean",
        "nms_family_fdp_mean", "base_family_fdp_mean",
        "nms_rej_mean", "base_rej_mean", "rej_ratio_nms_over_base",
        "nms_recall_object_mean", "base_recall_object_mean", "recall_ratio_object",
    ]
    print(agg[cols_show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
