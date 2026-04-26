"""Score-adaptive calibrated NMS-aware cluster e-values (Theory Section 5.3, Algorithm 2).

Unlike the uniform-weight convex form (Theorem 1), this driver uses a
score-adaptive cluster statistic R(C) (e.g., max phi) and validates via
CLUSTER-LEVEL self-normalization against a calibration pool of all-null
clusters. Validity mirrors the paper's candidate-level self-normalized
betting e-value, just applied at cluster granularity.

Usage:
    python scripts/run_nms_aware_calibrated.py \
        --candidate-table outputs/coco_groundingdino_gonogo_1000_mix_analysis/candidate_table.csv \
        --out-dir outputs/coco_gdino_nmsaware_calibrated_20seed \
        --iou-thresh 0.5 \
        --alphas 0.05,0.10,0.15,0.20 \
        --seeds 0,1,...,19 \
        --R-statistic max_phi
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

from ovd_hallucination_fdr.evalues import e_bh  # noqa: E402
from ovd_hallucination_fdr.matching import nms_clusters_xyxy  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-table", type=Path, default=None,
                    help="Pre-expanded candidate_table.csv")
    ap.add_argument("--candidate-jsonl", type=Path, default=None,
                    help="Alternative: raw candidates.jsonl (one family per line)")
    ap.add_argument("--match-iou-default", type=float, default=0.5,
                    help="IoU threshold for TP labeling when expanding from jsonl "
                    "(used if family record lacks match_iou field).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--alphas", type=str, default="0.05,0.10,0.15,0.20")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    ap.add_argument("--split-ratios", type=str, default="0.6,0.2,0.2")
    ap.add_argument("--num-bins", type=int, default=80)
    ap.add_argument("--clip-phi", type=float, default=1_000_000.0)
    ap.add_argument("--R-statistic", type=str, default="max_phi",
                    choices=["max_phi", "sum_phi", "top2_mean_phi", "max_evalue"])
    ap.add_argument("--stratify-by-size", action="store_true",
                    help="Use per-cluster-size calibration pools (Mondrian).")
    return ap.parse_args()


def load_candidates_jsonl(path: Path, match_iou_default: float) -> pd.DataFrame:
    """Flatten candidates.jsonl (one family per line) into a candidate-level DataFrame.

    Each row is one candidate with columns: family_id, image_id, category_id,
    prompt, is_prompt_absent, gt_count, score, x1, y1, x2, y2, max_iou, is_tp, is_null.
    """
    import json
    rows = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            image_id = rec.get("image_id")
            category_id = rec.get("category_id")
            prompt = rec.get("prompt", "")
            family_id = f"{image_id}__{category_id}__{prompt}"
            is_absent = bool(rec.get("is_prompt_absent", False))
            match_iou = float(rec.get("match_iou", match_iou_default))
            for ci, c in enumerate(rec.get("candidates", [])):
                box = c["box"]
                is_tp = bool(c.get("is_tp", False))
                max_iou = float(c.get("max_iou", 0.0))
                # is_null definition mirrors the formal pipeline:
                # - in absent families, every candidate is null (no TP by construction)
                # - in present families, candidates with max_iou < match_iou are null
                is_null = is_absent or (max_iou < match_iou)
                rows.append(dict(
                    family_id=family_id,
                    image_id=image_id,
                    category_id=category_id,
                    prompt=prompt,
                    is_prompt_absent=is_absent,
                    gt_count=int(rec.get("gt_count", 0)),
                    candidate_idx=ci,
                    score=float(c["score"]),
                    x1=float(box[0]), y1=float(box[1]),
                    x2=float(box[2]), y2=float(box[3]),
                    max_iou=max_iou,
                    is_tp=is_tp,
                    is_null=is_null,
                ))
    return pd.DataFrame(rows)


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


def fit_histogram_density_ratio(
    fit_df: pd.DataFrame, num_bins: int, clip_phi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram density ratio phi(s) = p_tp(s) / p_null(s) via empirical quantile bins."""
    scores_all = fit_df["score"].to_numpy(dtype=float)
    is_null = fit_df["is_null"].to_numpy(dtype=bool)
    is_tp = fit_df["is_tp"].to_numpy(dtype=bool)
    qs = np.quantile(scores_all, np.linspace(0, 1, num_bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    # deduplicate bin edges
    qs = np.unique(qs)
    if len(qs) < 2:
        qs = np.array([scores_all.min() - 1e-9, scores_all.max() + 1e-9])
    null_cnt, _ = np.histogram(scores_all[is_null], bins=qs)
    tp_cnt, _ = np.histogram(scores_all[is_tp], bins=qs)
    total_null = max(int(is_null.sum()), 1)
    total_tp = max(int(is_tp.sum()), 1)
    p0 = null_cnt / total_null  # null density
    p1 = tp_cnt / total_tp      # tp density
    # phi = p1 / p0 with laplace eps + clip
    phi = np.where(p0 > 0, p1 / np.maximum(p0, 1e-12), clip_phi)
    phi = np.minimum(phi, clip_phi)
    return qs, phi


def score_to_phi(scores: np.ndarray, qs: np.ndarray, phi: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(qs, scores, side="right") - 1, 0, len(phi) - 1)
    return phi[idx]


def cluster_statistic(R_stat: str, phi_vals: np.ndarray, evalues: np.ndarray | None = None) -> float:
    if R_stat == "max_phi":
        return float(np.max(phi_vals)) if len(phi_vals) else 0.0
    if R_stat == "sum_phi":
        return float(np.sum(phi_vals))
    if R_stat == "top2_mean_phi":
        if len(phi_vals) == 0:
            return 0.0
        top2 = np.sort(phi_vals)[::-1][:2]
        return float(np.mean(top2))
    if R_stat == "max_evalue":
        if evalues is None or len(evalues) == 0:
            return 0.0
        return float(np.max(evalues))
    raise ValueError(f"Unknown R statistic: {R_stat}")


def build_family_clusters(
    group: pd.DataFrame, iou_thresh: float
) -> list[tuple[list[int], np.ndarray, np.ndarray, np.ndarray]]:
    """Returns list of (member_positions, is_tp, is_null, phi) per cluster."""
    if len(group) == 0:
        return []
    boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    clusters = nms_clusters_xyxy(boxes, iou_thresh)
    out = []
    is_tp = group["is_tp"].to_numpy(dtype=bool)
    is_null = group["is_null"].to_numpy(dtype=bool)
    phi = group["_phi"].to_numpy(dtype=float)
    for members in clusters:
        out.append((members, is_tp[members], is_null[members], phi[members]))
    return out


def per_seed(
    df: pd.DataFrame,
    seed: int,
    ratios: tuple[float, float, float],
    iou_thresh: float,
    alphas: list[float],
    num_bins: int,
    clip_phi: float,
    R_stat: str,
    stratify_by_size: bool,
) -> list[dict]:
    split = split_families(df, seed=seed, ratios=ratios)
    df = df.assign(split=split)
    fit_df = df[df.split == "fit"].copy()
    cal_df = df[df.split == "cal"].copy()
    test_df = df[df.split == "test"].copy()

    qs, phi_table = fit_histogram_density_ratio(fit_df, num_bins, clip_phi)
    for part in (fit_df, cal_df, test_df):
        part["_phi"] = score_to_phi(part["score"].to_numpy(float), qs, phi_table)

    # Build cal all-null cluster pool
    cal_all_null_R: list[float] = []
    cal_all_null_sizes: list[int] = []
    for _fid, g in cal_df.groupby("family_id", sort=False):
        clusters = build_family_clusters(g, iou_thresh)
        for _mem, is_tp_c, is_null_c, phi_c in clusters:
            if is_null_c.all():
                R = cluster_statistic(R_stat, phi_c)
                cal_all_null_R.append(R)
                cal_all_null_sizes.append(len(phi_c))
    cal_R = np.asarray(cal_all_null_R, dtype=float)
    cal_sizes = np.asarray(cal_all_null_sizes, dtype=int)
    n_cal = len(cal_R)
    sum_R_cal = float(cal_R.sum())
    # size-stratified pool (Mondrian)
    if stratify_by_size:
        size_pools = {}
        for s, R in zip(cal_sizes, cal_R):
            size_pools.setdefault(int(s), []).append(R)
        size_pools = {k: (np.array(v), float(np.sum(v))) for k, v in size_pools.items()}

    # Baseline candidate-level e-values for side-by-side comparison
    test_df = test_df.copy()
    cal_null_df = cal_df[cal_df.is_null]
    sum_phi_cal = float(cal_null_df["_phi"].sum())
    n_cal_cand = int(len(cal_null_df))
    test_df["_evalue_cand"] = (n_cal_cand + 1) * test_df["_phi"].to_numpy(float) / np.maximum(
        test_df["_phi"].to_numpy(float) + sum_phi_cal, 1e-12
    )

    # Total TP clusters for recall denominator
    total_tp_clusters = 0
    per_family_test_clusters = []
    for fid, g in test_df.groupby("family_id", sort=False):
        clusters_info = build_family_clusters(g, iou_thresh)
        if not clusters_info:
            per_family_test_clusters.append((fid, g, []))
            continue
        for _mem, is_tp_c, _is_null_c, _phi_c in clusters_info:
            if is_tp_c.any():
                total_tp_clusters += 1
        per_family_test_clusters.append((fid, g, clusters_info))
    total_tp_candidates = int(test_df["is_tp"].sum())

    rows = []
    for alpha in alphas:
        # ------ score-adaptive calibrated NMS-aware ------
        sa_rej = sa_fp = sa_tp = 0
        family_fdps_sa: list[float] = []
        # ------ baseline candidate e-BH ------
        base_rej = base_fp = base_tp = 0
        base_cluster_hit_tp = 0
        family_fdps_base: list[float] = []

        for fid, g, clusters_info in per_family_test_clusters:
            # Score-adaptive: compute cluster e-value for each cluster
            if clusters_info:
                Rs = np.array([cluster_statistic(R_stat, phi_c)
                               for _, _, _, phi_c in clusters_info], dtype=float)
                if stratify_by_size:
                    Ecals = np.zeros_like(Rs)
                    for i, (_mem, _tp, _nl, phi_c) in enumerate(clusters_info):
                        sz = len(phi_c)
                        if sz in size_pools:
                            pool_arr, pool_sum = size_pools[sz]
                            n_l = len(pool_arr)
                            Ecals[i] = (n_l + 1) * Rs[i] / max(Rs[i] + pool_sum, 1e-12)
                        else:
                            Ecals[i] = (n_cal + 1) * Rs[i] / max(Rs[i] + sum_R_cal, 1e-12)
                else:
                    Ecals = (n_cal + 1) * Rs / np.maximum(Rs + sum_R_cal, 1e-12)
                rej_cluster_idx = e_bh(Ecals, alpha)
                fam_fp = fam_tp = 0
                for ci in rej_cluster_idx:
                    _mem, is_tp_c, is_null_c, _phi_c = clusters_info[ci]
                    if is_tp_c.any():
                        fam_tp += 1
                    elif is_null_c.all():
                        fam_fp += 1
                n_rej = len(rej_cluster_idx)
                sa_rej += n_rej
                sa_fp += fam_fp
                sa_tp += fam_tp
                family_fdps_sa.append(fam_fp / max(n_rej, 1) if n_rej > 0 else 0.0)
            else:
                family_fdps_sa.append(0.0)

            # Baseline: per-candidate e-BH, then project onto same cluster graph
            evs = g["_evalue_cand"].to_numpy(float)
            rej_cand = e_bh(evs, alpha)
            is_tp_v = g["is_tp"].to_numpy(bool)
            is_null_v = g["is_null"].to_numpy(bool)
            if rej_cand:
                sel_tp = int(is_tp_v[rej_cand].sum())
                sel_null = int(is_null_v[rej_cand].sum())
                base_rej += len(rej_cand)
                base_tp += sel_tp
                base_fp += sel_null
                family_fdps_base.append(sel_null / max(len(rej_cand), 1))
                # project onto cluster graph
                K = len(g)
                cand2cluster = np.full(K, -1, dtype=int)
                for c_idx, (mem, _, _, _) in enumerate(clusters_info):
                    for m in mem:
                        cand2cluster[m] = c_idx
                hit_clusters = set(int(cand2cluster[i]) for i in rej_cand if cand2cluster[i] >= 0)
                for ci in hit_clusters:
                    _mem, is_tp_c, _is_null_c, _phi_c = clusters_info[ci]
                    if is_tp_c.any():
                        base_cluster_hit_tp += 1
            else:
                family_fdps_base.append(0.0)

        rows.append(dict(
            seed=seed,
            alpha=alpha,
            R_statistic=R_stat,
            iou_thresh=iou_thresh,
            n_cal_all_null_clusters=n_cal,
            total_tp_clusters=total_tp_clusters,
            total_tp_candidates=total_tp_candidates,
            # score-adaptive calibrated
            sa_rej=sa_rej,
            sa_fp=sa_fp,
            sa_tp=sa_tp,
            sa_fdp_pooled=sa_fp / max(sa_rej, 1),
            sa_family_fdp_mean=float(np.mean(family_fdps_sa)) if family_fdps_sa else 0.0,
            sa_recall_object=sa_tp / max(total_tp_clusters, 1),
            # baseline candidate e-BH
            base_rej=base_rej,
            base_fp=base_fp,
            base_tp=base_tp,
            base_fdp_pooled=base_fp / max(base_rej, 1),
            base_family_fdp_mean=float(np.mean(family_fdps_base)) if family_fdps_base else 0.0,
            base_recall_candidate=base_tp / max(total_tp_candidates, 1),
            base_recall_object=base_cluster_hit_tp / max(total_tp_clusters, 1),
        ))
    return rows


def main():
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    ratios = tuple(float(x) for x in args.split_ratios.split(","))
    if len(ratios) != 3:
        raise SystemExit("--split-ratios must have 3 values")

    if args.candidate_table is not None:
        df = pd.read_csv(args.candidate_table)
        print(f"loaded {len(df)} candidates from {args.candidate_table}")
    elif args.candidate_jsonl is not None:
        df = load_candidates_jsonl(args.candidate_jsonl, args.match_iou_default)
        print(f"loaded {len(df)} candidates from {args.candidate_jsonl}")
    else:
        raise SystemExit("Provide --candidate-table or --candidate-jsonl")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in seeds:
        rows = per_seed(
            df, seed, ratios, args.iou_thresh, alphas,
            args.num_bins, args.clip_phi, args.R_statistic, args.stratify_by_size,
        )
        all_rows.extend(rows)
        print(f"  seed {seed}: {len(rows)} α rows, α=0.10 → "
              f"sa_recall_obj={[r for r in rows if abs(r['alpha']-0.10)<1e-6][0]['sa_recall_object']:.4f}, "
              f"base_recall_obj={[r for r in rows if abs(r['alpha']-0.10)<1e-6][0]['base_recall_object']:.4f}")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(f"written: {args.out_dir/'summary.csv'}")

    agg = summary.groupby("alpha").agg(
        n_seeds=("seed", "count"),
        sa_fdp_pooled_mean=("sa_fdp_pooled", "mean"),
        sa_fdp_pooled_std=("sa_fdp_pooled", "std"),
        sa_family_fdp_mean=("sa_family_fdp_mean", "mean"),
        sa_rej_mean=("sa_rej", "mean"),
        sa_recall_object_mean=("sa_recall_object", "mean"),
        base_fdp_pooled_mean=("base_fdp_pooled", "mean"),
        base_family_fdp_mean=("base_family_fdp_mean", "mean"),
        base_rej_mean=("base_rej", "mean"),
        base_recall_object_mean=("base_recall_object", "mean"),
    ).reset_index()
    agg["recall_ratio_sa_over_base_object"] = (
        agg["sa_recall_object_mean"] / agg["base_recall_object_mean"].replace(0, np.nan)
    ).round(4)
    agg.to_csv(args.out_dir / "aggregate.csv", index=False)
    print(f"written: {args.out_dir/'aggregate.csv'}")
    print("\n=== aggregate ===")
    cols = ["alpha", "n_seeds",
            "sa_fdp_pooled_mean", "base_fdp_pooled_mean",
            "sa_family_fdp_mean", "base_family_fdp_mean",
            "sa_rej_mean", "base_rej_mean",
            "sa_recall_object_mean", "base_recall_object_mean",
            "recall_ratio_sa_over_base_object"]
    print(agg[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
