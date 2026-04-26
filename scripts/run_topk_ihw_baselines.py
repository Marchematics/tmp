#!/usr/bin/env python3
"""Top-k per-family baseline (E2) and IHW-lite covariate-FDR baseline (E3).

Top-k baseline:
  Per family, return top-k candidates by detector score. k is chosen on
  the calibration split as the largest k satisfying mean cal family-FDP
  ≤ α; if no k passes, k=1 (most-conservative).

IHW-lite:
  Conformal p-values within family. Family-size K is the covariate.
  Bin K into 5 strata; per stratum, run BH at level alpha/pi0_hat
  with pi0 estimated globally via Storey (lambda=0.5).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

CONFIGS = [
    ("COCO val", "GDino",      "coco_groundingdino_gonogo_1000_mix_analysis",
     "coco_gdino_1000_20seed_formal_hist"),
    ("COCO val", "OWL-ViT",    "coco_owlvit_val_1000_mix_analysis",
     "coco_owlvit_1000_20seed_formal_hist"),
    ("COCO val", "YOLO-World", "coco_yoloworld_val_1000_mix_analysis",
     "coco_yoloworld_val_1000_20seed_formal_hist"),
    ("VOC 2012", "GDino",      "voc_1000_gdino_analysis",
     "voc_gdino_1000_20seed_formal_hist"),
    ("VOC 2012", "OWL-ViT",    "voc_1000_owlvit_analysis",
     "voc_owlvit_1000_20seed_formal_hist"),
    ("VOC 2012", "YOLO-World", "voc_yoloworld_1000_mix_analysis",
     "voc_yoloworld_1000_20seed_formal_hist"),
]
ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 20
K_VALUES = [1, 2, 3, 5, 10]  # top-k candidates per family
K_BIN_EDGES = [0, 1, 5, 10, 20, np.inf]  # IHW K-strata


def split_families(family_ids: np.ndarray, seed: int) -> tuple[set, set, set]:
    rng = np.random.default_rng(seed)
    fams = family_ids.copy()
    rng.shuffle(fams)
    n = len(fams)
    n_fit = int(round(0.6 * n))
    n_cal = int(round(0.2 * n))
    n_fit = max(1, min(n_fit, n - 2))
    n_cal = max(1, min(n_cal, n - n_fit - 1))
    return set(fams[:n_fit]), set(fams[n_fit:n_fit+n_cal]), set(fams[n_fit+n_cal:])


# =====================  E2: top-k baseline  =====================
def topk_select(df: pd.DataFrame, k: int) -> np.ndarray:
    """Return boolean mask: keep top-k by score within each family."""
    df = df.assign(_rank=df.groupby("family_id")["score"].rank(method="first", ascending=False))
    return (df["_rank"] <= k).to_numpy()


def evaluate_topk(df_full: pd.DataFrame, seed: int, alpha: float) -> dict:
    fams = df_full["family_id"].drop_duplicates().to_numpy()
    fit, cal, test = split_families(fams, seed)
    cal_df = df_full[df_full["family_id"].isin(cal)]
    test_df = df_full[df_full["family_id"].isin(test)]

    # Pick best k satisfying cal mean family-FDP ≤ α
    best_k = 1
    for k in K_VALUES:
        mask = topk_select(cal_df, k)
        sub = cal_df.assign(rejected=mask)
        sub_rej = sub[sub["rejected"]]
        if len(sub_rej) == 0:
            continue
        # mean family FDP on cal
        fam_fdp = (
            sub_rej.groupby("family_id")
            .apply(lambda g: float((~g["is_tp"]).mean()))
        ).mean()
        if fam_fdp <= alpha:
            best_k = k

    mask = topk_select(test_df, best_k)
    rej_df = test_df[mask]
    n_rej = len(rej_df)
    n_fp = int((~rej_df["is_tp"]).sum())
    n_tp = int(rej_df["is_tp"].sum())
    n_tp_total = int(test_df["is_tp"].sum())
    fam_fdp_test = (
        rej_df.groupby("family_id")
        .apply(lambda g: float((~g["is_tp"]).mean()))
    ).mean() if n_rej > 0 else 0.0
    return {
        "k": best_k,
        "rejections": n_rej,
        "fdp_pooled": n_fp / max(n_rej, 1),
        "fdp_family": fam_fdp_test,
        "recall": n_tp / max(n_tp_total, 1),
    }


# =====================  E3: IHW-lite  =====================
def bh(p: np.ndarray, alpha: float) -> np.ndarray:
    K = len(p)
    if K == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, K + 1)
    eligible = sorted_p <= alpha * ranks / K
    r = int(np.where(eligible)[0].max() + 1) if eligible.any() else 0
    mask = np.zeros(K, dtype=bool)
    mask[order[:r]] = True
    return mask


def ihw_lite(df_test: pd.DataFrame, alpha: float) -> np.ndarray:
    """Bin candidates by family-K covariate; run BH within each bin at adapted alpha."""
    K = df_test["family_size"].to_numpy()
    p = df_test["conformal_pvalue"].to_numpy()
    # global pi0 estimate (Storey)
    lam = 0.5
    pi0 = (np.sum(p > lam) + 1) / ((1 - lam) * len(p))
    pi0 = min(pi0, 1.0)

    bin_idx = np.digitize(K, K_BIN_EDGES) - 1
    rejected = np.zeros(len(p), dtype=bool)
    for b in range(len(K_BIN_EDGES) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        # weight: inverse of pi0 within bin (allow more rejections in bins with fewer nulls)
        local_pi0 = (np.sum(p[mask] > lam) + 1) / ((1 - lam) * mask.sum())
        local_pi0 = min(local_pi0, 1.0)
        bin_alpha = alpha / max(local_pi0, 1e-3)
        m = bh(p[mask], bin_alpha)
        idx = np.where(mask)[0]
        if m.any():
            rejected[idx[m]] = True
    return rejected


def evaluate_ihw(df_test: pd.DataFrame, alpha: float) -> dict:
    rej = ihw_lite(df_test, alpha)
    n_rej = int(rej.sum())
    is_tp = df_test["is_tp"].to_numpy()
    n_fp = int((rej & ~is_tp).sum())
    n_tp = int((rej & is_tp).sum())
    n_tp_total = int(is_tp.sum())
    if n_rej > 0:
        fam_ids = df_test["family_id"].to_numpy()[rej]
        is_fp = (~is_tp)[rej].astype(int)
        fdp_fam = pd.DataFrame({"f": fam_ids, "fp": is_fp}).groupby("f")["fp"].mean().mean()
    else:
        fdp_fam = 0.0
    return {
        "rejections": n_rej,
        "fdp_pooled": n_fp / max(n_rej, 1),
        "fdp_family": float(fdp_fam),
        "recall": n_tp / max(n_tp_total, 1),
    }


# =====================  Driver  =====================
def aggregate_config(
    dataset: str, detector: str, analysis_dir: Path, formal_dir: Path
) -> list[dict]:
    cand_path = analysis_dir / "candidate_table.csv"
    if not cand_path.exists():
        print(f"[skip] no candidate_table {cand_path}")
        return []
    df_full = pd.read_csv(cand_path)
    rows_topk: dict[float, list] = {a: [] for a in ALPHAS}
    rows_ihw: dict[float, list] = {a: [] for a in ALPHAS}

    for seed in range(N_SEEDS):
        # E2 top-k uses raw candidate_table directly (no e-value needed)
        for alpha in ALPHAS:
            rows_topk[alpha].append(evaluate_topk(df_full, seed, alpha))

        # E3 IHW reads pre-computed test_candidates_with_evalues.csv
        # (has conformal_pvalue + family_size)
        e_path = formal_dir / f"seed_{seed}" / "test_candidates_with_evalues.csv"
        if not e_path.exists():
            continue
        df_e = pd.read_csv(
            e_path,
            usecols=["family_id", "is_tp", "conformal_pvalue", "family_size"],
        )
        for alpha in ALPHAS:
            rows_ihw[alpha].append(evaluate_ihw(df_e, alpha))

    out = []
    for method_name, src in (("topk_calibrated", rows_topk),
                             ("ihw_lite_K", rows_ihw)):
        for alpha, samples in src.items():
            if not samples:
                continue
            sdf = pd.DataFrame(samples)
            pass_pooled = float((sdf["fdp_pooled"] <= alpha).mean())
            pass_family = float((sdf["fdp_family"] <= alpha).mean())
            row = {
                "table": "table2_baselines",
                "dataset": dataset,
                "detector": detector,
                "alpha": float(alpha),
                "method": method_name,
                "fdp_pooled_mean": float(sdf["fdp_pooled"].mean()),
                "fdp_pooled_std": float(sdf["fdp_pooled"].std(ddof=1)),
                "fdp_family_mean": float(sdf["fdp_family"].mean()),
                "recall_mean": float(sdf["recall"].mean()),
                "rejections_mean": float(sdf["rejections"].mean()),
                "phi_max_median": "",
                "loo_mean": "",
                "n_seeds": N_SEEDS,
                "source": (f"outputs/{analysis_dir.name}" if method_name == "topk_calibrated"
                           else f"outputs/{formal_dir.name}"),
                "pooled_bound_eps": "",
                "pooled_bound_delta": pass_pooled,
                "pooled_bound_covered": pass_family,
                "pooled_bound_eps_stratified": "",
            }
            out.append(row)
    return out


def main() -> None:
    all_rows = []
    for dataset, detector, ana, fml in CONFIGS:
        ana_dir = OUTPUTS / ana
        fml_dir = OUTPUTS / fml
        print(f"\n{dataset} / {detector}")
        rows = aggregate_config(dataset, detector, ana_dir, fml_dir)
        for r in rows:
            if r["alpha"] != 0.10:
                continue
            print(
                f"  {r['method']:18s}  rej={r['rejections_mean']:7.1f}  "
                f"FDP={r['fdp_pooled_mean']:.4f}  per-fam={r['fdp_family_mean']:.4f}  "
                f"recall={r['recall_mean']:.3f}"
            )
        all_rows.extend(rows)

    out_csv = OUTPUTS / "topk_ihw_baselines.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
