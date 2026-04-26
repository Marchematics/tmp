#!/usr/bin/env python3
"""
Validate k_min fix using the EXACT formal pipeline implementation.
Imports directly from the pipeline source to ensure consistency.

Mirrors run_formal_betting_pipeline.py logic with added k_min parameter.
"""
import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_mpl")

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
sys.path.insert(0, str(SRC))

from ovd_hallucination_fdr.evalues import e_bh  # noqa

CANDIDATE_TABLE = PROJECT / "outputs/lvis_groundingdino_1000_presentall_absentraremix_analysis/candidate_table.csv"
OUT_DIR = PROJECT / "outputs/lvis_kmin_formal_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA   = 0.10
N_SEEDS = 20
SMOOTHING = 0.0      # matches best LVIS run (smooth0)
NUM_BINS  = 80
CLIP_PHI  = 1e6
SPLIT_RATIOS = (0.6, 0.2, 0.2)

# ── exact copy of pipeline's quantile bin logic ──────────────────────────
def make_quantile_bins(scores: np.ndarray, num_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(scores, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        lo, hi = float(np.min(scores)), float(np.max(scores))
        if lo == hi:
            lo -= 1e-6; hi += 1e-6
        edges = np.linspace(lo, hi, min(num_bins, 10) + 1)
    edges[0]  = -np.inf
    edges[-1] =  np.inf
    return edges

def fit_phi_pipeline(fit_df, num_bins=NUM_BINS, smoothing=SMOOTHING, clip=CLIP_PHI):
    null_s = fit_df.loc[fit_df["is_null"],  "score"].to_numpy(float)
    tp_s   = fit_df.loc[fit_df["is_tp"],    "score"].to_numpy(float)
    edges  = make_quantile_bins(fit_df["score"].to_numpy(float), num_bins)
    nc, _  = np.histogram(null_s, bins=edges)
    tc, _  = np.histogram(tp_s,   bins=edges)
    bins   = len(edges) - 1
    p0 = (nc.astype(float) + smoothing) / (len(null_s) + smoothing * bins)
    p1 = (tc.astype(float) + smoothing) / (len(tp_s)   + smoothing * bins)
    # avoid 0/0 → treat as 0 phi
    ratio = np.where(p0 > 0, p1 / np.maximum(p0, 1e-300), 0.0)
    ratio  = np.minimum(ratio, clip)
    def phi(s):
        s = np.asarray(s, float)
        idx = np.searchsorted(edges, s, side="right") - 1
        idx = np.clip(idx, 0, bins - 1)
        return ratio[idx]
    return phi

def split_families(df, seed, ratios=SPLIT_RATIOS):
    rng   = np.random.default_rng(seed)
    fams  = df["family_id"].drop_duplicates().to_numpy()
    rng.shuffle(fams)
    n     = len(fams)
    n_fit = max(1, min(int(round(ratios[0] * n)), n - 2))
    n_cal = max(1, min(int(round(ratios[1] * n)), n - n_fit - 1))
    fit_s = set(fams[:n_fit])
    cal_s = set(fams[n_fit: n_fit + n_cal])
    test_s = set(fams) - fit_s - cal_s
    return fit_s, cal_s, test_s

def run_one(df, seed, k_min=1):
    fit_s, cal_s, test_s = split_families(df, seed)
    fit  = df[df["family_id"].isin(fit_s)]
    cal  = df[df["family_id"].isin(cal_s)]
    test = df[df["family_id"].isin(test_s)]

    phi = fit_phi_pipeline(fit)

    cal_null_scores = cal.loc[cal["is_null"], "score"].to_numpy(float)
    phi_cal = phi(cal_null_scores)
    sum_phi_cal = float(phi_cal.sum())
    n_cal = len(cal_null_scores)

    total_tp = int(test["is_tp"].sum())
    fp = tp = total_rej = 0

    for _, fam in test.groupby("family_id", sort=False):
        K = len(fam)
        if K < k_min:
            continue
        pv = phi(fam["score"].to_numpy(float))
        denom = pv + sum_phi_cal
        ev = np.where(denom > 0, (n_cal + 1) * pv / denom, 0.0)
        rej_pos = e_bh(ev, ALPHA)
        if not rej_pos:
            continue
        sel = fam.iloc[rej_pos]
        fp  += int(sel["is_null"].sum())
        tp  += int(sel["is_tp"].sum())
        total_rej += len(sel)

    fdp    = fp / max(total_rej, 1)
    recall = tp / max(total_tp, 1)
    return {"rejections": total_rej, "fp": fp, "tp": tp, "fdp": fdp, "recall": recall}

def sweep(df, k_min, label=""):
    rows = [run_one(df, s, k_min=k_min) for s in range(N_SEEDS)]
    r = pd.DataFrame(rows)
    tag = label or f"k_min={k_min}"
    print(f"  {tag:30s}  FDP={r['fdp'].mean():.4f}±{r['fdp'].std():.4f}  "
          f"recall={r['recall'].mean():.4f}±{r['recall'].std():.4f}  "
          f"rej={r['rejections'].mean():.1f}")
    return {"k_min": k_min, "label": tag,
            "mean_fdp": r["fdp"].mean(), "std_fdp": r["fdp"].std(),
            "mean_recall": r["recall"].mean(), "std_recall": r["recall"].std(),
            "mean_rej": r["rejections"].mean(),
            "n_seeds_fdp_above_alpha": int((r["fdp"] > ALPHA).sum()),
            "per_seed_fdp": r["fdp"].tolist()}

# ── load ──────────────────────────────────────────────────────────────────
print("Loading…")
df = pd.read_csv(CANDIDATE_TABLE)
print(f"  {len(df):,} candidates, {df['family_id'].nunique():,} families")
print(f"  Smoothing={SMOOTHING}, bins={NUM_BINS}, seeds={N_SEEDS}, alpha={ALPHA}\n")

results = []

print("=== k_min sweep (exact pipeline) ===")
for km in [1, 2, 3, 4, 5, 7, 10]:
    results.append(sweep(df, km))

# ── find minimum k_min that achieves mean FDP <= alpha ───────────────────
valid = [r for r in results if r["mean_fdp"] <= ALPHA]
print(f"\n{'='*60}")
print(f"Valid configs (mean FDP ≤ {ALPHA}):")
for r in valid:
    print(f"  k_min={r['k_min']:2d}: FDP={r['mean_fdp']:.4f}±{r['std_fdp']:.4f}  "
          f"recall={r['mean_recall']:.4f}  seeds_above_alpha={r['n_seeds_fdp_above_alpha']}/20")

# ── per-seed FDP distribution for best valid k_min ───────────────────────
if valid:
    best = min(valid, key=lambda x: -x["mean_recall"])
    print(f"\nPer-seed FDP for k_min={best['k_min']}:")
    for i, v in enumerate(best["per_seed_fdp"]):
        flag = " ← ABOVE α" if v > ALPHA else ""
        print(f"  seed {i:2d}: FDP={v:.4f}{flag}")

# ── ALSO run smoothing=1 for comparison ──────────────────────────────────
print(f"\n=== With smoothing=1.0 (original default) ===")
SMOOTHING_ORIG = 1.0

def fit_phi_smooth1(fit_df):
    return fit_phi_pipeline(fit_df, smoothing=1.0)

def run_one_s1(df, seed, k_min=1):
    fit_s, cal_s, test_s = split_families(df, seed)
    fit  = df[df["family_id"].isin(fit_s)]
    cal  = df[df["family_id"].isin(cal_s)]
    test = df[df["family_id"].isin(test_s)]
    phi  = fit_phi_smooth1(fit)
    cal_null_scores = cal.loc[cal["is_null"], "score"].to_numpy(float)
    phi_cal     = phi(cal_null_scores)
    sum_phi_cal = float(phi_cal.sum())
    n_cal       = len(cal_null_scores)
    total_tp    = int(test["is_tp"].sum())
    fp = tp = total_rej = 0
    for _, fam in test.groupby("family_id", sort=False):
        K = len(fam)
        if K < k_min:
            continue
        pv = phi(fam["score"].to_numpy(float))
        denom = pv + sum_phi_cal
        ev = np.where(denom > 0, (n_cal + 1) * pv / denom, 0.0)
        rej_pos = e_bh(ev, ALPHA)
        if not rej_pos:
            continue
        sel = fam.iloc[rej_pos]
        fp  += int(sel["is_null"].sum())
        tp  += int(sel["is_tp"].sum())
        total_rej += len(sel)
    fdp    = fp / max(total_rej, 1)
    recall = tp / max(total_tp, 1)
    return {"rejections": total_rej, "fp": fp, "tp": tp, "fdp": fdp, "recall": recall}

s1_results = []
for km in [1, 2, 3, 5]:
    rows = [run_one_s1(df, s, k_min=km) for s in range(N_SEEDS)]
    r = pd.DataFrame(rows)
    tag = f"smooth=1 k_min={km}"
    print(f"  {tag:30s}  FDP={r['fdp'].mean():.4f}±{r['fdp'].std():.4f}  "
          f"recall={r['recall'].mean():.4f}±{r['recall'].std():.4f}  "
          f"rej={r['rejections'].mean():.1f}")
    s1_results.append({"k_min": km, "smoothing": 1.0, "label": tag,
                        "mean_fdp": r["fdp"].mean(), "std_fdp": r["fdp"].std(),
                        "mean_recall": r["recall"].mean(), "std_recall": r["recall"].std(),
                        "mean_rej": r["rejections"].mean()})

# ── save ──────────────────────────────────────────────────────────────────
all_results = results + s1_results
out_df = pd.DataFrame(all_results).drop(columns=["per_seed_fdp"], errors="ignore")
out_df.to_csv(OUT_DIR / "kmin_formal_results.csv", index=False)
print(f"\nSaved → {OUT_DIR}/kmin_formal_results.csv")
