#!/usr/bin/env python3
"""
Three-pronged repair sweep for LVIS FDR violation.
Root cause: sparse-bin density-ratio blowup + small absent families.

Fix A: null_floor sweep  (regularize phi)
Fix B: category-conditional phi  (Mondrian on category)
Fix C: K_min filter  (skip tiny families)
Fix D: A+C combined
"""
import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_mpl")

_PROJECT = Path(__file__).resolve().parents[1]
CANDIDATE_TABLE = _PROJECT / "outputs/lvis_groundingdino_1000_presentall_absentraremix_analysis/candidate_table.csv"
OUT_DIR = _PROJECT / "outputs/lvis_repair_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA  = 0.10
N_BINS = 80
BINS   = np.linspace(0.05, 1.0, N_BINS + 1)
N_SEEDS = 20

# ── core helpers ─────────────────────────────────────────────────────────
def fit_phi(tp_scores, null_scores, null_floor=1e-6):
    tp_h,  _ = np.histogram(tp_scores,   bins=BINS, density=True)
    nu_h,  _ = np.histogram(null_scores, bins=BINS, density=True)
    nu_h = np.maximum(nu_h, null_floor)
    ratio = np.minimum(tp_h / nu_h, 1e6)
    def phi(s):
        idx = np.searchsorted(BINS[1:], s, side="right")
        return ratio[np.clip(idx, 0, N_BINS - 1)]
    return phi

def ebh(evalues, K, alpha):
    if K == 0 or len(evalues) == 0:
        return np.array([], dtype=int)
    order = np.argsort(evalues)[::-1]
    thresholds = K / (alpha * np.arange(1, K + 1))
    mask = order[order >= 0]  # dummy
    mask = evalues[order] >= thresholds
    if not mask.any():
        return np.array([], dtype=int)
    return order[: np.where(mask)[0][-1] + 1]

def run_one(df, seed, null_floor=1e-6, cat_mondrian=False, k_min=1):
    rng = np.random.default_rng(seed)
    fams = df["family_id"].unique().copy(); rng.shuffle(fams)
    n = len(fams)
    fit_f  = set(fams[:int(0.6*n)])
    cal_f  = set(fams[int(0.6*n):int(0.8*n)])
    test_f = set(fams[int(0.8*n):])

    fit  = df[df["family_id"].isin(fit_f)]
    cal  = df[df["family_id"].isin(cal_f)]
    test = df[df["family_id"].isin(test_f)]

    # global phi
    global_phi = fit_phi(fit[fit["is_tp"]]["score"].values,
                         fit[fit["is_null"]]["score"].values, null_floor)

    # category phi (Fix B)
    cat_phi = {}
    if cat_mondrian:
        for cat, grp in fit.groupby("category_name"):
            tp_s  = grp[grp["is_tp"]]["score"].values
            nu_s  = grp[grp["is_null"]]["score"].values
            if len(tp_s) >= 10 and len(nu_s) >= 20:
                cat_phi[cat] = fit_phi(tp_s, nu_s, null_floor)

    # global cal pool
    cal_null = cal[cal["is_null"]]["score"].values
    g_phi_cal_sum = global_phi(cal_null).sum()
    g_m = len(cal_null)

    # per-category cal pool
    cat_cal = {}
    if cat_mondrian:
        for cat, grp in cal[cal["is_null"]].groupby("category_name"):
            s = grp["score"].values
            if len(s) >= 10:
                cat_cal[cat] = s

    total_tp_pool = test[test["is_tp"]].shape[0]
    fp = tp = 0

    for fam_id, fam_df in test.groupby("family_id"):
        K = len(fam_df)
        if K < k_min:
            continue
        cat = fam_df["category_name"].iloc[0]
        scores = fam_df["score"].values

        # choose phi and cal pool
        if cat_mondrian and cat in cat_phi:
            p_fn = cat_phi[cat]
            cal_s = cat_cal.get(cat, cal_null)
            phi_sum = p_fn(cal_s).sum()
            m = len(cal_s)
        else:
            p_fn = global_phi
            phi_sum = g_phi_cal_sum
            m = g_m

        if m == 0 or phi_sum == 0:
            continue

        pv = p_fn(scores)
        ev = (m + 1) * pv / (pv + phi_sum)
        rej = ebh(ev, K, ALPHA)
        if len(rej) == 0:
            continue
        rej_rows = fam_df.iloc[rej]
        tp += int(rej_rows["is_tp"].sum())
        fp += int(rej_rows["is_null"].sum())

    total = tp + fp
    return {
        "rejections": total, "fp": fp, "tp": tp,
        "fdp": fp / max(total, 1),
        "recall": tp / max(total_tp_pool, 1),
    }

def sweep_seeds(df, label, seeds=N_SEEDS, **kwargs):
    rows = []
    for s in range(seeds):
        r = run_one(df, seed=s, **kwargs)
        rows.append(r)
    res = pd.DataFrame(rows)
    mean_fdp    = res["fdp"].mean()
    std_fdp     = res["fdp"].std()
    mean_recall = res["recall"].mean()
    std_recall  = res["recall"].std()
    mean_rej    = res["rejections"].mean()
    print(f"  {label:45s}  "
          f"FDP={mean_fdp:.4f}±{std_fdp:.4f}  "
          f"recall={mean_recall:.4f}±{std_recall:.4f}  "
          f"rej={mean_rej:.1f}")
    return {"label": label, "mean_fdp": mean_fdp, "std_fdp": std_fdp,
            "mean_recall": mean_recall, "std_recall": std_recall,
            "mean_rejections": mean_rej, **kwargs}

# ── load ──────────────────────────────────────────────────────────────────
print("Loading…")
df = pd.read_csv(CANDIDATE_TABLE)
print(f"  {len(df):,} candidates, {df['family_id'].nunique():,} families\n")

results = []

# ── Baseline ──────────────────────────────────────────────────────────────
print("=== Baseline (null_floor=1e-6, no mondrian, k_min=1) ===")
results.append(sweep_seeds(df, "baseline", null_floor=1e-6, cat_mondrian=False, k_min=1))

# ── Fix A: null_floor sweep ───────────────────────────────────────────────
print("\n=== Fix A: null_floor sweep ===")
for fl in [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
    results.append(sweep_seeds(df, f"null_floor={fl}", null_floor=fl, cat_mondrian=False, k_min=1))

# ── Fix B: category-conditional phi ──────────────────────────────────────
print("\n=== Fix B: category-conditional phi (floor=1e-6) ===")
results.append(sweep_seeds(df, "cat_mondrian (floor=1e-6)", null_floor=1e-6, cat_mondrian=True, k_min=1))

print("\n=== Fix B+A: category-conditional phi + floor=1e-2 ===")
results.append(sweep_seeds(df, "cat_mondrian + floor=1e-2", null_floor=1e-2, cat_mondrian=True, k_min=1))

# ── Fix C: K_min filter ───────────────────────────────────────────────────
print("\n=== Fix C: K_min filter (floor=1e-6) ===")
for km in [2, 3, 5, 10]:
    results.append(sweep_seeds(df, f"k_min={km}", null_floor=1e-6, cat_mondrian=False, k_min=km))

# ── Fix D: best A + C combined ────────────────────────────────────────────
print("\n=== Fix D: floor=1e-2 + k_min=3 ===")
results.append(sweep_seeds(df, "floor=1e-2 + k_min=3", null_floor=1e-2, cat_mondrian=False, k_min=3))

print("\n=== Fix D: floor=5e-3 + k_min=3 ===")
results.append(sweep_seeds(df, "floor=5e-3 + k_min=3", null_floor=5e-3, cat_mondrian=False, k_min=3))

# ── save ──────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv(OUT_DIR / "repair_sweep_results.csv", index=False)
print(f"\nSaved → {OUT_DIR}/repair_sweep_results.csv")

# ── summary: which configs achieve mean FDP <= alpha? ──────────────────────
valid = out_df[out_df["mean_fdp"] <= ALPHA].sort_values("mean_recall", ascending=False)
print(f"\n=== Valid configs (mean FDP ≤ {ALPHA}) sorted by recall ===")
print(valid[["label","mean_fdp","std_fdp","mean_recall","std_recall","mean_rejections"]].to_string(index=False))
