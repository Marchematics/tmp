#!/usr/bin/env python3
"""
Test per-image Mondrian calibration as a fix for LVIS FDR violation.
Hypothesis: within-image null score heterogeneity causes exchangeability failure.
"""
import os, json
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_mpl")

CANDIDATE_TABLE = Path(str(Path(__file__).resolve().parents[1] / "outputs/lvis_groundingdino_1000_presentall_absentraremix_analysis/candidate_table.csv"))
OUT_DIR = Path(str(Path(__file__).resolve().parents[1] / "outputs/lvis_within_image_heterogeneity"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
N_BINS = 80
BINS = np.linspace(0.05, 1.0, N_BINS + 1)
MONDRIAN_MIN = 30  # min cal nulls for per-image pool

def fit_phi(tp_scores, null_scores):
    tp_h, _ = np.histogram(tp_scores, bins=BINS, density=True)
    nu_h, _ = np.histogram(null_scores, bins=BINS, density=True)
    nu_h = np.maximum(nu_h, 1e-6)
    ratio = np.minimum(tp_h / nu_h, 1e6)
    def phi(s):
        idx = np.searchsorted(BINS[1:], s, side="right")
        return ratio[np.clip(idx, 0, N_BINS - 1)]
    return phi

def ebh(evalues, K, alpha):
    if K == 0 or len(evalues) == 0:
        return np.array([], dtype=int)
    order = np.argsort(evalues)[::-1]
    sorted_e = evalues[order]
    thresholds = K / (alpha * np.arange(1, K + 1))
    mask = sorted_e >= thresholds
    if not mask.any():
        return np.array([], dtype=int)
    return order[: np.where(mask)[0][-1] + 1]

def run_pipeline(df, seed, mondrian="global"):
    rng = np.random.default_rng(seed)
    families = df["family_id"].unique().copy()
    rng.shuffle(families)
    n = len(families)
    fit_fams  = set(families[:int(0.6 * n)])
    cal_fams  = set(families[int(0.6 * n):int(0.8 * n)])
    test_fams = set(families[int(0.8 * n):])

    fit_data  = df[df["family_id"].isin(fit_fams)]
    cal_data  = df[df["family_id"].isin(cal_fams)]
    test_data = df[df["family_id"].isin(test_fams)]

    phi_fn = fit_phi(
        fit_data[fit_data["is_tp"]]["score"].values,
        fit_data[fit_data["is_null"]]["score"].values,
    )

    # Global cal pool
    cal_null = cal_data[cal_data["is_null"]]
    global_phi_cal = phi_fn(cal_null["score"].values)
    global_phi_sum = global_phi_cal.sum()
    global_m = len(cal_null)

    # Per-image cal pools (for Mondrian)
    if mondrian == "image":
        img_cal = {}
        for img_id, grp in cal_null.groupby("image_id"):
            s = grp["score"].values
            if len(s) >= MONDRIAN_MIN:
                ph = phi_fn(s)
                img_cal[img_id] = (ph.sum(), len(s))

    total_tp = df[df["family_id"].isin(test_fams) & df["is_tp"]].shape[0]
    fp = tp = 0

    for fam_id, fam_df in test_data.groupby("family_id"):
        K = len(fam_df)
        scores = fam_df["score"].values

        if mondrian == "image":
            img_id = fam_df["image_id"].iloc[0]
            if img_id in img_cal:
                phi_sum, m = img_cal[img_id]
            else:
                phi_sum, m = global_phi_sum, global_m
        else:
            phi_sum, m = global_phi_sum, global_m

        pv = phi_fn(scores)
        ev = (m + 1) * pv / (pv + phi_sum)
        rej_idx = ebh(ev, K, ALPHA)
        if len(rej_idx) == 0:
            continue
        rej = fam_df.iloc[rej_idx]
        tp += rej["is_tp"].sum()
        fp += rej["is_null"].sum()

    total_rej = tp + fp
    fdp   = fp / max(total_rej, 1)
    recall = tp / max(total_tp, 1)
    return {"rejections": total_rej, "fdp": fdp, "recall": recall, "fp": fp, "tp": tp}

# ── load ──────────────────────────────────────────────────────────────────
print("Loading candidate table…")
df = pd.read_csv(CANDIDATE_TABLE)
print(f"  {len(df):,} candidates, {df['family_id'].nunique():,} families")

# ── Task 1: per-image null score analysis ──────────────────────────────────
print("\n=== Task 1: Per-image null score distribution ===")
null_df = df[df["is_null"]]
img_null_stats = null_df.groupby("image_id").agg(
    n_null=("score","count"),
    mean_score=("score","mean"),
    p95_score=("score", lambda x: np.percentile(x, 95)),
).reset_index()
global_p95 = np.percentile(null_df["score"].values, 95)
print(f"Global null p95 = {global_p95:.4f}")
high = img_null_stats[img_null_stats["p95_score"] > 2 * global_p95].sort_values("p95_score", ascending=False)
print(f"Images with p95 > 2×global_p95: {len(high)} / {len(img_null_stats)}")
print(high.head(20).to_string(index=False))
img_null_stats.to_csv(OUT_DIR / "per_image_null_stats.csv", index=False)

# ── Task 2-3: seed=0, global vs per-image Mondrian ────────────────────────
print("\n=== Task 2-3: Seed=0, global vs per-image Mondrian ===")
res_global = run_pipeline(df, seed=0, mondrian="global")
res_image  = run_pipeline(df, seed=0, mondrian="image")
print(f"  Global:           rejections={res_global['rejections']}  FDP={res_global['fdp']:.4f}  recall={res_global['recall']:.4f}")
print(f"  Per-image Mondrian: rejections={res_image['rejections']}  FDP={res_image['fdp']:.4f}  recall={res_image['recall']:.4f}")

# ── Task 4: 5-seed sweep ──────────────────────────────────────────────────
print("\n=== Task 4: 5-seed sweep, global vs per-image Mondrian ===")
rows = []
for seed in range(5):
    rg = run_pipeline(df, seed=seed, mondrian="global")
    ri = run_pipeline(df, seed=seed, mondrian="image")
    print(f"  seed={seed}  global: rej={rg['rejections']} FDP={rg['fdp']:.3f}  | image: rej={ri['rejections']} FDP={ri['fdp']:.3f}")
    rows.append({"seed": seed, "method": "global",    **rg})
    rows.append({"seed": seed, "method": "img_mondrian", **ri})

sweep_df = pd.DataFrame(rows)
for method, grp in sweep_df.groupby("method"):
    print(f"\n  {method}: mean FDP={grp['fdp'].mean():.4f}±{grp['fdp'].std():.4f}  mean recall={grp['recall'].mean():.4f}±{grp['recall'].std():.4f}")

sweep_df.to_csv(OUT_DIR / "seed_sweep.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────
summary = {
    "global_mean_fdp": sweep_df[sweep_df["method"]=="global"]["fdp"].mean(),
    "img_mondrian_mean_fdp": sweep_df[sweep_df["method"]=="img_mondrian"]["fdp"].mean(),
    "global_mean_recall": sweep_df[sweep_df["method"]=="global"]["recall"].mean(),
    "img_mondrian_mean_recall": sweep_df[sweep_df["method"]=="img_mondrian"]["recall"].mean(),
    "n_high_null_images": int(len(high)),
    "global_null_p95": float(global_p95),
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {OUT_DIR}/")
print(json.dumps(summary, indent=2))
