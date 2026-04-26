#!/usr/bin/env python3
"""Diagnostic: compare COCO-val vs COCO-train candidate tables
to explain why COCO-train has weaker FDR control (15/20 pass vs 20/20)."""

import pandas as pd
import numpy as np

# ---------- paths ----------
PATH_VAL = "coco_groundingdino_gonogo_1000_mix_analysis/candidate_table.csv"
PATH_TRAIN = "coco_train_1000_mix_analysis/candidate_table.csv"

# ---------- load ----------
val = pd.read_csv(PATH_VAL)
train = pd.read_csv(PATH_TRAIN)

# Ensure boolean columns are bool
for df in [val, train]:
    for col in ["is_prompt_absent", "is_tp", "is_null"]:
        df[col] = df[col].astype(bool)

datasets = {"COCO-val": val, "COCO-train": train}

# ====================================================================
# 1. Basic counts
# ====================================================================
print("=" * 80)
print("1. BASIC COUNTS")
print("=" * 80)
header = f"{'Metric':<45} {'COCO-val':>15} {'COCO-train':>15}"
print(header)
print("-" * 80)

for label, metric_fn in [
    ("Total candidates", lambda d: len(d)),
    ("Total families", lambda d: d["family_id"].nunique()),
    ("Total images", lambda d: d["image_id"].nunique()),
    ("Absent families (is_prompt_absent==True)", lambda d: d[d["is_prompt_absent"]]["family_id"].nunique()),
    ("Present families (is_prompt_absent==False)", lambda d: d[~d["is_prompt_absent"]]["family_id"].nunique()),
]:
    v = metric_fn(val)
    t = metric_fn(train)
    print(f"{label:<45} {v:>15} {t:>15}")

# ====================================================================
# 2. Null & TP fractions
# ====================================================================
print("\n" + "=" * 80)
print("2. NULL & TP FRACTIONS")
print("=" * 80)
print(header)
print("-" * 80)

for label, metric_fn in [
    ("Null fraction (overall)", lambda d: d["is_null"].mean()),
    ("Null fraction (absent families)", lambda d: d[d["is_prompt_absent"]]["is_null"].mean()),
    ("Null fraction (present families)", lambda d: d[~d["is_prompt_absent"]]["is_null"].mean()),
    ("TP fraction (overall)", lambda d: d["is_tp"].mean()),
    ("TP fraction (absent families)", lambda d: d[d["is_prompt_absent"]]["is_tp"].mean()),
    ("TP fraction (present families)", lambda d: d[~d["is_prompt_absent"]]["is_tp"].mean()),
]:
    v = metric_fn(val)
    t = metric_fn(train)
    print(f"{label:<45} {v:>15.4f} {t:>15.4f}")

# ====================================================================
# 3. Score distributions
# ====================================================================
print("\n" + "=" * 80)
print("3a. SCORE DISTRIBUTION — NULLS (is_null==True)")
print("=" * 80)
print(header)
print("-" * 80)

for label, stat_fn in [
    ("count", lambda s: len(s)),
    ("mean", lambda s: s.mean()),
    ("median", lambda s: s.median()),
    ("p75", lambda s: s.quantile(0.75)),
    ("p90", lambda s: s.quantile(0.90)),
    ("p95", lambda s: s.quantile(0.95)),
    ("max", lambda s: s.max()),
]:
    v = stat_fn(val[val["is_null"]]["score"])
    t = stat_fn(train[train["is_null"]]["score"])
    if label == "count":
        print(f"  {label:<43} {v:>15} {t:>15}")
    else:
        print(f"  {label:<43} {v:>15.6f} {t:>15.6f}")

print("\n" + "=" * 80)
print("3b. SCORE DISTRIBUTION — TPs (is_tp==True)")
print("=" * 80)
print(header)
print("-" * 80)

for label, stat_fn in [
    ("count", lambda s: len(s)),
    ("mean", lambda s: s.mean()),
    ("median", lambda s: s.median()),
    ("p25", lambda s: s.quantile(0.25)),
    ("p10", lambda s: s.quantile(0.10)),
    ("p05", lambda s: s.quantile(0.05)),
    ("min", lambda s: s.min()),
]:
    v = stat_fn(val[val["is_tp"]]["score"])
    t = stat_fn(train[train["is_tp"]]["score"])
    if label == "count":
        print(f"  {label:<43} {v:>15} {t:>15}")
    else:
        print(f"  {label:<43} {v:>15.6f} {t:>15.6f}")

# ====================================================================
# 3c. Score distributions for nulls — absent vs present
# ====================================================================
print("\n" + "=" * 80)
print("3c. NULL SCORES — ABSENT vs PRESENT families")
print("=" * 80)
sub_header = f"{'Metric':<35} {'Val-Abs':>10} {'Val-Pres':>10} {'Trn-Abs':>10} {'Trn-Pres':>10}"
print(sub_header)
print("-" * 80)

for label, stat_fn in [
    ("count", lambda s: len(s)),
    ("mean", lambda s: s.mean()),
    ("median", lambda s: s.median()),
    ("p90", lambda s: s.quantile(0.90)),
    ("p95", lambda s: s.quantile(0.95)),
    ("max", lambda s: s.max()),
]:
    vals = []
    for df in [val, train]:
        for absent in [True, False]:
            mask = df["is_null"] & (df["is_prompt_absent"] == absent)
            s = df[mask]["score"]
            vals.append(stat_fn(s))
    if label == "count":
        print(f"  {label:<33} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")
    else:
        print(f"  {label:<33} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")

# ====================================================================
# 4. Candidates per family
# ====================================================================
print("\n" + "=" * 80)
print("4. CANDIDATES PER FAMILY")
print("=" * 80)
print(header)
print("-" * 80)

for label, mask_fn in [
    ("Mean cands/family (overall)", lambda d: d),
    ("Mean cands/family (absent)", lambda d: d[d["is_prompt_absent"]]),
    ("Mean cands/family (present)", lambda d: d[~d["is_prompt_absent"]]),
    ("Median cands/family (overall)", lambda d: d),
]:
    if "Median" in label:
        v = mask_fn(val).groupby("family_id").size().median()
        t = mask_fn(train).groupby("family_id").size().median()
    else:
        v = mask_fn(val).groupby("family_id").size().mean()
        t = mask_fn(train).groupby("family_id").size().mean()
    print(f"{label:<45} {v:>15.2f} {t:>15.2f}")

# Fraction of absent-only families
for label, fn in [
    ("Frac absent-only families", lambda d: d[d["is_prompt_absent"]]["family_id"].nunique() / d["family_id"].nunique()),
]:
    v = fn(val)
    t = fn(train)
    print(f"{label:<45} {v:>15.4f} {t:>15.4f}")

# ====================================================================
# 5. phi_max computation
# ====================================================================
print("\n" + "=" * 80)
print("5. PHI_MAX (likelihood ratio max)")
print("=" * 80)

bins = np.linspace(0.05, 1.0, 81)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

for name, df in datasets.items():
    tp_scores = df[df["is_tp"]]["score"].values
    null_scores = df[df["is_null"]]["score"].values

    tp_hist, _ = np.histogram(tp_scores, bins=bins, density=True)
    null_hist, _ = np.histogram(null_scores, bins=bins, density=True)

    # floor null density
    null_hist = np.maximum(null_hist, 1e-6)

    phi = np.clip(tp_hist / null_hist, 0, 1e6)
    phi_max = phi.max()
    phi_argmax = bin_centers[np.argmax(phi)]

    print(f"  {name}: phi_max = {phi_max:.2f}  (at score ~ {phi_argmax:.4f})")

    # Also report where phi > 100
    high_phi = bin_centers[phi > 100]
    if len(high_phi) > 0:
        print(f"    phi > 100 in score range [{high_phi.min():.3f}, {high_phi.max():.3f}]")
    else:
        print(f"    phi never exceeds 100")

# ====================================================================
# 6. Overlap diagnostic
# ====================================================================
print("\n" + "=" * 80)
print("6. OVERLAP DIAGNOSTIC")
print("=" * 80)
print(header)
print("-" * 80)

for name, df in datasets.items():
    tp_scores = df[df["is_tp"]]["score"]
    null_scores = df[df["is_null"]]["score"]
    tp_med = tp_scores.median()
    null_med = null_scores.median()
    print(f"  {name}:")
    print(f"    Median TP score:    {tp_med:.6f}")
    print(f"    Median null score:  {null_med:.6f}")
    print(f"    Frac nulls > median(TP):   {(null_scores > tp_med).mean():.6f}")
    print(f"    Frac TPs  < median(null):  {(tp_scores < null_med).mean():.6f}")
    print()

# ====================================================================
# 7. Additional: per-family max-null-score distribution
# ====================================================================
print("=" * 80)
print("7. PER-FAMILY MAX NULL SCORE (drives false discoveries)")
print("=" * 80)
print(header)
print("-" * 80)

for label, stat_fn in [
    ("mean", lambda s: s.mean()),
    ("median", lambda s: s.median()),
    ("p75", lambda s: s.quantile(0.75)),
    ("p90", lambda s: s.quantile(0.90)),
    ("p95", lambda s: s.quantile(0.95)),
    ("max", lambda s: s.max()),
]:
    v_max_nulls = val[val["is_null"]].groupby("family_id")["score"].max()
    t_max_nulls = train[train["is_null"]].groupby("family_id")["score"].max()
    v = stat_fn(v_max_nulls)
    t = stat_fn(t_max_nulls)
    print(f"  {label:<43} {v:>15.6f} {t:>15.6f}")

# ====================================================================
# 8. Category-level analysis: which categories are problematic?
# ====================================================================
print("\n" + "=" * 80)
print("8. ABSENT-FAMILY CATEGORIES WITH HIGHEST MAX NULL SCORES (COCO-train)")
print("=" * 80)

train_absent_nulls = train[train["is_prompt_absent"] & train["is_null"]]
fam_max = train_absent_nulls.groupby(["family_id", "category_name"])["score"].max().reset_index()
fam_max = fam_max.sort_values("score", ascending=False)
print(f"{'family_id':<30} {'category':<20} {'max_null_score':>15}")
print("-" * 70)
for _, row in fam_max.head(20).iterrows():
    print(f"{row['family_id']:<30} {row['category_name']:<20} {row['score']:>15.6f}")

# Same for val
print(f"\n{'':=<80}")
print("8b. ABSENT-FAMILY CATEGORIES WITH HIGHEST MAX NULL SCORES (COCO-val)")
print("=" * 80)

val_absent_nulls = val[val["is_prompt_absent"] & val["is_null"]]
fam_max_v = val_absent_nulls.groupby(["family_id", "category_name"])["score"].max().reset_index()
fam_max_v = fam_max_v.sort_values("score", ascending=False)
print(f"{'family_id':<30} {'category':<20} {'max_null_score':>15}")
print("-" * 70)
for _, row in fam_max_v.head(20).iterrows():
    print(f"{row['family_id']:<30} {row['category_name']:<20} {row['score']:>15.6f}")

# ====================================================================
# 9. Key summary
# ====================================================================
print("\n" + "=" * 80)
print("9. KEY SUMMARY")
print("=" * 80)

val_nulls = val[val["is_null"]]["score"]
train_nulls = train[train["is_null"]]["score"]
val_tps = val[val["is_tp"]]["score"]
train_tps = train[train["is_tp"]]["score"]

print(f"""
COCO-val:  {len(val_nulls)} nulls, {len(val_tps)} TPs
COCO-train: {len(train_nulls)} nulls, {len(train_tps)} TPs

Score separation (median TP - median null):
  COCO-val:  {val_tps.median() - val_nulls.median():.4f}
  COCO-train: {train_tps.median() - train_nulls.median():.4f}

Null score upper tail (p95):
  COCO-val:  {val_nulls.quantile(0.95):.4f}
  COCO-train: {train_nulls.quantile(0.95):.4f}

TP score lower tail (p05):
  COCO-val:  {val_tps.quantile(0.05):.4f}
  COCO-train: {train_tps.quantile(0.05):.4f}

Overlap zone (null p95 > TP p05?):
  COCO-val:  null_p95={val_nulls.quantile(0.95):.4f}  vs  tp_p05={val_tps.quantile(0.05):.4f}  gap={val_tps.quantile(0.05) - val_nulls.quantile(0.95):.4f}
  COCO-train: null_p95={train_nulls.quantile(0.95):.4f}  vs  tp_p05={train_tps.quantile(0.05):.4f}  gap={train_tps.quantile(0.05) - train_nulls.quantile(0.95):.4f}
""")
