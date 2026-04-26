#!/usr/bin/env python3
"""
Trace exactly which LVIS-1000 families contribute FP rejections (seed=0).
Diagnoses within-image null score heterogeneity as root cause of FDR violation.
"""
import os, json
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path

import os, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
try:
    from ovd_hallucination_fdr.paths import DATASETS as _DS
    _COCO_ROOT = _DS.get('coco_root', pathlib.Path('./data/coco2017'))
    _LVIS_ROOT = _DS.get('lvis_root', pathlib.Path('./data/lvis'))
    _OBJECTS365_ROOT = _DS.get('objects365_root', pathlib.Path('./data/objects365'))
    _OPEN_IMAGES_ROOT = _DS.get('open_images_root', pathlib.Path('./data/open_images_v7'))
except Exception:
    _COCO_ROOT = pathlib.Path(os.environ.get('COCO_ROOT', './data/coco2017'))
    _LVIS_ROOT = pathlib.Path(os.environ.get('LVIS_ROOT', './data/lvis'))
    _OBJECTS365_ROOT = pathlib.Path(os.environ.get('OBJECTS365_ROOT', './data/objects365'))
    _OPEN_IMAGES_ROOT = pathlib.Path(os.environ.get('OPEN_IMAGES_ROOT', './data/open_images_v7'))


os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovd_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CANDIDATE_TABLE = Path(str(Path(__file__).resolve().parents[1] / "outputs/lvis_groundingdino_1000_presentall_absentraremix_analysis/candidate_table.csv"))
IMG_ROOT = _COCO_ROOT
OUT_DIR = Path(str(Path(__file__).resolve().parents[1] / "outputs/lvis_fp_trace_seed0"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
N_BINS = 80
BINS = np.linspace(0.05, 1.0, N_BINS + 1)

# ── helpers ──────────────────────────────────────────────────────────────────
def fit_phi(tp_scores, null_scores):
    tp_h, _ = np.histogram(tp_scores, bins=BINS, density=True)
    nu_h, _ = np.histogram(null_scores, bins=BINS, density=True)
    nu_h = np.maximum(nu_h, 1e-6)
    ratio = np.minimum(tp_h / nu_h, 1e6)
    def phi(s):
        idx = np.searchsorted(BINS[1:], s, side="right")
        return ratio[np.clip(idx, 0, N_BINS - 1)]
    return phi

def compute_evalues(test_scores, phi_fn, phi_cal_sum, m):
    pv = phi_fn(test_scores)
    return (m + 1) * pv / (pv + phi_cal_sum)

def ebh(evalues, K, alpha):
    if K == 0 or len(evalues) == 0:
        return np.array([], dtype=int)
    order = np.argsort(evalues)[::-1]
    sorted_e = evalues[order]
    thresholds = K / (alpha * np.arange(1, K + 1))
    mask = sorted_e >= thresholds
    if not mask.any():
        return np.array([], dtype=int)
    k_hat = np.where(mask)[0][-1]
    return order[: k_hat + 1]

# ── load data ──────────────────────────────────────────────────────────────
print("Loading candidate table…")
df = pd.read_csv(CANDIDATE_TABLE)
print(f"  {len(df):,} candidates, {df['family_id'].nunique():,} families")

# ── seed-0 split ───────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
families = df["family_id"].unique().copy()
rng.shuffle(families)
n = len(families)
fit_fams  = set(families[:int(0.6 * n)])
cal_fams  = set(families[int(0.6 * n):int(0.8 * n)])
test_fams = set(families[int(0.8 * n):])

fit_data  = df[df["family_id"].isin(fit_fams)]
cal_data  = df[df["family_id"].isin(cal_fams)]
test_data = df[df["family_id"].isin(test_fams)]

# ── fit phi ───────────────────────────────────────────────────────────────
phi_fn = fit_phi(
    fit_data[fit_data["is_tp"]]["score"].values,
    fit_data[fit_data["is_null"]]["score"].values,
)

cal_null_scores = cal_data[cal_data["is_null"]]["score"].values
phi_cal = phi_fn(cal_null_scores)
phi_cal_sum = phi_cal.sum()
m = len(cal_null_scores)
print(f"  cal null pool: m={m}, phi_cal_sum={phi_cal_sum:.2f}")

# ── run e-BH on each test family ──────────────────────────────────────────
fp_records = []
tp_records = []

for fam_id, fam_df in test_data.groupby("family_id"):
    K = len(fam_df)
    scores = fam_df["score"].values
    ev = compute_evalues(scores, phi_fn, phi_cal_sum, m)
    rej_idx = ebh(ev, K, ALPHA)
    if len(rej_idx) == 0:
        continue
    rej_rows = fam_df.iloc[rej_idx]
    for _, row in rej_rows.iterrows():
        rec = {
            "family_id": fam_id,
            "image_id": row["image_id"],
            "file_name": row["file_name"],
            "category_name": row["category_name"],
            "is_prompt_absent": row["is_prompt_absent"],
            "score": row["score"],
            "max_iou": row["max_iou"],
            "e_value": ev[fam_df.index.get_loc(row.name)],
            "family_K": K,
            "is_null": row["is_null"],
            "is_tp": row["is_tp"],
        }
        if row["is_null"]:
            fp_records.append(rec)
        else:
            tp_records.append(rec)

fp_df = pd.DataFrame(fp_records)
tp_df = pd.DataFrame(tp_records)
total_rej = len(fp_df) + len(tp_df)
fdp = len(fp_df) / max(total_rej, 1)
recall = len(tp_df) / df[df["family_id"].isin(test_fams) & df["is_tp"]].shape[0]

print(f"\n=== Seed-0 results (alpha={ALPHA}) ===")
print(f"  Total rejections: {total_rej}  TP={len(tp_df)}  FP={len(fp_df)}")
print(f"  FDP={fdp:.4f}  Recall={recall:.4f}")

# ── Analysis A: FP by is_prompt_absent ──────────────────────────────────
print("\n=== A. FP breakdown by is_prompt_absent ===")
if len(fp_df):
    print(fp_df["is_prompt_absent"].value_counts().rename({True:"absent", False:"present"}))

# ── Analysis B: FP by max_iou bucket ─────────────────────────────────────
print("\n=== B. FP breakdown by max_iou bucket ===")
if len(fp_df):
    bins_iou = [-0.001, 0.1, 0.3, 0.4, 0.5]
    labels = ["[0,0.1)", "[0.1,0.3)", "[0.3,0.4)", "[0.4,0.5)"]
    fp_df["iou_bucket"] = pd.cut(fp_df["max_iou"], bins=bins_iou, labels=labels)
    print(fp_df["iou_bucket"].value_counts().sort_index())

# ── Analysis C: top-10 images by FP count ────────────────────────────────
print("\n=== C. Top-10 images by FP count ===")
if len(fp_df):
    img_fp = fp_df.groupby("image_id").agg(
        n_fp=("is_null","sum"),
        categories=("category_name", lambda x: list(x.unique())),
        file_name=("file_name","first"),
        mean_score=("score","mean"),
    ).sort_values("n_fp", ascending=False).head(10)
    print(img_fp[["n_fp","mean_score","categories"]].to_string())

# ── Per-image null score stats (whole dataset) ────────────────────────────
print("\n=== D. Per-image null score: top-20 highest-null-score images ===")
img_null = df[df["is_null"]].groupby("image_id").agg(
    n_null=("score","count"),
    mean_score=("score","mean"),
    p95_score=("score", lambda x: np.percentile(x, 95)),
).reset_index()
global_p95 = np.percentile(df[df["is_null"]]["score"].values, 95)
print(f"  Global null p95 = {global_p95:.4f}")
high_null_imgs = img_null[img_null["p95_score"] > 2 * global_p95].sort_values("p95_score", ascending=False)
print(f"  Images with p95 > 2*global_p95: {len(high_null_imgs)}")
print(high_null_imgs.head(20).to_string(index=False))

# ── What fraction of FPs come from high-null-score images? ────────────────
if len(fp_df):
    top10pct_imgs = set(img_null.nlargest(int(0.1 * len(img_null)), "p95_score")["image_id"])
    fp_from_top = fp_df["image_id"].isin(top10pct_imgs).sum()
    print(f"\n  FPs from top-10%-null-score images: {fp_from_top}/{len(fp_df)} = {fp_from_top/len(fp_df):.1%}")

# ── Save FP CSV ───────────────────────────────────────────────────────────
fp_df.to_csv(OUT_DIR / "fp_rejections_seed0.csv", index=False)
print(f"\nSaved FP CSV → {OUT_DIR}/fp_rejections_seed0.csv")

# ── Analysis E: render top-12 FP boxes ───────────────────────────────────
print("\n=== E. Rendering top-12 FP boxes ===")
if len(fp_df) == 0:
    print("  No FP rejections to render.")
else:
    top_fps = fp_df.sort_values("score", ascending=False).head(12)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    rendered = 0
    for ax in axes:
        ax.axis("off")

    for _, row in top_fps.iterrows():
        img_path = IMG_ROOT / row["file_name"]
        if not img_path.exists():
            print(f"  Missing: {img_path}")
            continue
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        ax = axes[rendered]
        ax.imshow(img)
        x1, y1, x2, y2 = row["score"], row["max_iou"], 0, 0  # placeholder init
        # get box from original df
        match = df[(df["family_id"] == row["family_id"]) &
                   (df["score"].round(6) == round(row["score"], 6)) &
                   (df["is_null"] == True)]
        if len(match):
            r = match.iloc[0]
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                      linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
        absent_str = "ABSENT" if row["is_prompt_absent"] else "present"
        label = f"{row['category_name']}\ns={row['score']:.2f} iou={row['max_iou']:.2f}\n{absent_str}"
        ax.set_title(label, fontsize=8, color="red")
        rendered += 1
        if rendered >= 12:
            break

    plt.suptitle("Top-12 FP rejections (LVIS seed=0, sorted by score)", fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / "fp_top12_visual.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Rendered {rendered} boxes → {out_png}")

# ── Summary JSON ─────────────────────────────────────────────────────────
summary = {
    "seed": 0, "alpha": ALPHA,
    "total_rejections": total_rej,
    "fp": len(fp_df), "tp": len(tp_df),
    "fdp": round(fdp, 4), "recall": round(recall, 4),
    "fp_absent": int(fp_df["is_prompt_absent"].sum()) if len(fp_df) else 0,
    "fp_present": int((~fp_df["is_prompt_absent"]).sum()) if len(fp_df) else 0,
    "fp_iou_below_0.1": int((fp_df["max_iou"] < 0.1).sum()) if len(fp_df) else 0,
    "fp_iou_0.3_to_0.5": int(((fp_df["max_iou"] >= 0.3) & (fp_df["max_iou"] < 0.5)).sum()) if len(fp_df) else 0,
}
with open(OUT_DIR / "summary_seed0.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary → {OUT_DIR}/summary_seed0.json")
print(json.dumps(summary, indent=2))
