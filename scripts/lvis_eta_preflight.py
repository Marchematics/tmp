"""
A1 pre-flight: Estimate the η̄ (non-exhaustiveness rate) range on existing LVIS
candidates BEFORE designing the full lvis_aux_eta_pipeline.

Goal: decide whether α(1-η̄) correction is feasible.
- If η̄ < 0.2: A1 is viable, recall should stay > 10%
- If η̄ in [0.2, 0.3]: marginal, need conservative quantile
- If η̄ > 0.3: re-design needed

Method:
- LVIS v1 val per-image annotations include 'not_exhaustive_category_ids' and
  'neg_category_ids'. Per the LVIS protocol:
    * If a category is in `neg_category_ids` → confirmed absent (any detector
      hit on the image for this category is a guaranteed FP).
    * If a category is in image's positive ann set → present.
    * Otherwise (category neither in pos, neg, nor exhaustive) → status unknown
      → "non-exhaustive" → empirical FP may be inflated.
- Build per-(image,category) status from raw LVIS json.
- For each candidate in the pilot jsonl, classify by status.
- Compare empirical FP rate of candidates whose category is exhaustive
  (= confirmed-absent OR confirmed-present-with-all-instances-labelled)
  vs candidates whose category is non-exhaustive.
- η̄ proxy = max(0, 1 - FP_rate_nonexh / FP_rate_neg) ≈ fraction of "FP" calls
  on non-exhaustive families that could in fact be hidden TPs.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LVIS_JSON = REPO_ROOT / "data_raw/a_project_2/data/lvis/lvis_v1_val.json"
PILOT_JSONL = REPO_ROOT / "data_raw/a_project_2/outputs/lvis_groundingdino_1000_presentall_absentraremix.jsonl"


def main():
    print(f"[load] {LVIS_JSON}")
    with open(LVIS_JSON, "r") as f:
        lvis = json.load(f)

    images = {im["id"]: im for im in lvis["images"]}
    # LVIS image fields: not_exhaustive_category_ids, neg_category_ids
    # Build per-image: pos_set (cats with at least one annotation in this image)
    pos_per_image = defaultdict(set)
    for ann in lvis["annotations"]:
        pos_per_image[ann["image_id"]].add(ann["category_id"])

    print(f"[ok] LVIS val: {len(images)} images, {len(lvis['annotations'])} anns")

    # Per-(image, cat) status enum:
    #   'NEG'        → in neg_category_ids (confirmed absent)
    #   'POS_EXH'    → in positive set AND NOT in not_exhaustive_category_ids
    #   'POS_NONEXH' → in positive set AND in not_exhaustive_category_ids
    #   'UNKNOWN'    → neither in pos, neg, exh-list, nor non-exh-list
    def status(image_id: int, cat_id: int) -> str:
        im = images[image_id]
        neg = set(im.get("neg_category_ids", []))
        non_exh = set(im.get("not_exhaustive_category_ids", []))
        pos = pos_per_image[image_id]
        if cat_id in neg:
            return "NEG"
        if cat_id in pos:
            return "POS_NONEXH" if cat_id in non_exh else "POS_EXH"
        if cat_id in non_exh:
            return "UNKNOWN_NONEXH"
        return "UNKNOWN"  # treated as confirmed-absent under LVIS federated eval

    # Walk the pilot jsonl
    print(f"[load] {PILOT_JSONL}")
    counts = defaultdict(lambda: dict(n_cand=0, n_pos_match=0, n_neg_match=0))
    fam_count = 0
    with open(PILOT_JSONL, "r") as f:
        for line in f:
            fam = json.loads(line)
            fam_count += 1
            st = status(fam["image_id"], fam["category_id"])
            cands = fam.get("candidates", [])
            counts[st]["n_cand"] += len(cands)
            for c in cands:
                # Heuristic for FP/TP under existing GT:
                # candidate dict has 'matched' field if pipeline already labeled.
                # Otherwise fall back to our own IoU check.
                matched = c.get("is_tp", c.get("matched", None))
                if matched is None:
                    # If absent prompt, gt_count is 0, no candidate can be TP
                    if fam["gt_count"] == 0:
                        matched = False
                    else:
                        # naive: if any IoU >= match_iou we consider TP — keep it simple,
                        # mark unknown as FP which is the conservative choice
                        matched = False
                if matched:
                    counts[st]["n_pos_match"] += 1
                else:
                    counts[st]["n_neg_match"] += 1

    print(f"[ok] processed {fam_count} families")
    print()
    print(f"{'Status':22s} {'#fam (cand)':>14s} {'#FP':>8s} {'#TP':>8s} {'FP rate':>10s}")
    print("-" * 70)
    for st in ["NEG", "POS_EXH", "POS_NONEXH", "UNKNOWN", "UNKNOWN_NONEXH"]:
        d = counts[st]
        n = d["n_cand"]
        fp_rate = d["n_neg_match"] / n if n else 0.0
        print(f"{st:22s} {n:>14d} {d['n_neg_match']:>8d} {d['n_pos_match']:>8d} {fp_rate:>10.4f}")
    print()

    # η̄ estimation:
    # On NEG families: every candidate is a guaranteed FP under LVIS protocol → reference FP rate = 1.0
    # On UNKNOWN families: per LVIS federated eval, treated as absent, but truly some hidden TPs
    #   → empirical FP rate underestimates true TP rate
    #   → η̄ ≈ fraction of "FP" calls on UNKNOWN families that are actually hidden TPs
    # Direct estimation needs an oracle; instead use a proxy:
    # For absent prompts (gt_count == 0) on NON-NEG categories, the "FP" count is
    # potentially contaminated by hidden TPs.

    print("=== η̄ proxy estimation ===")
    # Group by family-level absent/present
    absent_neg_cands = 0
    absent_unknown_cands = 0
    absent_unknown_nonexh_cands = 0
    with open(PILOT_JSONL, "r") as f:
        for line in f:
            fam = json.loads(line)
            if fam["gt_count"] != 0:
                continue
            st = status(fam["image_id"], fam["category_id"])
            n = len(fam.get("candidates", []))
            if st == "NEG":
                absent_neg_cands += n
            elif st == "UNKNOWN":
                absent_unknown_cands += n
            elif st == "UNKNOWN_NONEXH":
                absent_unknown_nonexh_cands += n

    print(f"Absent-prompt families:")
    print(f"  NEG (clean):          {absent_neg_cands} candidates")
    print(f"  UNKNOWN (federated):  {absent_unknown_cands} candidates")
    print(f"  UNKNOWN_NONEXH:       {absent_unknown_nonexh_cands} candidates")
    print()
    print(f"Note: η̄ proxy requires oracle to count hidden TPs in UNKNOWN/UNKNOWN_NONEXH")
    print(f"      → A1 should pick aux subset from images where ALL queried prompts have status NEG or POS_EXH")
    print()

    # Coverage feasibility check: how many images in the 1000-pilot have all prompts as NEG or POS_EXH?
    print("=== aux-subset feasibility on the existing 1000-image pilot ===")
    by_image_clean = defaultdict(lambda: dict(total=0, clean=0))
    with open(PILOT_JSONL, "r") as f:
        for line in f:
            fam = json.loads(line)
            st = status(fam["image_id"], fam["category_id"])
            by_image_clean[fam["image_id"]]["total"] += 1
            if st in ("NEG", "POS_EXH"):
                by_image_clean[fam["image_id"]]["clean"] += 1
    n_images = len(by_image_clean)
    n_full_clean = sum(1 for d in by_image_clean.values() if d["clean"] == d["total"])
    n_partial = sum(1 for d in by_image_clean.values() if 0 < d["clean"] < d["total"])
    print(f"  Total images in pilot: {n_images}")
    print(f"  Fully clean (all prompts NEG/POS_EXH): {n_full_clean}")
    print(f"  Partially clean (≥1 unknown prompt):    {n_partial}")
    print(f"  Fully unknown:                          {n_images - n_full_clean - n_partial}")
    print()
    print(f"  → A1 aux subset target: {min(n_full_clean, 200)} images viable from existing pilot")


if __name__ == "__main__":
    main()
