"""
A1 fast path: filter existing LVIS pilot jsonl to keep only families with
status ∈ {NEG, POS_EXH} → a "clean aux subset" without re-running detector
inference.

Output: a sister jsonl that the existing formal_betting_pipeline can consume.
This gives us:
- Honest empirical FDP baseline on prompts whose ground-truth is reliable
- Reference point for η̄ = (full_FDP - clean_FDP) / full_FDP
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LVIS_JSON = REPO_ROOT / "data_raw/a_project_2/data/lvis/lvis_v1_val.json"
PILOT_JSONL = REPO_ROOT / "data_raw/a_project_2/outputs/lvis_groundingdino_1000_presentall_absentraremix.jsonl"
OUT_DIR = REPO_ROOT / "data_raw/a_project_2/outputs"
OUT_JSONL = OUT_DIR / "lvis_groundingdino_aux_clean_negposexh.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter an LVIS pilot jsonl to NEG+POS_EXH families only."
    )
    parser.add_argument("--ann-json", type=Path, default=LVIS_JSON)
    parser.add_argument("--input-jsonl", type=Path, default=PILOT_JSONL)
    parser.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[load] {args.ann_json}")
    with open(args.ann_json, "r") as f:
        lvis = json.load(f)

    images = {im["id"]: im for im in lvis["images"]}
    pos_per_image = defaultdict(set)
    for ann in lvis["annotations"]:
        pos_per_image[ann["image_id"]].add(ann["category_id"])

    def status(image_id, cat_id):
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
        return "UNKNOWN"

    n_in, n_out = 0, 0
    n_by_status = defaultdict(int)
    images_kept = set()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input_jsonl, "r") as f_in, open(args.out_jsonl, "w") as f_out:
        for line in f_in:
            n_in += 1
            fam = json.loads(line)
            st = status(fam["image_id"], fam["category_id"])
            n_by_status[st] += 1
            if st in ("NEG", "POS_EXH"):
                f_out.write(line)
                n_out += 1
                images_kept.add(fam["image_id"])

    # Write meta
    meta = {
        "source_jsonl": str(args.input_jsonl),
        "ann_json": str(args.ann_json),
        "filter": "status ∈ {NEG, POS_EXH}",
        "num_families_in": n_in,
        "num_families_out": n_out,
        "by_status": dict(n_by_status),
        "num_unique_images": len(images_kept),
    }
    with open(str(args.out_jsonl) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] in={n_in}, kept={n_out} (NEG+POS_EXH)")
    print(f"[ok] unique images covered: {len(images_kept)}")
    print(f"[ok] written: {args.out_jsonl}")
    print(f"[meta] {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
