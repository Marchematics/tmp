#!/usr/bin/env python3
"""
Filter an OpenImages pilot jsonl using human image-level labels.

Keep a family iff:
- absent prompt: the prompt class MID is human-verified negative for the image
- present prompt: the prompt class MID is human-verified positive for the image
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


REPO_ROOT = Path(__file__).resolve().parents[1]
OI_ANN_DIR = _OPEN_IMAGES_ROOT / "annotations"
DEFAULT_COCO_JSON = OI_ANN_DIR / "oi_val_coco_5000.json"
DEFAULT_CLASS_CSV = OI_ANN_DIR / "class-descriptions-boxable.csv"
DEFAULT_IMAGE_LABELS = OI_ANN_DIR / "oidv7-val-annotations-human-imagelabels.csv"
DEFAULT_INPUT = REPO_ROOT / "data_raw/a_project_2/outputs/oi_1000_gdino.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data_raw/a_project_2/outputs/oi_1000_gdino_aux_clean_imagelabels.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter OpenImages candidates to human-verified image-label families."
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ann-json", type=Path, default=DEFAULT_COCO_JSON)
    parser.add_argument("--class-descriptions", type=Path, default=DEFAULT_CLASS_CSV)
    parser.add_argument("--image-labels-csv", type=Path, default=DEFAULT_IMAGE_LABELS)
    return parser.parse_args()


def load_cat_id_to_mid(class_csv: Path, ann_json: Path) -> dict[int, str]:
    mids: list[str] = []
    names: list[str] = []
    with class_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            mids.append(row[0])
            names.append(row[1])

    with ann_json.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    cat_id_to_mid: dict[int, str] = {}
    mismatches: list[tuple[int, str, str]] = []
    for cat in coco["categories"]:
        cat_id = int(cat["id"])
        if cat_id < 1 or cat_id > len(mids):
            raise ValueError(f"category id out of class-description range: {cat_id}")
        cat_id_to_mid[cat_id] = mids[cat_id - 1]
        expected_name = names[cat_id - 1]
        if cat["name"] != expected_name:
            mismatches.append((cat_id, cat["name"], expected_name))

    if mismatches:
        sample = mismatches[:5]
        raise ValueError(f"category/class-description mismatch, sample={sample}")
    return cat_id_to_mid


def collect_needed_images(input_jsonl: Path) -> set[str]:
    needed: set[str] = set()
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            fam = json.loads(line)
            needed.add(Path(fam["file_name"]).stem)
    return needed


def load_image_label_sets(
    image_labels_csv: Path, needed_image_ids: set[str]
) -> tuple[dict[str, set[str]], dict[str, set[str]], int]:
    pos_by_image: dict[str, set[str]] = defaultdict(set)
    neg_by_image: dict[str, set[str]] = defaultdict(set)
    rows_used = 0
    with image_labels_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["ImageID"]
            if image_id not in needed_image_ids:
                continue
            confidence = float(row["Confidence"])
            label = row["LabelName"]
            if confidence == 1.0:
                pos_by_image[image_id].add(label)
                rows_used += 1
            elif confidence == 0.0:
                neg_by_image[image_id].add(label)
                rows_used += 1
    return pos_by_image, neg_by_image, rows_used


def main() -> None:
    args = parse_args()
    print(f"[load] class map: {args.class_descriptions}")
    cat_id_to_mid = load_cat_id_to_mid(args.class_descriptions, args.ann_json)

    print(f"[scan] needed images from {args.input_jsonl}")
    needed_images = collect_needed_images(args.input_jsonl)
    print(f"[scan] needed images: {len(needed_images)}")

    print(f"[load] image labels: {args.image_labels_csv}")
    pos_by_image, neg_by_image, label_rows_used = load_image_label_sets(
        args.image_labels_csv, needed_images
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    by_status: dict[str, int] = defaultdict(int)
    images_kept: set[str] = set()
    with args.input_jsonl.open("r", encoding="utf-8") as f_in, args.out_jsonl.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            n_in += 1
            fam = json.loads(line)
            image_id = Path(fam["file_name"]).stem
            cat_mid = cat_id_to_mid[int(fam["category_id"])]
            if fam["is_prompt_absent"]:
                status = "ABSENT_NEG" if cat_mid in neg_by_image.get(image_id, set()) else "ABSENT_UNVERIFIED"
            else:
                status = "PRESENT_POS" if cat_mid in pos_by_image.get(image_id, set()) else "PRESENT_UNVERIFIED"
            by_status[status] += 1

            if status in {"ABSENT_NEG", "PRESENT_POS"}:
                f_out.write(line)
                n_out += 1
                images_kept.add(image_id)

    meta = {
        "source_jsonl": str(args.input_jsonl),
        "ann_json": str(args.ann_json),
        "class_descriptions": str(args.class_descriptions),
        "image_labels_csv": str(args.image_labels_csv),
        "filter": "absent in human negative labels OR present in human positive labels",
        "num_families_in": n_in,
        "num_families_out": n_out,
        "by_status": dict(by_status),
        "num_needed_images": len(needed_images),
        "num_images_with_positive_labels": len(pos_by_image),
        "num_images_with_negative_labels": len(neg_by_image),
        "num_image_label_rows_used": label_rows_used,
        "num_unique_images_kept": len(images_kept),
    }
    meta_path = Path(str(args.out_jsonl) + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    print(f"[ok] in={n_in}, kept={n_out}")
    print(f"[ok] written: {args.out_jsonl}")
    print(f"[meta] {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
