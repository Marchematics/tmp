#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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


import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ovd_hallucination_fdr.annotations import (  # noqa: E402
    family_to_record_base,
    load_annotation_index,
    sample_query_families,
)
from ovd_hallucination_fdr.detectors import GroundingDinoHF, OwlViTHF, YoloWorldUltralytics  # noqa: E402
from ovd_hallucination_fdr.io import ensure_dir  # noqa: E402
from ovd_hallucination_fdr.matching import max_iou_to_gt  # noqa: E402


DEFAULT_COCO_ANN = Path(str(_COCO_ROOT / "annotations/instances_val2017.json"))
DEFAULT_COCO_IMG_DIR = Path(str(_COCO_ROOT / "val2017"))
DEFAULT_OBJECTS365_ANN = _OBJECTS365_ROOT / "annotations/zhiyuan_objv2_val.json"
DEFAULT_OBJECTS365_IMG_DIR = _OBJECTS365_ROOT / "val"
DEFAULT_OPENIMAGES_ANN = _OPEN_IMAGES_ROOT / "annotations/oi_val_coco_5000.json"
DEFAULT_OPENIMAGES_IMG_DIR = _OPEN_IMAGES_ROOT / "val"
DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_OWLVIT_MODEL_ID = "google/owlv2-base-patch16-ensemble"
DEFAULT_YOLOWORLD_MODEL_ID = "yolov8x-worldv2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Grounding DINO prompted-present/absent hallucination pilot."
    )
    parser.add_argument("--dataset", choices=["coco", "lvis", "objects365", "openimages"], default="coco")
    parser.add_argument("--detector", choices=["groundingdino", "owlvit", "yoloworld"], default="groundingdino")
    parser.add_argument("--ann-json", type=Path, default=DEFAULT_COCO_ANN)
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_COCO_IMG_DIR)
    parser.add_argument("--out-jsonl", type=Path, default=PROJECT_ROOT / "outputs" / "groundingdino_candidates.jsonl")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit-images", type=int, default=100)
    parser.add_argument("--prompt-pool", choices=["all", "rare"], default="all")
    parser.add_argument("--present-prompt-pool", choices=["all", "rare"], default=None)
    parser.add_argument("--absent-prompt-pool", choices=["all", "rare"], default=None)
    parser.add_argument("--absent-prompts-per-image", type=int, default=1)
    parser.add_argument("--present-prompts-per-image", type=int, default=1)
    parser.add_argument("--prompt-style", choices=["category", "freeform"], default="category")
    parser.add_argument("--prompt-template", type=str, default="{category}")
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    # Resolve dataset-specific defaults when user didn't explicitly pass --ann-json / --img-dir
    if args.ann_json == DEFAULT_COCO_ANN and args.dataset == "objects365":
        args.ann_json = DEFAULT_OBJECTS365_ANN
        args.img_dir = DEFAULT_OBJECTS365_IMG_DIR
    elif args.ann_json == DEFAULT_COCO_ANN and args.dataset == "openimages":
        args.ann_json = DEFAULT_OPENIMAGES_ANN
        args.img_dir = DEFAULT_OPENIMAGES_IMG_DIR
    if not args.ann_json.exists():
        raise FileNotFoundError(f"Missing annotation JSON: {args.ann_json}")
    if not args.img_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {args.img_dir}")
    if args.absent_prompts_per_image < 0 or args.present_prompts_per_image < 0:
        raise ValueError("Prompt counts must be nonnegative")
    if args.absent_prompts_per_image + args.present_prompts_per_image <= 0:
        raise ValueError("At least one prompt per image is required")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    validate_args(args)

    total_start = time.perf_counter()
    logging.info("Loading %s annotations from %s", args.dataset, args.ann_json)
    index = load_annotation_index(
        ann_json=args.ann_json,
        img_dir=args.img_dir,
        dataset=args.dataset,
        limit_images=args.limit_images,
    )
    families = sample_query_families(
        index,
        prompt_template=args.prompt_template,
        prompt_style=args.prompt_style,
        prompt_pool=args.prompt_pool,
        present_prompt_pool=args.present_prompt_pool,
        absent_prompt_pool=args.absent_prompt_pool,
        absent_prompts_per_image=args.absent_prompts_per_image,
        present_prompts_per_image=args.present_prompts_per_image,
        seed=args.seed,
    )
    logging.info("Built %d image-query families from %d images", len(families), len(index.images))

    if args.detector == "owlvit":
        model_id = args.model_id if args.model_id != DEFAULT_MODEL_ID else DEFAULT_OWLVIT_MODEL_ID
        detector = OwlViTHF(model_id=model_id, device=args.device)
    elif args.detector == "yoloworld":
        model_id = args.model_id if args.model_id != DEFAULT_MODEL_ID else DEFAULT_YOLOWORLD_MODEL_ID
        detector = YoloWorldUltralytics(model_id=model_id, device=args.device)
    else:
        detector = GroundingDinoHF(model_id=args.model_id, device=args.device)
    ensure_dir(args.out_jsonl.parent)

    num_written = 0
    num_candidates = 0
    num_missing = 0
    generation_start = time.perf_counter()
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for family in tqdm(families, desc="Grounding DINO families"):
            if not family.image_path.exists():
                num_missing += 1
                if args.skip_missing_images:
                    continue
                raise FileNotFoundError(f"Missing image: {family.image_path}")

            detections = detector.predict(
                family.image_path,
                family.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                nms_iou=args.nms_iou,
                max_det=args.max_det,
            )
            boxes = [list(det.box) for det in detections]
            max_ious = max_iou_to_gt(boxes, family.gt_boxes)

            candidates: list[dict[str, object]] = []
            for det, max_iou in zip(detections, max_ious):
                candidates.append(
                    {
                        "box": [round(float(v), 4) for v in det.box],
                        "score": float(det.score),
                        "label": det.label,
                        "max_iou": float(max_iou),
                        "is_tp": bool(float(max_iou) >= args.match_iou),
                    }
                )

            record = family_to_record_base(family)
            record.update(
                {
                    "model_id": args.model_id,
                    "box_threshold": args.box_threshold,
                    "text_threshold": args.text_threshold,
                    "nms_iou": args.nms_iou,
                    "max_det": args.max_det,
                    "match_iou": args.match_iou,
                    "candidates": candidates,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            num_written += 1
            num_candidates += len(candidates)

            if args.log_every and num_written % args.log_every == 0:
                logging.info(
                    "Processed %d families, candidates=%d, mean K=%.2f",
                    num_written,
                    num_candidates,
                    num_candidates / max(num_written, 1),
                )

    meta_path = args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".meta.json")
    generation_seconds = time.perf_counter() - generation_start
    total_seconds = time.perf_counter() - total_start
    meta = {
        "dataset": args.dataset,
        "ann_json": str(args.ann_json),
        "img_dir": str(args.img_dir),
        "out_jsonl": str(args.out_jsonl),
        "detector": args.detector,
        "model_id": args.model_id,
        "device": args.device,
        "limit_images": args.limit_images,
        "prompt_pool": args.prompt_pool,
        "present_prompt_pool": args.present_prompt_pool,
        "absent_prompt_pool": args.absent_prompt_pool,
        "absent_prompts_per_image": args.absent_prompts_per_image,
        "present_prompts_per_image": args.present_prompts_per_image,
        "prompt_style": args.prompt_style,
        "prompt_template": args.prompt_template,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "match_iou": args.match_iou,
        "seed": args.seed,
        "num_families_written": num_written,
        "num_candidates": num_candidates,
        "mean_candidates_per_family": float(num_candidates / max(num_written, 1)),
        "num_missing_images": num_missing,
        "candidate_generation_seconds": float(generation_seconds),
        "total_wall_seconds": float(total_seconds),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s and %s", args.out_jsonl, meta_path)


if __name__ == "__main__":
    main()
