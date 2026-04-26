#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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

from typing import Any

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
from ovd_hallucination_fdr.io import ensure_dir  # noqa: E402
from ovd_hallucination_fdr.matching import max_iou_to_gt  # noqa: E402


DEFAULT_COCO_ANN = Path(str(_COCO_ROOT / "annotations/instances_val2017.json"))
DEFAULT_COCO_IMG_DIR = Path(str(_COCO_ROOT / "val2017"))
DEFAULT_OBJECTS365_ANN = _OBJECTS365_ROOT / "annotations/zhiyuan_objv2_val.json"
DEFAULT_OBJECTS365_IMG_DIR = _OBJECTS365_ROOT / "val"
DEFAULT_OPENIMAGES_ANN = _OPEN_IMAGES_ROOT / "annotations/oi_val_coco_5000.json"
DEFAULT_OPENIMAGES_IMG_DIR = _OPEN_IMAGES_ROOT / "val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a BIRDet/MMDetection OV detector on the same image-prompt families "
            "as run_groundingdino_pilot.py and emit compatible jsonl candidates."
        )
    )
    parser.add_argument("--dataset", choices=["coco", "lvis", "objects365", "openimages"], default="coco")
    parser.add_argument("--ann-json", type=Path, default=DEFAULT_COCO_ANN)
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_COCO_IMG_DIR)
    parser.add_argument("--out-jsonl", type=Path, default=PROJECT_ROOT / "outputs" / "coco_birdet_1000_mix.jsonl")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional cloned BIRDet repo to prepend to sys.path.")
    parser.add_argument("--config", type=Path, required=True, help="MMDetection config for BIRDet.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="BIRDet checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--limit-images", type=int, default=1000)
    parser.add_argument("--prompt-pool", choices=["all", "rare"], default="all")
    parser.add_argument("--present-prompt-pool", choices=["all", "rare"], default=None)
    parser.add_argument("--absent-prompt-pool", choices=["all", "rare"], default=None)
    parser.add_argument("--absent-prompts-per-image", type=int, default=1)
    parser.add_argument("--present-prompts-per-image", type=int, default=1)
    parser.add_argument("--prompt-style", choices=["category", "freeform"], default="category")
    parser.add_argument("--prompt-template", type=str, default="{category}")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--vanilla-threshold", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--class-source", choices=["auto", "model", "annotation"], default="auto")
    parser.add_argument("--pos-oar-threshold", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
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
    if not args.config.exists():
        raise FileNotFoundError(f"Missing BIRDet/MMDetection config: {args.config}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing BIRDet checkpoint: {args.checkpoint}")
    if args.absent_prompts_per_image < 0 or args.present_prompts_per_image < 0:
        raise ValueError("Prompt counts must be nonnegative")
    if args.absent_prompts_per_image + args.present_prompts_per_image <= 0:
        raise ValueError("At least one prompt per image is required")
    if args.score_threshold < 0.0:
        raise ValueError("--score-threshold must be nonnegative")
    if args.max_det <= 0:
        raise ValueError("--max-det must be positive")


def normalize_label(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").split())


def load_mmdet_model(args: argparse.Namespace) -> Any:
    if args.repo_root is not None:
        repo_root = args.repo_root.resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"Missing BIRDet repo root: {repo_root}")
        sys.path.insert(0, str(repo_root))

    try:
        from mmdet.apis import init_detector
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError(
            "Could not import mmdet.apis.init_detector. Activate the BIRDet env "
            "and install the official repository requirements before running this script."
        ) from exc

    return init_detector(str(args.config), str(args.checkpoint), device=args.device)


def get_model_classes(model: Any) -> list[str] | None:
    dataset_meta = getattr(model, "dataset_meta", None)
    if isinstance(dataset_meta, dict):
        classes = dataset_meta.get("classes") or dataset_meta.get("CLASSES")
        if classes:
            return [str(v) for v in classes]
    classes = getattr(model, "CLASSES", None)
    if classes:
        return [str(v) for v in classes]
    return None


def tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def detections_from_mmdet_result(result: Any) -> list[dict[str, Any]]:
    if hasattr(result, "pred_instances"):
        inst = result.pred_instances
        bboxes = tensor_to_numpy(getattr(inst, "bboxes", None)).reshape(-1, 4)
        scores = tensor_to_numpy(getattr(inst, "scores", None)).reshape(-1)
        labels = tensor_to_numpy(getattr(inst, "labels", None)).reshape(-1)
        return [
            {"box": bboxes[i].astype(float).tolist(), "score": float(scores[i]), "label_index": int(labels[i])}
            for i in range(len(scores))
        ]

    if isinstance(result, tuple) and result:
        result = result[0]

    if isinstance(result, list):
        detections: list[dict[str, Any]] = []
        for label_index, class_rows in enumerate(result):
            arr = tensor_to_numpy(class_rows)
            if arr.size == 0:
                continue
            arr = arr.reshape(-1, arr.shape[-1])
            for row in arr:
                if len(row) < 5:
                    continue
                detections.append(
                    {
                        "box": [float(v) for v in row[:4]],
                        "score": float(row[4]),
                        "label_index": int(label_index),
                    }
                )
        return detections

    raise TypeError(f"Unsupported MMDetection inference result type: {type(result)!r}")


def overlap_area_ratio(candidate: list[float], selected: list[float]) -> float:
    x1 = max(float(candidate[0]), float(selected[0]))
    y1 = max(float(candidate[1]), float(selected[1]))
    x2 = min(float(candidate[2]), float(selected[2]))
    y2 = min(float(candidate[3]), float(selected[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, float(candidate[2]) - float(candidate[0])) * max(0.0, float(candidate[3]) - float(candidate[1]))
    return inter / area if area > 0 else 0.0


def apply_pos(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for det in detections:
        by_label[int(det["label_index"])].append(det)

    kept: list[dict[str, Any]] = []
    for label_dets in by_label.values():
        selected: list[dict[str, Any]] = []
        for det in sorted(label_dets, key=lambda item: float(item["score"]), reverse=True):
            if all(overlap_area_ratio(det["box"], prev["box"]) < threshold for prev in selected):
                selected.append(det)
        kept.extend(selected)
    return kept


def build_class_lookup(index_categories: tuple[Any, ...], model_classes: list[str] | None, class_source: str) -> dict[str, int]:
    if class_source == "model":
        if not model_classes:
            raise ValueError("--class-source=model requested, but the model exposes no class names")
        class_names = model_classes
    elif class_source == "annotation":
        class_names = [cat.name for cat in index_categories]
    else:
        class_names = model_classes or [cat.name for cat in index_categories]
    lookup: dict[str, int] = {}
    for idx, name in enumerate(class_names):
        lookup.setdefault(normalize_label(name), idx)
    return lookup


def infer_image(model: Any, image_path: Path, score_threshold: float, max_det: int, pos_oar_threshold: float | None) -> list[dict[str, Any]]:
    try:
        from mmdet.apis import inference_detector
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError("Could not import mmdet.apis.inference_detector") from exc

    detections = detections_from_mmdet_result(inference_detector(model, str(image_path)))
    detections = [det for det in detections if float(det["score"]) >= score_threshold]
    if pos_oar_threshold is not None:
        detections = apply_pos(detections, pos_oar_threshold)
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return detections[:max_det]


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

    model = load_mmdet_model(args)
    model_classes = get_model_classes(model)
    class_lookup = build_class_lookup(index.categories, model_classes, args.class_source)
    ensure_dir(args.out_jsonl.parent)

    families_by_image: dict[int, list[Any]] = defaultdict(list)
    for family in families:
        families_by_image[family.image_id].append(family)

    num_written = 0
    num_candidates = 0
    num_missing = 0
    num_unmatched_classes = 0
    generation_start = time.perf_counter()
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for image in tqdm(index.images, desc="BIRDet images"):
            image_families = families_by_image.get(image.id, [])
            if not image_families:
                continue
            if not image.image_path.exists():
                num_missing += 1
                if args.skip_missing_images:
                    continue
                raise FileNotFoundError(f"Missing image: {image.image_path}")

            image_detections = infer_image(
                model,
                image.image_path,
                score_threshold=args.score_threshold,
                max_det=args.max_det,
                pos_oar_threshold=args.pos_oar_threshold,
            )

            for family in image_families:
                target_index = class_lookup.get(normalize_label(family.category_name))
                if target_index is None:
                    num_unmatched_classes += 1
                    family_detections: list[dict[str, Any]] = []
                else:
                    family_detections = [
                        det for det in image_detections if int(det["label_index"]) == int(target_index)
                    ][: args.max_det]

                boxes = [det["box"] for det in family_detections]
                max_ious = max_iou_to_gt(boxes, family.gt_boxes)
                candidates: list[dict[str, object]] = []
                for det, max_iou in zip(family_detections, max_ious):
                    candidates.append(
                        {
                            "box": [round(float(v), 4) for v in det["box"]],
                            "score": float(det["score"]),
                            "label": family.category_name,
                            "label_index": int(det["label_index"]),
                            "max_iou": float(max_iou),
                            "is_tp": bool(float(max_iou) >= args.match_iou),
                        }
                    )

                record = family_to_record_base(family)
                record.update(
                    {
                        "model_id": "BIRDet",
                        "detector": "birdet_mmdet",
                        "config": str(args.config),
                        "checkpoint": str(args.checkpoint),
                        "score_threshold": args.score_threshold,
                        "vanilla_threshold": args.vanilla_threshold,
                        "max_det": args.max_det,
                        "match_iou": args.match_iou,
                        "pos_oar_threshold": args.pos_oar_threshold,
                        "class_source": args.class_source,
                        "candidates": candidates,
                    }
                )
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                num_written += 1
                num_candidates += len(candidates)

            if args.log_every and num_written and num_written % args.log_every == 0:
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
        "detector": "birdet_mmdet",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "limit_images": args.limit_images,
        "prompt_pool": args.prompt_pool,
        "present_prompt_pool": args.present_prompt_pool,
        "absent_prompt_pool": args.absent_prompt_pool,
        "absent_prompts_per_image": args.absent_prompts_per_image,
        "present_prompts_per_image": args.present_prompts_per_image,
        "prompt_style": args.prompt_style,
        "prompt_template": args.prompt_template,
        "score_threshold": args.score_threshold,
        "vanilla_threshold": args.vanilla_threshold,
        "max_det": args.max_det,
        "match_iou": args.match_iou,
        "seed": args.seed,
        "class_source": args.class_source,
        "model_classes": model_classes,
        "pos_oar_threshold": args.pos_oar_threshold,
        "num_families_written": num_written,
        "num_candidates": num_candidates,
        "mean_candidates_per_family": float(num_candidates / max(num_written, 1)),
        "num_missing_images": num_missing,
        "num_unmatched_classes": num_unmatched_classes,
        "candidate_generation_seconds": float(generation_seconds),
        "total_wall_seconds": float(total_seconds),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s and %s", args.out_jsonl, meta_path)


if __name__ == "__main__":
    main()
