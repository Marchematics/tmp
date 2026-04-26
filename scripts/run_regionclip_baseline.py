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

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
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
DEFAULT_REGIONCLIP_ROOT = REPO_ROOT / "external" / "regionclip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official RegionCLIP COCO open-vocabulary detector and emit compatible jsonl candidates."
    )
    parser.add_argument("--dataset", choices=["coco"], default="coco")
    parser.add_argument("--ann-json", type=Path, default=DEFAULT_COCO_ANN)
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_COCO_IMG_DIR)
    parser.add_argument("--out-jsonl", type=Path, default=PROJECT_ROOT / "outputs" / "coco_regionclip_1000_mix.jsonl")
    parser.add_argument("--regionclip-root", type=Path, default=DEFAULT_REGIONCLIP_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--rpn-weights", type=Path, default=None)
    parser.add_argument("--base-text-emb", type=Path, default=None)
    parser.add_argument("--openset-text-emb", type=Path, default=None)
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
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def default_regionclip_paths(args: argparse.Namespace) -> None:
    root = args.regionclip_root
    if args.config is None:
        args.config = root / "configs" / "COCO-InstanceSegmentation" / "CLIP_fast_rcnn_R_50_C4_ovd.yaml"
    if args.weights is None:
        args.weights = root / "pretrained_ckpt" / "regionclip" / "regionclip_finetuned-coco_rn50.pth"
    if args.rpn_weights is None:
        args.rpn_weights = root / "pretrained_ckpt" / "rpn" / "rpn_coco_48.pth"
    if args.base_text_emb is None:
        args.base_text_emb = root / "pretrained_ckpt" / "concept_emb" / "coco_48_base_cls_emb.pth"
    if args.openset_text_emb is None:
        args.openset_text_emb = root / "pretrained_ckpt" / "concept_emb" / "coco_65_cls_emb.pth"


def validate_args(args: argparse.Namespace) -> None:
    default_regionclip_paths(args)
    required = {
        "annotation JSON": args.ann_json,
        "image directory": args.img_dir,
        "RegionCLIP root": args.regionclip_root,
        "RegionCLIP config": args.config,
        "RegionCLIP weights": args.weights,
        "RegionCLIP RPN weights": args.rpn_weights,
        "RegionCLIP base text embeddings": args.base_text_emb,
        "RegionCLIP openset text embeddings": args.openset_text_emb,
    }
    for label, path in required.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
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


class RegionClipPredictor:
    def __init__(self, args: argparse.Namespace) -> None:
        sys.path.insert(0, str(args.regionclip_root.resolve()))
        import torch
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import get_cfg
        from detectron2.data import detection_utils as d2_utils
        from detectron2.data.transforms import ResizeShortestEdge
        from detectron2.modeling import build_model

        self.torch = torch
        self.d2_utils = d2_utils

        cfg = get_cfg()
        cfg.merge_from_file(str(args.config))
        cfg.merge_from_list(
            [
                "MODEL.WEIGHTS",
                str(args.weights),
                "MODEL.CLIP.OFFLINE_RPN_CONFIG",
                str(args.regionclip_root / "configs" / "COCO-InstanceSegmentation" / "mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"),
                "MODEL.CLIP.BB_RPN_WEIGHTS",
                str(args.rpn_weights),
                "MODEL.CLIP.TEXT_EMB_PATH",
                str(args.base_text_emb),
                "MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH",
                str(args.openset_text_emb),
                "MODEL.ROI_HEADS.SOFT_NMS_ENABLED",
                "True",
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                str(args.score_threshold),
                "TEST.DETECTIONS_PER_IMAGE",
                str(args.max_det),
                "MODEL.DEVICE",
                args.device,
            ]
        )
        cfg.freeze()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        if (
            cfg.MODEL.META_ARCHITECTURE in ["CLIPRCNN", "CLIPFastRCNN", "PretrainFastRCNN"]
            and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None
            and cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN"
        ):
            DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
                cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
            )
        self.aug = ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        if self.input_format not in ["RGB", "BGR"]:
            raise ValueError(f"Unsupported RegionCLIP input format: {self.input_format}")

    def __call__(self, image_path: Path) -> dict[str, Any]:
        with self.torch.no_grad():
            original_image = self.d2_utils.read_image(str(image_path), format="BGR")
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image_tensor = self.torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            return self.model([{"image": image_tensor, "height": height, "width": width}])[0]


def detections_from_instances(outputs: dict[str, Any], class_names: list[str], max_det: int) -> list[dict[str, Any]]:
    instances = outputs.get("instances")
    if instances is None:
        return []
    instances = instances.to("cpu")
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []
    labels = instances.pred_classes.numpy() if instances.has("pred_classes") else []
    detections: list[dict[str, Any]] = []
    for box, score, label_idx in zip(boxes, scores, labels):
        label_idx = int(label_idx)
        label = class_names[label_idx] if 0 <= label_idx < len(class_names) else str(label_idx)
        detections.append(
            {
                "box": [float(v) for v in box.tolist()],
                "score": float(score),
                "label_index": label_idx,
                "label": label,
            }
        )
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return detections[:max_det]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    validate_args(args)

    sys.path.insert(0, str(args.regionclip_root.resolve()))
    from detectron2.data.datasets.coco_zeroshot_categories import COCO_OVD_ALL_CLS

    class_names = [str(v) for v in COCO_OVD_ALL_CLS]
    class_lookup = {normalize_label(name): idx for idx, name in enumerate(class_names)}

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

    predictor = RegionClipPredictor(args)
    ensure_dir(args.out_jsonl.parent)

    families_by_image: dict[int, list[Any]] = defaultdict(list)
    for family in families:
        families_by_image[family.image_id].append(family)

    num_written = 0
    num_candidates = 0
    num_missing = 0
    num_unsupported_classes = 0
    generation_start = time.perf_counter()
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for image in tqdm(index.images, desc="RegionCLIP images"):
            image_families = families_by_image.get(image.id, [])
            if not image_families:
                continue
            if not image.image_path.exists():
                num_missing += 1
                if args.skip_missing_images:
                    continue
                raise FileNotFoundError(f"Missing image: {image.image_path}")

            image_detections = detections_from_instances(predictor(image.image_path), class_names, args.max_det)
            for family in image_families:
                target_index = class_lookup.get(normalize_label(family.category_name))
                if target_index is None:
                    num_unsupported_classes += 1
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
                            "label": str(det["label"]),
                            "label_index": int(det["label_index"]),
                            "max_iou": float(max_iou),
                            "is_tp": bool(float(max_iou) >= args.match_iou),
                        }
                    )

                record = family_to_record_base(family)
                record.update(
                    {
                        "model_id": "microsoft/RegionCLIP:regionclip_finetuned-coco_rn50",
                        "detector": "regionclip",
                        "config": str(args.config),
                        "weights": str(args.weights),
                        "rpn_weights": str(args.rpn_weights),
                        "base_text_emb": str(args.base_text_emb),
                        "openset_text_emb": str(args.openset_text_emb),
                        "score_threshold": args.score_threshold,
                        "vanilla_threshold": args.vanilla_threshold,
                        "max_det": args.max_det,
                        "match_iou": args.match_iou,
                        "supported_by_regionclip_coco65": target_index is not None,
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
        "detector": "regionclip",
        "model_id": "microsoft/RegionCLIP:regionclip_finetuned-coco_rn50",
        "regionclip_root": str(args.regionclip_root),
        "config": str(args.config),
        "weights": str(args.weights),
        "rpn_weights": str(args.rpn_weights),
        "base_text_emb": str(args.base_text_emb),
        "openset_text_emb": str(args.openset_text_emb),
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
        "class_names": class_names,
        "num_families_written": num_written,
        "num_candidates": num_candidates,
        "mean_candidates_per_family": float(num_candidates / max(num_written, 1)),
        "num_missing_images": num_missing,
        "num_unsupported_classes": num_unsupported_classes,
        "candidate_generation_seconds": float(generation_seconds),
        "total_wall_seconds": float(total_seconds),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s and %s", args.out_jsonl, meta_path)


if __name__ == "__main__":
    main()
