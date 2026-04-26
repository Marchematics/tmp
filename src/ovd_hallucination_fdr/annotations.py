from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    frequency: str | None = None


@dataclass(frozen=True)
class BoxAnn:
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    iscrowd: bool = False


@dataclass(frozen=True)
class ImageRecord:
    id: int
    file_name: str
    width: int
    height: int
    image_path: Path
    annotations: tuple[BoxAnn, ...]


@dataclass(frozen=True)
class QueryFamily:
    dataset: str
    image_id: int
    file_name: str
    width: int
    height: int
    image_path: Path
    category_id: int
    category_name: str
    prompt: str
    is_prompt_absent: bool
    gt_boxes: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class AnnotationIndex:
    dataset: str
    categories: tuple[Category, ...]
    images: tuple[ImageRecord, ...]


FREEFORM_PRESENT_TEMPLATES = (
    "a visible {category} in the scene",
    "a photo of a {category}",
    "a {category} object",
    "the {category} in this image",
)
FREEFORM_ABSENT_TEMPLATES = (
    "a photo containing a {category}",
    "a small {category} near the center",
    "a {category} in the scene",
    "a close-up of a {category}",
)


def coco_xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box]
    return x, y, x + max(w, 0.0), y + max(h, 0.0)


def image_file_name(row: dict[str, Any]) -> str:
    if row.get("file_name"):
        return str(row["file_name"])
    coco_url = str(row.get("coco_url", ""))
    for split in ("train2017", "val2017", "test2017"):
        marker = f"/{split}/"
        if marker in coco_url:
            return f"{split}/{coco_url.rsplit(marker, 1)[1]}"
    return Path(coco_url).name


def load_annotation_index(
    *,
    ann_json: Path,
    img_dir: Path,
    dataset: str,
    limit_images: int | None = None,
) -> AnnotationIndex:
    payload = json.loads(ann_json.read_text(encoding="utf-8"))
    categories = tuple(
        Category(
            id=int(row["id"]),
            name=str(row["name"]),
            frequency=str(row["frequency"]) if "frequency" in row else None,
        )
        for row in sorted(payload["categories"], key=lambda item: int(item["id"]))
    )

    image_rows = sorted(payload["images"], key=lambda item: int(item["id"]))
    if limit_images is not None:
        image_rows = image_rows[: int(limit_images)]
    allowed_image_ids = {int(row["id"]) for row in image_rows}

    anns_by_image: dict[int, list[BoxAnn]] = {image_id: [] for image_id in allowed_image_ids}
    for ann in payload["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in allowed_image_ids:
            continue
        if "bbox" not in ann:
            continue
        anns_by_image.setdefault(image_id, []).append(
            BoxAnn(
                category_id=int(ann["category_id"]),
                bbox_xyxy=coco_xywh_to_xyxy(ann["bbox"]),
                iscrowd=bool(ann.get("iscrowd", 0)),
            )
        )

    images: list[ImageRecord] = []
    for row in image_rows:
        image_id = int(row["id"])
        file_name = image_file_name(row)
        image_path = img_dir / file_name
        images.append(
            ImageRecord(
                id=image_id,
                file_name=file_name,
                width=int(row.get("width", 0)),
                height=int(row.get("height", 0)),
                image_path=image_path,
                annotations=tuple(anns_by_image.get(image_id, [])),
            )
        )

    return AnnotationIndex(dataset=dataset, categories=categories, images=tuple(images))


def _category_pool(categories: tuple[Category, ...], prompt_pool: str) -> list[Category]:
    if prompt_pool == "all":
        return list(categories)
    if prompt_pool == "rare":
        rare = [cat for cat in categories if cat.frequency == "r"]
        if rare:
            return rare
        return list(categories)
    raise ValueError(f"Unknown prompt pool: {prompt_pool!r}")


def sample_query_families(
    index: AnnotationIndex,
    *,
    prompt_template: str,
    prompt_style: str = "category",
    prompt_pool: str,
    present_prompt_pool: str | None = None,
    absent_prompt_pool: str | None = None,
    absent_prompts_per_image: int,
    present_prompts_per_image: int,
    seed: int,
) -> list[QueryFamily]:
    rng = random.Random(seed)
    category_by_id = {cat.id: cat for cat in index.categories}
    present_pool = _category_pool(index.categories, present_prompt_pool or prompt_pool)
    absent_pool = _category_pool(index.categories, absent_prompt_pool or prompt_pool)
    present_pool_ids = {cat.id for cat in present_pool}

    families: list[QueryFamily] = []
    for image in index.images:
        gt_by_category: dict[int, list[tuple[float, float, float, float]]] = {}
        for ann in image.annotations:
            if ann.iscrowd:
                continue
            gt_by_category.setdefault(ann.category_id, []).append(ann.bbox_xyxy)

        present_ids = sorted(cat_id for cat_id in gt_by_category if cat_id in present_pool_ids)
        absent_ids = sorted(cat.id for cat in absent_pool if cat.id not in gt_by_category)
        rng.shuffle(present_ids)
        rng.shuffle(absent_ids)

        selected: list[tuple[int, bool]] = []
        selected.extend((cat_id, False) for cat_id in present_ids[:present_prompts_per_image])
        selected.extend((cat_id, True) for cat_id in absent_ids[:absent_prompts_per_image])

        for category_id, is_absent in selected:
            category = category_by_id[category_id]
            category_name = category.name
            if prompt_style == "category":
                prompt = prompt_template.format(category=category_name)
            elif prompt_style == "freeform":
                templates = FREEFORM_ABSENT_TEMPLATES if is_absent else FREEFORM_PRESENT_TEMPLATES
                prompt = rng.choice(templates).format(category=category_name)
            else:
                raise ValueError(f"Unknown prompt style: {prompt_style!r}")
            families.append(
                QueryFamily(
                    dataset=index.dataset,
                    image_id=image.id,
                    file_name=image.file_name,
                    width=image.width,
                    height=image.height,
                    image_path=image.image_path,
                    category_id=category_id,
                    category_name=category_name,
                    prompt=prompt,
                    is_prompt_absent=is_absent,
                    gt_boxes=tuple(gt_by_category.get(category_id, ())),
                )
            )

    return families


def family_to_record_base(family: QueryFamily) -> dict[str, Any]:
    return {
        "dataset": family.dataset,
        "image_id": family.image_id,
        "file_name": family.file_name,
        "width": family.width,
        "height": family.height,
        "image_path": str(family.image_path),
        "category_id": family.category_id,
        "category_name": family.category_name,
        "prompt": family.prompt,
        "is_prompt_absent": family.is_prompt_absent,
        "gt_count": len(family.gt_boxes),
        "gt_boxes": [list(box) for box in family.gt_boxes],
    }
