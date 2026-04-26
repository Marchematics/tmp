from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .matching import nms_xyxy


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    label: str | None = None


def _from_pretrained_cached_first(factory: Any, model_id: str) -> Any:
    try:
        return factory.from_pretrained(model_id, local_files_only=True)
    except Exception:
        return factory.from_pretrained(model_id)


class GroundingDinoHF:
    """Small adapter around Hugging Face Grounding DINO style models."""

    def __init__(self, model_id: str, device: str) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Grounding DINO inference requires torch and transformers. "
                "Install this pilot's requirements.txt first."
            ) from exc
        try:
            from transformers import AutoModelForZeroShotObjectDetection

            model_cls = AutoModelForZeroShotObjectDetection
        except ImportError:
            from transformers import GroundingDinoForObjectDetection

            model_cls = GroundingDinoForObjectDetection

        self.torch = torch
        self.device = device
        self.processor = _from_pretrained_cached_first(AutoProcessor, model_id)
        self.model = _from_pretrained_cached_first(model_cls, model_id).to(device)
        self.model.eval()

    def predict(
        self,
        image_path: Path,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
        nms_iou: float,
        max_det: int,
    ) -> list[Detection]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        text = prompt if prompt.endswith(".") else f"{prompt}."
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([image.size[::-1]], device=self.device)
        result = self._post_process(outputs, inputs, target_sizes, box_threshold, text_threshold)
        detections = _result_to_detections(result)
        detections = [det for det in detections if det.score >= box_threshold]
        if not detections:
            return []

        keep = nms_xyxy(
            [list(det.box) for det in detections],
            [det.score for det in detections],
            iou_threshold=nms_iou,
            max_det=max_det,
        )
        return [detections[idx] for idx in keep]

    def _post_process(
        self,
        outputs: Any,
        inputs: dict[str, Any],
        target_sizes: Any,
        box_threshold: float,
        text_threshold: float,
    ) -> dict[str, Any]:
        processor = self.processor
        input_ids = inputs.get("input_ids")

        if hasattr(processor, "post_process_grounded_object_detection"):
            try:
                return processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]
            except TypeError:
                pass
            try:
                return processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]
            except TypeError:
                return processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]

        if hasattr(processor, "post_process_object_detection"):
            return processor.post_process_object_detection(
                outputs,
                threshold=box_threshold,
                target_sizes=target_sizes,
            )[0]

        raise AttributeError("Processor has no known object-detection post-process method")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class OwlViTHF:
    """Adapter for OWL-ViT v2 (google/owlv2-*) via Hugging Face transformers."""

    def __init__(self, model_id: str, device: str) -> None:
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:
            raise ImportError(
                "OWL-ViT inference requires torch and transformers>=4.35. "
                "Install this pilot's requirements.txt first."
            ) from exc

        self.torch = torch
        self.device = device
        self.processor = _from_pretrained_cached_first(Owlv2Processor, model_id)
        self.model = _from_pretrained_cached_first(Owlv2ForObjectDetection, model_id).to(device)
        self.model.eval()

    def predict(
        self,
        image_path: Path,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
        nms_iou: float,
        max_det: int,
    ) -> list[Detection]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[[prompt]], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([image.size[::-1]], device=self.device)
        # post_process_grounded_object_detection returns list[dict] with keys scores/boxes/labels
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=box_threshold,
                target_sizes=target_sizes,
            )
        except (AttributeError, TypeError):
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=box_threshold,
                target_sizes=target_sizes,
            )
        result = results[0]

        detections = _result_to_detections(result)
        detections = [det for det in detections if det.score >= box_threshold]
        if not detections:
            return []

        keep = nms_xyxy(
            [list(det.box) for det in detections],
            [det.score for det in detections],
            iou_threshold=nms_iou,
            max_det=max_det,
        )
        return [detections[idx] for idx in keep]


class YoloWorldUltralytics:
    """Adapter for YOLO-World via ultralytics."""

    def __init__(self, model_id: str, device: str) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise ImportError(
                "YOLO-World inference requires ultralytics>=8.1. "
                "pip install ultralytics"
            ) from exc

        self.model = YOLOWorld(model_id)
        self.model.to(device)
        self.device = device
        self._last_prompt: str | None = None

    def predict(
        self,
        image_path: Path,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
        nms_iou: float,
        max_det: int,
    ) -> list[Detection]:
        if prompt != self._last_prompt:
            self.model.set_classes([prompt])
            self._last_prompt = prompt

        results = self.model.predict(
            str(image_path),
            conf=box_threshold,
            iou=nms_iou,
            max_det=max_det,
            verbose=False,
        )
        if not results:
            return []

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        detections: list[Detection] = []
        for box, score in zip(boxes, scores):
            detections.append(
                Detection(
                    box=tuple(float(v) for v in box),
                    score=float(score),
                    label=prompt,
                )
            )
        return detections


def _result_to_detections(result: dict[str, Any]) -> list[Detection]:
    boxes = _as_list(result.get("boxes"))
    scores = _as_list(result.get("scores"))
    label_payload = result.get("labels")
    if label_payload is None:
        label_payload = result.get("text_labels")
    labels = _as_list(label_payload)
    if not labels:
        labels = [None] * len(boxes)

    detections: list[Detection] = []
    for box, score, label in zip(boxes, scores, labels):
        if len(box) != 4:
            continue
        detections.append(
            Detection(
                box=tuple(float(v) for v in box),
                score=float(score),
                label=None if label is None else str(label),
            )
        )
    return detections
