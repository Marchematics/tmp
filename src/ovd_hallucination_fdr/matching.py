from __future__ import annotations

import numpy as np


def box_iou_xyxy(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a.astype(np.float32, copy=False)
    b = boxes_b.astype(np.float32, copy=False)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def max_iou_to_gt(
    candidate_boxes: list[list[float]] | np.ndarray,
    gt_boxes: list[list[float]] | tuple[tuple[float, float, float, float], ...],
) -> np.ndarray:
    cand = np.asarray(candidate_boxes, dtype=np.float32).reshape((-1, 4))
    gt = np.asarray(gt_boxes, dtype=np.float32).reshape((-1, 4))
    if len(cand) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(gt) == 0:
        return np.zeros((len(cand),), dtype=np.float32)
    return box_iou_xyxy(cand, gt).max(axis=1)


def nms_clusters_xyxy(
    boxes: list[list[float]] | np.ndarray,
    iou_threshold: float = 0.7,
) -> list[list[int]]:
    """Partition candidate boxes into NMS-connected-components.

    Two boxes share an edge iff their IoU >= iou_threshold. Connected components
    of this graph form the NMS clusters used by the cluster-level e-value.
    The routine is O(n^2) in the number of boxes but n is typically < 200 per
    (image, prompt) family so the cost is negligible relative to detection.

    Returns a list of clusters; each cluster is a list of original indices.
    Singleton boxes (no IoU edges) return as length-1 clusters.
    """
    boxes_arr = np.asarray(boxes, dtype=np.float32).reshape((-1, 4))
    n = boxes_arr.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    iou = box_iou_xyxy(boxes_arr, boxes_arr)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if iou[i, j] >= iou_threshold:
                union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        buckets.setdefault(root, []).append(i)
    return list(buckets.values())


def nms_xyxy(
    boxes: list[list[float]] | np.ndarray,
    scores: list[float] | np.ndarray,
    iou_threshold: float,
    max_det: int | None = None,
) -> list[int]:
    boxes_arr = np.asarray(boxes, dtype=np.float32).reshape((-1, 4))
    scores_arr = np.asarray(scores, dtype=np.float32)
    if len(boxes_arr) == 0:
        return []

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]
    areas = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    order = scores_arr.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if max_det is not None and len(keep) >= max_det:
            break
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.clip(xx2 - xx1, 0.0, None)
        h = np.clip(yy2 - yy1, 0.0, None)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]

    return keep
