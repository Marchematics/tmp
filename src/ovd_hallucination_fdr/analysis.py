from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import numpy as np

from .evalues import conformal_binary_evalue, e_bh


def stable_unit_interval_key(*parts: object) -> float:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def flatten_candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_idx, family in enumerate(records):
        family_id = f"{family.get('dataset')}:{family.get('image_id')}:{family.get('category_id')}:{family_idx}"
        candidates = family.get("candidates", [])
        for cand_idx, cand in enumerate(candidates):
            box = cand.get("box", [math.nan, math.nan, math.nan, math.nan])
            rows.append(
                {
                    "family_id": family_id,
                    "dataset": family.get("dataset"),
                    "image_id": family.get("image_id"),
                    "file_name": family.get("file_name"),
                    "category_id": family.get("category_id"),
                    "category_name": family.get("category_name"),
                    "prompt": family.get("prompt"),
                    "is_prompt_absent": bool(family.get("is_prompt_absent")),
                    "gt_count": int(family.get("gt_count", 0)),
                    "candidate_idx": cand_idx,
                    "score": float(cand.get("score", 0.0)),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "max_iou": float(cand.get("max_iou", 0.0)),
                    "is_tp": bool(cand.get("is_tp", False)),
                    "is_null": not bool(cand.get("is_tp", False)),
                }
            )
    return rows


def fdp_recall_curve(rows: list[dict[str, Any]], *, grid_size: int = 100) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    total_tp = sum(1 for row in ordered if row["is_tp"])
    curve: list[dict[str, Any]] = []
    checkpoints = sorted(set(max(1, int(round(len(ordered) * frac / grid_size))) for frac in range(1, grid_size + 1)))
    for n_selected in checkpoints:
        selected = ordered[:n_selected]
        fp = sum(1 for row in selected if row["is_null"])
        tp = n_selected - fp
        curve.append(
            {
                "selected": n_selected,
                "selected_fraction": n_selected / len(ordered),
                "threshold": float(selected[-1]["score"]),
                "fp": fp,
                "tp": tp,
                "fdp": fp / max(n_selected, 1),
                "recall": tp / total_tp if total_tp > 0 else 0.0,
            }
        )
    return curve


def top_fraction_fdp(rows: list[dict[str, Any]], fraction: float) -> float | None:
    if not rows:
        return None
    n_selected = max(1, int(math.ceil(len(rows) * fraction)))
    selected = sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:n_selected]
    return sum(1 for row in selected if row["is_null"]) / n_selected


def build_evalue_tables(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    cal_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cal_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for row in rows:
        split_key = stable_unit_interval_key(row["family_id"])
        if split_key < cal_fraction:
            cal_rows.append(row)
        else:
            test_rows.append(row)

    null_scores = np.asarray([row["score"] for row in cal_rows if row["is_null"]], dtype=np.float64)
    if null_scores.size == 0:
        raise ValueError("No calibration null scores found; increase data or cal_fraction.")

    evalue_rows: list[dict[str, Any]] = []
    for row in test_rows:
        evalue = conformal_binary_evalue(float(row["score"]), null_scores, alpha)
        out = dict(row)
        out["evalue"] = float(evalue)
        out["split"] = "test"
        evalue_rows.append(out)

    family_rows: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evalue_rows:
        by_family[str(row["family_id"])].append(row)

    for family_id, family_candidates in sorted(by_family.items()):
        rejected_idx = set(e_bh([row["evalue"] for row in family_candidates], alpha))
        selected = [row for idx, row in enumerate(family_candidates) if idx in rejected_idx]
        fp = sum(1 for row in selected if row["is_null"])
        tp = len(selected) - fp
        family_rows.append(
            {
                "family_id": family_id,
                "dataset": family_candidates[0]["dataset"],
                "image_id": family_candidates[0]["image_id"],
                "category_id": family_candidates[0]["category_id"],
                "category_name": family_candidates[0]["category_name"],
                "prompt": family_candidates[0]["prompt"],
                "is_prompt_absent": family_candidates[0]["is_prompt_absent"],
                "num_candidates": len(family_candidates),
                "num_rejected": len(selected),
                "fp": fp,
                "tp": tp,
                "fdp": fp / max(len(selected), 1),
            }
        )

    null_test_evalues = [row["evalue"] for row in evalue_rows if row["is_null"]]
    summary = {
        "alpha": alpha,
        "cal_fraction": cal_fraction,
        "num_cal_candidates": len(cal_rows),
        "num_test_candidates": len(test_rows),
        "num_cal_null_scores": int(null_scores.size),
        "cal_null_score_threshold_min": float(np.min(null_scores)),
        "cal_null_score_threshold_max": float(np.max(null_scores)),
        "test_null_evalue_mean": float(np.mean(null_test_evalues)) if null_test_evalues else None,
        "e_bh_num_families": len(family_rows),
        "e_bh_total_rejections": int(sum(row["num_rejected"] for row in family_rows)),
        "e_bh_total_fp": int(sum(row["fp"] for row in family_rows)),
        "e_bh_mean_family_fdp": float(np.mean([row["fdp"] for row in family_rows])) if family_rows else None,
    }
    return evalue_rows, family_rows, summary
