from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def conformal_binary_evalue(score: float, null_scores: np.ndarray, alpha: float) -> float:
    """Binary conformal e-value from an exchangeable null-score pool.

    The threshold follows the theorem sketch:

        q = ceil((1 - alpha) * (m + 1)).

    If q=m+1, the finite calibration pool cannot realize that order statistic, so
    the test is conservative and returns zero.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.asarray(null_scores, dtype=np.float64)
    m = int(scores.size)
    if m == 0:
        raise ValueError("At least one calibration null score is required")

    q = int(math.ceil((1.0 - alpha) * (m + 1)))
    denom = int(math.ceil(alpha * (m + 1)))
    if denom <= 0:
        raise ValueError("Invalid alpha/calibration combination")
    if q > m:
        return 0.0

    threshold = float(np.sort(scores)[q - 1])
    scale = float((m + 1) / denom)
    return scale if float(score) >= threshold else 0.0


def conformal_binary_evalues(scores: Iterable[float], null_scores: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray([conformal_binary_evalue(score, null_scores, alpha) for score in scores], dtype=np.float64)


def conformal_pvalues(scores: Iterable[float], null_scores: np.ndarray) -> np.ndarray:
    """Conformal p-values from an exchangeable null-score pool."""
    scores = np.asarray(list(scores), dtype=np.float64)
    null_scores = np.asarray(null_scores, dtype=np.float64)
    m = int(null_scores.size)
    if m == 0:
        raise ValueError("At least one calibration null score is required")
    null_sorted = np.sort(null_scores)
    # For each score s, p = (1 + #{null >= s}) / (m + 1)
    counts = m - np.searchsorted(null_sorted, scores, side="left")
    return (1.0 + counts) / (m + 1)


def bh(pvalues: Iterable[float], alpha: float) -> list[int]:
    """Run Benjamini-Hochberg at level alpha and return rejected indices."""
    p = np.asarray(list(pvalues), dtype=np.float64)
    m = int(p.size)
    if m == 0:
        return []
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = alpha * np.arange(1, m + 1) / m
    below = np.where(sorted_p <= thresholds)[0]
    if below.size == 0:
        return []
    cutoff = float(sorted_p[below[-1]])
    return [int(idx) for idx, value in enumerate(p) if float(value) <= cutoff]


def cluster_evalues_convex(
    candidate_evalues: np.ndarray,
    clusters: list[list[int]],
    weights_per_cluster: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Aggregate candidate-level e-values into cluster-level e-values.

    Implements Theorem 1 (predictable convex form) of
    `docs/theory_notes/nms_aware_evalue.md`:

        E(C) = sum_{i in C} w_i * e_i,  sum_i w_i = 1, w_i >= 0, predictable.

    Default weights are uniform (1/|C|), which is strictly predictable
    (depends only on cluster cardinality, not on e_i values) and is
    therefore e-value valid under the null without any adaptivity caveat.

    Parameters
    ----------
    candidate_evalues : np.ndarray, shape (n,)
        Base candidate e-values (e.g., self-normalized betting e-values).
    clusters : list[list[int]]
        Partition of range(n) — each inner list is candidate indices forming
        one NMS-connected cluster.
    weights_per_cluster : list[np.ndarray] | None
        Optional per-cluster weight vectors (each summing to 1). If None,
        uniform weights are used.

    Returns
    -------
    np.ndarray, shape (len(clusters),) — one convex e-value per cluster.
    """
    e = np.asarray(candidate_evalues, dtype=np.float64)
    out = np.zeros(len(clusters), dtype=np.float64)
    for c_idx, members in enumerate(clusters):
        if not members:
            continue
        if weights_per_cluster is None:
            w = np.full(len(members), 1.0 / len(members), dtype=np.float64)
        else:
            w = np.asarray(weights_per_cluster[c_idx], dtype=np.float64)
            if w.size != len(members):
                raise ValueError(
                    f"cluster {c_idx}: weights size {w.size} != members {len(members)}"
                )
            if w.min() < 0:
                raise ValueError(f"cluster {c_idx}: negative weight")
            s = float(w.sum())
            if s <= 0:
                raise ValueError(f"cluster {c_idx}: weights sum to 0")
            w = w / s
        out[c_idx] = float(np.dot(w, e[np.asarray(members, dtype=int)]))
    return out


def nms_aware_e_bh(
    candidate_evalues: np.ndarray,
    clusters: list[list[int]],
    alpha: float,
    weights_per_cluster: list[np.ndarray] | None = None,
    rank_within_cluster: np.ndarray | None = None,
) -> tuple[list[int], list[int]]:
    """Cluster-level e-BH — returns (rejected_clusters, rejected_candidates).

    Steps:
      1. Aggregate candidate e-values into M cluster e-values (convex form).
      2. Run e-BH at level alpha on the M cluster e-values.
      3. Map each rejected cluster back to ONE representative candidate
         (the one with the highest score / e-value / first index inside the
         cluster, configurable via ``rank_within_cluster``). This
         respects the theorem: each cluster contributes at most one rejection
         to the downstream pipeline, regardless of cluster size.

    Parameters
    ----------
    candidate_evalues : np.ndarray
        Base e-values.
    clusters : list[list[int]]
        NMS partition over range(len(candidate_evalues)).
    alpha : float
        FDR level.
    weights_per_cluster : list[np.ndarray] | None
        Optional predictable weights (must not depend on the cluster's own
        e-values if strict Theorem-1 validity is required).
    rank_within_cluster : np.ndarray | None, shape (n,)
        Optional per-candidate tiebreaker score used to pick the cluster
        representative (higher = preferred). Defaults to candidate_evalues.

    Returns
    -------
    (rejected_cluster_indices, rejected_candidate_indices)
    """
    cluster_e = cluster_evalues_convex(
        candidate_evalues, clusters, weights_per_cluster
    )
    rej_clusters = e_bh(cluster_e, alpha)
    tiebreak = (
        np.asarray(rank_within_cluster, dtype=np.float64)
        if rank_within_cluster is not None
        else np.asarray(candidate_evalues, dtype=np.float64)
    )
    rejected_candidates: list[int] = []
    for c_idx in rej_clusters:
        members = clusters[c_idx]
        if not members:
            continue
        # pick the candidate with highest tiebreak; stable on ties
        best = members[0]
        best_val = tiebreak[best]
        for m in members[1:]:
            if tiebreak[m] > best_val:
                best = m
                best_val = tiebreak[m]
        rejected_candidates.append(int(best))
    return rej_clusters, rejected_candidates


def e_bh(evalues: Iterable[float], alpha: float) -> list[int]:
    """Run e-BH at level alpha and return rejected indices.

    Wang-Ramdas e-BH sorts e-values descending and finds the largest k such that
    e_(k) >= K / (alpha * k), equivalently k * e_(k) / K >= 1 / alpha.
    """
    e = np.asarray(list(evalues), dtype=np.float64)
    k_total = int(e.size)
    if k_total == 0:
        return []
    order = np.argsort(e)[::-1]
    sorted_e = e[order]

    selected_k = 0
    for rank, value in enumerate(sorted_e, start=1):
        if (rank * float(value) / k_total) >= (1.0 / alpha):
            selected_k = rank

    if selected_k == 0:
        return []
    cutoff = float(sorted_e[selected_k - 1])
    return [int(idx) for idx, value in enumerate(e) if float(value) >= cutoff]
