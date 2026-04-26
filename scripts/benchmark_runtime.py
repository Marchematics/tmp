#!/usr/bin/env python3
"""Latency / compute benchmark (E1).

Wraps the candidate-table-based phi fit / e-value computation / e-BH selection
pipeline with timing instrumentation. Reports per-stage wall time across 6
configs (mean over 5 seeds).

Detector inference time is not measured here (already documented in the
checklist as ~280s per RTX 3090); we focus on the wrapper overhead which is
what reviewers care about.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

CONFIGS = [
    ("COCO val", "GDino",      "coco_groundingdino_gonogo_1000_mix_analysis"),
    ("COCO val", "OWL-ViT",    "coco_owlvit_val_1000_mix_analysis"),
    ("COCO val", "YOLO-World", "coco_yoloworld_val_1000_mix_analysis"),
    ("VOC 2012", "GDino",      "voc_1000_gdino_analysis"),
    ("VOC 2012", "OWL-ViT",    "voc_1000_owlvit_analysis"),
    ("VOC 2012", "YOLO-World", "voc_yoloworld_1000_mix_analysis"),
]
N_SEEDS = 5
N_BINS = 80


def split_families(family_ids: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    fams = family_ids.copy()
    rng.shuffle(fams)
    n = len(fams)
    n_fit = int(round(0.6 * n))
    n_cal = int(round(0.2 * n))
    return (set(fams[:n_fit]),
            set(fams[n_fit:n_fit+n_cal]),
            set(fams[n_fit+n_cal:]))


def fit_phi(fit_null: np.ndarray, fit_tp: np.ndarray):
    edges = np.linspace(min(fit_null.min(), fit_tp.min() if len(fit_tp) else 0),
                        max(fit_null.max(), fit_tp.max() if len(fit_tp) else 1) + 1e-9,
                        N_BINS + 1)
    nc, _ = np.histogram(fit_null, bins=edges)
    tc, _ = np.histogram(fit_tp, bins=edges)
    p0 = nc.astype(float) / max(nc.sum(), 1)
    p1 = tc.astype(float) / max(tc.sum(), 1)
    phi = np.where(p0 > 0, p1 / np.maximum(p0, 1e-12), 0.0)
    expected = float((p0 * phi).sum())
    if expected > 0:
        phi /= expected
    return edges, phi


def apply_phi(s: np.ndarray, edges, phi):
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(phi) - 1)
    return phi[idx]


def within_family_ebh(e: np.ndarray, alpha: float):
    K = len(e)
    if K == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-e, kind="stable")
    se = e[order]
    ranks = np.arange(1, K + 1)
    eligible = se >= K / (alpha * ranks)
    r = int(np.where(eligible)[0].max() + 1) if eligible.any() else 0
    mask = np.zeros(K, dtype=bool)
    mask[order[:r]] = True
    return mask


def benchmark_one(df_full: pd.DataFrame, seed: int, alpha: float = 0.10) -> dict:
    fams = df_full["family_id"].drop_duplicates().to_numpy()
    fit_set, cal_set, test_set = split_families(fams, seed)

    fit = df_full[df_full["family_id"].isin(fit_set)]
    cal = df_full[df_full["family_id"].isin(cal_set)]
    test = df_full[df_full["family_id"].isin(test_set)]

    times = {}

    t0 = time.perf_counter()
    edges, phi = fit_phi(
        fit[fit["is_null"]]["score"].to_numpy(),
        fit[fit["is_tp"]]["score"].to_numpy(),
    )
    times["phi_fit_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cal_phi = apply_phi(cal[cal["is_null"]]["score"].to_numpy(), edges, phi)
    sum_phi_cal = float(cal_phi.sum())
    m_cal = len(cal_phi)
    times["calibration_aggregate_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    test_scores = test["score"].to_numpy()
    test_phi = apply_phi(test_scores, edges, phi)
    denom = test_phi + sum_phi_cal
    e = np.zeros_like(test_phi)
    valid = denom > 0
    e[valid] = (m_cal + 1) * test_phi[valid] / denom[valid]
    times["evalue_total_ms"] = (time.perf_counter() - t0) * 1000

    n_test_fam = len(test_set)
    times["evalue_per_family_us"] = times["evalue_total_ms"] * 1000 / max(n_test_fam, 1)

    t0 = time.perf_counter()
    fam_ids = test["family_id"].to_numpy()
    unique_fams, inverse = np.unique(fam_ids, return_inverse=True)
    rejected = np.zeros(len(test), dtype=bool)
    for i in range(len(unique_fams)):
        idx = np.where(inverse == i)[0]
        m = within_family_ebh(e[idx], alpha)
        if m.any():
            rejected[idx[m]] = True
    times["ebh_total_ms"] = (time.perf_counter() - t0) * 1000
    times["ebh_per_family_us"] = times["ebh_total_ms"] * 1000 / max(n_test_fam, 1)

    times["total_wrapper_ms"] = (
        times["phi_fit_ms"]
        + times["calibration_aggregate_ms"]
        + times["evalue_total_ms"]
        + times["ebh_total_ms"]
    )
    times["n_test_families"] = n_test_fam
    times["n_test_candidates"] = len(test)
    times["n_cal_null"] = m_cal
    return times


def main() -> None:
    rows = []
    for dataset, detector, ana in CONFIGS:
        path = OUTPUTS / ana / "candidate_table.csv"
        if not path.exists():
            print(f"[skip] {ana}")
            continue
        df = pd.read_csv(path)
        print(f"\n{dataset} / {detector}: {len(df)} rows, {df['family_id'].nunique()} families")
        seed_rows = []
        for s in range(N_SEEDS):
            t = benchmark_one(df, s)
            seed_rows.append(t)
        # Aggregate
        agg = {k: float(np.mean([r[k] for r in seed_rows]))
               for k in seed_rows[0].keys()}
        rows.append({
            "dataset": dataset,
            "detector": detector,
            **{k: round(v, 3) for k, v in agg.items()},
        })
        print(
            f"  phi_fit={agg['phi_fit_ms']:.2f}ms  "
            f"cal_agg={agg['calibration_aggregate_ms']:.2f}ms  "
            f"evalue={agg['evalue_total_ms']:.2f}ms ({agg['evalue_per_family_us']:.1f}us/fam)  "
            f"ebh={agg['ebh_total_ms']:.2f}ms ({agg['ebh_per_family_us']:.1f}us/fam)"
        )
        print(
            f"  total_wrapper={agg['total_wrapper_ms']:.2f}ms  "
            f"({agg['n_test_families']:.0f} fams, {agg['n_test_candidates']:.0f} cands)"
        )

    out_csv = OUTPUTS / "runtime_benchmark.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
