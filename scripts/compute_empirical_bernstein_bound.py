#!/usr/bin/env python3
"""Compute Empirical Bernstein bound for pooled FDP and compare with Hoeffding.

For each config and seed, loads per-candidate e-values, applies e-BH per family,
computes X_g = V_g - α·R_g, and uses the empirical variance to get a tighter bound.
"""
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Configs to analyze
CONFIGS = [
    ("COCO full", "GDino", "coco_gdino_fullscale_20seed_formal_hist"),
    ("COCO val", "GDino", "coco_gdino_1000_20seed_formal_hist"),
    ("VOC 2012", "GDino", "voc_gdino_1000_20seed_formal_hist"),
    ("COCO val", "OWL-ViT", "coco_owlvit_1000_20seed_formal_hist"),
    ("VOC 2012", "OWL-ViT", "voc_owlvit_1000_20seed_formal_hist"),
    ("COCO val", "YOLO-World", "coco_yoloworld_val_1000_20seed_formal_hist"),
    ("VOC 2012", "YOLO-World", "voc_yoloworld_1000_20seed_formal_hist"),
]

ALPHAS = [0.05, 0.10, 0.15, 0.20]
DELTA = 0.05
N_SEEDS = 20


def e_bh(evalues, alpha):
    """Apply e-BH procedure. Returns indices of rejected candidates."""
    K = len(evalues)
    if K == 0:
        return []
    indexed = sorted(enumerate(evalues), key=lambda x: -x[1])
    rejected = []
    for rank_minus1, (idx, ev) in enumerate(indexed):
        rank = rank_minus1 + 1
        threshold = K / (alpha * rank)
        if ev >= threshold:
            rejected.append(idx)
        else:
            break
    return rejected


def compute_per_family_stats(candidates_path, alpha):
    """Load candidates, group by family, apply e-BH, return per-family (V_g, R_g, K_g)."""
    families = defaultdict(list)
    with open(candidates_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "test":
                continue
            fam_id = row["family_id"]
            families[fam_id].append({
                "evalue": float(row["betting_evalue"]),
                "is_tp": row["is_tp"] == "True",
            })

    results = []
    for fam_id, members in families.items():
        K_g = len(members)
        evalues = [m["evalue"] for m in members]
        is_tp = [m["is_tp"] for m in members]

        rejected_indices = e_bh(evalues, alpha)
        R_g = len(rejected_indices)
        V_g = sum(1 for i in rejected_indices if not is_tp[i])

        results.append({"K_g": K_g, "R_g": R_g, "V_g": V_g})

    return results


def hoeffding_bound(N_fam, K_max, R_bar, delta):
    """Original Hoeffding bound from the paper."""
    if R_bar <= 0:
        return float("inf")
    eps = K_max * math.sqrt(math.log(1 / delta) / (2 * N_fam)) / R_bar
    return eps


def empirical_bernstein_bound(X_values, b, R_bar, delta):
    """Empirical Bernstein bound.

    Given X_g = V_g - α·R_g with |X_g| ≤ b, and R_bar = mean rejections,
    compute ε such that P(FDP_pool ≤ α + ε) ≥ 1 - δ.
    """
    N = len(X_values)
    if N == 0 or R_bar <= 0:
        return float("inf")

    X_arr = np.array(X_values, dtype=float)
    sigma2 = np.var(X_arr, ddof=1)  # sample variance

    L = math.log(1 / delta)

    # Solve N·t² - (2Lb/3)·t - 2L·σ² = 0
    a_coef = N
    b_coef = -(2 * L * b / 3)
    c_coef = -(2 * L * sigma2)

    discriminant = b_coef**2 - 4 * a_coef * c_coef
    if discriminant < 0:
        return float("inf")

    t = (-b_coef + math.sqrt(discriminant)) / (2 * a_coef)
    if t < 0:
        t = 0

    # ε = t / R_bar (convert from sum-level to FDP-level)
    eps = t / R_bar
    return eps


def main():
    print(f"{'Config':<25} {'α':>5} {'ε_hoef':>10} {'ε_bern':>10} {'ratio':>8} "
          f"{'σ²(X)':>10} {'K_max':>6} {'R_bar':>6} {'N_fam':>6}")
    print("-" * 100)

    for dataset, detector, dirname in CONFIGS:
        output_dir = PROJECT_ROOT / "outputs" / dirname
        if not output_dir.exists():
            print(f"  SKIP {dirname} (not found)")
            continue

        for alpha in ALPHAS:
            # Collect stats across seeds
            eps_hoef_list = []
            eps_bern_list = []
            sigma2_list = []

            for seed in range(N_SEEDS):
                candidates_path = output_dir / f"seed_{seed}" / "test_candidates_with_evalues.csv"
                if not candidates_path.exists():
                    continue

                family_stats = compute_per_family_stats(candidates_path, alpha)
                N_fam = len(family_stats)
                K_max = max(s["K_g"] for s in family_stats)
                total_R = sum(s["R_g"] for s in family_stats)
                R_bar = total_R / N_fam if N_fam > 0 else 0

                # X_g = V_g - α·R_g
                X_values = [s["V_g"] - alpha * s["R_g"] for s in family_stats]
                b = 2 * K_max  # range bound on X_g

                eps_h = hoeffding_bound(N_fam, K_max, R_bar, DELTA)
                eps_b = empirical_bernstein_bound(X_values, b, R_bar, DELTA)

                eps_hoef_list.append(eps_h)
                eps_bern_list.append(eps_b)
                sigma2_list.append(np.var(X_values, ddof=1))

            if not eps_hoef_list:
                continue

            mean_hoef = np.mean(eps_hoef_list)
            mean_bern = np.mean(eps_bern_list)
            mean_sigma2 = np.mean(sigma2_list)
            ratio = mean_hoef / mean_bern if mean_bern > 0 else float("inf")

            # Get representative stats from last seed
            R_bar_repr = R_bar
            N_fam_repr = N_fam
            K_max_repr = K_max

            label = f"{dataset}/{detector}"
            print(f"{label:<25} {alpha:>5.2f} {mean_hoef:>10.3f} {mean_bern:>10.3f} "
                  f"{ratio:>8.2f}x {mean_sigma2:>10.4f} {K_max_repr:>6} "
                  f"{R_bar_repr:>6.3f} {N_fam_repr:>6}")

    print()
    print("=" * 100)
    print("STRATIFIED ANALYSIS: K ≤ 5 families only")
    print("=" * 100)
    print(f"{'Config':<25} {'α':>5} {'ε_hoef':>10} {'ε_bern':>10} {'ratio':>8} "
          f"{'σ²(X)':>10} {'K_max':>6} {'R_bar':>6} {'N_fam':>6}")
    print("-" * 100)

    for dataset, detector, dirname in CONFIGS:
        output_dir = PROJECT_ROOT / "outputs" / dirname
        if not output_dir.exists():
            continue

        for alpha in [0.10]:
            eps_hoef_list = []
            eps_bern_list = []

            for seed in range(N_SEEDS):
                candidates_path = output_dir / f"seed_{seed}" / "test_candidates_with_evalues.csv"
                if not candidates_path.exists():
                    continue

                family_stats = compute_per_family_stats(candidates_path, alpha)
                # Filter to small families
                small_stats = [s for s in family_stats if s["K_g"] <= 5]
                if not small_stats:
                    continue

                N_fam = len(small_stats)
                K_max = max(s["K_g"] for s in small_stats)
                total_R = sum(s["R_g"] for s in small_stats)
                R_bar = total_R / N_fam if N_fam > 0 else 0

                X_values = [s["V_g"] - alpha * s["R_g"] for s in small_stats]
                b = 2 * K_max

                eps_h = hoeffding_bound(N_fam, K_max, R_bar, DELTA)
                eps_b = empirical_bernstein_bound(X_values, b, R_bar, DELTA)

                eps_hoef_list.append(eps_h)
                eps_bern_list.append(eps_b)

            if not eps_hoef_list:
                continue

            mean_hoef = np.mean(eps_hoef_list)
            mean_bern = np.mean(eps_bern_list)
            ratio = mean_hoef / mean_bern if mean_bern > 0 else float("inf")

            label = f"{dataset}/{detector}"
            print(f"{label:<25} {alpha:>5.2f} {mean_hoef:>10.3f} {mean_bern:>10.3f} "
                  f"{ratio:>8.2f}x {np.mean([np.var([s['V_g'] - alpha * s['R_g'] for s in [ss for ss in compute_per_family_stats(output_dir / f'seed_0' / 'test_candidates_with_evalues.csv', alpha) if ss['K_g'] <= 5]], ddof=1)]):>10.4f} "
                  f"{K_max:>6} {R_bar:>6.3f} {N_fam:>6}")

    print()
    print("If ε < 1: bound is non-vacuous (pooled FDP ≤ α + ε with prob ≥ 1-δ)")
    print("If ε ≥ 1: bound is vacuous (trivially true since FDP ∈ [0,1])")
    print(f"δ = {DELTA}")


if __name__ == "__main__":
    main()
