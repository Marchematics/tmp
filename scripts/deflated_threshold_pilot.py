#!/usr/bin/env python3
"""Quick pilot: deflated score threshold at multiple alpha levels.

Tests whether a concentration-corrected score threshold can achieve
valid pooled FDP control and non-trivial recall.
"""
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_TABLE = (
    PROJECT_ROOT / "outputs" / "coco_groundingdino_gonogo_1000_mix.jsonl"
)
# Use the pre-computed test_candidates from seed_0 for a quick test
SEED_DATA_DIR = (
    PROJECT_ROOT / "outputs" / "coco_gdino_1000_20seed_formal_hist"
)

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DELTA = 0.05


def split_families(family_ids, seed, ratios=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    unique_fams = np.array(sorted(set(family_ids)))
    rng.shuffle(unique_fams)
    n = len(unique_fams)
    n_fit = max(1, int(round(ratios[0] * n)))
    n_cal = max(1, int(round(ratios[1] * n)))
    n_cal = min(n_cal, n - n_fit - 1)
    fit_set = set(unique_fams[:n_fit])
    cal_set = set(unique_fams[n_fit:n_fit + n_cal])
    splits = {}
    for fam in family_ids:
        if fam in fit_set:
            splits[fam] = "fit"
        elif fam in cal_set:
            splits[fam] = "cal"
        else:
            splits[fam] = "test"
    return splits


def evaluate_threshold(df, threshold, split_map, target_split):
    """Evaluate a score threshold on the target split."""
    mask = df["family_id"].map(split_map) == target_split
    subset = df[mask]

    total_tp_available = int(subset["is_tp"].sum())
    selected = subset[subset["score"] >= threshold]

    if selected.empty:
        return {"rejections": 0, "fp": 0, "tp": 0, "fdp_pooled": 0.0,
                "recall": 0.0, "fdp_family": 0.0}

    fp = int((~selected["is_tp"]).sum())
    tp = int(selected["is_tp"].sum())
    rej = len(selected)
    fdp_pooled = fp / max(rej, 1)
    recall = tp / max(total_tp_available, 1)

    # Per-family FDP
    family_fdps = []
    for _, grp in selected.groupby("family_id"):
        fam_fp = int((~grp["is_tp"]).sum())
        family_fdps.append(fam_fp / max(len(grp), 1))
    fdp_family = np.mean(family_fdps) if family_fdps else 0.0

    return {"rejections": rej, "fp": fp, "tp": tp, "fdp_pooled": fdp_pooled,
            "recall": recall, "fdp_family": fdp_family}


def calibrate_deflated(df, split_map, alpha, delta=0.05):
    """Find threshold with deflated guarantee on cal split.

    Returns threshold and deflation info for multiple deflation modes.
    """
    mask = df["family_id"].map(split_map) == "cal"
    cal = df[mask].copy()

    if cal.empty:
        return {}

    # Get unique scores for threshold grid
    scores = np.sort(cal["score"].unique())[::-1]

    # Cal family info
    cal_families = cal.groupby("family_id")
    N_fam_cal = cal["family_id"].nunique()
    K_max_cal = cal.groupby("family_id").size().max()

    results = {}

    for mode in ["naive", "crc_plus1", "hoeffding_fam", "hoeffding_cand"]:
        best = None
        cum_selected = 0
        cum_fp = 0

        # Track per-family stats for family-level deflation
        fam_selected = defaultdict(int)
        fam_fp = defaultdict(int)

        for threshold in scores:
            # Add candidates at this threshold
            at_thresh = cal[cal["score"] == threshold]
            for _, row in at_thresh.iterrows():
                fam = row["family_id"]
                fam_selected[fam] += 1
                cum_selected += 1
                if not row["is_tp"]:
                    fam_fp[fam] += 1
                    cum_fp += 1

            if cum_selected == 0:
                continue

            cal_fdp = cum_fp / cum_selected

            # Compute deflation margin
            if mode == "naive":
                margin = 0.0  # no deflation
            elif mode == "crc_plus1":
                # CRC: (n*L + 1)/(n+1)
                # Per-family loss
                losses = []
                for fam in cal.groupby("family_id").groups:
                    if fam_selected[fam] > 0:
                        losses.append(fam_fp[fam] / fam_selected[fam])
                    else:
                        losses.append(0.0)
                mean_loss = np.mean(losses)
                bound = (N_fam_cal * mean_loss + 1) / (N_fam_cal + 1)
                margin = bound - cal_fdp  # effective margin above cal_fdp
                # For CRC: check if bound ≤ α directly
                if bound > alpha:
                    continue
                # CRC passed: use cal_fdp as "effective" and margin=0
                margin = 0.0
                cal_fdp = bound  # use the CRC bound as the effective FDP
            elif mode == "hoeffding_fam":
                # Family-level: ε = K_max * sqrt(log(2/δ)/(2*N_fam)) / R_bar
                R_bar = cum_selected / N_fam_cal
                if R_bar <= 0:
                    continue
                margin = K_max_cal * math.sqrt(
                    math.log(2 / delta) / (2 * N_fam_cal)) / R_bar
            elif mode == "hoeffding_cand":
                # Candidate-level Hoeffding (invalid but optimistic)
                n_cal_cand = len(cal)
                margin = math.sqrt(math.log(2 / delta) / (2 * n_cal_cand))

            # Check if threshold passes
            if cal_fdp + margin <= alpha:
                candidate = {
                    "threshold": float(threshold),
                    "cal_fdp": cal_fdp,
                    "margin": margin,
                    "cal_selected": cum_selected,
                    "mode": mode,
                }
                if best is None or cum_selected > best["cal_selected"]:
                    best = candidate

        results[mode] = best

    return results


def main():
    print("Loading candidate data from pre-processed CSVs...")

    print()
    print(f"{'Mode':<20} {'α':>5} {'Thresh':>8} {'Cal FDP':>8} {'Margin':>8} "
          f"{'Rej':>6} {'Test FDP':>9} {'Fam FDP':>9} {'Recall':>8} {'Pass':>5}")
    print("-" * 100)

    for seed in [0, 1, 2]:
        # Load the pre-processed CSV which has ALL candidates (fit/cal/test)
        csv_path = SEED_DATA_DIR / f"seed_{seed}" / "test_candidates_with_evalues.csv"
        if not csv_path.exists():
            print(f"  SKIP seed {seed}")
            continue

        df = pd.read_csv(csv_path)
        df["is_tp"] = df["is_tp"].astype(str).str.lower() == "true"

        # Use the split column already in the CSV
        split_map = dict(zip(df["family_id"], df["split"]))

        for alpha in ALPHAS:
            cal_results = calibrate_deflated(df, split_map, alpha, DELTA)

            for mode in ["naive", "crc_plus1", "hoeffding_cand", "hoeffding_fam"]:
                info = cal_results.get(mode)
                if info is None:
                    print(f"s{seed} {mode:<20} {alpha:>5.2f} {'---':>8} {'---':>8} "
                          f"{'---':>8} {'---':>6} {'---':>9} {'---':>9} {'---':>8} {'N/A':>5}")
                    continue

                test_res = evaluate_threshold(
                    df, info["threshold"], split_map, "test")

                passed = "✓" if test_res["fdp_pooled"] <= alpha else "✗"
                print(f"s{seed} {mode:<20} {alpha:>5.2f} {info['threshold']:>8.3f} "
                      f"{info['cal_fdp']:>8.3f} {info['margin']:>8.4f} "
                      f"{test_res['rejections']:>6} {test_res['fdp_pooled']:>8.3f} "
                      f"{test_res['fdp_family']:>8.3f} "
                      f"{test_res['recall']:>8.3f} {passed:>5}")

        print()  # separator between seeds


if __name__ == "__main__":
    main()
