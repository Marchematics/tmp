"""
K-stratified pooled-FDP bound: take the sharpening path from
theory_notes/pooled_fdr_bound.md §Interpretation point 1.

Original (non-stratified, sequence form):
  eps_seq = (1/rho_R) * sqrt( 2 chi log(1/delta) * sum_g K_g^2 / (sum_g K_g)^2 )

Stratified by K ∈ {K<=5, 5<K<=30, K>30}:
  Within each stratum l:
    rho_l = S_R^(l) / S_K^(l)
    eps_seq^(l) = (1/rho_l) * sqrt( 2 chi log(m/delta) * sum_g^(l) K_g^2 / (S_K^(l))^2 )
    eps_worst^(l) = (K_max^(l) / (rho_l K_min^(l))) * sqrt( 2 chi log(m/delta) / G_l )
  Stratum weights:
    w_l = S_R^(l) / S_R_total   (since pooled FDP = sum_l w_l * F_pool^(l))
  Overall stratified bound:
    eps_strat_seq = sum_l w_l * eps_seq^(l)
    eps_strat_worst = sum_l w_l * eps_worst^(l)
  Coverage: F_pool <= alpha + eps_strat.

We union-bound across m=3 strata with delta/m each (hence log(m/delta)).

For each source dir in the main CSV, compute per-(alpha, seed):
  - eps_seq (non-strat, reference)
  - eps_strat_seq
  - eps_strat_worst
  - covered_* flags
then aggregate per-(source, alpha) and write to a side CSV
`outputs/pooled_bound_stratified/summary.csv`.

Optionally back-fill a new column `pooled_bound_eps_stratified` into the
main CSV; `pooled_bound_covered` already captured non-stratified coverage.
"""
from __future__ import annotations

import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "data_raw/a_project_2"
MAIN_CSV = PROJECT / "outputs/paper_ground_truth_table_2026-04-14.csv"
OUT_DIR = PROJECT / "outputs/pooled_bound_stratified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 0.05
DEP_FACTOR = 1.0
RHO_FLOOR = 1e-12


def assign_stratum(k: float) -> str:
    if k <= 5:
        return "K<=5"
    if k <= 30:
        return "5<K<=30"
    return "K>30"


def per_seed_family_sizes(seed_dir: Path) -> pd.DataFrame | None:
    cand = seed_dir / "test_candidates_with_evalues.csv"
    if not cand.exists():
        return None
    try:
        fam = pd.read_csv(cand, usecols=["family_id", "family_size"])
    except ValueError:
        return None
    fam = fam.drop_duplicates("family_id").reset_index(drop=True)
    fam["stratum"] = fam["family_size"].apply(assign_stratum)
    return fam


def strat_aggregates(fam: pd.DataFrame) -> dict:
    """Aggregate per stratum from per-family sizes (test-split families)."""
    out = {}
    for stratum, g in fam.groupby("stratum"):
        K = g["family_size"].astype(float).to_numpy()
        out[stratum] = {
            "G_l": int(K.size),
            "K_min": float(K.min()) if K.size else None,
            "K_max": float(K.max()) if K.size else None,
            "S_K_l": float(K.sum()),
            "S_K_sq_l": float((K * K).sum()),
        }
    return out


def compute_bounds_for_dir(source_dir: Path, alphas=(0.05, 0.10, 0.15, 0.20)) -> list[dict]:
    strata_csv = source_dir / "family_strata_results.csv"
    if not strata_csv.exists():
        return []
    strata = pd.read_csv(strata_csv)
    # Pick the canonical method
    methods = strata["method"].astype(str).unique()
    method = None
    for m in ["betting"] + [x for x in methods if "betting" in x]:
        if m in methods:
            method = m
            break
    if method is None:
        return []

    # Per-seed family stats cache
    seed_family_cache: dict[int, dict] = {}
    seeds = sorted(strata["seed"].astype(int).unique())
    for seed in seeds:
        fam = per_seed_family_sizes(source_dir / f"seed_{seed}")
        if fam is None:
            continue
        seed_family_cache[seed] = strat_aggregates(fam)

    rows = []
    for alpha in alphas:
        sub = strata[
            np.isclose(strata["alpha"].astype(float), alpha)
            & (strata["method"].astype(str) == method)
        ]
        if sub.empty:
            continue

        m_strata = 3  # union bound count
        log_term_strat = math.log(m_strata / DELTA)
        log_term_base = math.log(1.0 / DELTA)

        per_seed_rows = []
        for seed in sorted(sub["seed"].astype(int).unique()):
            sub_s = sub[sub["seed"] == seed]
            S_R_total = float(sub_s["total_rejections"].sum())
            fp_total = float(sub_s["fp"].sum())
            fdp_pool = fp_total / max(S_R_total, 1.0)
            fam_stats = seed_family_cache.get(seed)
            if fam_stats is None:
                continue

            # Non-stratified sequence bound (baseline)
            all_S_K = sum(s["S_K_l"] for s in fam_stats.values())
            all_S_K_sq = sum(s["S_K_sq_l"] for s in fam_stats.values())
            all_G = sum(s["G_l"] for s in fam_stats.values())
            if all_S_K <= 0 or S_R_total <= 0:
                continue
            rho_global = S_R_total / all_S_K
            rho_eff_global = max(rho_global, RHO_FLOOR)
            eps_seq_global = (
                (1.0 / rho_eff_global)
                * math.sqrt(2.0 * DEP_FACTOR * log_term_base * all_S_K_sq / (all_S_K ** 2))
            )

            # Stratified bounds
            eps_strat_seq = 0.0
            eps_strat_worst = 0.0
            for stratum_label, stats_l in fam_stats.items():
                row_l = sub_s[sub_s["stratum"] == stratum_label]
                if row_l.empty:
                    continue
                G_l = stats_l["G_l"]
                S_K_l = stats_l["S_K_l"]
                S_K_sq_l = stats_l["S_K_sq_l"]
                K_min_l = stats_l["K_min"]
                K_max_l = stats_l["K_max"]
                S_R_l = float(row_l["total_rejections"].sum())
                if G_l <= 0 or S_K_l <= 0:
                    continue
                w_l = S_R_l / S_R_total if S_R_total > 0 else 0.0
                rho_l = S_R_l / S_K_l if S_K_l > 0 else 0.0
                rho_l_eff = max(rho_l, RHO_FLOOR)
                # sequence form within stratum
                if S_K_l > 0:
                    eps_l_seq = (
                        (1.0 / rho_l_eff)
                        * math.sqrt(2.0 * DEP_FACTOR * log_term_strat * S_K_sq_l / (S_K_l ** 2))
                    )
                else:
                    eps_l_seq = float("inf")
                # worst-case within stratum
                if K_min_l and K_min_l > 0 and G_l > 0:
                    eps_l_worst = (
                        (K_max_l / (rho_l_eff * K_min_l))
                        * math.sqrt(2.0 * DEP_FACTOR * log_term_strat / G_l)
                    )
                else:
                    eps_l_worst = float("inf")
                eps_strat_seq += w_l * eps_l_seq
                eps_strat_worst += w_l * eps_l_worst

            per_seed_rows.append(
                dict(
                    source=str(source_dir.relative_to(PROJECT)),
                    alpha=alpha,
                    seed=seed,
                    fdp_pool=fdp_pool,
                    S_R_total=S_R_total,
                    G_total=all_G,
                    eps_seq=eps_seq_global,
                    eps_strat_seq=eps_strat_seq,
                    eps_strat_worst=eps_strat_worst,
                    raw_pass=bool(fdp_pool <= alpha),
                    covered_seq=bool(fdp_pool <= alpha + eps_seq_global),
                    covered_strat_seq=bool(fdp_pool <= alpha + eps_strat_seq),
                    covered_strat_worst=bool(fdp_pool <= alpha + eps_strat_worst),
                )
            )

        if not per_seed_rows:
            continue
        df_s = pd.DataFrame(per_seed_rows)
        # Aggregate per (source, alpha)
        agg = {
            "source": str(source_dir.relative_to(PROJECT)),
            "alpha": alpha,
            "n_seeds": int(len(df_s)),
            "fdp_pool_mean": float(df_s["fdp_pool"].mean()),
            "raw_pass_k": int(df_s["raw_pass"].sum()),
            "eps_seq_mean": float(df_s["eps_seq"].mean()),
            "eps_seq_median": float(df_s["eps_seq"].median()),
            "eps_strat_seq_mean": float(df_s["eps_strat_seq"].mean()),
            "eps_strat_seq_median": float(df_s["eps_strat_seq"].median()),
            "eps_strat_worst_mean": float(df_s["eps_strat_worst"].mean()),
            "cov_seq": int(df_s["covered_seq"].sum()),
            "cov_strat_seq": int(df_s["covered_strat_seq"].sum()),
            "cov_strat_worst": int(df_s["covered_strat_worst"].sum()),
        }
        rows.append(agg)
    return rows


def main():
    df_main = pd.read_csv(MAIN_CSV)
    # Unique source dirs (exclude raw csv sources)
    sources = sorted(
        set(s for s in df_main["source"].astype(str).unique() if not s.endswith(".csv"))
    )
    print(f"unique source dirs: {len(sources)}")

    all_aggs = []
    per_seed_details = []
    for src in sources:
        src_dir = PROJECT / src
        if not src_dir.is_dir():
            continue
        rows = compute_bounds_for_dir(src_dir)
        all_aggs.extend(rows)
        print(f"  {src}: {len(rows)} alpha rows processed")

    if not all_aggs:
        print("no rows processed")
        return

    agg_df = pd.DataFrame(all_aggs)
    agg_df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nwritten {OUT_DIR/'summary.csv'}: {len(agg_df)} rows")

    # Print top highlights at α=0.10
    print("\n=== Stratified vs non-stratified (α=0.10) ===")
    cols = [
        "source",
        "fdp_pool_mean",
        "eps_seq_mean",
        "eps_strat_seq_mean",
        "eps_strat_worst_mean",
        "cov_seq",
        "cov_strat_seq",
        "cov_strat_worst",
        "raw_pass_k",
        "n_seeds",
    ]
    a = agg_df[agg_df["alpha"] == 0.10][cols].copy()
    a["shrink_seq_vs_strat"] = (a["eps_seq_mean"] / a["eps_strat_seq_mean"]).round(3)
    print(a.round(4).to_string(index=False))

    # Back-fill column into main CSV
    df_main["pooled_bound_eps_stratified"] = np.nan
    for _, r in agg_df.iterrows():
        mask = (df_main["source"] == r["source"]) & np.isclose(df_main["alpha"], r["alpha"])
        # Pick the tighter of strat_seq and strat_worst for reporting
        tighter = min(r["eps_strat_seq_mean"], r["eps_strat_worst_mean"])
        df_main.loc[mask, "pooled_bound_eps_stratified"] = round(tighter, 6)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MAIN_CSV.with_suffix(f".bak.{ts}.csv")
    shutil.copy(MAIN_CSV, backup)
    df_main.to_csv(MAIN_CSV, index=False)
    print(f"\nback-filled {MAIN_CSV.name} (new col pooled_bound_eps_stratified)")
    print(f"backup: {backup}")


if __name__ == "__main__":
    main()
