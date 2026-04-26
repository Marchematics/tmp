"""
Batch back-fill pooled_bound_eps / pooled_bound_delta / pooled_bound_covered
in paper_ground_truth_table_2026-04-14.csv.

For each unique (source, alpha) in the main CSV that points to a
formal_hist output dir with family_strata_results.csv + seed_*/
test_candidates_with_evalues.csv, re-use the logic in validate_pooled_bound.py
to compute:
  - epsilon_sequence (the tighter, data-dependent bound)
  - covered_sequence (bool per seed)
then aggregate to per-(source,alpha):
  - pooled_bound_eps := mean(epsilon_sequence across seeds)
  - pooled_bound_delta := 0.05
  - pooled_bound_covered := "k/N" string, e.g. "20/20"

Rows whose source is a raw CSV or has no seed_*/ dirs are left empty.

Writes updated CSV in-place, backs up original.
"""
from __future__ import annotations

import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
# Re-use helpers from the canonical script
from validate_pooled_bound import (  # type: ignore
    candidate_family_stats,
    fallback_family_stats,
    load_seed_pooled_rows,
)

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data_raw/a_project_2/outputs/paper_ground_truth_table_2026-04-14.csv"
DELTA = 0.05
DEP_FACTOR = 1.0
RHO_FLOOR = 1e-12
METHOD_DEFAULT = "betting"


def compute_bound_for_dir_alpha(source_dir: Path, alpha: float) -> dict | None:
    """Return per-seed epsilon + coverage summary, or None if data missing."""
    strata = source_dir / "family_strata_results.csv"
    if not strata.exists():
        return None
    # Try the default method first; if empty, enumerate available methods
    # containing 'betting'
    try:
        rows = load_seed_pooled_rows(strata, alpha, METHOD_DEFAULT)
    except ValueError:
        # Find a method column value starting with 'betting'
        df = pd.read_csv(strata)
        methods = [m for m in df["method"].astype(str).unique() if "betting" in m]
        if not methods:
            return None
        rows = load_seed_pooled_rows(strata, alpha, methods[0])

    eps_seq_vals = []
    covered_flags = []
    total_seeds = len(rows)
    for _, row in rows.iterrows():
        seed = int(row["seed"])
        stats = candidate_family_stats(source_dir, seed)
        if stats is None:
            stats = fallback_family_stats(
                strata, seed, float(row["alpha"]), str(row["method"]), None
            )
        if stats is None or stats.k_sum <= 0 or stats.num_families <= 0:
            continue
        total_rejections = float(row["total_rejections"])
        rho = total_rejections / stats.k_sum
        rho_eff = max(rho, RHO_FLOOR)
        log_term = math.log(1.0 / DELTA)
        eps_seq = (
            (1.0 / rho_eff)
            * math.sqrt(
                2.0 * DEP_FACTOR * log_term * stats.k_sum_sq / (stats.k_sum * stats.k_sum)
            )
        )
        eps_seq_vals.append(eps_seq)
        fdp = float(row["fdp_pooled"])
        covered_flags.append(bool(fdp <= alpha + eps_seq))

    if not eps_seq_vals:
        return None
    return {
        "eps_mean": float(np.mean(eps_seq_vals)),
        "eps_median": float(np.median(eps_seq_vals)),
        "n_seeds": total_seeds,
        "n_eval": len(eps_seq_vals),
        "covered_k": int(sum(covered_flags)),
        "covered_n": len(covered_flags),
    }


def main():
    df = pd.read_csv(CSV)
    print(f"rows: {len(df)} cols: {list(df.columns)}")
    for col in ["pooled_bound_eps", "pooled_bound_delta", "pooled_bound_covered"]:
        if col not in df.columns:
            raise RuntimeError(f"expected column {col} in CSV")

    # Group by (source, alpha) for efficiency
    cache: dict[tuple[str, float], dict | None] = {}
    updated = 0
    skipped_no_dir = 0
    skipped_no_data = 0

    for idx, row in df.iterrows():
        src = str(row["source"]).strip()
        alpha = float(row["alpha"])
        key = (src, alpha)
        if key not in cache:
            # Resolve source path: relative to project_2 dir
            if src.endswith(".csv"):
                cache[key] = None
                skipped_no_dir += 1
                continue
            src_dir = REPO / "data_raw/a_project_2" / src
            if not src_dir.exists() or not src_dir.is_dir():
                cache[key] = None
                skipped_no_dir += 1
                continue
            cache[key] = compute_bound_for_dir_alpha(src_dir, alpha)
        r = cache[key]
        if r is None:
            skipped_no_data += 1
            continue
        df.at[idx, "pooled_bound_eps"] = round(r["eps_mean"], 6)
        df.at[idx, "pooled_bound_delta"] = DELTA
        df.at[idx, "pooled_bound_covered"] = f"{r['covered_k']}/{r['covered_n']}"
        updated += 1

    # Backup and write
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CSV.with_suffix(f".bak.{ts}.csv")
    shutil.copy(CSV, backup)
    df.to_csv(CSV, index=False)
    print(f"updated rows: {updated}")
    print(f"skipped (no dir / raw csv): {skipped_no_dir}")
    print(f"skipped (no seed data): {skipped_no_data}")
    print(f"backup: {backup}")
    print(f"wrote:  {CSV}")


if __name__ == "__main__":
    main()
