#!/usr/bin/env python3
"""
Per-class breakdown: family counts, present rate, macro vs micro FDP/recall.
Usage:
  python analyze_per_class.py \
    --input-jsonl outputs/voc_smoke200_gdino.jsonl \
    --formal-dir  outputs/voc_smoke200_formal
"""
import argparse, json, csv, sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--formal-dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    # ── 1. Per-class family counts from JSONL ──────────────────────────────
    class_present = defaultdict(int)   # cat → #present families
    class_absent  = defaultdict(int)   # cat → #absent families
    total_families = 0
    total_absent   = 0

    with open(args.input_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            cat = rec["category_name"]
            if rec["is_prompt_absent"]:
                class_absent[cat] += 1
                total_absent += 1
            else:
                class_present[cat] += 1
            total_families += 1

    absent_frac = total_absent / max(total_families, 1)
    print(f"\n{'='*60}")
    print(f"FAMILY COUNTS  total={total_families}  absent_frac={absent_frac:.2f}")
    print(f"{'='*60}")

    all_cats = sorted(set(class_present) | set(class_absent))
    print(f"\n{'Category':<22} | present | absent | total | present%")
    print("-"*60)
    rows = []
    for c in all_cats:
        p = class_present[c]; a = class_absent[c]; t = p + a
        rows.append((t, c, p, a))
    rows.sort(reverse=True)
    for t, c, p, a in rows:
        print(f"  {c:<20} | {p:7d} | {a:6d} | {t:5d} | {p/max(t,1)*100:.0f}%")

    # ── 2. Macro vs Micro FDP/recall from formal pipeline ─────────────────
    strata_file = args.formal_dir / "family_strata_results.csv"
    mean_file   = args.formal_dir / "mean_results_by_alpha.csv"

    if not mean_file.exists():
        print("\nFormal pipeline results not yet available.")
        return

    # Micro (global)
    micro_fdp = micro_recall = None
    with open(mean_file) as f:
        for row in csv.DictReader(f):
            if row["method"] == "betting" and abs(float(row["alpha"]) - args.alpha) < 0.001:
                micro_fdp    = float(row["fdp"])
                micro_recall = float(row["recall"])
                break

    print(f"\n{'='*60}")
    print(f"MICRO (global) at alpha={args.alpha:.2f}")
    print(f"  FDP={micro_fdp*100:.2f}%  recall={micro_recall*100:.2f}%"
          f"  {'PASS' if micro_fdp <= args.alpha else 'FAIL'}")

    # Per-class breakdown from strata if available
    if strata_file.exists():
        class_fdp    = defaultdict(list)
        class_recall = defaultdict(list)
        with open(strata_file) as f:
            for row in csv.DictReader(f):
                if row.get("method") != "betting": continue
                if abs(float(row.get("alpha", 0)) - args.alpha) > 0.001: continue
                cat = row.get("stratum", row.get("category_name", "?"))
                fdp_val = row.get("fdp", row.get("mean_fdp", None))
                rec_val = row.get("recall", row.get("mean_recall", None))
                if fdp_val is not None:
                    class_fdp[cat].append(float(fdp_val))
                    class_recall[cat].append(float(rec_val) if rec_val else 0.0)

        if class_fdp:
            macro_fdp    = sum(sum(v)/len(v) for v in class_fdp.values()) / len(class_fdp)
            macro_recall = sum(sum(v)/len(v) for v in class_recall.values()) / len(class_recall)
            print(f"\nMACRO (per-class mean) at alpha={args.alpha:.2f}")
            print(f"  FDP={macro_fdp*100:.2f}%  recall={macro_recall*100:.2f}%"
                  f"  {'PASS' if macro_fdp <= args.alpha else 'FAIL'}")
            print(f"\n  micro-macro FDP gap = {(micro_fdp-macro_fdp)*100:+.2f}pp")

            print(f"\n{'Category':<22} | FDP%   | recall% | status")
            print("-"*55)
            per_class_avg = {c: (sum(class_fdp[c])/len(class_fdp[c]),
                                  sum(class_recall[c])/len(class_recall[c]))
                              for c in class_fdp}
            for c, (fdp, rec) in sorted(per_class_avg.items(), key=lambda x: -x[1][0]):
                s = "PASS" if fdp <= args.alpha else "FAIL"
                print(f"  {c:<20} | {fdp*100:5.2f}% | {rec*100:6.2f}%  | {s}")
        else:
            print("\n  (strata file has no per-class rows matching filter)")
    else:
        print(f"\n  (strata file not found: {strata_file})")

    print()


if __name__ == "__main__":
    main()
