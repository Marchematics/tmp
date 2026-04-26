# OVD Hallucination FDR

Anonymous code release accompanying the NeurIPS 2026 submission
*Post-hoc Family-Level FDR Control for Open-Vocabulary Object Detection
via Self-Normalized Conformal Betting e-Values*.

The method is a post-hoc reliability wrapper for frozen open-vocabulary
detectors. It controls per-family false discovery rate for category-name
or fixed-template prompts whose absence labels are auditable. The core
ingredients are self-normalized conformal betting e-values (finite-sample
valid under null-score exchangeability) and within-family e-BH (FDR
control under arbitrary within-family dependence).

## Repository layout

```
config.yaml                     dataset roots & detector model ids
src/ovd_hallucination_fdr/      core library (e-values, detectors,
                                annotations, matching, paths, ...)
scripts/                        65 experiment / aggregation / plot scripts
outputs/                        pre-computed CSVs reproducing every
                                table in the paper, plus 1 demo seed
paper/                          submission tex + compiled pdf
requirements.txt / pyproject.toml
LICENSE                         MIT
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core
pip install -e ".[detectors]"            # optional: torch / transformers / ultralytics
```

`src/ovd_hallucination_fdr/paths.py` reads `config.yaml`; either edit
the yaml in place or override at runtime via environment variables:

```bash
export OVD_PROJECT_ROOT="$PWD"
export COCO_ROOT=/path/to/coco2017
export VOC_ROOT=/path/to/voc2012
export LVIS_ROOT=/path/to/lvis
```

## 2. Quick demo (no data download required)

```bash
python scripts/demo_minimal.py
```

This runs within-family e-BH on the bundled single-seed COCO/GroundingDINO
candidate table at α = 0.10 and prints the resulting (rejections, pooled
FDP, recall) alongside the published 20-seed mean. Expected output is OK
within seed variance.

## 3. Reproducing the paper tables

All numbers in the main text and appendix are stored in
`outputs/paper_ground_truth_table_2026-04-14.csv` (1142 rows, 18 columns).
The aggregator scripts are pure functions of the bundled CSVs:

| Paper artefact | Script |
|----------------|--------|
| Table 1 (full-COCO/GDino) main result | `scripts/run_formal_betting_pipeline.py` |
| Table 2 (baseline comparison) | `scripts/run_adaptive_fdr_baselines.py`, `scripts/run_topk_ihw_baselines.py`, `scripts/run_crc_all_configs.py` |
| Table 3 (score floor) | `scripts/adaptive_score_floor.py` |
| Table 4 (cluster vs candidate) | `scripts/build_object_vs_candidate_table.py` |
| App. K-stratified | `scripts/build_k_stratified_table.py` |
| App. Conditional validity | `scripts/build_conditional_validity_tables.py` |
| App. Estimator ablation | `scripts/estimator_ablation_aggregate.py` |
| App. Cal-pool sweep | `scripts/cal_size_sweep.py` |
| App. Shift stress | `scripts/exchangeability_shift.py` |
| App. Latency | `scripts/benchmark_runtime.py` |

The 6 (dataset × detector) configurations are bundled as
`outputs/<config>/candidate_table.csv`. The full 20-seed e-value
artefacts (≈ 1.3 GB) are NOT bundled; only seed 0 of COCO/GDino is
included for the quick demo. To regenerate the full 20-seed evidence,
run:

```bash
python scripts/run_formal_betting_pipeline.py \
    --candidates outputs/coco_groundingdino_gonogo_1000_mix_analysis/candidate_table.csv \
    --out outputs/coco_gdino_1000_20seed_formal_hist \
    --n-seeds 20
```

Detector inference scripts (`run_groundingdino_pilot.py`,
`run_birdet_baseline.py`, `run_regionclip_baseline.py`) require the raw
datasets to be downloaded and the corresponding `*_ROOT` env var to be
set.

## 4. Three-mode deployment

The wrapper supports three deployment modes:

* **Mode A** — per-family FDR only (default; finite-sample, dependence-robust).
* **Mode B** — Mode A plus LOO-mean empirical pooled-FDP monitoring.
* **Mode C** — Mode A plus a Bernstein-certified pooled-FDP upper bound on
  the K ≤ 5 stratum (`scripts/compute_empirical_bernstein_bound.py`).

## 5. License

MIT. See `LICENSE`.
