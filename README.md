<div align="center">

# OVD Hallucination FDR

**Post-hoc family-level FDR control for open-vocabulary object detection**
*via self-normalized conformal betting e-values and dependence-robust e-BH*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## TL;DR

Open-vocabulary detectors return high-confidence boxes for categories
that are not present (in our COCO absent-query pilot, the top-decile
detections are 100% hallucinations). This repository implements a
**post-hoc reliability wrapper** that controls the per-family false
discovery rate at a user-chosen level α on a frozen detector, for
category-name or fixed-template prompts whose absence labels are
auditable. The method is finite-sample valid, distribution-free,
and tolerates arbitrary within-family dependence (NMS, shared
proposals).

A reviewer can reproduce every number in the paper from the bundled
pre-computed outputs (≈1.6 GB) **without re-running any detector**.

---

## Key results (α = 0.10)

| Configuration | Per-family FDP | Pooled FDP | Recall | Pooled precision |
|---------------|---------------:|-----------:|-------:|-----------------:|
| COCO/GDino (full)            | **0.78%** | 6.44% | 16.4% | **93.6%** (vs 72.9% for p-BH) |
| COCO/GDino + adaptive floor  | 0.65%     | 4.0%  | 18.7% | 96.0% |
| COCO/GDino + cluster e-BH    | —         | 4.1%  | **32.1%** (1.71×) | — |
| COCO/RegionCLIP + cluster    | —         | —     | **21.6%** (5.8×)  | — |

A non-vacuous Bernstein pooled-FDP certificate (ε = 0.074) holds on
the K ≤ 5 stratum (57% of families).

---

## Quick start

```bash
git clone <repo-url> ovd_fdr && cd ovd_fdr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce the α = 0.10 row of Table 1 in <2 seconds
OVD_PROJECT_ROOT=$PWD python scripts/demo_minimal.py
```

Expected output:

```
metric             demo (seed 0)    paper (20-seed mean)
-------------    ---------------  ----------------------
rejections                  133                   134.0
pooled FDP                3.01%                   3.96%
recall                   18.04%                  18.71%

Demo OK
```

---

## Repository layout

```
.
├── config.yaml                     # dataset roots & detector model ids
├── src/ovd_hallucination_fdr/      # core library
│   ├── evalues.py                  # self-normalized betting e-values + e-BH
│   ├── detectors.py                # GDino / OWL-ViT / YOLO-World wrappers
│   ├── annotations.py              # GT loaders for COCO/VOC/LVIS/OI
│   ├── matching.py                 # IoU greedy matching, family construction
│   ├── analysis.py                 # FDP / recall / pooled-precision metrics
│   ├── paths.py                    # config-driven path resolution
│   └── io.py
├── scripts/                        # 65 experiment / aggregation / plot scripts
├── outputs/                        # pre-computed CSVs + intermediate artefacts
├── requirements.txt / pyproject.toml
└── LICENSE                         # MIT
```

---

## Reproducing the paper

All numbers in the main text + appendix are sourced from
`outputs/paper_ground_truth_table_2026-04-14.csv` (1142 rows, 18 cols).

| Paper artefact | Script | Required inputs (all bundled) |
|---|---|---|
| Main result (Table 1) | `scripts/run_formal_betting_pipeline.py` | `outputs/coco_groundingdino_gonogo_1000_mix_analysis/candidate_table.csv` |
| Baselines (Table 2) | `scripts/run_adaptive_fdr_baselines.py`, `scripts/run_topk_ihw_baselines.py`, `scripts/run_crc_all_configs.py` | candidate tables for all 6 configs |
| Score floor (Table 3) | `scripts/adaptive_score_floor.py` | 20-seed formal-pipeline outputs |
| Cluster e-BH (Table 4) | `scripts/build_object_vs_candidate_table.py` | NMS-aware run outputs |
| K-stratified bound | `scripts/build_k_stratified_table.py` | 20-seed e-value runs |
| Conditional validity | `scripts/build_conditional_validity_tables.py` | 20-seed e-value runs |
| Estimator ablation | `scripts/estimator_ablation_aggregate.py` | bundled estimator runs |
| Cal-pool sweep | `scripts/cal_size_sweep.py` | bundled sweep outputs |
| Shift stress | `scripts/exchangeability_shift.py` | candidate tables |
| Latency benchmark | `scripts/benchmark_runtime.py` | candidate tables |

Each script reads from `outputs/` (set via `OVD_OUTPUTS_DIR` or
`config.yaml`) and writes a result CSV back into `outputs/`.

### End-to-end detector inference (optional)

To regenerate candidate tables from raw images:

```bash
export COCO_ROOT=/path/to/coco2017
export VOC_ROOT=/path/to/voc2012
export LVIS_ROOT=/path/to/lvis

python scripts/run_groundingdino_pilot.py --dataset coco --n-images 1000
python scripts/run_owlvit_pilot.py        --dataset coco --n-images 1000
python scripts/run_yoloworld_pilot.py     --dataset coco --n-images 1000
```

Datasets are not redistributed; download them from the official sources
(COCO 2017, Pascal VOC 2012, LVIS v1, Open Images v7).

---

## Three deployment modes

| Mode | Guarantee | Cost |
|------|-----------|------|
| **A** | Per-family FDR ≤ α (Theorem 1) | None — finite-sample, exact |
| **B** | Mode A + LOO-mean pooled-FDP monitoring | Compute LOO mean per release |
| **C** | Mode A + Bernstein-certified pooled-FDP bound on K ≤ 5 (Cor. 4) | Stratified concentration audit |

---

## Configuration

Edit `config.yaml` directly, or set environment variables at runtime
(env vars take precedence):

```yaml
project_root: ${OVD_PROJECT_ROOT:-./}
outputs_dir:  ${OVD_OUTPUTS_DIR:-./outputs}
datasets:
  coco_root:        ${COCO_ROOT:-./data/coco2017}
  voc_root:         ${VOC_ROOT:-./data/voc2012}
  lvis_root:        ${LVIS_ROOT:-./data/lvis}
  objects365_root:  ${OBJECTS365_ROOT:-./data/objects365}
  open_images_root: ${OPEN_IMAGES_ROOT:-./data/open_images_v7}
detector_models:
  groundingdino: IDEA-Research/grounding-dino-tiny
  owlvit:        google/owlv2-base-patch16-ensemble
  yoloworld:     yolov8x-worldv2
```

---

## Citation

Anonymous submission to NeurIPS 2026. Citation block will be added
after the review period.

## License

MIT — see [`LICENSE`](LICENSE).
