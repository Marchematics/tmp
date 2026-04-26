# B3 + B2 Robustness Log

Generated on 2026-04-16 from `data_raw/a_project_2`.

## B3.1 Cross-Family LOO KS/MMD

- Input table: `outputs/coco_groundingdino_gonogo_1000_mix_analysis/candidate_table.csv`
- Seeds: `0,1,2,3,4`
- Pairwise family comparisons: `657750`
- KS Bonferroni reject count at 0.05: `227` (`0.0345%`)
- Mean KS statistic: `0.5812`; median: `0.5000`
- Mean RBF MMD^2: `0.3097`; median: `0.1711`
- Outputs:
  - `outputs/robustness_analysis/cross_family_ks_mmd.csv`
  - `outputs/robustness_analysis/cross_family_ks_mmd.pdf`
  - `outputs/robustness_analysis/cross_family_ks_mmd_seed_summary.csv`

Note: the prompt named `outputs/coco_groundingdino_gonogo_1000_mix_betting_oracle/` as containing per-seed `test_candidates_with_evalues.csv`, but this tree has a single oracle output rather than per-seed formal splits. The analysis reconstructs the formal family splits from the canonical candidate table and computes calibration-null LOO e-values directly.

## B3.2 YOLO-World Phi Laplace

- Input table: `outputs/coco_yoloworld_val_1000_mix_analysis/candidate_table.csv`
- Seeds: `0..19`
- Vanilla command uses `--smoothing 0.0 --clip-phi 1000000 --phi-laplace-eps 0`
- Laplace command uses `--smoothing 0.0 --clip-phi 1000000 --phi-laplace-eps 1e-3`
- `coco_yoloworld_val_pilot.jsonl` was not present in the repository; this run uses the existing analyzed candidate table that backs the 20-seed YOLO-World formal history.

Summary:

| variant | max phi | clipped-bin seeds | FDP@0.10 | Recall@0.10 | rejections@0.10 |
|---|---:|---:|---:|---:|---:|
| vanilla | 1000000.0 | 20/20 | 0.030820 | 0.152653 | 62.50 |
| laplace | 42967.690334 | 20/20 vanilla-zero-bin exposure, regularized before clip | 0.030835 | 0.152513 | 62.45 |

Interpretation: Laplace removes the effective `phi_max=1e6` hard-clipped outlier without material FDP/recall movement.

Outputs:

- `outputs/robustness_analysis/yoloworld_vanilla/`
- `outputs/robustness_analysis/yoloworld_laplace_eps1e-3/`
- `outputs/robustness_analysis/robustness_yoloworld_floor.csv`

## B3.3 Per-Family LOO Distribution

- Datasets: COCO, VOC, LVIS
- Seeds: `0..19`
- Output figure: `outputs/robustness_analysis/loo_per_family_distribution.pdf`
- Output table: `outputs/robustness_analysis/loo_per_family_distribution.csv`

Unweighted per-family mean LOO summary:

| dataset | families x seeds | mean | median | std |
|---|---:|---:|---:|---:|
| COCO | 10247 | 1.4511 | 0.4735 | 13.2637 |
| VOC | 10391 | 2.2136 | 0.4725 | 23.0164 |
| LVIS | 11584 | 1.4634 | 0.8945 | 2.4527 |

Global calibration-null LOO mean remains 1 by construction; the table above is unweighted over families, so tiny high-phi families can move the mean while the median stays near or below 1.

## B2 Template Pool and Template-Count Sweep

Generated template files:

- `data/templates/freeform_50.txt`
- `data/templates/freeform_100.txt`
- `data/templates/freeform_200.txt`

Sweep input:

- Candidate table: `outputs/coco_gdino_freeform_500/analysis/candidate_table.csv`
- Seeds: `0..19`
- Template counts: `7,50,100,200`
- Mode: deterministic prompt/template relabeling on the existing free-form candidate scores. This isolates the calibration-pool shrinkage effect requested in B2.2; it does not rerun detector inference for each new prompt string.

Per-template rows are logged in:

- `outputs/robustness_analysis/freeform_template_diagnostics_by_template.csv`
- `outputs/robustness_analysis/freeform_template_count_sweep/template_count_sweep_diagnostics.csv`

Summary at alpha 0.10:

| templates | strategy | mean cal families/template | FDP | recall | rejections | phi_max_max |
|---:|---|---:|---:|---:|---:|---:|
| 7 | pertemplate | 42.7143 | 0.083333 | 0.001109 | 0.65 | 26.9684 |
| 50 | pertemplate | 5.9800 | 0.000000 | 0.000150 | 0.05 | 26.6132 |
| 100 | pertemplate | 2.9900 | 0.000000 | 0.000133 | 0.05 | 24.7200 |
| 200 | pertemplate | 1.4950 | 0.100000 | 0.000000 | 0.10 | 30.4348 |

Output summary:

- `outputs/robustness_analysis/freeform_templates_sweep.csv`
- `outputs/robustness_analysis/freeform_template_count_sweep/template_count_sweep_curve.pdf`
