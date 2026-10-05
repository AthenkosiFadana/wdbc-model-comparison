# Logistic Regression vs Decision Tree — WDBC Dataset

A self-contained model-comparison study on the Wisconsin Diagnostic Breast Cancer
dataset, evaluated with repeated stratified 5-fold cross-validation (5 repeats
x 5 folds = 50 evaluations per model, fixed seed 42, identical folds for both
models).

## Research question

How do logistic regression and decision tree models compare in classifying
benign and malignant samples in the WDBC dataset, when evaluated using
repeated stratified 5-fold cross-validation?

## Headline results (mean ± SD across 50 folds)

| Metric | Logistic regression | Decision tree (depth 3) |
|---|---|---|
| Balanced accuracy | 0.9746 ± 0.0179 | 0.9184 ± 0.0263 |
| Macro F1 | 0.9772 ± 0.0156 | 0.9215 ± 0.0252 |
| Malignant recall | 0.9576 ± 0.0339 | 0.8840 ± 0.0515 |

Full results and limitations: [REPORT.md](REPORT.md).

## Layout

```
run_comparison.py   main experiment: fixed settings, 50 shared folds, writes results/
verify.py           verification: determinism, fold identity, leakage check, baselines
                    (7/7 checks pass -> results/verification.json)
REPORT.md           full write-up: abstract, intro, related work, method, results, limitations, references
ANALYSIS_LOG.md     dated log of decisions, a priori vs post-hoc
results/            fold_metrics.csv, summary.csv, confusion_matrices.csv, dataset_info.json
```

## Reproduce

```
pip install -r requirements.txt
python run_comparison.py
python verify.py
```

Python 3.11. Verified on Windows 11. All outputs are deterministic: reruns
produce byte-identical result files (see `verify.py`).

## Verification status

- Re-running the experiment produces identical files (SHA-256 matched).
- Both models scored on the identical set of 50 folds.
- Standardisation fitted on training folds only (checked against raw train
  statistics; differs from full-data statistics).
- Majority-class and stratified-dummy baselines evaluated on the same folds
  (`results/baselines.csv`).

## Scope

Results are descriptive for these two models, this dataset, and this setup.
The 50 fold results are not 50 independent experiments; spread is reported as
variation across folds, not as a statistical significance claim. No clinical
usefulness is claimed.
