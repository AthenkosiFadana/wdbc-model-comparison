# Logistic Regression vs Decision Tree on the Wisconsin Diagnostic Breast Cancer Dataset

## Research question

How do logistic regression and decision tree models compare in classifying benign and malignant samples in the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, when evaluated using repeated stratified 5-fold cross-validation?

Breast tumour classification is the case study; the project itself is centred on the model comparison.

## Method

- **Evaluation design:** 5-fold stratified cross-validation, repeated 10 times, `random_state = 42` → 50 train-validation evaluations per model.
- **Identical folds:** the 50 splits are materialised once and shared by both models, so every fold comparison is paired.
- **No information leakage:** standardisation sits inside the logistic-regression pipeline and is fitted only on each fold's training data.
- **Class labelling:** `malignant` is class 0 (212 samples), `benign` is class 1 (357 samples), 569 samples and 30 features in total. Malignant recall is reported separately because it measures the share of malignant samples the model actually catches.
- **Statistics reported:** mean, standard deviation (variation across folds), min and max for every metric, plus confusion matrices pooled over all 50 folds.

### Settings fixed before running (a priori, no tuning)

| Model | Settings |
|---|---|
| Logistic regression | `StandardScaler` → `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)` |
| Decision tree | `DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)`, no pruning or depth tuning |

Neither model was tuned on this data; the settings were chosen in advance and reported as-is.

## Results

### Headline comparison (mean ± standard deviation across 50 folds)

| Metric | Logistic regression | Decision tree | Gap (LR − tree) |
|---|---|---|---|
| Balanced accuracy | 0.9746 ± 0.0179 | 0.9184 ± 0.0263 | +0.0562 |
| Macro precision | 0.9806 ± 0.0130 | 0.9271 ± 0.0255 | +0.0535 |
| Macro recall | 0.9746 ± 0.0179 | 0.9184 ± 0.0263 | +0.0562 |
| Macro F1 | 0.9772 ± 0.0156 | 0.9215 ± 0.0252 | +0.0557 |
| **Malignant recall** | **0.9576 ± 0.0339** | **0.8840 ± 0.0515** | **+0.0736** |
| Malignant precision | 0.9856 ± 0.0153 | 0.9206 ± 0.0463 | +0.0650 |
| Malignant F1 | 0.9711 ± 0.0200 | 0.9004 ± 0.0321 | +0.0707 |
| Benign recall | 0.9916 ± 0.0089 | 0.9529 ± 0.0307 | +0.0387 |
| Accuracy | 0.9789 ± 0.0143 | 0.9272 ± 0.0234 | +0.0517 |

Full per-fold values: `results/fold_metrics.csv`; summary statistics: `results/summary.csv`.

### Fold-to-fold spread

| Metric | LR min–max | Tree min–max |
|---|---|---|
| Balanced accuracy | 0.9302 – 1.0000 | 0.8718 – 0.9762 |
| Malignant recall | 0.8605 – 1.0000 | 0.7619 – 0.9767 |

The tree's malignant recall varies more across folds (±0.0515 vs ±0.0339) and its worst fold caught only 76.2% of malignant samples.

### Confusion matrices (pooled over 50 folds; each sample is predicted 10 times, 5690 predictions per model)

Logistic regression

|  | predicted malignant | predicted benign |
|---|---|---|
| true malignant | 2030 | 90 |
| true benign | 30 | 3540 |

Decision tree (depth 3)

|  | predicted malignant | predicted benign |
|---|---|---|
| true malignant | 1874 | 246 |
| true benign | 168 | 3402 |

Error profile: the logistic regression missed 90 of 2120 malignant predictions (4.2%) and over-called 30 benign samples (0.8%). The tree missed 246 malignant predictions (11.6%) and over-called 168 benign samples (4.7%). The tree's errors are concentrated in false negatives — malignant samples labelled benign — which is the costlier error in this case study.

Machine-readable version: `results/confusion_matrices.csv`.

### Baselines (post-hoc, same 50 folds)

| Model | Accuracy | Balanced accuracy | Malignant recall |
|---|---|---|---|
| Majority class | 0.6274 ± 0.0040 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| Stratified random | 0.5536 ± 0.0372 | 0.5115 ± 0.0402 | 0.3465 ± 0.0512 |

Both compared models clear the trivial baselines by a wide margin; the
majority-class baseline also shows why accuracy alone is misleading here
(62.7% for always predicting "benign", with malignant recall of 0).
Added after the first run as a reference point — see `ANALYSIS_LOG.md`.
Stored in `results/baselines.csv`.

## Verification

`python verify.py` re-checks the experiment and writes
`results/verification.json`. All 7 checks pass:

1. Re-running `run_comparison.py` produces byte-identical result files (SHA-256 level).
2. Each model has exactly 50 fold evaluations.
3. Both models were scored on matching train/test sizes for every fold.
4. Recorded folds match folds regenerated from seed 42 (one shared split set).
5. Scaler statistics match raw training-fold statistics exactly (max diff 0.0).
6. Scaler statistics differ from full-dataset statistics (no leakage from test folds).
7. Logistic regression beats the majority-class baseline on balanced accuracy.

## Interpretation

- Under this evaluation setup, logistic regression with fold-wise standardisation scores higher than a depth-3 decision tree on every reported metric, with the largest advantage in malignant recall (+7.4 points) and the smallest in benign recall (+3.9 points).
- The dominant practical difference is error type: the tree produces roughly 2.7× as many missed malignant predictions as logistic regression.
- The tree's wider fold-to-fold spread indicates its performance is more sensitive to which samples land in the training fold, consistent with a small, hard-split model.

## Limitations

- The 50 fold results are **not 50 independent experiments** — training sets overlap heavily — so the spreads above are described as variation across folds, not as a basis for a significance claim or a definitive statistical win. The gap is descriptive for this run.
- The dataset is small (569 samples), comes from a single source (UCI WDBC, digitised fine-needle aspirates), and the conclusion is limited to these two models, this dataset, and this evaluation setup.
- No clinical usefulness is claimed. Results do not establish diagnostic performance, generalisation to other populations, or readiness for deployment.
- Settings were fixed rather than tuned, so the comparison reflects these pre-specified configurations, not each model's best achievable performance.

## Reproduce

```
python run_comparison.py
```

Requires Python 3.11+, `scikit-learn`, `pandas`, `numpy`. Outputs go to `results/`.
