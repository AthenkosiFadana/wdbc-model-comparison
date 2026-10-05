# Logistic Regression vs Decision Tree on the Wisconsin Diagnostic Breast Cancer Dataset

## Abstract

**Aim:** compare two standard classifiers — logistic regression and decision tree — in classifying benign and malignant breast tumour samples under a fixed, leakage-controlled evaluation.

**Methods:** 10 × 5 repeated stratified cross-validation (50 train-validation evaluations per model) on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset (569 samples, 30 features, 212 malignant), with random seed 42 and one shared set of folds for both models. Model settings were fixed before the first run and not tuned: logistic regression with standardisation inside the pipeline (fit per training fold), and a decision tree with `max_depth = 3`. We report macro precision/recall/F1, balanced accuracy, malignant-class recall separately, fold-to-fold variation, pooled confusion matrices, and majority-class/stratified baselines. The experiment is deterministic and passes 7 automated verification checks.

**Results:** logistic regression scored higher on every metric — balanced accuracy 0.9746 ± 0.0179 vs 0.9184 ± 0.0263, macro F1 0.9772 ± 0.0156 vs 0.9215 ± 0.0252, and malignant recall 0.9576 ± 0.0339 vs 0.8840 ± 0.0515 (mean ± SD across 50 folds). Pooled over all folds, logistic regression missed 4.2% of malignant predictions versus 11.6% for the tree.

**Conclusions:** under this evaluation setup, logistic regression outperformed the depth-3 decision tree, with the practically important difference being fewer missed malignant samples. The finding is descriptive for these two model settings, this dataset, and this protocol; the 50 fold results are not 50 independent experiments, and no clinical usefulness is claimed.

## 1. Introduction

Comparing machine learning models is the most common activity in applied classification research, and breast tumour classification is one of its most-used case studies. Yet reported model comparisons are strongly shaped by evaluation choices: how the data is split, whether preprocessing is fitted inside or outside cross-validation, whether hyperparameters were tuned on the same data used for scoring, and whether a single split or repeated folds were used. Two studies can report the same model with accuracies differing by several percentage points purely because of protocol differences.

This project treats the comparison itself as the research object. Breast tumour classification on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset serves as the case study, and the methodological question — how to compare two models fairly and report the comparison honestly — is given equal weight to the empirical result. Three principles guide the design:

1. **Pre-specification.** Model settings, seed, fold structure, and metrics were fixed before any result was seen, and are reported as-is (`ANALYSIS_LOG.md` separates a-priori decisions from post-hoc additions).
2. **Leakage control.** Standardisation lives inside the logistic-regression pipeline so it is fitted only on each fold's training data.
3. **Honest reporting.** Spread across folds is reported as variation, not as 50 independent experiments; conclusions are scoped to these models, this dataset, and this setup.

The report proceeds as: related work (Section 2), research question (Section 3), data (Section 4), method (Section 5), results (Section 6), verification (Section 7), exploratory robustness check (Section 8), interpretation (Section 9), limitations (Section 10), conclusion (Section 11), and references.

## 2. Related work

**Origin of the dataset.** WDBC was introduced by Street, Wolberg, and Mangasarian (1993), who computed ten nuclear morphometry features (mean, standard error, and worst value of each) from digitised fine-needle aspirate images, yielding 30 numeric predictors for 569 samples. Using ten-fold cross-validation, they reported **97% accuracy** with a single linear separating plane on just three of the thirty features, and the system was deployed at University of Wisconsin Hospitals. The dataset is distributed through the UCI Machine Learning Repository.

**Typical reported performance.** Later studies on WDBC consistently report high scores, but with heterogeneous protocols:

- Akinwamide, Fele, and Ojo (2025) compared six classifiers on a single 80/20 split and found logistic regression best at **97.90% accuracy**, with the decision tree at 93.71% — the same ordering we observe, under a different (and less stable) evaluation design.
- Pati et al. (2024), in a peer-reviewed PLOS ONE study, combined recursive feature elimination with grey-wolf optimisation and seven classifiers, reaching **98.25% accuracy** on WDBC for their best hybrid configuration.
- Across the literature, accuracies from the low 90s (shallow single trees, untuned) to the high 90s (linear models, tuned ensembles) are routine, which itself is a caution: on this dataset almost any reasonable pipeline scores well, so **the interesting differences are in error structure and stability, not headline accuracy alone**.

**Methodological literature.** Two findings from the evaluation literature shape our reporting. First, cross-validation folds are dependent — their training sets overlap heavily — so fold scores cannot be treated as independent replications (Bates, Hastie, and Tibshirani, 2024). Second, using cross-validation both to select a model and to report its performance optimistically biases the estimate (Varma and Simon, 2006); repeated CV reduces variance but does not remove the need for a held-out or nested design when tuning. Because our settings were fixed in advance and no tuning was performed, we avoid the second problem entirely; because we describe spread rather than run significance tests on fold scores, we avoid over-claiming on the first.

**Positioning.** Unlike most of the studies above, this project does not tune models, select features, or claim a best classifier in general. It asks a narrower, fully reproducible question: given two pre-specified configurations and one shared set of repeated stratified folds, how do they differ in average performance, in stability across folds, and — critically for the case study — in which kinds of errors they make?

## 3. Research question

How do logistic regression and decision tree models compare in classifying benign and malignant samples in the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, when evaluated using repeated stratified 5-fold cross-validation?

Breast tumour classification is the case study; the project itself is centred on the model comparison.

## 4. Data

| Property | Value |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository |
| Samples | 569 |
| Features | 30 numeric (mean, standard error, and worst value of 10 nuclear characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) |
| Classes | malignant = 212 (37.3%), benign = 357 (62.7%) |
| Missing values | none |
| Source | digitised fine-needle aspirate images, University of Wisconsin Hospitals |

Loaded via `sklearn.datasets.load_breast_cancer`, which encodes malignant as 0 and benign as 1; this report uses those labels explicitly throughout. The 62.7 / 35.3 class split is mildly imbalanced: an always-"benign" classifier scores 62.7% accuracy while catching no malignant samples, which is why balanced accuracy and malignant recall are reported alongside accuracy.

## 5. Method

- **Evaluation design:** 5-fold stratified cross-validation, repeated 10 times, `random_state = 42` → 50 train-validation evaluations per model.
- **Identical folds:** the 50 splits are materialised once and shared by both models, so every fold comparison is paired.
- **No information leakage:** standardisation sits inside the logistic-regression pipeline and is fitted only on each fold's training data.
- **Class labelling:** `malignant` is class 0 (212 samples), `benign` is class 1 (357 samples). Malignant recall is reported separately because it measures the share of malignant samples the model actually catches.
- **Statistics reported:** mean, standard deviation (variation across folds), min and max for every metric, plus confusion matrices pooled over all 50 folds.

### Settings fixed before running (a priori, no tuning)

| Model | Settings |
|---|---|
| Logistic regression | `StandardScaler` → `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)` |
| Decision tree | `DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)`, no pruning or depth tuning |

Neither model was tuned on this data; the settings were chosen in advance and reported as-is. Logistic regression (Cox, 1958) is a linear probabilistic classifier; the decision tree (Breiman et al., 1984) is a depth-limited axis-aligned rule set, included as the interpretable, non-linear counterpart. Depth 3 was chosen a priori as a shallow, interpretable tree rather than a tuned optimum. Software: scikit-learn 1.6.1 (Pedregosa et al., 2011).

## 6. Results

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

## 7. Verification

`python verify.py` re-checks the experiment and writes
`results/verification.json`. All 7 checks pass:

1. Re-running `run_comparison.py` produces byte-identical result files (SHA-256 level).
2. Each model has exactly 50 fold evaluations.
3. Both models were scored on matching train/test sizes for every fold.
4. Recorded folds match folds regenerated from seed 42 (one shared split set).
5. Scaler statistics match raw training-fold statistics exactly (max diff 0.0).
6. Scaler statistics differ from full-dataset statistics (no leakage from test folds).
7. Logistic regression beats the majority-class baseline on balanced accuracy.

## 8. Exploratory robustness check (post-hoc)

Run **after** the primary results were known, and labelled exploratory throughout (`ANALYSIS_LOG.md`). Two questions: was the gap an artefact of seed 42, and was it an artefact of the pre-specified `max_depth = 3`? Script: `robustness.py`.

### Seed sensitivity (pre-specified settings, 5 seeds × 50 folds each)

| Seed | LR balanced acc. | Tree balanced acc. | LR malignant recall | Tree malignant recall |
|---|---|---|---|---|
| 7 | 0.9721 | 0.9189 | 0.9579 | 0.8818 |
| 42 (primary) | 0.9746 | 0.9184 | 0.9576 | 0.8840 |
| 123 | 0.9760 | 0.9190 | 0.9615 | 0.8827 |
| 777 | 0.9732 | 0.9171 | 0.9580 | 0.8840 |
| 2026 | 0.9745 | 0.9178 | 0.9598 | 0.8830 |

Logistic regression scored higher on balanced accuracy and on malignant recall in **5 of 5 seeds**, with the balanced-accuracy gap ranging only from 0.0532 to 0.0570. The direction and size of the gap are not an artefact of the seed.

### Tree depth sensitivity (seed-42 folds, 50 folds)

| `max_depth` | Balanced accuracy | Malignant recall |
|---|---|---|
| 2 | 0.9133 ± 0.0289 | 0.8901 ± 0.0591 |
| 3 (pre-specified) | 0.9184 ± 0.0263 | 0.8840 ± 0.0515 |
| 4 | 0.9226 ± 0.0271 | 0.8953 ± 0.0518 |
| 5 | 0.9279 ± 0.0230 | 0.9062 ± 0.0477 |
| unlimited | 0.9253 ± 0.0279 | 0.9071 ± 0.0507 |

Deeper trees do improve on the pre-specified depth 3 (best observed: depth 5, balanced accuracy 0.9279), but even the best tree variant remains 0.047 below logistic regression (0.9746) on the same folds. Two caveats: this sweep both selects and evaluates the "best" depth on the same folds, which biases the tree's numbers *upward* (Varma and Simon, 2006) — so 0.9279 is a generous figure for the tree; and it is exploratory, not part of the pre-specified analysis.

Files: `results/robustness_seeds.csv`, `results/robustness_tree_depths.csv`.

## 9. Interpretation

- Under this evaluation setup, logistic regression with fold-wise standardisation scores higher than a depth-3 decision tree on every reported metric, with the largest advantage in malignant recall (+7.4 points) and the smallest in benign recall (+3.9 points).
- The dominant practical difference is error type: the tree produces roughly 2.7× as many missed malignant predictions as logistic regression.
- The tree's wider fold-to-fold spread indicates its performance is more sensitive to which samples land in the training fold, consistent with a small, hard-split model.
- Our ordering (logistic regression above decision tree) matches the single-split comparison of Akinwamide et al. (2025), and both models sit within the 92–99% band typical of published WDBC results — consistent with the view that WDBC is a "easy" benchmark where protocol differences move numbers as much as model differences do.
- The exploratory checks (Section 8) support the stability of the finding: the gap persisted across five seeds and under every tree depth tested, so it is not attributable to the particular seed or the depth-3 limit.

## 10. Limitations

- The 50 fold results are **not 50 independent experiments** — training sets overlap heavily — so the spreads above are described as variation across folds, not as a basis for a significance claim or a definitive statistical win. The gap is descriptive for this run (Bates, Hastie, and Tibshirani, 2024).
- The dataset is small (569 samples), comes from a single source (UCI WDBC, digitised fine-needle aspirates from one hospital system), and the conclusion is limited to these two models, this dataset, and this evaluation setup.
- No clinical usefulness is claimed. Results do not establish diagnostic performance, generalisation to other populations, or readiness for deployment.
- Settings were fixed rather than tuned, so the comparison reflects these pre-specified configurations, not each model's best achievable performance. A tuned tree or a different regularisation for logistic regression could narrow or widen the gap.
- The literature comparison in Section 2 is indicative rather than systematic; no formal systematic review was performed.
- The robustness check (Section 8) was exploratory and limited to the seeds and tree depths tested; it does not cover other model families (random forest, SVM, tuned logistic regression) or a nested design, and its depth sweep selects and evaluates on the same folds.

## 11. Conclusion

Under a pre-specified, leakage-controlled, repeated stratified 5-fold protocol on WDBC, logistic regression (`C=1.0`, with fold-wise standardisation) outperformed a depth-3 decision tree on all nine reported metrics. The practically important difference is in error structure: logistic regression identified 95.8% of malignant samples on average versus 88.4% for the tree, and its pooled false-negative rate was 4.2% versus 11.6%. The tree was also less stable across folds. These findings are descriptive for these two configurations, this dataset, and this evaluation setup; stronger claims would require external validation datasets, nested evaluation for any tuning, and uncertainty estimates that account for dependence between folds.

## 12. Reproduce

```
pip install -r requirements.txt
python run_comparison.py
python verify.py
python robustness.py
```

Requires Python 3.11+. Outputs go to `results/`.

## References

Akinwamide, S., Fele, T. and Ojo, O.A. (2025). Comparative evaluation of supervised machine learning algorithms for breast cancer prediction using the Wisconsin diagnostic dataset. *GJETA*. Available at: https://gjeta.com/sites/default/files/fulltext_pdf/GJETA-2025-0246.pdf [Accessed 5 Oct. 2026].

Bates, S., Hastie, T. and Tibshirani, R. (2024). Cross-validation: what does it estimate and how well does it do it? *Journal of the American Statistical Association*, 119(545), pp. 477–489.

Breiman, L., Friedman, J.H., Olshen, R.A. and Stone, C.J. (1984). *Classification and Regression Trees*. Belmont, CA: Wadsworth.

Chicco, D. and Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21, 6.

Cox, D.R. (1958). The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B*, 20(2), pp. 215–242.

Dietterich, T.G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), pp. 1895–1923.

Mangasarian, O.L., Street, W.N. and Wolberg, W.H. (1995). Breast cancer diagnosis and prognosis via linear programming. *Operations Research*, 43(4), pp. 570–577.

Pati, A., Panigrahi, A., Parhi, M., Giri, J., Qin, H., Mallik, S., Pattanayak, S.R. and Agrawal, U.K. (2024). Performance assessment of hybrid machine learning approaches for breast cancer and recurrence prediction. *PLOS ONE*, 19(8), e0304768. https://doi.org/10.1371/journal.pone.0304768

Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Street, W.N., Wolberg, W.H. and Mangasarian, O.L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology*, 1905, pp. 861–870. https://doi.org/10.1117/12.148698

UCI Machine Learning Repository (n.d.). *Breast Cancer Wisconsin (Diagnostic) Data Set*. https://doi.org/10.24432/C5DW2B [Accessed 5 Oct. 2026].

Varma, S. and Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.
