# Analysis log

Dated record of decisions. Entries marked **a priori** were made before any
results were seen; entries marked **post-hoc** were added after the first run.

## 2026-10-05 — a priori (before first run)

- **Research question fixed:** compare logistic regression vs decision tree on
  WDBC under repeated stratified 5-fold CV; breast tumour classification is the
  case study, the model comparison is the project.
- **Evaluation design fixed:** 5 folds x 10 repeats, `random_state = 42`
  (50 evaluations per model). The 50 splits are materialised once from a single
  seed and shared by both models so every fold comparison is paired.
- **Model settings fixed, no tuning:**
  - Logistic regression: `StandardScaler -> LogisticRegression(C=1.0,
    solver="lbfgs", max_iter=1000, random_state=42)`, scaler inside the
    pipeline so it fits on each training fold only.
  - Decision tree: `criterion="gini", max_depth=3, random_state=42`, no
    pruning or depth search.
  - Rationale: report pre-specified configurations rather than tune on the
    evaluation data; tuning would require a nested CV design, which is out of
    scope for this first comparison.
- **Metrics fixed:** accuracy, balanced accuracy, macro precision/recall/F1,
  plus malignant-class precision/recall/F1 and benign recall reported
  separately. Confusion matrices pooled over all 50 folds.
- **Class labelling fixed:** malignant = 0 (212 samples), benign = 1 (357
  samples). Malignant recall called out explicitly (costlier error).
- **Reporting rules fixed:** describe spread as variation across folds; do not
  treat the 50 folds as independent experiments; no significance claims; no
  clinical claims; conclusion limited to these models, this dataset, this
  setup.

## 2026-10-05 — first run

- Experiment executed as specified; results written to `results/`.
- Headline: logistic regression higher on every metric; largest gap in
  malignant recall (0.9576 vs 0.8840).

## 2026-10-05 — post-hoc (after first run, verification phase)

Added at the user's request as quality control, clearly after seeing results:

- **Determinism check:** re-ran `run_comparison.py` and compared SHA-256 of
  all result files against the first run.
- **Fold-identity check:** confirmed both models were scored on the identical
  50 splits (same train/test indices per fold, same per-fold test sizes).
- **Leakage check:** fitted the logistic-regression pipeline scaler on a
  training fold and compared its `mean_`/`scale_` against raw training-fold
  statistics and against full-dataset statistics; must match training-fold and
  differ from full-data.
- **Baselines:** evaluated majority-class and stratified-dummy predictors on
  the same 50 folds (`results/baselines.csv`) so model scores have a reference
  point. Baselines were not part of the original plan.

Nothing above changed the experiment, the model settings, or the reported
metrics for the two compared models.

- Outcome: 7/7 checks pass (`results/verification.json`).
- One bug found during verification was in the *check itself* (the fold-size
  assertion compared train size against test size instead of model against
  model); the check was corrected and re-run. The experiment code was not
  changed.
- Baseline scores: majority class balanced accuracy 0.5000, malignant recall
  0.0000; stratified random 0.5115 ± 0.0402 balanced accuracy. Both compared
  models far exceed these.
