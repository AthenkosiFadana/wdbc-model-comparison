# Portfolio & resume material

## Portfolio project entry

**Title:** Fair Model Comparison: Logistic Regression vs Decision Tree on the
Wisconsin Diagnostic Breast Cancer Dataset

**One-paragraph description:**

A self-directed research project on *how* to compare machine learning models,
not just which one wins. Two classic classifiers were evaluated on the WDBC
dataset (569 samples, 30 features) under a pre-specified, leakage-controlled
protocol: 10×5 repeated stratified cross-validation with one shared set of 50
folds per model, standardisation fitted inside the pipeline per fold, and all
hyperparameters fixed before the first run. Logistic regression outperformed
the depth-3 decision tree on all nine metrics, with malignant recall at 95.8%
vs 88.4% and a pooled false-negative rate of 4.2% vs 11.6%. The result was
stress-tested across 5 random seeds and a tree depth sweep, verified by 7
automated checks (determinism, fold identity, leakage), and documented from
start to finish in a dated analysis log separating a-priori decisions from
post-hoc ones. Full report includes related work, baselines, and limitations.

**Tech / keywords:**
Python, scikit-learn, pandas, NumPy, logistic regression, decision trees,
repeated stratified k-fold cross-validation, data-leakage prevention,
pipeline preprocessing, class-imbalance-aware metrics (balanced accuracy,
macro F1, malignant recall), confusion-matrix error analysis, baselines,
reproducibility, git, automated verification.

**Links:** GitHub repo · full report (`REPORT.md`) · analysis log
(`ANALYSIS_LOG.md`)

---

## Resume bullets (pick 4–5)

- Designed and executed a reproducible model-comparison study (logistic
  regression vs decision tree, WDBC dataset: 569 samples / 30 features) using
  10×5 repeated stratified cross-validation — 50 paired evaluations per model
  on one shared, fixed-seed fold set.
- Prevented data leakage by fitting standardisation inside the
  cross-validation pipeline per training fold; confirmed with automated
  verification checks (determinism, fold provenance, scaler-fit statistics) —
  7/7 passing.
- Evaluated models with macro precision/recall/F1, balanced accuracy, and
  malignant-class recall; pooled confusion-matrix analysis showed 4.2% vs
  11.6% false-negative rates on malignant samples.
- Demonstrated result robustness via post-hoc stress tests (5 random seeds, a
  tree-depth sweep) and documented every decision in a dated analysis log
  separating pre-registered choices from exploratory ones.
- Built a fully reproducible Python project from scratch: pinned dependencies,
  deterministic outputs, CI-ready scripts, baseline comparisons, and a
  research write-up with related work and references.

**One-liner (for a profile or cover letter):**

"Built a fully reproducible, leakage-controlled model-comparison study on the
Wisconsin breast cancer dataset — including pre-specified settings, shared
cross-validation folds, automated verification, and honest limitations — as a
practitioner's approach to research methodology."

---

## Suggested talking points (if asked about it in an interview)

1. **Leakage:** why standardisation belongs inside the pipeline, and how I
   proved it was (scaler stats match the training fold exactly, differ from
   full-data stats).
2. **Fairness:** shared folds — both models sit the same exam.
3. **Honest statistics:** 50 folds ≠ 50 experiments; reported as variation,
   no significance claims.
4. **Error structure over headline accuracy:** the interesting difference was
   false negatives (missed malignancies), not the 5-point accuracy gap.
5. **Process discipline:** dated analysis log, a-priori vs post-hoc,
   verification script anyone can re-run.
