# Post: plain-language abstract

Ready to paste into LinkedIn, a blog, GitHub README, or a portfolio entry.

---

## The short version

Two of the most classic machine learning models — logistic regression and a
decision tree — were put head-to-head on a well-known breast tumour dataset,
using a deliberately careful evaluation: 50 repeated cross-validation runs per
model, identical data splits for both models, and every setting fixed in
advance before any results were seen.

Logistic regression came out ahead on every measure. Most importantly, it
correctly identified **95.8% of malignant samples** on average, versus **88.4%
for the decision tree**. Pooled across all runs, it missed 4.2% of malignant
predictions compared with the tree's 11.6%.

The result wasn't a fluke of one lucky setup: it held across 5 different random
seeds, and no decision tree depth I tested closed the gap.

The claim is deliberately narrow: these two model settings, this dataset, this
evaluation setup. It says nothing about clinical diagnosis — but it's a clean,
fully reproducible example of how to compare models honestly, including the
part most write-ups skip: saying exactly what the numbers don't mean.

---

## Longer version (for a blog post intro)

Most model comparisons you read report one accuracy number and a winner. This
project started from a different question: *how much of a "win" is the model,
and how much is the evaluation?*

I compared logistic regression against a decision tree on the Wisconsin
Diagnostic Breast Cancer dataset (569 samples, 30 numeric features, labels:
benign or malignant). The setup was designed to be as fair and leakage-free as
I could make it:

- **Same exam for both models.** One shared set of 50 stratified cross-validation
  folds (5 folds × 10 repeats, fixed seed), so every comparison is paired.
- **No peeking.** Standardisation was fitted only on each fold's training data,
  inside the pipeline. Settings were fixed before the first run and never tuned
  on the evaluation data.
- **Honest reporting.** Every metric reported as mean ± variation across folds,
  with an explicit note that the 50 folds are *not* 50 independent experiments.
- **Checked, not assumed.** Seven automated verification checks (determinism,
  shared folds, no leakage), plus majority-class baselines so the scores have a
  reference point.

**Results:** logistic regression scored higher on all nine metrics — balanced
accuracy 0.975 vs 0.918, and malignant recall (the share of malignant samples
caught) 0.958 vs 0.884. Its error pattern mattered too: pooled over all folds,
it labelled 4.2% of malignant samples as benign, versus 11.6% for the tree.

**Robustness:** the gap held in 5 of 5 random seeds (range 0.053–0.057 on
balanced accuracy), and the best-performing tree depth I tested still finished
~0.05 below logistic regression.

**What it doesn't mean:** no clinical claim, no "best model" claim — just a
careful, reproducible answer to a narrow question, with every decision and
caveat written down.

---

## Hashtags / tags

machine-learning, model-evaluation, cross-validation, data-leakage,
reproducible-research, logistic-regression, decision-tree, scikit-learn,
breast-cancer-dataset, data-science
