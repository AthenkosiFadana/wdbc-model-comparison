import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import run_comparison as exp

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

SEEDS = [7, 42, 123, 2026, 777]
DEPTHS = [2, 3, 4, 5, None]
METRICS = ["balanced_accuracy", "f1_macro", "malignant_recall"]


def make_folds(y, seed):
    cv = RepeatedStratifiedKFold(
        n_splits=exp.N_SPLITS, n_repeats=exp.N_REPEATS, random_state=seed
    )
    return list(cv.split(np.zeros(len(y)), y))


def evaluate(estimator, X, y, folds):
    scores = {m: [] for m in METRICS}
    for train_idx, test_idx in folds:
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        row = exp.score_predictions(y.iloc[test_idx], y_pred)
        for m in METRICS:
            scores[m].append(row[m])
    return {
        f"{m}_mean": float(np.mean(v)) for m, v in scores.items()
    } | {
        f"{m}_std": float(np.std(v, ddof=1)) for m, v in scores.items()
    }


def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    lr = exp.MODELS["logistic_regression"]

    seed_folds = {s: make_folds(y, s) for s in SEEDS}

    seed_rows = []
    for s in SEEDS:
        lr_scores = evaluate(lr, X, y, seed_folds[s])
        tree = clone(exp.MODELS["decision_tree"])
        tree_scores = evaluate(tree, X, y, seed_folds[s])
        seed_rows.append({"seed": s, "model": "logistic_regression", **lr_scores})
        seed_rows.append({"seed": s, "model": "decision_tree", **tree_scores})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(RESULTS_DIR / "robustness_seeds.csv", index=False)

    depth_folds = seed_folds[exp.SEED]
    depth_rows = []
    for depth in DEPTHS:
        tree = DecisionTreeClassifier(
            criterion="gini", max_depth=depth, random_state=exp.SEED
        )
        label = "unlimited" if depth is None else str(depth)
        scores = evaluate(tree, X, y, depth_folds)
        depth_rows.append(
            {
                "max_depth": label,
                "n_leaves_bound": "none" if depth is None else 2 ** depth,
                **scores,
            }
        )
    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(RESULTS_DIR / "robustness_tree_depths.csv", index=False)

    print("=== Seed sensitivity (pre-specified settings, 5 seeds x 50 folds) ===")
    pivot = seed_df.pivot(
        index="seed",
        columns="model",
        values=["balanced_accuracy_mean", "malignant_recall_mean"],
    ).round(4)
    print(pivot.to_string())

    lr_ba = seed_df[seed_df.model == "logistic_regression"][
        "balanced_accuracy_mean"
    ].to_numpy()
    tree_ba = seed_df[seed_df.model == "decision_tree"][
        "balanced_accuracy_mean"
    ].to_numpy()
    lr_mr = seed_df[seed_df.model == "logistic_regression"][
        "malignant_recall_mean"
    ].to_numpy()
    tree_mr = seed_df[seed_df.model == "decision_tree"][
        "malignant_recall_mean"
    ].to_numpy()

    wins = int((lr_ba > tree_ba).sum()), int((lr_mr > tree_mr).sum())
    print(
        f"\nLR > tree on balanced accuracy in {wins[0]}/{len(SEEDS)} seeds, "
        f"on malignant recall in {wins[1]}/{len(SEEDS)} seeds"
    )
    print(f"Balanced accuracy gap (LR - tree) across seeds: "
          f"min {(lr_ba - tree_ba).min():.4f}, max {(lr_ba - tree_ba).max():.4f}")

    print("\n=== Tree depth sensitivity (seed-42 folds, 50 folds) ===")
    print(
        depth_df[
            [
                "max_depth",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "malignant_recall_mean",
                "malignant_recall_std",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    best_tree = depth_df.loc[depth_df.balanced_accuracy_mean.idxmax()]
    print(
        f"\nBest tree depth on these folds: {best_tree['max_depth']} "
        f"(balanced accuracy {best_tree.balanced_accuracy_mean:.4f}) "
        f"vs logistic regression {lr_ba[SEEDS.index(exp.SEED)]:.4f}"
    )

    lr_still_wins = bool(
        lr_ba[SEEDS.index(exp.SEED)] > best_tree.balanced_accuracy_mean
    )
    print(
        "LR still above best tree depth on seed-42 folds: "
        + ("yes" if lr_still_wins else "NO")
    )
    print("\nWrote", RESULTS_DIR / "robustness_seeds.csv")
    print("Wrote", RESULTS_DIR / "robustness_tree_depths.csv")


if __name__ == "__main__":
    main()
