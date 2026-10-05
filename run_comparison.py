import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
N_SPLITS = 5
N_REPEATS = 10
MALIGANT_LABEL = 0
BENIGN_LABEL = 1

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

MODELS = {
    "logistic_regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    ),
    "decision_tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=SEED,
    ),
}


def build_folds(y):
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    return list(cv.split(np.zeros(len(y)), y))


def score_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "malignant_precision": precision_score(
            y_true, y_pred, pos_label=MALIGANT_LABEL
        ),
        "malignant_recall": recall_score(
            y_true, y_pred, pos_label=MALIGANT_LABEL
        ),
        "malignant_f1": f1_score(y_true, y_pred, pos_label=MALIGANT_LABEL),
        "benign_recall": recall_score(y_true, y_pred, pos_label=BENIGN_LABEL),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    dataset_info = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_counts": {
            "malignant": int((y == MALIGANT_LABEL).sum()),
            "benign": int((y == BENIGN_LABEL).sum()),
        },
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "n_fold_evaluations": N_SPLITS * N_REPEATS,
        "model_settings": {
            "logistic_regression": {
                "pipeline": "StandardScaler -> LogisticRegression",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": SEED,
                "standardization_fit_on_training_fold_only": True,
            },
            "decision_tree": {
                "criterion": "gini",
                "max_depth": 3,
                "random_state": SEED,
                "no_pruning_tuning": True,
            },
        },
    }

    folds = build_folds(y)

    rows = []
    cms = {name: np.zeros((2, 2), dtype=int) for name in MODELS}

    for model_name, estimator in MODELS.items():
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            repeat = fold_idx // N_SPLITS + 1
            split = fold_idx % N_SPLITS + 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = clone(estimator)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores = score_predictions(y_test, y_pred)
            rows.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "split": split,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **scores,
                }
            )
            cms[model_name] += confusion_matrix(
                y_test, y_pred, labels=[MALIGANT_LABEL, BENIGN_LABEL]
            )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "malignant_precision",
        "malignant_recall",
        "malignant_f1",
        "benign_recall",
    ]

    summary_rows = []
    for model_name, group in fold_df.groupby("model", sort=False):
        for metric in metric_cols:
            values = group[metric].to_numpy()
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std_across_folds": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    cm_rows = []
    label_names = ["malignant", "benign"]
    for model_name, cm in cms.items():
        for i, true_label in enumerate(label_names):
            for j, pred_label in enumerate(label_names):
                cm_rows.append(
                    {
                        "model": model_name,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(cm[i, j]),
                    }
                )
    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(RESULTS_DIR / "confusion_matrices.csv", index=False)

    pooled = (
        fold_df.groupby("model", sort=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    print(json.dumps(dataset_info, indent=2))
    print("\n=== Per-model metrics across 50 folds (mean +/- std) ===")
    for model_name in MODELS:
        print(f"\n{model_name}")
        sub = summary_df[summary_df["model"] == model_name]
        for _, r in sub.iterrows():
            print(
                f"  {r['metric']:<22} {r['mean']:.4f} +/- {r['std_across_folds']:.4f}"
                f"  (min {r['min']:.4f}, max {r['max']:.4f})"
            )

    print("\n=== Confusion matrices pooled over 50 folds ===")
    for model_name, cm in cms.items():
        print(f"\n{model_name} (rows = true, cols = predicted)")
        print(f"{'':>12} pred_malignant  pred_benign")
        print(
            f"{'true_malign':>12} {cm[0, 0]:>14} {cm[0, 1]:>12}"
        )
        print(f"{'true_benign':>12} {cm[1, 0]:>14} {cm[1, 1]:>12}")

    (RESULTS_DIR / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2), encoding="utf-8"
    )

    print("\nWrote results to", RESULTS_DIR)
    return pooled


if __name__ == "__main__":
    main()
