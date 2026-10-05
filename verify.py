import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

import run_comparison as exp

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

CHECKS = []


def record(name, passed, detail):
    CHECKS.append({"check": name, "passed": bool(passed), "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")


def check_determinism():
    before = {
        p.name: p.read_bytes()
        for p in sorted(RESULTS_DIR.glob("*"))
        if p.is_file()
    }
    subprocess.run(
        [sys.executable, str(ROOT / "run_comparison.py")],
        check=True,
        capture_output=True,
        cwd=ROOT,
    )
    after = {
        p.name: p.read_bytes()
        for p in sorted(RESULTS_DIR.glob("*"))
        if p.is_file()
    }
    identical = before == after and set(before) == set(after)
    record(
        "determinism_rerun",
        identical,
        f"{len(after)} result files byte-identical after rerun"
        if identical
        else f"files differ: {set(before) ^ set(after) or 'content mismatch'}",
    )


def check_fold_identity():
    df = pd.read_csv(RESULTS_DIR / "fold_metrics.csv")
    n_expected = exp.N_SPLITS * exp.N_REPEATS
    counts = df.groupby("model").size()
    sizes_ok = True
    for _, group in df.groupby("model"):
        sizes_ok &= len(group) == n_expected
    pivot = df.pivot_table(
        index=["repeat", "split"],
        columns="model",
        values=["n_train", "n_test"],
        aggfunc="first",
    )
    same_sizes = bool(
        (pivot["n_train"].nunique(axis=1) == 1).all()
        and (pivot["n_test"].nunique(axis=1) == 1).all()
    )

    data = load_breast_cancer(as_frame=True)
    folds = exp.build_folds(data.target)
    regenerated = sorted(
        (len(tr), len(te)) for tr, te in folds
    )
    recorded = sorted(
        (r.n_train, r.n_test)
        for r in df[df.model == df.model.iloc[0]].itertuples()
    )
    matches_seed = regenerated == recorded

    record(
        "fold_count_per_model",
        bool((counts == n_expected).all()) and sizes_ok,
        f"each model has {n_expected} fold evaluations: "
        + ", ".join(f"{k}={v}" for k, v in counts.items()),
    )
    record(
        "identical_test_sizes_per_fold",
        same_sizes,
        "both models evaluated on matching train/test sizes for every fold",
    )
    record(
        "folds_match_recorded_seed",
        matches_seed,
        "recorded fold sizes match folds regenerated from seed "
        f"{exp.SEED} (shared by both models)",
    )


def check_scaler_leakage():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    folds = exp.build_folds(y)
    pipeline = clone(exp.MODELS["logistic_regression"])
    scaler = pipeline.named_steps["scaler"]

    max_train_diff = 0.0
    matches_full_data = True
    for train_idx, test_idx in folds[:10]:
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_mean = X.iloc[train_idx].mean().to_numpy()
        train_std = X.iloc[train_idx].std(ddof=0).to_numpy()
        full_mean = X.mean().to_numpy()
        max_train_diff = max(
            max_train_diff,
            float(np.abs(scaler.mean_ - train_mean).max()),
            float(np.abs(scaler.scale_ - train_std).max()),
        )
        matches_full_data &= bool(
            np.allclose(scaler.mean_, full_mean, atol=1e-9)
        )

    record(
        "scaler_matches_training_fold",
        max_train_diff < 1e-9,
        f"max |scaler stat - train-fold stat| = {max_train_diff:.2e} over 10 folds",
    )
    record(
        "scaler_not_fit_on_full_data",
        not matches_full_data,
        "scaler statistics differ from full-dataset statistics (not leaked)",
    )


def check_baselines():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    folds = exp.build_folds(y)

    strategies = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "stratified_random": DummyClassifier(
            strategy="stratified", random_state=exp.SEED
        ),
    }
    rows = []
    for name, base in strategies.items():
        accs, baccs, mrecs = [], [], []
        for train_idx, test_idx in folds:
            model = clone(base)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict(X.iloc[test_idx])
            y_true = y.iloc[test_idx]
            accs.append(accuracy_score(y_true, y_pred))
            baccs.append(balanced_accuracy_score(y_true, y_pred))
            mrecs.append(
                recall_score(y_true, y_pred, pos_label=exp.MALIGANT_LABEL)
            )
        rows.append(
            {
                "model": name,
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs, ddof=1),
                "balanced_accuracy_mean": np.mean(baccs),
                "balanced_accuracy_std": np.std(baccs, ddof=1),
                "malignant_recall_mean": np.mean(mrecs),
                "malignant_recall_std": np.std(mrecs, ddof=1),
            }
        )

    baseline_df = pd.DataFrame(rows).round(4)
    baseline_df.to_csv(RESULTS_DIR / "baselines.csv", index=False)

    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    lr_bacc = summary[
        (summary.model == "logistic_regression")
        & (summary.metric == "balanced_accuracy")
    ]["mean"].iloc[0]
    majority = baseline_df[baseline_df.model == "majority_class"].iloc[0]
    beats_majority = lr_bacc > majority.balanced_accuracy_mean
    record(
        "beats_majority_baseline",
        bool(beats_majority),
        f"LR balanced accuracy {lr_bacc:.4f} > majority {majority.balanced_accuracy_mean:.4f}",
    )
    print("\nBaselines (same 50 folds):")
    print(baseline_df.to_string(index=False))


def main():
    print("Verifying", RESULTS_DIR, "\n")
    check_determinism()
    check_fold_identity()
    check_scaler_leakage()
    check_baselines()

    (RESULTS_DIR / "verification.json").write_text(
        json.dumps(CHECKS, indent=2), encoding="utf-8"
    )

    failed = [c for c in CHECKS if not c["passed"]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
