"""Extended evaluation for the CIC-IDS2017 web-attack ML-AIDS benchmark.

The original notebook (notebooks/Final_Project.ipynb) is kept unchanged. This
module re-runs the same 13 model configurations and adds the checks that
accuracy-style metrics cannot show on a heavily imbalanced dataset:

* a majority-class baseline row (always predict "Benign"),
* per-class recall and F1, plus macro-F1,
* the scaler is fit inside each CV fold (the notebook fits it on all data),
* K-Means / EM / SOM cluster ids are mapped to class labels by majority vote
  on the training fold. Raw cluster ids have no relation to the label ids
  0-3, so comparing them directly (as the notebook does) is not meaningful,
* a duplicate-row count and an optional ``--dedupe`` run,
* a flag when a model hit its iteration cap (ConvergenceWarning).

Usage (from the repo root):

    python -m src.ids_eval --csv data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    python -m src.ids_eval --csv <csv> --dedupe --models forest,tree
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASS_NAMES = ["Benign", "Brute Force", "XSS", "SQL Injection"]
CLASS_IDS = [0, 1, 2, 3]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def map_label(label):
    """Map a raw CIC-IDS2017 label to a class id 0-3 (NaN for anything else).

    Substring matching is deliberate. Some copies of the CSV (e.g. on Kaggle)
    corrupt the dash in "Web Attack - Brute Force" into a different character,
    so an exact-string match silently drops every attack row.
    """
    if not isinstance(label, str):
        return np.nan
    if label == "BENIGN":
        return 0
    if "Brute Force" in label:
        return 1
    if "XSS" in label:
        return 2
    if "Sql Injection" in label:
        return 3
    return np.nan


def load_web_attacks(csv_path, dedupe=False):
    """Load and clean the CSV. Returns (X, y, info).

    X is *unscaled*: scaling happens inside each CV fold (see make_preprocessor).
    Cleaning matches the notebook: strip column names, replace +/-inf with NaN,
    drop rows with NaN, keep only the four target classes.
    """
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["Label"] = df["Label"].map(map_label)
    df = df[df["Label"].notna()].copy()
    df["Label"] = df["Label"].astype(int)

    rows_after_cleaning = len(df)
    duplicate_rows = int(df.duplicated().sum())
    if dedupe:
        df = df.drop_duplicates()

    y = df["Label"].to_numpy()
    X = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
    info = {
        "rows_after_cleaning": rows_after_cleaning,
        "duplicate_rows": duplicate_rows,
        "deduplicated": bool(dedupe),
        "rows_used": int(len(df)),
        "n_features": int(X.shape[1]),
        "class_counts": {
            CLASS_NAMES[i]: int((y == i).sum()) for i in CLASS_IDS
        },
    }
    return X.to_numpy(dtype=float), y, info


def _clip(a):
    return np.clip(a, -3.0, 3.0)


def make_preprocessor():
    """StandardScaler followed by clipping to [-3, 3] (the paper's interval)."""
    return Pipeline(
        [("scale", StandardScaler()), ("clip", FunctionTransformer(_clip))]
    )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class ClusterLabelMapper(BaseEstimator, ClassifierMixin):
    """Wrap an unsupervised model so its output can be scored against labels.

    The wrapped model is fit on features only. Labels are used afterwards,
    on the training fold only, to name each cluster after the majority class
    among the training points assigned to it. Test labels are never used.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        try:
            self.model_ = clone(self.estimator)
        except TypeError:  # estimator that is not fully sklearn-compatible
            self.model_ = copy.deepcopy(self.estimator)
        self.model_.fit(X)
        clusters = np.asarray(self.model_.predict(X))
        y = np.asarray(y)
        self.mapping_ = {}
        for c in np.unique(clusters):
            labels, counts = np.unique(y[clusters == c], return_counts=True)
            self.mapping_[int(c)] = int(labels[np.argmax(counts)])
        self.default_ = int(np.bincount(y).argmax())
        return self

    def predict(self, X):
        clusters = np.asarray(self.model_.predict(X))
        return np.array(
            [self.mapping_.get(int(c), self.default_) for c in clusters]
        )


def build_models(n_features, seed=42):
    """The notebook's 13 configurations (same hyperparameters)."""
    models = {
        "ANN": MLPClassifier(max_iter=500, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=None, class_weight="balanced",
            random_state=seed,
        ),
    }
    for k in range(1, 6):
        models[f"k-NN (k={k})"] = KNeighborsClassifier(n_neighbors=k)
    models["Naive Bayes"] = GaussianNB()
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=100, max_depth=None, class_weight="balanced",
        random_state=seed,
    )
    models["SVM"] = SVC(
        kernel="rbf", class_weight="balanced", max_iter=1000, random_state=seed
    )
    models["K-Means"] = ClusterLabelMapper(KMeans(n_clusters=4, random_state=seed))
    models["EM"] = ClusterLabelMapper(
        GaussianMixture(n_components=4, random_state=seed)
    )
    try:
        from sklearn_som.som import SOM

        models["SOM"] = ClusterLabelMapper(SOM(m=2, n=2, dim=n_features))
    except ImportError:
        print("sklearn-som not installed; skipping SOM (pip install sklearn-som)")
    return models


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def summarize(y_true, y_pred):
    """Overall and per-class metrics for one set of predictions."""
    _, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_IDS, zero_division=0
    )
    row = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(
            y_true, y_pred, labels=CLASS_IDS, average="macro", zero_division=0
        ),
        "Weighted-F1": f1_score(
            y_true, y_pred, labels=CLASS_IDS, average="weighted", zero_division=0
        ),
    }
    for i, name in enumerate(CLASS_NAMES):
        row[f"Recall {name}"] = recall[i]
    for i, name in enumerate(CLASS_NAMES):
        row[f"F1 {name}"] = f1[i]
    return row


def majority_baseline(y):
    """Metrics for a classifier that always predicts the most common class."""
    y = np.asarray(y)
    pred = np.full_like(y, np.bincount(y).argmax())
    return summarize(y, pred)


def evaluate_cv(models, X, y, n_splits=5, seed=42):
    """Stratified k-fold CV with out-of-fold predictions pooled for metrics.

    Returns (results DataFrame, {model name: confusion matrix}).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows, matrices = [], {}

    base = majority_baseline(y)
    rows.append({"Algorithm": "Majority class (always Benign)", **base})

    for name, model in models.items():
        print(f"Evaluating {name}", flush=True)
        y_true_all, y_pred_all = [], []
        t_fit, t_pred = [], []
        hit_cap = False
        for train_idx, test_idx in skf.split(X, y):
            pre = make_preprocessor()
            X_train = pre.fit_transform(X[train_idx])
            X_test = pre.transform(X[test_idx])
            est = clone(model)

            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                est.fit(X_train, y[train_idx])
            t_fit.append(time.perf_counter() - t0)
            hit_cap |= any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )

            t0 = time.perf_counter()
            pred = est.predict(X_test)
            t_pred.append(time.perf_counter() - t0)

            y_true_all.append(y[test_idx])
            y_pred_all.append(np.asarray(pred))

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        rows.append(
            {
                "Algorithm": name,
                **summarize(y_true_all, y_pred_all),
                "T1(s)": float(np.mean(t_fit)),
                "T2(s)": float(np.mean(t_pred)),
                "Hit iteration cap": hit_cap,
            }
        )
        matrices[name] = confusion_matrix(
            y_true_all, y_pred_all, labels=CLASS_IDS
        )
    return pd.DataFrame(rows), matrices


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True, help="path to the Thursday web-attacks CSV")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--dedupe", action="store_true", help="drop exact duplicate rows first")
    ap.add_argument(
        "--models", default="",
        help="comma-separated name fragments to run (e.g. 'forest,tree'); default: all",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    X, y, info = load_web_attacks(args.csv, dedupe=args.dedupe)
    print(json.dumps(info, indent=2))

    models = build_models(X.shape[1], seed=args.seed)
    wanted = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if wanted:
        models = {
            n: m for n, m in models.items() if any(w in n.lower() for w in wanted)
        }
        if not models:
            raise SystemExit(f"No model name matches: {args.models}")

    table, matrices = evaluate_cv(
        models, X, y, n_splits=args.folds, seed=args.seed
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_dedup" if args.dedupe else ""
    table.round(4).to_csv(out / f"extended_cv{suffix}.csv", index=False)
    with open(out / f"confusion_matrices{suffix}.json", "w") as f:
        json.dump(
            {
                "labels": CLASS_NAMES,
                "matrices": {k: v.tolist() for k, v in matrices.items()},
            },
            f, indent=2,
        )
    with open(out / f"data_report{suffix}.json", "w") as f:
        json.dump(info, f, indent=2)

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.round(4).to_string(index=False))
    print(f"\nWrote results to {out.resolve()}")


if __name__ == "__main__":
    main()
