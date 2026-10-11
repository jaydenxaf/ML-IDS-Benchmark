"""Tests on small synthetic data. They check the evaluation logic, not the
real CIC-IDS2017 results."""
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from src import ids_eval


def make_blobs_imbalanced(seed=0, counts=(3000, 150, 60, 12), d=8):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(4, d))
    X = np.vstack(
        [centers[k] + rng.normal(0, 1, size=(n, d)) for k, n in enumerate(counts)]
    )
    y = np.concatenate([np.full(n, k) for k, n in enumerate(counts)])
    return X, y, centers


def test_map_label_handles_corrupted_dash():
    assert ids_eval.map_label("BENIGN") == 0
    assert ids_eval.map_label("Web Attack � Brute Force") == 1
    assert ids_eval.map_label("Web Attack – XSS") == 2
    assert ids_eval.map_label("Web Attack � Sql Injection") == 3
    assert np.isnan(ids_eval.map_label("DDoS"))
    assert np.isnan(ids_eval.map_label(np.nan))


def test_majority_baseline_equals_benign_share():
    _, y, _ = make_blobs_imbalanced()
    base = ids_eval.majority_baseline(y)
    assert base["Accuracy"] == pytest.approx((y == 0).mean())
    assert base["Recall Benign"] == 1.0
    assert base["Recall SQL Injection"] == 0.0


def test_cluster_mapper_fixes_label_permutation():
    X, y, centers = make_blobs_imbalanced()
    # Start the centers permuted so cluster ids do NOT line up with label ids.
    init = centers[[1, 2, 3, 0]]
    raw = KMeans(n_clusters=4, init=init, n_init=1, random_state=0).fit(X).predict(X)
    assert accuracy_score(y, raw) < 0.2  # raw cluster ids look terrible

    mapped = ids_eval.ClusterLabelMapper(
        KMeans(n_clusters=4, init=init, n_init=1, random_state=0)
    ).fit(X, y).predict(X)
    assert accuracy_score(y, mapped) > 0.99  # same clusters, correctly named


def test_loader_cleans_and_dedupes(tmp_path):
    text = (
        " Flow Duration, Feature B, Label\n"
        "1,2,BENIGN\n"
        "3,4,BENIGN\n"
        "3,4,BENIGN\n"  # exact duplicate
        "inf,5,BENIGN\n"  # dropped (inf)
        "6,,BENIGN\n"  # dropped (NaN)
        "7,8,Web Attack – Brute Force\n"
        "9,10,Web Attack – XSS\n"
        "11,12,Web Attack – Sql Injection\n"
        "13,14,DDoS\n"  # not a target class
    )
    path = tmp_path / "toy.csv"
    path.write_bytes(text.encode("cp1252"))  # en dash is byte 0x96, invalid UTF-8

    X, y, info = ids_eval.load_web_attacks(path)
    assert X.shape == (6, 2)
    assert info["duplicate_rows"] == 1
    assert info["class_counts"] == {
        "Benign": 3, "Brute Force": 1, "XSS": 1, "SQL Injection": 1,
    }

    X2, y2, info2 = ids_eval.load_web_attacks(path, dedupe=True)
    assert X2.shape == (5, 2) and info2["rows_used"] == 5


def test_evaluate_cv_reports_per_class_metrics():
    X, y, _ = make_blobs_imbalanced()
    models = ids_eval.build_models(X.shape[1])
    keep = {k: v for k, v in models.items() if k in {"Decision Tree", "K-Means"}}
    table, matrices = ids_eval.evaluate_cv(keep, X, y)

    assert table["Algorithm"].tolist()[0].startswith("Majority class")
    for col in ["Accuracy", "Macro-F1", "Recall SQL Injection", "T1(s)"]:
        assert col in table.columns
    tree = table.set_index("Algorithm").loc["Decision Tree"]
    assert tree["Accuracy"] > 0.95
    assert tree["Recall Benign"] > 0.95
    assert matrices["Decision Tree"].sum() == len(y)  # every row predicted once
