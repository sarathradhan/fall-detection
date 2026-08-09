from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.severity_anomaly import (
    ANOMALY_CONTAMINATION,
    ANOMALY_N_ESTIMATORS,
    compute_anomaly_flags,
    compute_anomaly_scores,
    fit_isolation_forest,
)
from src.data.severity_labeling import (
    NON_APPLICABLE,
    SEVERITY_LABELS,
    assign_severity_labels,
    compute_cluster_statistics,
    extract_window_features,
    fit_feature_scaler,
    fit_kmeans,
    map_clusters_to_severity,
)


def test_isolation_forest_train_only_and_deterministic() -> None:
    train_features = np.vstack([np.zeros((10, 58)), np.ones((10, 58))])
    model = fit_isolation_forest(train_features, contamination=0.1, random_state=42, n_estimators=10)

    val_features = np.full((5, 58), 2.0)
    scores_1 = compute_anomaly_scores(model, val_features)
    flags_1 = compute_anomaly_flags(model, val_features)

    model_2 = fit_isolation_forest(train_features, contamination=0.1, random_state=42, n_estimators=10)
    scores_2 = compute_anomaly_scores(model_2, val_features)
    flags_2 = compute_anomaly_flags(model_2, val_features)

    assert isinstance(model, IsolationForest)
    assert scores_1.shape == (5,)
    assert flags_1.shape == (5,)
    assert np.array_equal(flags_1, flags_2)
    assert np.all(np.isfinite(scores_1))


def test_kmeans_severity_labels_remain_unchanged_after_anomaly_flagging() -> None:
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    fall_indices = np.array([1, 2], dtype=np.int64)
    cluster_ids = np.array([1, 0], dtype=np.int64)
    cluster_to_severity = {0: 1, 1: 2}

    severity = assign_severity_labels(labels, fall_indices, cluster_ids, cluster_to_severity)
    refined = np.array(severity, copy=True)
    anomaly_flags = np.array([0, 1], dtype=np.int64)
    anomaly_mask = np.zeros_like(refined, dtype=bool)
    anomaly_mask[fall_indices] = anomaly_flags == 1
    refined[anomaly_mask] = NON_APPLICABLE

    assert severity.tolist() == [NON_APPLICABLE, 2, 1, NON_APPLICABLE]
    assert refined.tolist() == [NON_APPLICABLE, 2, NON_APPLICABLE, NON_APPLICABLE]
    assert severity[1] == 2
    assert refined[1] == 2
    assert refined[2] == NON_APPLICABLE


def test_anomaly_scores_and_flags_are_produced_for_all_fall_windows() -> None:
    windows = np.tile(np.arange(64 * 6, dtype=np.float64).reshape(1, 64, 6), (5, 1, 1))
    features, _ = extract_window_features(windows)
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    model = fit_isolation_forest(scaled, contamination=0.1, random_state=42, n_estimators=10)

    scores = compute_anomaly_scores(model, scaled)
    flags = compute_anomaly_flags(model, scaled)

    assert scores.shape == (5,)
    assert flags.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert set(flags.tolist()).issubset({0, 1})


def test_isolation_forest_only_uses_train_fall_windows() -> None:
    train_features = np.vstack([np.zeros((10, 58)), np.ones((10, 58))])
    val_features = np.full((2, 58), 5.0)
    model = fit_isolation_forest(train_features, contamination=0.05, random_state=42, n_estimators=10)

    train_scores = compute_anomaly_scores(model, train_features)
    val_scores = compute_anomaly_scores(model, val_features)

    assert train_scores.shape == (20,)
    assert val_scores.shape == (2,)
    assert np.all(np.isfinite(train_scores))
    assert np.all(np.isfinite(val_scores))
    assert not np.array_equal(train_scores, val_scores)
