from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_extract_window_features_shape_and_finiteness() -> None:
    windows = np.vstack(
        [
            np.arange(64 * 6, dtype=np.float32).reshape(1, 64, 6),
            np.arange(64 * 6, dtype=np.float32).reshape(1, 64, 6) + 1,
        ]
    )
    features, names = extract_window_features(windows)

    assert features.shape == (2, len(names))
    assert all(np.isfinite(features).flat)
    assert "acc1_x_mean" in names
    assert "acc_mag_peak" in names


def test_kmeans_fitted_on_train_only_and_assigns_validated_splits() -> None:
    train_features = np.vstack([np.zeros((3, 58)), np.ones((3, 58))])
    val_features = np.full((2, 58), 2.0)
    scaler = StandardScaler().fit(train_features)
    scaled_train = scaler.transform(train_features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=5).fit(scaled_train)

    val_scaled = scaler.transform(val_features)
    val_labels = kmeans.predict(val_scaled)

    assert np.array_equal(kmeans.labels_, kmeans.predict(scaled_train))
    assert val_labels.shape == (2,)


def test_assign_severity_labels_uses_non_applicable_for_adl() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    fall_indices = np.array([1, 3], dtype=np.int64)
    cluster_ids = np.array([0, 2], dtype=np.int64)
    cluster_to_severity = {0: 0, 2: 2}

    severity = assign_severity_labels(labels, fall_indices, cluster_ids, cluster_to_severity)
    assert severity.tolist() == [NON_APPLICABLE, 0, NON_APPLICABLE, 2]


def test_map_clusters_to_severity_deterministic() -> None:
    cluster_stats = pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "acc_mag_peak_mean": [1.0, 3.0, 2.0],
            "gyro_mag_peak_mean": [1.0, 2.0, 3.0],
        }
    )
    mapping = map_clusters_to_severity(cluster_stats)

    assert mapping == {0: 0, 2: 1, 1: 2}


def test_compute_cluster_statistics_uses_train_cluster_features() -> None:
    cluster_ids = {
        "train": np.array([0, 0, 1, 1], dtype=np.int64),
        "val": np.array([0, 1], dtype=np.int64),
        "test": np.array([1, 1], dtype=np.int64),
    }
    feature_names = [f"feature_{i}" for i in range(58)]
    feature_names[-8:] = [
        "acc_mag_mean",
        "acc_mag_std",
        "acc_mag_min",
        "acc_mag_max",
        "acc_mag_range",
        "acc_mag_rms",
        "acc_mag_peak",
        "acc_mag_energy",
    ]
    feature_names[-16:-8] = [
        "gyro_mag_mean",
        "gyro_mag_std",
        "gyro_mag_min",
        "gyro_mag_max",
        "gyro_mag_range",
        "gyro_mag_rms",
        "gyro_mag_peak",
        "gyro_mag_energy",
    ]
    train_features = np.zeros((4, 58), dtype=np.float64)
    train_features[:, feature_names.index("acc_mag_peak")] = [1.0, 1.2, 2.5, 2.7]
    train_features[:, feature_names.index("gyro_mag_peak")] = [1.5, 1.6, 2.2, 2.4]

    stats = compute_cluster_statistics(cluster_ids, {"train": train_features}, feature_names)
    assert stats.loc[stats["cluster_id"] == 0, "acc_mag_peak_mean"].item() == 1.1
    assert stats.loc[stats["cluster_id"] == 1, "gyro_mag_peak_mean"].item() == 2.3


def test_existing_binary_labels_unaffected_by_severity_assignment() -> None:
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    fall_indices = np.array([1, 2], dtype=np.int64)
    cluster_ids = np.array([1, 0], dtype=np.int64)
    cluster_to_severity = {0: 1, 1: 2}
    severity = assign_severity_labels(labels, fall_indices, cluster_ids, cluster_to_severity)

    assert labels.tolist() == [0, 1, 1, 0]
    assert severity.tolist() == [NON_APPLICABLE, 2, 1, NON_APPLICABLE]
