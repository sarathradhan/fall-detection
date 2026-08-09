from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data.severity_anomaly import (
    ANOMALY_CONTAMINATION,
    ANOMALY_N_ESTIMATORS,
    compute_anomaly_flags,
    compute_anomaly_scores,
    fit_isolation_forest,
)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SEVERITY_DIR = PROCESSED_DIR / "severity"
CLUSTER_FEATURES_DIR = SEVERITY_DIR / "clustering_features"

SEVERITY_LABELS = {
    -1: "Non-applicable",
    0: "Mild",
    1: "Moderate",
    2: "Severe",
}

NON_APPLICABLE = -1

IMU_CHANNELS = [
    "acc1_x",
    "acc1_y",
    "acc1_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

FEATURE_AGGREGATES = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "rms",
    "peak_abs",
]

MAGNITUDE_AGGREGATES = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "rms",
    "peak",
    "energy",
]


@dataclass
class SeverityPipelineResult:
    severity_labels: dict[str, np.ndarray]
    refined_severity_labels: dict[str, np.ndarray]
    anomaly_scores: dict[str, np.ndarray]
    anomaly_flags: dict[str, np.ndarray]
    train_cluster_ids: np.ndarray
    cluster_to_severity: dict[int, int]
    severity_summary: dict[str, Any]
    cluster_stats: pd.DataFrame
    anomaly_summary: pd.DataFrame
    anomaly_by_severity: pd.DataFrame
    anomaly_by_subject: pd.DataFrame
    feature_names: list[str]


def _ensure_output_dirs() -> None:
    SEVERITY_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTER_FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_split(split_name: str, processed_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    else:
        processed_dir = Path(processed_dir).expanduser().resolve()

    windows = np.load(Path(processed_dir) / f"{split_name}.npy")
    labels = np.load(Path(processed_dir) / f"{split_name}_labels.npy")
    subject_ids = np.load(Path(processed_dir) / f"{split_name}_subject_ids.npy")
    recording_ids = np.load(Path(processed_dir) / f"{split_name}_recording_ids.npy")

    if not (len(windows) == len(labels) == len(subject_ids) == len(recording_ids)):
        raise ValueError(f"Processed split {split_name} contains mismatched array lengths.")

    return windows, labels, subject_ids, recording_ids


def extract_window_features(windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if windows.ndim != 3 or windows.shape[2] != len(IMU_CHANNELS):
        raise ValueError("Expected windows array of shape (n_windows, window_size, 6).")

    channel_means = windows.mean(axis=1)
    channel_stds = windows.std(axis=1)
    channel_mins = windows.min(axis=1)
    channel_maxs = windows.max(axis=1)
    channel_ranges = channel_maxs - channel_mins
    channel_rms = np.sqrt(np.mean(np.square(windows), axis=1))
    channel_peak_abs = np.max(np.abs(windows), axis=1)

    features = []
    names: list[str] = []
    for idx, channel in enumerate(IMU_CHANNELS):
        features.append(channel_means[:, idx])
        names.append(f"{channel}_mean")
        features.append(channel_stds[:, idx])
        names.append(f"{channel}_std")
        features.append(channel_mins[:, idx])
        names.append(f"{channel}_min")
        features.append(channel_maxs[:, idx])
        names.append(f"{channel}_max")
        features.append(channel_ranges[:, idx])
        names.append(f"{channel}_range")
        features.append(channel_rms[:, idx])
        names.append(f"{channel}_rms")
        features.append(channel_peak_abs[:, idx])
        names.append(f"{channel}_peak_abs")

    acc_magnitude = np.linalg.norm(windows[:, :, :3], axis=2)
    gyro_magnitude = np.linalg.norm(windows[:, :, 3:], axis=2)

    for prefix, magnitude in (("acc_mag", acc_magnitude), ("gyro_mag", gyro_magnitude)):
        magnitude_mean = magnitude.mean(axis=1)
        magnitude_std = magnitude.std(axis=1)
        magnitude_min = magnitude.min(axis=1)
        magnitude_max = magnitude.max(axis=1)
        magnitude_range = magnitude_max - magnitude_min
        magnitude_rms = np.sqrt(np.mean(np.square(magnitude), axis=1))
        magnitude_peak = magnitude_max
        magnitude_energy = np.mean(np.square(magnitude), axis=1)

        features.extend(
            [
                magnitude_mean,
                magnitude_std,
                magnitude_min,
                magnitude_max,
                magnitude_range,
                magnitude_rms,
                magnitude_peak,
                magnitude_energy,
            ]
        )
        names.extend(
            [
                f"{prefix}_mean",
                f"{prefix}_std",
                f"{prefix}_min",
                f"{prefix}_max",
                f"{prefix}_range",
                f"{prefix}_rms",
                f"{prefix}_peak",
                f"{prefix}_energy",
            ]
        )

    feature_matrix = np.stack(features, axis=1).astype(np.float64)

    if feature_matrix.shape[1] != len(names):
        raise AssertionError("Feature names length does not match feature matrix columns.")

    if not np.isfinite(feature_matrix).all():
        raise ValueError("Extracted features contain non-finite values.")

    return feature_matrix, names


def _fall_window_indices(labels: np.ndarray) -> np.ndarray:
    return np.where(labels == 1)[0]


def fit_feature_scaler(features: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def fit_kmeans(features: np.ndarray, n_clusters: int = 3, random_state: int = 42, n_init: int = 20) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    kmeans.fit(features)
    return kmeans


def cluster_intensity_scores(cluster_stats: pd.DataFrame) -> pd.Series:
    return cluster_stats["acc_mag_peak_mean"] + cluster_stats["gyro_mag_peak_mean"]


def map_clusters_to_severity(cluster_stats: pd.DataFrame) -> dict[int, int]:
    ranking = cluster_stats.sort_values(
        by=["acc_mag_peak_mean", "gyro_mag_peak_mean"],
        ascending=[True, True],
        ignore_index=False,
    )
    mapping: dict[int, int] = {}
    for severity_label, cluster_id in enumerate(ranking["cluster_id"].tolist()):
        mapping[int(cluster_id)] = severity_label
    return mapping


def assign_severity_labels(
    labels: np.ndarray,
    fall_indices: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_to_severity: dict[int, int],
) -> np.ndarray:
    severity = np.full(len(labels), NON_APPLICABLE, dtype=np.int64)
    if len(fall_indices) != len(cluster_ids):
        raise ValueError("Number of fall clusters must match number of fall windows.")
    for window_idx, cluster_id in zip(fall_indices.tolist(), cluster_ids.tolist()):
        severity[window_idx] = cluster_to_severity[int(cluster_id)]
    return severity


def compute_cluster_statistics(
    split_cluster_ids: dict[str, np.ndarray],
    split_features: dict[str, np.ndarray],
    feature_names: list[str],
) -> pd.DataFrame:
    if not feature_names:
        raise ValueError("Feature names are required to compute cluster statistics.")

    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    required_names = [
        "acc_mag_mean",
        "acc_mag_rms",
        "acc_mag_peak",
        "gyro_mag_mean",
        "gyro_mag_rms",
        "gyro_mag_peak",
    ]
    missing = [name for name in required_names if name not in feature_index]
    if missing:
        raise ValueError(f"Missing required feature names for cluster statistics: {missing}")

    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(np.unique(split_cluster_ids["train"])):
        row = {
            "cluster_id": int(cluster_id),
        }
        total_count = 0
        for split in ["train", "val", "test"]:
            split_count = int((split_cluster_ids[split] == cluster_id).sum())
            row[f"{split}_count"] = split_count
            total_count += split_count
        row["total_count"] = total_count

        train_mask = split_cluster_ids["train"] == cluster_id
        train_features = split_features["train"][train_mask]
        if train_features.size == 0:
            row.update(
                {
                    "acc_mag_mean": float("nan"),
                    "acc_mag_rms_mean": float("nan"),
                    "acc_mag_peak_mean": float("nan"),
                    "gyro_mag_mean": float("nan"),
                    "gyro_mag_rms_mean": float("nan"),
                    "gyro_mag_peak_mean": float("nan"),
                    "cluster_intensity_score": float("nan"),
                }
            )
        else:
            row.update(
                {
                    "acc_mag_mean": float(train_features[:, feature_index["acc_mag_mean"]].mean()),
                    "acc_mag_rms_mean": float(train_features[:, feature_index["acc_mag_rms"]].mean()),
                    "acc_mag_peak_mean": float(train_features[:, feature_index["acc_mag_peak"]].mean()),
                    "gyro_mag_mean": float(train_features[:, feature_index["gyro_mag_mean"]].mean()),
                    "gyro_mag_rms_mean": float(train_features[:, feature_index["gyro_mag_rms"]].mean()),
                    "gyro_mag_peak_mean": float(train_features[:, feature_index["gyro_mag_peak"]].mean()),
                    "cluster_intensity_score": float(
                        float(train_features[:, feature_index["acc_mag_peak"]].mean())
                        + float(train_features[:, feature_index["gyro_mag_peak"]].mean())
                    ),
                }
            )
        rows.append(row)
    cluster_stats = pd.DataFrame(rows)
    return cluster_stats


def _save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _save_cluster_statistics(cluster_stats: pd.DataFrame, path: Path) -> None:
    cluster_stats.to_csv(path, index=False)


def save_severity_artifacts(
    result: SeverityPipelineResult,
    output_dir: str | Path | None = None,
) -> None:
    if output_dir is None:
        output_dir = SEVERITY_DIR
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, labels in result.severity_labels.items():
        np.save(output_dir / f"{split}_severity_labels.npy", labels)
        np.save(output_dir / f"{split}_refined_severity_labels.npy", result.refined_severity_labels[split])
        np.save(output_dir / f"{split}_anomaly_scores.npy", result.anomaly_scores[split])
        np.save(output_dir / f"{split}_anomaly_flags.npy", result.anomaly_flags[split])

    _save_json(
        {
            "cluster_to_severity": {str(k): int(v) for k, v in result.cluster_to_severity.items()},
            "severity_labels": SEVERITY_LABELS,
            "note": "Mild/Moderate/Severe severity labels are derived from train-fall-window KMeans clustering of normalized IMU window features. Anomalies are identified by a train-only Isolation Forest and flagged separately."
        },
        output_dir / "severity_mapping.json",
    )

    _save_json(result.severity_summary, output_dir / "severity_summary.json")
    _save_cluster_statistics(result.cluster_stats, output_dir / "cluster_statistics.csv")
    result.anomaly_summary.to_csv(output_dir / "anomaly_summary.csv", index=False)
    result.anomaly_by_severity.to_csv(output_dir / "anomaly_by_severity.csv", index=False)
    result.anomaly_by_subject.to_csv(output_dir / "anomaly_by_subject.csv", index=False)

    _save_json(
        {
            "contamination": ANOMALY_CONTAMINATION,
            "n_estimators": ANOMALY_N_ESTIMATORS,
            "random_state": 42,
            "decision_function_note": "Higher scores are more normal and lower scores are more anomalous. Anomaly flags are assigned where IsolationForest.predict == -1.",
        },
        output_dir / "isolation_forest_config.json",
    )

    feature_names_path = output_dir / "feature_names.json"
    _save_json(result.feature_names, feature_names_path)


def save_clustering_features(
    split_name: str,
    features: np.ndarray,
    fall_indices: np.ndarray,
    path: Path | None = None,
) -> None:
    if path is None:
        path = CLUSTER_FEATURES_DIR
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f"{split_name}_fall_features.npy", features)
    np.save(path / f"{split_name}_fall_indices.npy", fall_indices)


def run_severity_pipeline(
    processed_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 20,
) -> SeverityPipelineResult:
    _ensure_output_dirs()

    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    else:
        processed_dir = Path(processed_dir).expanduser().resolve()

    if output_dir is None:
        output_dir = SEVERITY_DIR
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data: dict[str, dict[str, Any]] = {}
    split_fall_features: dict[str, np.ndarray] = {}
    split_fall_indices: dict[str, np.ndarray] = {}
    all_labels: dict[str, np.ndarray] = {}
    all_subject_ids: dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        windows, labels, subject_ids, recording_ids = load_processed_split(split, processed_dir)
        fall_indices = _fall_window_indices(labels)
        fall_windows = windows[fall_indices]
        if len(fall_windows) == 0:
            split_fall_features[split] = np.empty((0, 0), dtype=np.float64)
            split_fall_indices[split] = fall_indices
        else:
            features, feature_names = extract_window_features(fall_windows)
            split_fall_features[split] = features
            split_fall_indices[split] = fall_indices
            save_clustering_features(split, features, fall_indices)

        split_data[split] = {
            "windows": windows,
            "labels": labels,
            "subject_ids": subject_ids,
            "recording_ids": recording_ids,
            "fall_indices": fall_indices,
        }
        all_labels[split] = labels
        all_subject_ids[split] = subject_ids

    if split_fall_features["train"].shape[0] == 0:
        raise ValueError("Training split contains no fall windows for severity clustering.")

    feature_scaler = fit_feature_scaler(split_fall_features["train"])
    train_scaled = feature_scaler.transform(split_fall_features["train"])
    kmeans_model = fit_kmeans(train_scaled, n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    isolation_forest = fit_isolation_forest(
        train_scaled,
        contamination=ANOMALY_CONTAMINATION,
        random_state=random_state,
        n_estimators=ANOMALY_N_ESTIMATORS,
    )

    split_cluster_ids: dict[str, np.ndarray] = {}
    split_anomaly_scores: dict[str, np.ndarray] = {}
    split_anomaly_flags: dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        if split_fall_features[split].shape[0] == 0:
            split_cluster_ids[split] = np.empty((0,), dtype=np.int64)
            split_anomaly_scores[split] = np.empty((0,), dtype=np.float64)
            split_anomaly_flags[split] = np.empty((0,), dtype=np.int64)
        else:
            scaled = feature_scaler.transform(split_fall_features[split])
            split_cluster_ids[split] = kmeans_model.predict(scaled)
            split_anomaly_scores[split] = compute_anomaly_scores(isolation_forest, scaled)
            split_anomaly_flags[split] = compute_anomaly_flags(isolation_forest, scaled)

    cluster_stats = compute_cluster_statistics(split_cluster_ids, split_fall_features, feature_names)
    cluster_to_severity = map_clusters_to_severity(cluster_stats)
    cluster_stats["severity_label"] = cluster_stats["cluster_id"].map(cluster_to_severity)
    cluster_stats["severity_name"] = cluster_stats["severity_label"].map(SEVERITY_LABELS)
    cluster_stats = cluster_stats.sort_values(by=["severity_label", "cluster_id"]).reset_index(drop=True)

    severity_labels: dict[str, np.ndarray] = {}
    refined_severity_labels: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        severity_labels[split] = assign_severity_labels(
            labels=all_labels[split],
            fall_indices=split_fall_indices[split],
            cluster_ids=split_cluster_ids[split],
            cluster_to_severity=cluster_to_severity,
        )
        refined_severity = np.array(severity_labels[split], copy=True)
        anomaly_mask = np.zeros_like(refined_severity, dtype=bool)
        anomaly_mask[split_fall_indices[split]] = split_anomaly_flags[split] == 1
        refined_severity[anomaly_mask] = NON_APPLICABLE
        refined_severity_labels[split] = refined_severity

    anomaly_summary_rows: list[dict[str, Any]] = []
    anomaly_by_severity_rows: list[dict[str, Any]] = []
    anomaly_by_subject_rows: list[dict[str, Any]] = []

    for split in ["train", "val", "test"]:
        total = int(len(split_fall_indices[split]))
        anomalies = int(split_anomaly_flags[split].sum())
        anomaly_summary_rows.append(
            {
                "split": split,
                "fall_windows": total,
                "anomalous_windows": anomalies,
                "anomaly_rate": float(anomalies / total if total else 0.0),
            }
        )

        fall_severity_labels = severity_labels[split][split_fall_indices[split]]
        for severity_label, severity_name in [(0, "Mild"), (1, "Moderate"), (2, "Severe")]:
            severity_fall_mask = fall_severity_labels == severity_label
            severity_count = int(severity_fall_mask.sum())
            if severity_count == 0:
                anomaly_rate = 0.0
                anomalies_in_severity = 0
            else:
                anomalies_in_severity = int(split_anomaly_flags[split][severity_fall_mask].sum())
                anomaly_rate = float(anomalies_in_severity / severity_count)
            anomaly_by_severity_rows.append(
                {
                    "split": split,
                    "severity_label": severity_label,
                    "severity_name": severity_name,
                    "fall_windows": severity_count,
                    "anomalous_windows": anomalies_in_severity,
                    "anomaly_rate": anomaly_rate,
                }
            )

        subject_ids = all_subject_ids[split][split_fall_indices[split]]
        for subject_id in np.unique(subject_ids):
            subject_mask = subject_ids == subject_id
            subject_total = int(subject_mask.sum())
            subject_anomalies = int(split_anomaly_flags[split][subject_mask].sum())
            anomaly_by_subject_rows.append(
                {
                    "split": split,
                    "subject_id": str(subject_id),
                    "fall_windows": subject_total,
                    "anomalous_windows": subject_anomalies,
                    "anomaly_rate": float(subject_anomalies / subject_total if subject_total else 0.0),
                }
            )

    anomaly_summary = pd.DataFrame(anomaly_summary_rows)
    anomaly_by_severity = pd.DataFrame(anomaly_by_severity_rows)
    anomaly_by_subject = pd.DataFrame(anomaly_by_subject_rows)

    severity_summary: dict[str, Any] = {
        "split_counts": {},
        "cluster_to_severity": {str(cluster_id): SEVERITY_LABELS[severity] for cluster_id, severity in cluster_to_severity.items()},
        "anomaly_contamination": ANOMALY_CONTAMINATION,
        "anomaly_n_estimators": ANOMALY_N_ESTIMATORS,
        "anomaly_note": "Anomaly scores from Isolation Forest are decision_function outputs; larger values mean more normal samples. anomaly_flag == 1 denotes suspicious fall windows. Refined severity sets anomalous fall windows to -1 (uncertain).",
    }
    for split in ["train", "val", "test"]:
        severity_counts = {label: int((severity_labels[split] == int(label)).sum()) for label in [0, 1, 2]}
        severity_summary["split_counts"][split] = {
            "fall_windows": int(len(split_fall_indices[split])),
            "mild": severity_counts[0],
            "moderate": severity_counts[1],
            "severe": severity_counts[2],
        }

    save_severity_artifacts(
        SeverityPipelineResult(
            severity_labels=severity_labels,
            refined_severity_labels=refined_severity_labels,
            anomaly_scores=split_anomaly_scores,
            anomaly_flags=split_anomaly_flags,
            train_cluster_ids=split_cluster_ids["train"],
            cluster_to_severity=cluster_to_severity,
            severity_summary=severity_summary,
            cluster_stats=cluster_stats,
            anomaly_summary=anomaly_summary,
            anomaly_by_severity=anomaly_by_severity,
            anomaly_by_subject=anomaly_by_subject,
            feature_names=feature_names,
        ),
        output_dir=output_dir,
    )

    with (Path(output_dir) / "feature_scaler.pkl").open("wb") as handle:
        import pickle

        pickle.dump(feature_scaler, handle)
    with (Path(output_dir) / "kmeans_model.pkl").open("wb") as handle:
        import pickle

        pickle.dump(kmeans_model, handle)
    with (Path(output_dir) / "isolation_forest.pkl").open("wb") as handle:
        import pickle

        pickle.dump(isolation_forest, handle)

    return SeverityPipelineResult(
        severity_labels=severity_labels,
        refined_severity_labels=refined_severity_labels,
        anomaly_scores=split_anomaly_scores,
        anomaly_flags=split_anomaly_flags,
        train_cluster_ids=split_cluster_ids["train"],
        cluster_to_severity=cluster_to_severity,
        severity_summary=severity_summary,
        cluster_stats=cluster_stats,
        anomaly_summary=anomaly_summary,
        anomaly_by_severity=anomaly_by_severity,
        anomaly_by_subject=anomaly_by_subject,
        feature_names=feature_names,
    )
