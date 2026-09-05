from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SEVERITY_DIR = PROCESSED_DIR / "severity"
CLUSTER_FEATURES_DIR = SEVERITY_DIR / "clustering_features"
OUTPUT_DIR = SEVERITY_DIR / "reports"
SPLITS = ["train", "val", "test"]

CLASS_NAMES = {
    0: "ADL / non-fall",
    1: "Fall",
}

SEVERITY_NAMES = {
    -1: "Non-applicable / uncertain",
    0: "Mild",
    1: "Moderate",
    2: "Severe",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_models() -> tuple[object, object, dict[int, int]]:
    with (SEVERITY_DIR / "feature_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (SEVERITY_DIR / "kmeans_model.pkl").open("rb") as handle:
        kmeans = pickle.load(handle)
    mapping = load_json(SEVERITY_DIR / "severity_mapping.json")
    cluster_to_severity = {int(key): int(value) for key, value in mapping["cluster_to_severity"].items()}
    return scaler, kmeans, cluster_to_severity


def export_split(split: str, scaler: object, kmeans: object, cluster_to_severity: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.load(PROCESSED_DIR / f"{split}_labels.npy")
    subject_ids = np.load(PROCESSED_DIR / f"{split}_subject_ids.npy", allow_pickle=True)
    recording_ids = np.load(PROCESSED_DIR / f"{split}_recording_ids.npy", allow_pickle=True)

    severity_labels = np.load(SEVERITY_DIR / f"{split}_severity_labels.npy")
    refined_severity_labels = np.load(SEVERITY_DIR / f"{split}_refined_severity_labels.npy")
    fall_indices = np.load(CLUSTER_FEATURES_DIR / f"{split}_fall_indices.npy")
    fall_features = np.load(CLUSTER_FEATURES_DIR / f"{split}_fall_features.npy")
    anomaly_scores = np.load(SEVERITY_DIR / f"{split}_anomaly_scores.npy")
    anomaly_flags = np.load(SEVERITY_DIR / f"{split}_anomaly_flags.npy")

    if not (len(fall_indices) == len(fall_features) == len(anomaly_scores) == len(anomaly_flags)):
        raise ValueError(f"{split} fall-window arrays have mismatched lengths.")
    if not (len(labels) == len(subject_ids) == len(recording_ids) == len(severity_labels) == len(refined_severity_labels)):
        raise ValueError(f"{split} full-window arrays have mismatched lengths.")

    cluster_ids = kmeans.predict(scaler.transform(fall_features)) if len(fall_features) else np.empty((0,), dtype=np.int64)

    fall_lookup: dict[int, dict[str, object]] = {}
    fall_rows: list[dict[str, object]] = []
    for fall_order, window_index in enumerate(fall_indices.tolist()):
        cluster_id = int(cluster_ids[fall_order])
        severity_label = int(severity_labels[window_index])
        refined_label = int(refined_severity_labels[window_index])
        anomaly_flag = int(anomaly_flags[fall_order])
        row = {
            "split": split,
            "window_index": int(window_index),
            "fall_window_order": fall_order,
            "subject_id": str(subject_ids[window_index]),
            "recording_id": str(recording_ids[window_index]),
            "class_label": int(labels[window_index]),
            "class_name": CLASS_NAMES.get(int(labels[window_index]), "Unknown"),
            "cluster_id": cluster_id,
            "cluster_mapped_severity_label": int(cluster_to_severity[cluster_id]),
            "severity_label": severity_label,
            "severity_class": SEVERITY_NAMES[severity_label],
            "anomaly_score": float(anomaly_scores[fall_order]),
            "anomaly_flag": anomaly_flag,
            "anomaly_marking": "Anomalous" if anomaly_flag == 1 else "Normal",
            "refined_severity_label": refined_label,
            "refined_severity_class": SEVERITY_NAMES[refined_label],
        }
        fall_lookup[int(window_index)] = row
        fall_rows.append(row)

    all_rows: list[dict[str, object]] = []
    for window_index in range(len(labels)):
        severity_label = int(severity_labels[window_index])
        refined_label = int(refined_severity_labels[window_index])
        fall_info = fall_lookup.get(window_index)
        all_rows.append(
            {
                "split": split,
                "window_index": window_index,
                "subject_id": str(subject_ids[window_index]),
                "recording_id": str(recording_ids[window_index]),
                "class_label": int(labels[window_index]),
                "class_name": CLASS_NAMES.get(int(labels[window_index]), "Unknown"),
                "is_fall_window": bool(int(labels[window_index]) == 1),
                "fall_window_order": None if fall_info is None else fall_info["fall_window_order"],
                "cluster_id": None if fall_info is None else fall_info["cluster_id"],
                "severity_label": severity_label,
                "severity_class": SEVERITY_NAMES[severity_label],
                "anomaly_score": None if fall_info is None else fall_info["anomaly_score"],
                "anomaly_flag": None if fall_info is None else fall_info["anomaly_flag"],
                "anomaly_marking": "Not evaluated" if fall_info is None else fall_info["anomaly_marking"],
                "refined_severity_label": refined_label,
                "refined_severity_class": SEVERITY_NAMES[refined_label],
            }
        )

    return pd.DataFrame(all_rows), pd.DataFrame(fall_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler, kmeans, cluster_to_severity = load_models()

    all_window_tables = []
    fall_window_tables = []
    for split in SPLITS:
        all_windows, fall_windows = export_split(split, scaler, kmeans, cluster_to_severity)
        all_window_tables.append(all_windows)
        fall_window_tables.append(fall_windows)

    all_windows_df = pd.concat(all_window_tables, ignore_index=True)
    fall_windows_df = pd.concat(fall_window_tables, ignore_index=True)

    all_windows_path = OUTPUT_DIR / "post_isolation_forest_all_windows.csv"
    fall_windows_path = OUTPUT_DIR / "post_isolation_forest_fall_windows.csv"
    all_windows_df.to_csv(all_windows_path, index=False)
    fall_windows_df.to_csv(fall_windows_path, index=False)

    summary = (
        fall_windows_df.groupby(["split", "severity_class", "anomaly_marking"], observed=True)
        .size()
        .reset_index(name="window_count")
        .sort_values(["split", "severity_class", "anomaly_marking"])
    )
    summary_path = OUTPUT_DIR / "post_isolation_forest_window_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved all-window listing: {all_windows_path}")
    print(f"Saved fall-window listing: {fall_windows_path}")
    print(f"Saved summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
