"""Evaluate the trained CNN at window, recording, and subject levels.

This script only reads existing processed arrays and the trained model. It does
not retrain the model or modify files under data/processed/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.keras_model_loading import load_keras_model  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "results" / "cnn_baseline" / "cnn_baseline_best.keras"
METADATA_PATH = ROOT / "results" / "cnn_baseline" / "run_metadata.json"
OUTPUT_DIR = ROOT / "results" / "cnn_baseline" / "recording_level_eval"
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 64


def load_test_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = ["test.npy", "test_labels.npy", "test_recording_ids.npy", "test_subject_ids.npy"]
    arrays = {name: np.load(DATA_DIR / name, allow_pickle=True) for name in names}
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Test arrays are not row-aligned; mismatched lengths: {details}")

    features = arrays["test.npy"]
    labels = arrays["test_labels.npy"]
    recording_ids = arrays["test_recording_ids.npy"].astype(str)
    subject_ids = arrays["test_subject_ids.npy"].astype(str)
    if features.ndim != 3:
        raise ValueError(f"test.npy must be 3D, found shape {features.shape}.")
    if not np.isfinite(features).all():
        raise ValueError("test.npy contains NaN or Inf values.")
    if labels.ndim != 1 or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("test_labels.npy must be a finite one-dimensional binary array.")
    return features.astype(np.float32, copy=False), labels.astype(np.int32, copy=False), recording_ids, subject_ids


def load_chosen_threshold() -> tuple[float | None, str | None]:
    if not METADATA_PATH.exists():
        return None, f"Metadata file not found: {METADATA_PATH}"
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        chosen = metadata.get("chosen_threshold", {}).get("threshold")
        if chosen is None:
            return None, "run_metadata.json does not contain chosen_threshold.threshold"
        threshold = float(chosen)
        if not 0.0 <= threshold <= 1.0:
            return None, f"Chosen threshold must be between 0 and 1, found {threshold}"
        return threshold, None
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return None, f"Could not read chosen threshold from {METADATA_PATH}: {exc}"


def threshold_name(threshold: float) -> str:
    return "threshold_0_5" if np.isclose(threshold, DEFAULT_THRESHOLD) else f"threshold_{threshold:g}".replace(".", "_")


def window_metrics(labels: np.ndarray, predictions: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def aggregate_recordings(
    labels: np.ndarray,
    predictions: np.ndarray,
    recording_ids: np.ndarray,
    subject_ids: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for recording_id in pd.unique(recording_ids):
        mask = recording_ids == recording_id
        recording_labels = labels[mask]
        recording_predictions = predictions[mask]
        row = {
            "recording_id": recording_id,
            "subject_id": pd.unique(subject_ids[mask])[0],
            "true_label": int(np.any(recording_labels == 1)),
            "num_windows": int(mask.sum()),
            "num_fall_windows": int((recording_labels == 1).sum()),
            f"predicted_positive_{threshold_name(threshold)}": int(np.any(recording_predictions == 1)),
            f"num_detected_fall_windows_{threshold_name(threshold)}": int(
                ((recording_labels == 1) & (recording_predictions == 1)).sum()
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_recording_metrics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    name = threshold_name(threshold)
    predicted_column = f"predicted_positive_{name}"
    frame[f"detected_fall_recording_{name}"] = ((frame["true_label"] == 1) & (frame[predicted_column] == 1)).astype(int)
    frame[f"fully_missed_fall_recording_{name}"] = ((frame["true_label"] == 1) & (frame[predicted_column] == 0)).astype(int)
    frame[f"false_positive_recording_{name}"] = ((frame["true_label"] == 0) & (frame[predicted_column] == 1)).astype(int)
    return frame


def recording_summary(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    name = threshold_name(threshold)
    fall = frame[frame["true_label"] == 1]
    adl = frame[frame["true_label"] == 0]
    detected = int(fall[f"detected_fall_recording_{name}"].sum())
    missed = int(fall[f"fully_missed_fall_recording_{name}"].sum())
    false_positive = int(adl[f"false_positive_recording_{name}"].sum())
    return {
        "threshold": float(threshold),
        "total_recordings": int(len(frame)),
        "total_fall_recordings": int(len(fall)),
        "total_adl_recordings": int(len(adl)),
        "fall_recordings_fully_missed": missed,
        "fall_recordings_at_least_partially_detected": detected,
        "recording_level_recall": float(detected / len(fall)) if len(fall) else 0.0,
        "adl_recordings_falsely_triggered": false_positive,
        "recording_level_false_positive_rate": float(false_positive / len(adl)) if len(adl) else 0.0,
    }


def aggregate_subjects(recordings: pd.DataFrame, threshold: float) -> pd.DataFrame:
    name = threshold_name(threshold)
    rows: list[dict[str, Any]] = []
    for subject_id, group in recordings.groupby("subject_id", sort=True):
        fall = group[group["true_label"] == 1]
        missed = int(fall[f"fully_missed_fall_recording_{name}"].sum())
        total = len(fall)
        rows.append(
            {
                "subject_id": subject_id,
                "total_recordings": int(len(group)),
                "fall_recordings": int(total),
                "adl_recordings": int((group["true_label"] == 0).sum()),
                f"fall_recordings_missed_{name}": missed,
                f"fall_recordings_detected_{name}": int(total - missed),
                f"subject_recall_{name}": float((total - missed) / total) if total else np.nan,
                f"review_below_90_percent_{name}": bool(total > 0 and (total - missed) / total < 0.90),
            }
        )
    return pd.DataFrame(rows)


def merge_subject_metrics(base: pd.DataFrame, addition: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return addition
    return base.merge(addition, on=["subject_id", "total_recordings", "fall_recordings", "adl_recordings"], how="outer")


def missed_recordings(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    name = threshold_name(threshold)
    columns = ["subject_id", "recording_id", "num_fall_windows"]
    return frame.loc[frame[f"fully_missed_fall_recording_{name}"] == 1, columns].copy()


def comparison_row(level: str, default_recall: float, chosen_recall: float | None) -> dict[str, float | str]:
    return {
        "level": level,
        "threshold_0_5_recall": default_recall,
        "chosen_threshold_recall": chosen_recall if chosen_recall is not None else np.nan,
    }


def main() -> None:
    features, labels, recording_ids, subject_ids = load_test_arrays()
    chosen_threshold, warning_message = load_chosen_threshold()
    if warning_message:
        warnings.warn(f"{warning_message}; chosen-threshold comparison will be skipped.", stacklevel=1)
        print(f"WARNING: {warning_message}; chosen-threshold comparison skipped.")
    thresholds = [DEFAULT_THRESHOLD] if chosen_threshold is None or np.isclose(chosen_threshold, DEFAULT_THRESHOLD) else [DEFAULT_THRESHOLD, chosen_threshold]

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    import tensorflow as tf

    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_cnn_baseline import build_model

    model = load_keras_model(tf, MODEL_PATH, build_model)
    probabilities = model.predict(features, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = {threshold: (probabilities >= threshold).astype(np.int32) for threshold in thresholds}

    mapping = pd.DataFrame({"recording_id": recording_ids, "subject_id": subject_ids}).drop_duplicates()
    inconsistent = mapping.groupby("recording_id")["subject_id"].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if not inconsistent.empty:
        warnings.warn(f"Found recording IDs mapped to multiple subjects: {inconsistent.index.tolist()}", stacklevel=1)
        print(f"WARNING: inconsistent recording-to-subject mappings: {inconsistent.index.tolist()}")

    recording_frame: pd.DataFrame | None = None
    subject_frame: pd.DataFrame | None = None
    recording_summaries: dict[str, dict[str, Any]] = {}
    subject_summaries: dict[str, dict[str, Any]] = {}
    missed_by_threshold: dict[str, pd.DataFrame] = {}
    comparison_rows: list[dict[str, float | str]] = []
    default_window = window_metrics(labels, predictions[DEFAULT_THRESHOLD], DEFAULT_THRESHOLD)

    for threshold in thresholds:
        current = aggregate_recordings(labels, predictions[threshold], recording_ids, subject_ids, threshold)
        current = add_recording_metrics(current, threshold)
        recording_frame = current if recording_frame is None else recording_frame.merge(current, on=["recording_id", "subject_id", "true_label", "num_windows", "num_fall_windows"], how="outer")
        summary = recording_summary(current, threshold)
        recording_summaries[threshold_name(threshold)] = summary
        missed_by_threshold[threshold_name(threshold)] = missed_recordings(current, threshold)

        current_subjects = aggregate_subjects(current, threshold)
        subject_frame = merge_subject_metrics(subject_frame if subject_frame is not None else pd.DataFrame(), current_subjects)
        subject_recall = current_subjects[f"subject_recall_{threshold_name(threshold)}"].dropna()
        subject_summaries[threshold_name(threshold)] = {
            "mean_subject_recall": float(subject_recall.mean()) if len(subject_recall) else 0.0,
            "min_subject_recall": float(subject_recall.min()) if len(subject_recall) else 0.0,
            "max_subject_recall": float(subject_recall.max()) if len(subject_recall) else 0.0,
            "subjects_below_90_percent": current_subjects.loc[current_subjects[f"review_below_90_percent_{threshold_name(threshold)}"], "subject_id"].tolist(),
        }

        comparison_rows.append(comparison_row("recording", summary["recording_level_recall"], summary["recording_level_recall"] if threshold != DEFAULT_THRESHOLD else None))
        if threshold == DEFAULT_THRESHOLD:
            comparison_rows.append(comparison_row("window", default_window["recall"], None))
    if chosen_threshold is not None and not np.isclose(chosen_threshold, DEFAULT_THRESHOLD):
        chosen_window = window_metrics(labels, predictions[chosen_threshold], chosen_threshold)
        chosen_recording = recording_summaries[threshold_name(chosen_threshold)]
        comparison_rows = [
            comparison_row("window", default_window["recall"], chosen_window["recall"]),
            comparison_row("recording", recording_summaries[threshold_name(DEFAULT_THRESHOLD)]["recording_level_recall"], chosen_recording["recording_level_recall"]),
        ]
    else:
        comparison_rows = [comparison_rows[1], comparison_rows[0]]

    default_subject_column = f"subject_recall_{threshold_name(DEFAULT_THRESHOLD)}"
    chosen_subject_column = f"subject_recall_{threshold_name(chosen_threshold)}" if chosen_threshold is not None else None
    comparison_rows.append(
        comparison_row(
            "subject",
            float(subject_frame[default_subject_column].dropna().mean()),
            float(subject_frame[chosen_subject_column].dropna().mean()) if chosen_subject_column and chosen_subject_column in subject_frame else None,
        )
    )
    comparison = pd.DataFrame(comparison_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recording_frame.to_csv(OUTPUT_DIR / "recording_level_metrics.csv", index=False)
    subject_frame.to_csv(OUTPUT_DIR / "subject_level_metrics.csv", index=False)
    if chosen_threshold is not None and not np.isclose(chosen_threshold, DEFAULT_THRESHOLD):
        missed = missed_by_threshold[threshold_name(DEFAULT_THRESHOLD)].merge(
            missed_by_threshold[threshold_name(chosen_threshold)], on=["subject_id", "recording_id", "num_fall_windows"], how="outer", suffixes=("_threshold_0_5", "_chosen_threshold")
        )
    else:
        missed = missed_by_threshold[threshold_name(DEFAULT_THRESHOLD)]
    missed.to_csv(OUTPUT_DIR / "missed_fall_recordings.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "summary_comparison.csv", index=False)

    total_missed = sum(len(frame) for frame in missed_by_threshold.values())
    concentration: dict[str, Any] = {}
    for threshold in thresholds:
        name = threshold_name(threshold)
        missed_subjects = missed_by_threshold[name].groupby("subject_id").size().sort_values(ascending=False)
        total = int(len(missed_by_threshold[name]))
        concentration[name] = {
            "missed_falls_by_subject": {str(subject): int(count) for subject, count in missed_subjects.items()},
            "largest_subject_missed_count": int(missed_subjects.iloc[0]) if len(missed_subjects) else 0,
            "largest_subject_missed_fraction": float(missed_subjects.iloc[0] / total) if len(missed_subjects) and total else 0.0,
            "any_subject_accounts_for_more_than_30_percent": bool(len(missed_subjects) and total and missed_subjects.iloc[0] / total > 0.30),
        }
    summary = {
        "thresholds_used": {"default": DEFAULT_THRESHOLD, "chosen": chosen_threshold},
        "window_level_recall": {threshold_name(DEFAULT_THRESHOLD): default_window["recall"]},
        "recording_level": recording_summaries,
        "subject_level": subject_summaries,
        "subject_missed_fall_concentration": concentration,
        "subjects_flagged_below_90_percent": sorted({subject for data in subject_summaries.values() for subject in data["subjects_below_90_percent"]}),
        "inconsistent_recording_subject_mappings": inconsistent.index.tolist(),
        "outputs": {"directory": str(OUTPUT_DIR), "missed_fall_recordings_rows_combined": total_missed},
    }
    if chosen_threshold is not None:
        summary["window_level_recall"][threshold_name(chosen_threshold)] = window_metrics(labels, predictions[chosen_threshold], chosen_threshold)["recall"]
    (OUTPUT_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nRecording/subject-level evaluation summary")
    for row in comparison.itertuples(index=False):
        chosen_text = "skipped" if pd.isna(row.chosen_threshold_recall) else f"{row.chosen_threshold_recall:.4f}"
        print(f"{row.level:>10}: threshold 0.5 recall={row.threshold_0_5_recall:.4f}; chosen threshold recall={chosen_text}")
    for threshold in thresholds:
        name = threshold_name(threshold)
        missed_ids = missed_by_threshold[name]["recording_id"].tolist()
        print(f"Fully missed fall recordings at threshold {threshold:g}: {missed_ids}")
    print(f"Saved evaluation outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()