"""Verify that recording-level detections include the labeled impact window.

This is a read-only diagnostic. It reuses recording-level metrics when the
existing evaluator already exported the required columns and only falls back
to model inference when those columns are unavailable.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.keras_model_loading import load_keras_model  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
BASELINE_DIR = ROOT / "results" / "cnn_baseline"
EVAL_DIR = BASELINE_DIR / "recording_level_eval"
RECORDING_METRICS_PATH = EVAL_DIR / "recording_level_metrics.csv"
MODEL_PATH = BASELINE_DIR / "cnn_baseline_best.keras"
METADATA_PATH = BASELINE_DIR / "run_metadata.json"
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 64


def threshold_name(threshold: float) -> str:
    return "threshold_0_5" if np.isclose(threshold, DEFAULT_THRESHOLD) else f"threshold_{threshold:g}".replace(".", "_")


def load_chosen_threshold() -> tuple[float | None, str | None]:
    if not METADATA_PATH.exists():
        return None, f"Metadata file not found: {METADATA_PATH}"
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        value = metadata.get("chosen_threshold", {}).get("threshold")
        if value is None:
            return None, "run_metadata.json does not contain chosen_threshold.threshold"
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            return None, f"Chosen threshold must be between 0 and 1, found {threshold}"
        return threshold, None
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return None, f"Could not read chosen threshold: {exc}"


def load_existing_metrics() -> tuple[pd.DataFrame, str]:
    if not RECORDING_METRICS_PATH.exists():
        raise FileNotFoundError(f"Existing recording metrics not found: {RECORDING_METRICS_PATH}")
    frame = pd.read_csv(RECORDING_METRICS_PATH)
    required = {"subject_id", "recording_id", "true_label", "num_fall_windows"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"recording_level_metrics.csv is missing required columns: {missing}")
    return frame, "existing recording_level_metrics.csv"


def load_raw_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = ["test.npy", "test_labels.npy", "test_recording_ids.npy", "test_subject_ids.npy"]
    arrays = {name: np.load(DATA_DIR / name, allow_pickle=True) for name in names}
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Raw test arrays are not aligned: {lengths}")
    features = arrays["test.npy"]
    labels = arrays["test_labels.npy"].astype(np.int32)
    recording_ids = arrays["test_recording_ids.npy"].astype(str)
    subject_ids = arrays["test_subject_ids.npy"].astype(str)
    if not np.isfinite(features).all():
        raise ValueError("test.npy contains NaN or Inf values.")
    return features.astype(np.float32), labels, recording_ids, subject_ids


def recompute_metrics(thresholds: list[float]) -> tuple[pd.DataFrame, str]:
    features, labels, recording_ids, subject_ids = load_raw_arrays()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found for recomputation: {MODEL_PATH}")
    import tensorflow as tf

    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_cnn_baseline import build_model

    model = load_keras_model(tf, MODEL_PATH, build_model)
    probabilities = model.predict(features, batch_size=BATCH_SIZE, verbose=0).ravel()
    rows: list[dict[str, Any]] = []
    for recording_id in pd.unique(recording_ids):
        mask = recording_ids == recording_id
        row: dict[str, Any] = {
            "recording_id": recording_id,
            "subject_id": pd.unique(subject_ids[mask])[0],
            "true_label": int(np.any(labels[mask] == 1)),
            "num_fall_windows": int((labels[mask] == 1).sum()),
        }
        for threshold in thresholds:
            name = threshold_name(threshold)
            predictions = probabilities[mask] >= threshold
            row[f"predicted_positive_{name}"] = int(np.any(predictions))
            row[f"num_detected_fall_windows_{name}"] = int(np.sum((labels[mask] == 1) & predictions))
        rows.append(row)
    return pd.DataFrame(rows), "recomputed from test arrays and saved CNN model"


def ensure_detection_columns(frame: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, str]:
    names = [threshold_name(threshold) for threshold in thresholds]
    required = {
        f"{field}_{name}"
        for name in names
        for field in ["predicted_positive", "num_detected_fall_windows"]
    }
    if required.issubset(frame.columns):
        return frame.copy(), "reused existing recording-level CSV columns"
    print("Required impact-detection columns are missing; recomputing from raw test arrays.")
    return recompute_metrics(thresholds)


def build_breakdown(frame: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, list[str]]]:
    falls = frame.loc[frame["true_label"].astype(int) == 1].copy()
    output_columns = ["subject_id", "recording_id", "num_fall_windows"]
    summary: dict[str, dict[str, Any]] = {}
    false_trigger_ids: dict[str, list[str]] = {}
    for threshold in thresholds:
        name = threshold_name(threshold)
        predicted = falls[f"predicted_positive_{name}"].astype(bool)
        detected_windows = falls[f"num_detected_fall_windows_{name}"].astype(int)
        genuine = predicted & (detected_windows > 0)
        false_trigger_only = predicted & (detected_windows == 0)
        missed = ~predicted
        falls[f"predicted_positive_{name}"] = predicted.astype(int)
        falls[f"num_detected_fall_windows_{name}"] = detected_windows
        falls[f"detection_type_{name}"] = "unknown"
        falls.loc[genuine, f"detection_type_{name}"] = "genuine_impact"
        falls.loc[false_trigger_only, f"detection_type_{name}"] = "false_trigger_only"
        falls.loc[missed, f"detection_type_{name}"] = "missed"
        total = len(falls)
        original_recall = float(predicted.mean()) if total else 0.0
        genuine_count = int(genuine.sum())
        false_trigger_count = int(false_trigger_only.sum())
        missed_count = int(missed.sum())
        summary[name] = {
            "threshold": float(threshold),
            "total_fall_recordings": int(total),
            "genuine_impact_detections": genuine_count,
            "genuine_impact_detection_rate": float(genuine_count / total) if total else 0.0,
            "false_trigger_only_detections": false_trigger_count,
            "false_trigger_only_rate": float(false_trigger_count / total) if total else 0.0,
            "fully_missed_recordings": missed_count,
            "original_any_window_recording_recall": original_recall,
            "corrected_true_impact_detection_recall": float(genuine_count / total) if total else 0.0,
            "recall_difference_corrected_minus_original": float(genuine_count / total - original_recall) if total else 0.0,
        }
        false_trigger_ids[name] = falls.loc[false_trigger_only, "recording_id"].astype(str).tolist()
        output_columns.extend([f"predicted_positive_{name}", f"num_detected_fall_windows_{name}", f"detection_type_{name}"])
    return falls[output_columns], summary, false_trigger_ids


def main() -> None:
    chosen_threshold, threshold_warning = load_chosen_threshold()
    if threshold_warning:
        warnings.warn(f"{threshold_warning}; chosen-threshold diagnostic will be skipped.", stacklevel=1)
        print(f"WARNING: {threshold_warning}; chosen-threshold diagnostic skipped.")
    thresholds = [DEFAULT_THRESHOLD] if chosen_threshold is None or np.isclose(chosen_threshold, DEFAULT_THRESHOLD) else [DEFAULT_THRESHOLD, chosen_threshold]

    existing, source = load_existing_metrics()
    metrics, source = ensure_detection_columns(existing, thresholds)
    breakdown, threshold_summary, false_trigger_ids = build_breakdown(metrics, thresholds)

    false_trigger_frames = []
    for threshold in thresholds:
        name = threshold_name(threshold)
        selected = breakdown[breakdown[f"detection_type_{name}"] == "false_trigger_only"].copy()
        if not selected.empty:
            selected.insert(0, "threshold", threshold)
            selected = selected[["threshold", "subject_id", "recording_id", "num_fall_windows"]]
            false_trigger_frames.append(selected)
    false_trigger_output = pd.concat(false_trigger_frames, ignore_index=True) if false_trigger_frames else pd.DataFrame(columns=["threshold", "subject_id", "recording_id", "num_fall_windows"])

    cross_check_warnings: list[str] = []
    original_summary_path = EVAL_DIR / "evaluation_summary.json"
    if original_summary_path.exists():
        original = json.loads(original_summary_path.read_text(encoding="utf-8"))
        for threshold in thresholds:
            name = threshold_name(threshold)
            expected = original.get("recording_level", {}).get(name, {}).get("fall_recordings_fully_missed")
            actual = threshold_summary[name]["fully_missed_recordings"]
            if expected is not None and int(expected) != actual:
                message = f"Fully missed count mismatch at threshold {threshold:g}: existing evaluator={expected}, diagnostic={actual}"
                cross_check_warnings.append(message)
                warnings.warn(message, stacklevel=1)
                print(f"WARNING: {message}")

    summary = {
        "source_path": source,
        "thresholds_used": {"default": DEFAULT_THRESHOLD, "chosen": chosen_threshold},
        "thresholds": threshold_summary,
        "false_trigger_only_recordings": false_trigger_ids,
        "cross_check_warnings": cross_check_warnings,
        "interpretation": "Genuine impact detections require predicted_positive and at least one correctly predicted label-1 impact window."
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(EVAL_DIR / "impact_window_verification.csv", index=False)
    false_trigger_output.to_csv(EVAL_DIR / "false_trigger_only_recordings.csv", index=False)
    (EVAL_DIR / "impact_verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nImpact-window verification summary")
    for threshold in thresholds:
        name = threshold_name(threshold)
        result = threshold_summary[name]
        print(f"Threshold {threshold:g}:")
        print(f"  Any-window recording recall: {result['original_any_window_recording_recall']:.2%}")
        print(f"  True-impact-window recording recall: {result['corrected_true_impact_detection_recall']:.2%}")
        print(f"  Genuine impact detections: {result['genuine_impact_detections']}/{result['total_fall_recordings']}")
        print(f"  False-trigger-only detections: {result['false_trigger_only_detections']}/{result['total_fall_recordings']}")
        print(f"  Fully missed recordings: {result['fully_missed_recordings']}")
        difference = result["recall_difference_corrected_minus_original"]
        print(f"  Measures {'match' if np.isclose(difference, 0.0) else 'diverge'} by {abs(difference):.2%} (corrected minus original).")
        print(f"  False-trigger-only recording IDs: {false_trigger_ids[name]}")
    print(f"Source used: {source}")
    print(f"Saved diagnostic outputs to: {EVAL_DIR}")


if __name__ == "__main__":
    main()
