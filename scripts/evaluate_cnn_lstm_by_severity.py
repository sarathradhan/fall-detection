from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "results" / "cnn_lstm_baseline" / "cnn_lstm_best.keras"
OUTPUT_DIR = ROOT / "results" / "cnn_lstm_baseline" / "severity_eval"
THRESHOLD = 0.50
SEVERITY_NAMES = {-1: "Uncertain (-1)", 0: "Mild", 1: "Moderate", 2: "Severe"}


def load_severity_names(severity_dir: Path) -> dict[int, str]:
    summary_path = severity_dir / "severity_summary.json"
    if not summary_path.exists():
        return dict(SEVERITY_NAMES)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    names = summary.get("severity_names")
    if not isinstance(names, dict):
        return dict(SEVERITY_NAMES)
    parsed = {int(label): str(name) for label, name in names.items()}
    if -1 in parsed:
        parsed[-1] = "Uncertain (-1)"
    return parsed


def evaluate(severity_dir: Path, output_dir: Path) -> None:
    arrays = {name: np.load(DATA_DIR / name, allow_pickle=True) for name in ["test.npy", "test_labels.npy", "test_recording_ids.npy", "test_subject_ids.npy"]}
    refined = np.load(severity_dir / "test_refined_severity_labels.npy")
    lengths = [len(value) for value in [*arrays.values(), refined]]
    if len(set(lengths)) != 1:
        raise ValueError(f"Test and severity arrays are not aligned: {lengths}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model(arrays["test.npy"], training=False).numpy().ravel()
    predictions = probabilities >= THRESHOLD
    labels = arrays["test_labels.npy"].astype(int)
    recording_ids = arrays["test_recording_ids.npy"].astype(str)
    subject_ids = arrays["test_subject_ids.npy"].astype(str)
    severity_names = load_severity_names(severity_dir)
    present_severities = [-1, *sorted(label for label in severity_names if label >= 0)]
    rows: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []

    def has_consecutive_alert(mask: np.ndarray) -> bool:
        positions = np.flatnonzero(mask)
        return bool(len(positions) > 1 and np.any(np.diff(positions) == 1))

    def binary_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
        tp = int(np.sum(target & predicted))
        fp = int(np.sum(~target & predicted))
        fn = int(np.sum(target & ~predicted))
        tn = int(np.sum(~target & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}

    overall_window = binary_metrics(labels == 1, predictions)
    all_recording_rows: list[dict[str, Any]] = []
    for recording_id in np.unique(recording_ids):
        recording_mask = recording_ids == recording_id
        recording_prediction = predictions[recording_mask]
        all_recording_rows.append(
            {
                "recording_id": recording_id,
                "subject_id": np.unique(subject_ids[recording_mask])[0],
                "true_fall_recording": bool(np.any(labels[recording_mask] == 1)),
                "predicted_positive": bool(np.any(recording_prediction)),
                "two_consecutive_alert": has_consecutive_alert(recording_prediction),
            }
        )
    all_recordings = pd.DataFrame(all_recording_rows)
    fall_recordings = all_recordings[all_recordings["true_fall_recording"]]

    def recording_policy_metrics(frame: pd.DataFrame, policy_column: str) -> dict[str, float | int]:
        fall = frame[frame["true_fall_recording"]]
        adl = frame[~frame["true_fall_recording"]]
        detected = int(fall[policy_column].sum())
        false_positive = int(adl[policy_column].sum())
        return {
            "fall_recordings": int(len(fall)),
            "detected_fall_recordings": detected,
            "recording_recall": detected / len(fall) if len(fall) else 0.0,
            "adl_recordings": int(len(adl)),
            "adl_false_trigger_recordings": false_positive,
            "adl_false_trigger_rate": false_positive / len(adl) if len(adl) else 0.0,
        }

    for severity in present_severities:
        severity_name = severity_names[severity]
        group = (labels == 1) & (refined == severity)
        detected = int(np.sum(predictions[group]))
        total = int(np.sum(group))
        window_detail = binary_metrics(group, predictions)
        rows.append({"severity_label": severity, "severity": severity_name, "windows": total, "detected_windows": detected, "window_recall": detected / total if total else 0.0, "window_precision": window_detail["precision"], "window_f1": window_detail["f1"]})
        group_recordings = np.unique(recording_ids[group])
        for recording_id in group_recordings:
            recording_mask = recording_ids == recording_id
            severity_mask = recording_mask & group
            recording_rows.append({"severity_label": severity, "severity": severity_name, "recording_id": recording_id, "subject_id": np.unique(subject_ids[recording_mask])[0], "severity_windows": int(severity_mask.sum()), "detected_severity_windows": int(np.sum(predictions[severity_mask])), "recording_detected": bool(np.any(predictions[severity_mask])), "impact_verified": bool(np.any(predictions[severity_mask])), "two_consecutive_alert": has_consecutive_alert(severity_mask & predictions)})
        for subject_id in np.unique(subject_ids[group]):
            subject_group = group & (subject_ids == subject_id)
            count = int(subject_group.sum())
            subject_rows.append({"severity_label": severity, "severity": severity_name, "subject_id": subject_id, "severity_windows": count, "detected_severity_windows": int(np.sum(predictions[subject_group])), "subject_recall": float(np.sum(predictions[subject_group]) / count) if count else 0.0, "impact_verified": bool(np.any(predictions[subject_group]))})

    recording_frame = pd.DataFrame(recording_rows)
    subject_frame = pd.DataFrame(subject_rows)
    recording_summary = recording_frame.groupby(["severity_label", "severity"], as_index=False).agg(recordings=("recording_id", "nunique"), detected_recordings=("recording_detected", "sum"), impact_verified_recordings=("impact_verified", "sum"), two_consecutive_recordings=("two_consecutive_alert", "sum"))
    subject_summary = subject_frame.groupby(["severity_label", "severity"], as_index=False).agg(subjects=("subject_id", "nunique"), mean_subject_recall=("subject_recall", "mean"), min_subject_recall=("subject_recall", "min"), impact_verified_subjects=("impact_verified", "sum"))
    summary = {
        "threshold": THRESHOLD,
        "severity_dir": str(severity_dir),
        "window_metrics": rows,
        "recording_metrics_by_severity": recording_summary.to_dict("records"),
        "subject_metrics_by_severity": subject_summary.to_dict("records"),
        "overall_metrics": {"window": overall_window, "recording": recording_policy_metrics(all_recordings, "predicted_positive"), "subject_count": int(all_recordings["subject_id"].nunique()), "impact_verified_recordings": int(fall_recordings["predicted_positive"].sum())},
        "future_two_consecutive_window_policy": {"status": "analysis_only; default predictions unchanged", "rule": "alert only when two adjacent windows are predicted positive within one recording", "recording": recording_policy_metrics(all_recordings, "two_consecutive_alert")},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "window_metrics_by_severity.csv", index=False)
    recording_frame.to_csv(output_dir / "recording_metrics_by_severity.csv", index=False)
    subject_frame.to_csv(output_dir / "subject_metrics_by_severity.csv", index=False)
    all_recordings.to_csv(output_dir / "overall_recording_metrics.csv", index=False)
    recording_summary.to_csv(output_dir / "recording_summary_by_severity.csv", index=False)
    subject_summary.to_csv(output_dir / "subject_summary_by_severity.csv", index=False)
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--severity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    evaluate(args.severity_dir.resolve(), args.output_dir.resolve())
