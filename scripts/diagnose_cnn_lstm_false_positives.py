"""Diagnose CNN-LSTM ADL false-trigger recordings without retraining."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sisfall_preprocessing import ACTIVITY_MAPPING  # noqa: E402


DATA_DIR = ROOT / "data" / "processed"
CNN_DIR = ROOT / "results" / "cnn_baseline"
CNN_LSTM_DIR = ROOT / "results" / "cnn_lstm_baseline"
OUTPUT_DIR = CNN_LSTM_DIR / "false_positive_diagnostics"
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 64


@dataclass(frozen=True)
class ThresholdSpec:
    label: str
    value: float
    name: str


def threshold_name(threshold: float) -> str:
    return "threshold_0_5" if np.isclose(threshold, DEFAULT_THRESHOLD) else f"threshold_{threshold:g}".replace(".", "_")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file is missing: {path}")


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_columns(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")


def require_field(data: dict[str, Any], path: Path, fields: list[str]) -> Any:
    current: Any = data
    traversed: list[str] = []
    for field in fields:
        traversed.append(field)
        if not isinstance(current, dict) or field not in current:
            raise KeyError(f"{path} is missing expected field: {'.'.join(traversed)}")
        current = current[field]
    return current


def load_chosen_threshold(model_dir: Path) -> float:
    metadata_path = model_dir / "run_metadata.json"
    metadata = read_json(metadata_path)
    value = require_field(metadata, metadata_path, ["chosen_threshold", "threshold"])
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{metadata_path} chosen_threshold.threshold must be in [0, 1], found {threshold}")
    return threshold


def parse_activity_code(recording_id: str) -> str:
    match = re.search(r"(?:^|:)([DF]\d{2})_", recording_id)
    if not match:
        raise ValueError(f"Could not parse D/F activity code from recording_id: {recording_id}")
    return match.group(1)


def activity_judgment(activity_code: str, activity_name: str) -> tuple[str, str]:
    if activity_code.startswith("F"):
        return "data_integrity_bug", "Resolved fall activity code among ADL false-trigger recordings."
    borderline_codes = {"D02", "D03", "D04", "D06", "D08", "D10", "D11", "D13", "D18", "D19"}
    if activity_code in borderline_codes:
        return "plausible_borderline", "Rapid, abrupt, stumbling, collapse-into-chair, or jumping motion can resemble fall dynamics."
    return "unexpected", "Lower-impact ADL pattern should be less fall-like."


def find_false_trigger_recordings(metrics_path: Path, thresholds: list[ThresholdSpec]) -> pd.DataFrame:
    require_file(metrics_path)
    frame = pd.read_csv(metrics_path)
    required = ["recording_id", "subject_id", "true_label", "num_windows", "num_fall_windows"]
    for threshold in thresholds:
        required.extend([f"predicted_positive_{threshold.name}", f"false_positive_recording_{threshold.name}"])
    require_columns(frame, metrics_path, required)

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted_col = f"predicted_positive_{threshold.name}"
        false_positive_col = f"false_positive_recording_{threshold.name}"
        selected = frame.loc[(frame["true_label"].astype(int) == 0) & (frame[predicted_col].astype(int) == 1)].copy()
        flagged = frame.loc[frame[false_positive_col].astype(int) == 1, "recording_id"].astype(str).tolist()
        if sorted(selected["recording_id"].astype(str).tolist()) != sorted(flagged):
            raise ValueError(
                f"{metrics_path} has inconsistent predicted_positive and false_positive columns for {threshold.name}."
            )
        for row in selected.itertuples(index=False):
            recording_id = str(row.recording_id)
            activity_code = parse_activity_code(recording_id)
            activity_name = ACTIVITY_MAPPING.get(activity_code)
            if activity_name is None:
                raise KeyError(f"Activity code {activity_code} parsed from {recording_id} is absent from ACTIVITY_MAPPING.")
            judgment, rationale = activity_judgment(activity_code, activity_name)
            rows.append(
                {
                    "threshold_label": threshold.label,
                    "threshold": threshold.value,
                    "threshold_name": threshold.name,
                    "subject_id": str(row.subject_id),
                    "recording_id": recording_id,
                    "true_label": int(row.true_label),
                    "activity_code": activity_code,
                    "activity_name": activity_name,
                    "data_integrity_bug": bool(activity_code.startswith("F")),
                    "activity_judgment": judgment,
                    "judgment_rationale": rationale,
                    "total_windows": int(row.num_windows),
                    "num_fall_windows": int(row.num_fall_windows),
                }
            )
    columns = [
        "threshold_label",
        "threshold",
        "threshold_name",
        "subject_id",
        "recording_id",
        "true_label",
        "activity_code",
        "activity_name",
        "data_integrity_bug",
        "activity_judgment",
        "judgment_rationale",
        "total_windows",
        "num_fall_windows",
    ]
    return pd.DataFrame(rows, columns=columns)


def load_test_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        "features": DATA_DIR / "test.npy",
        "labels": DATA_DIR / "test_labels.npy",
        "recording_ids": DATA_DIR / "test_recording_ids.npy",
        "subject_ids": DATA_DIR / "test_subject_ids.npy",
    }
    for path in paths.values():
        require_file(path)
    features = np.load(paths["features"]).astype(np.float32, copy=False)
    labels = np.load(paths["labels"]).astype(np.int32, copy=False)
    recording_ids = np.load(paths["recording_ids"], allow_pickle=True).astype(str)
    subject_ids = np.load(paths["subject_ids"], allow_pickle=True).astype(str)
    lengths = {name: len(array) for name, array in {"features": features, "labels": labels, "recording_ids": recording_ids, "subject_ids": subject_ids}.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Test arrays are not row-aligned: {lengths}")
    if features.ndim != 3 or tuple(features.shape[1:]) != (64, 6):
        raise ValueError(f"test.npy expected shape (N, 64, 6), found {features.shape}")
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("test_labels.npy must be a one-dimensional binary array.")
    return features, labels, recording_ids, subject_ids


def positive_segments(indices: np.ndarray) -> tuple[int, int]:
    if len(indices) == 0:
        return 0, 0
    starts = np.r_[True, np.diff(indices) != 1]
    segment_ids = np.cumsum(starts)
    lengths = pd.Series(segment_ids).value_counts().to_numpy()
    return int(len(lengths)), int(lengths.max())


def classify_blip(total_windows: int, positive_indices: np.ndarray) -> str:
    _, longest_run = positive_segments(positive_indices)
    positive_count = int(len(positive_indices))
    positive_fraction = positive_count / total_windows if total_windows else 0.0
    # Sustained means either consecutive false positives or a broad recording-level trigger; otherwise it is a blip.
    if longest_run >= 2 or positive_fraction > 0.15:
        return "sustained_misprediction"
    return "single_window_blip"


def diagnose_window_patterns(activity_frame: pd.DataFrame, thresholds: list[ThresholdSpec]) -> pd.DataFrame:
    features, labels, recording_ids, subject_ids = load_test_arrays()
    unique_recordings = sorted(activity_frame["recording_id"].astype(str).unique().tolist())
    model_path = CNN_LSTM_DIR / "cnn_lstm_best.keras"
    require_file(model_path)
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    rows: list[dict[str, Any]] = []
    classifications: dict[str, dict[str, str]] = {}
    for recording_id in unique_recordings:
        mask = recording_ids == recording_id
        if not mask.any():
            raise ValueError(f"False-trigger recording {recording_id} not found in test_recording_ids.npy")
        if not np.all(labels[mask] == 0):
            raise ValueError(f"False-trigger recording {recording_id} contains non-ADL labels in test_labels.npy")
        subjects = np.unique(subject_ids[mask])
        if len(subjects) != 1:
            raise ValueError(f"False-trigger recording {recording_id} maps to multiple subjects: {subjects.tolist()}")
        probabilities = model.predict(features[mask], batch_size=BATCH_SIZE, verbose=0).ravel()
        total_windows = int(mask.sum())
        classifications[recording_id] = {}
        activity_code = parse_activity_code(recording_id)
        activity_name = ACTIVITY_MAPPING[activity_code]
        for threshold in thresholds:
            positive_indices = np.flatnonzero(probabilities >= threshold.value)
            segment_count, longest_run = positive_segments(positive_indices)
            classification = classify_blip(total_windows, positive_indices)
            classifications[recording_id][threshold.name] = classification
            rows.append(
                {
                    "threshold_label": threshold.label,
                    "threshold": threshold.value,
                    "threshold_name": threshold.name,
                    "subject_id": str(subjects[0]),
                    "recording_id": recording_id,
                    "activity_code": activity_code,
                    "activity_name": activity_name,
                    "total_windows": total_windows,
                    "positive_window_count": int(len(positive_indices)),
                    "positive_window_fraction": float(len(positive_indices) / total_windows),
                    "positive_window_indices": " ".join(str(int(index)) for index in positive_indices),
                    "positive_segment_count": segment_count,
                    "longest_positive_run": longest_run,
                    "max_probability": float(probabilities.max()) if len(probabilities) else np.nan,
                    "blip_classification": classification,
                    "classification_changes_between_thresholds": False,
                }
            )
    frame = pd.DataFrame(rows)
    for recording_id, values in classifications.items():
        changed = len(set(values.values())) > 1
        frame.loc[frame["recording_id"] == recording_id, "classification_changes_between_thresholds"] = changed
    return frame


def metric_value(metadata: dict[str, Any], path: Path, metric_key: str, field: str) -> float:
    value = require_field(metadata, path, ["metrics", metric_key, field])
    return float(value)


def metric_f1(metadata: dict[str, Any], path: Path, metric_key: str) -> float:
    value = require_field(metadata, path, ["metrics", metric_key, "classification_report", "fall", "f1-score"])
    return float(value)


def build_comparison_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configs = [
        ("CNN_baseline", CNN_DIR, DEFAULT_THRESHOLD, "test_threshold_0.5"),
        ("CNN_baseline", CNN_DIR, load_chosen_threshold(CNN_DIR), "test_chosen_threshold"),
        ("CNN_LSTM", CNN_LSTM_DIR, DEFAULT_THRESHOLD, "test_threshold_0.5"),
        ("CNN_LSTM", CNN_LSTM_DIR, load_chosen_threshold(CNN_LSTM_DIR), "test_chosen_threshold"),
    ]
    for model_name, model_dir, threshold, metric_key in configs:
        threshold_label = f"{model_name}@{threshold:.2f}"
        name = threshold_name(threshold)
        metadata_path = model_dir / "run_metadata.json"
        eval_path = model_dir / "recording_level_eval" / "evaluation_summary.json"
        impact_path = model_dir / "recording_level_eval" / "impact_verification_summary.json"
        metadata = read_json(metadata_path)
        evaluation = read_json(eval_path)
        impact = read_json(impact_path)
        rows.append(
            {
                "model_threshold": threshold_label,
                "precision": metric_value(metadata, metadata_path, metric_key, "precision"),
                "recall": metric_value(metadata, metadata_path, metric_key, "recall"),
                "F1": metric_f1(metadata, metadata_path, metric_key),
                "F2": metric_value(metadata, metadata_path, metric_key, "f2"),
                "recording_recall": float(require_field(evaluation, eval_path, ["recording_level", name, "recording_level_recall"])),
                "impact_verified_recall": float(require_field(impact, impact_path, ["thresholds", name, "corrected_true_impact_detection_recall"])),
                "adl_false_trigger_count": int(require_field(evaluation, eval_path, ["recording_level", name, "adl_recordings_falsely_triggered"])),
                "source": "reused",
            }
        )
    return pd.DataFrame(rows)


def verdict(activity_frame: pd.DataFrame, severity_frame: pd.DataFrame, comparison: pd.DataFrame) -> str:
    unexpected = int((activity_frame["activity_judgment"] == "unexpected").sum())
    integrity_bugs = int(activity_frame["data_integrity_bug"].sum())
    sustained = int((severity_frame["blip_classification"] == "sustained_misprediction").sum())
    cnn_05 = comparison.loc[comparison["model_threshold"] == "CNN_baseline@0.50"].iloc[0]
    lstm_05 = comparison.loc[comparison["model_threshold"] == "CNN_LSTM@0.50"].iloc[0]
    lstm_chosen = comparison.loc[comparison["model_threshold"] == "CNN_LSTM@0.25"].iloc[0]
    if integrity_bugs:
        return "Do not treat the false positives as model findings until the F-code data-integrity issue is resolved."
    if unexpected and sustained:
        return (
            "The CNN-LSTM recall gain is useful but should be treated cautiously because at least one false trigger is "
            "both unexpected by activity type and sustained at the window level; inspect those recordings before "
            "preferring the chosen threshold."
        )
    if unexpected:
        return (
            "The CNN-LSTM recall gain is useful but should be treated cautiously: the false triggers are only "
            "single-window blips, but at least one is an unexpected low-impact ADL activity. Prefer threshold 0.5 "
            "unless the extra chosen-threshold window recall is more important than the two added ADL trigger recordings."
        )
    if sustained:
        return (
            "The CNN-LSTM recall gain is useful but should be treated cautiously because at least one false trigger is "
            "sustained across adjacent or many windows; inspect those recordings before preferring the chosen threshold."
        )
    return (
        "The CNN-LSTM tradeoff appears worth it: compared with the CNN at 0.5, it raises window recall from "
        f"{cnn_05['recall']:.4f} to {lstm_05['recall']:.4f} and reaches {lstm_05['recording_recall']:.4f} "
        f"recording/impact recall, while its chosen threshold reaches {lstm_chosen['recall']:.4f} window recall. "
        "The added ADL false triggers are plausible borderline activities and single-window/non-sustained blips."
    )


def main() -> None:
    chosen = load_chosen_threshold(CNN_LSTM_DIR)
    thresholds = [
        ThresholdSpec("0.5", DEFAULT_THRESHOLD, threshold_name(DEFAULT_THRESHOLD)),
        ThresholdSpec("chosen", chosen, threshold_name(chosen)),
    ]
    activity = find_false_trigger_recordings(CNN_LSTM_DIR / "recording_level_eval" / "recording_level_metrics.csv", thresholds)
    severity = diagnose_window_patterns(activity, thresholds) if not activity.empty else pd.DataFrame()
    comparison = build_comparison_table()
    summary = {
        "thresholds_evaluated": {spec.label: spec.value for spec in thresholds},
        "false_trigger_recordings_by_threshold": activity.groupby("threshold_name")["recording_id"].nunique().to_dict() if not activity.empty else {},
        "activity_judgments": activity["activity_judgment"].value_counts().to_dict() if not activity.empty else {},
        "blip_classifications": severity["blip_classification"].value_counts().to_dict() if not severity.empty else {},
        "recordings_changing_classification_between_thresholds": sorted(severity.loc[severity["classification_changes_between_thresholds"], "recording_id"].unique().tolist()) if not severity.empty else [],
        "verdict": verdict(activity, severity, comparison),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    activity.to_csv(OUTPUT_DIR / "false_positive_activity_breakdown.csv", index=False)
    severity.to_csv(OUTPUT_DIR / "false_positive_severity_classification.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "full_window_level_comparison.csv", index=False)
    (OUTPUT_DIR / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPART 1 - Activity breakdown")
    print(activity.to_string(index=False) if not activity.empty else "No CNN-LSTM ADL false-trigger recordings found.")
    print("\nPART 2 - Blip vs sustained")
    print(severity.to_string(index=False) if not severity.empty else "No false-trigger windows to diagnose.")
    print("\nVerdict")
    print(summary["verdict"])
    print("\nPART 3 - Full precision/recall/F1/F2 comparison")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
