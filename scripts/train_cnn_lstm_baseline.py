"""Train and evaluate the second, compact CNN-LSTM fall-detection baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "cnn_lstm_baseline"
EVAL_DIR = RESULTS_DIR / "recording_level_eval"
SEED = 42
INPUT_SHAPE = (64, 6)
BATCH_SIZE = 64
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
DEFAULT_THRESHOLD = 0.5
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)
CLASS_NAMES = ["ADL/background", "fall"]


def set_seeds() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(SEED)
    np.random.seed(SEED)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(DATA_DIR / f"{split}.npy")
    labels = np.load(DATA_DIR / f"{split}_labels.npy")
    if features.ndim != 3 or tuple(features.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"{split}: expected data shape (N, 64, 6), found {features.shape}.")
    if labels.ndim != 1 or len(features) != len(labels):
        raise ValueError(f"{split}: data and labels are not aligned: {features.shape} vs {labels.shape}.")
    if not np.isfinite(features).all():
        raise ValueError(f"{split}: data contains NaN or Inf values.")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{split}: labels must be finite binary values 0/1.")
    return features.astype(np.float32, copy=False), labels.astype(np.int32, copy=False)


def load_test_metadata() -> tuple[np.ndarray, np.ndarray]:
    recording_ids = np.load(DATA_DIR / "test_recording_ids.npy", allow_pickle=True).astype(str)
    subject_ids = np.load(DATA_DIR / "test_subject_ids.npy", allow_pickle=True).astype(str)
    if len(recording_ids) != len(subject_ids):
        raise ValueError(f"test_recording_ids and test_subject_ids are not aligned: {len(recording_ids)} vs {len(subject_ids)}")
    return recording_ids, subject_ids


def print_balance(split: str, features: np.ndarray, labels: np.ndarray) -> None:
    counts = np.bincount(labels, minlength=2)
    print(f"{split:>5}: data={features.shape}, labels={labels.shape}, ADL/background={counts[0]:,} ({counts[0] / len(labels) * 100:.2f}%), fall={counts[1]:,} ({counts[1] / len(labels) * 100:.2f}%)")


def build_model(tf: Any):
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv1D(32, 5, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling1D(2),
            layers.LSTM(64, return_sequences=True, unroll=True),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_lstm_baseline",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int32)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
    }


def choose_threshold(table: pd.DataFrame) -> dict[str, float]:
    row = table.sort_values(["f2", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
    return {key: float(row[key]) for key in ["threshold", "precision", "recall", "f1", "f2"]}


def threshold_name(threshold: float) -> str:
    return "threshold_0_5" if np.isclose(threshold, DEFAULT_THRESHOLD) else f"threshold_{threshold:g}".replace(".", "_")


def detection_type_labels(genuine: pd.Series, false_trigger: pd.Series, missed: pd.Series) -> np.ndarray:
    labels = np.full(len(genuine), "unknown", dtype=object)
    labels[np.asarray(genuine, dtype=bool)] = "genuine_impact"
    labels[np.asarray(false_trigger, dtype=bool)] = "false_trigger_only"
    labels[np.asarray(missed, dtype=bool)] = "missed"
    return labels


def save_plots(history: pd.DataFrame, matrix: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric, title in zip(axes.flat, ["loss", "accuracy", "precision", "recall"], ["Loss", "Accuracy", "Precision", "Recall"]):
        axis.plot(history["epoch"], history[metric], label="train")
        axis.plot(history["epoch"], history[f"val_{metric}"], label="validation")
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "training_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title="CNN-LSTM test confusion matrix at 0.5")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrix_test_0.5.png", dpi=160)
    plt.close(figure)


def evaluate_test(model: Any, test_x: np.ndarray, test_y: np.ndarray, threshold: float) -> dict[str, Any]:
    probabilities = model.predict(test_x, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype(np.int32)
    metrics = threshold_metrics(test_y, probabilities, threshold)
    metrics["confusion_matrix"] = confusion_matrix(test_y, predictions, labels=[0, 1]).tolist()
    metrics["classification_report"] = classification_report(test_y, predictions, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0, output_dict=True)
    return metrics


def run_recording_evaluation(model: Any, test_x: np.ndarray, test_y: np.ndarray, recording_ids: np.ndarray, subject_ids: np.ndarray, chosen: float) -> dict[str, Any]:
    probabilities = model.predict(test_x, batch_size=BATCH_SIZE, verbose=0).ravel()
    thresholds = [DEFAULT_THRESHOLD, chosen] if not np.isclose(chosen, DEFAULT_THRESHOLD) else [DEFAULT_THRESHOLD]
    rows: list[dict[str, Any]] = []
    for recording_id in pd.unique(recording_ids):
        mask = recording_ids == recording_id
        row = {"recording_id": recording_id, "subject_id": pd.unique(subject_ids[mask])[0], "true_label": int(np.any(test_y[mask] == 1)), "num_windows": int(mask.sum()), "num_fall_windows": int((test_y[mask] == 1).sum())}
        for threshold in thresholds:
            name = threshold_name(threshold)
            prediction = probabilities[mask] >= threshold
            row[f"predicted_positive_{name}"] = int(np.any(prediction))
            row[f"num_detected_fall_windows_{name}"] = int(np.sum((test_y[mask] == 1) & prediction))
        rows.append(row)
    recording = pd.DataFrame(rows)
    summaries: dict[str, Any] = {}
    subject_frame: pd.DataFrame | None = None
    missed_by_threshold: dict[str, pd.DataFrame] = {}
    for threshold in thresholds:
        name = threshold_name(threshold)
        recording[f"detected_fall_recording_{name}"] = ((recording.true_label == 1) & (recording[f"predicted_positive_{name}"] == 1)).astype(int)
        recording[f"fully_missed_fall_recording_{name}"] = ((recording.true_label == 1) & (recording[f"predicted_positive_{name}"] == 0)).astype(int)
        recording[f"false_positive_recording_{name}"] = ((recording.true_label == 0) & (recording[f"predicted_positive_{name}"] == 1)).astype(int)
        falls = recording[recording.true_label == 1]
        adl = recording[recording.true_label == 0]
        detected = int(falls[f"detected_fall_recording_{name}"].sum())
        missed = int(falls[f"fully_missed_fall_recording_{name}"].sum())
        false_positive = int(adl[f"false_positive_recording_{name}"].sum())
        summaries[name] = {"threshold": float(threshold), "total_recordings": int(len(recording)), "total_fall_recordings": int(len(falls)), "total_adl_recordings": int(len(adl)), "fall_recordings_fully_missed": missed, "fall_recordings_at_least_partially_detected": detected, "recording_level_recall": detected / len(falls), "adl_recordings_falsely_triggered": false_positive, "recording_level_false_positive_rate": false_positive / len(adl)}
        missed_by_threshold[name] = falls.loc[falls[f"fully_missed_fall_recording_{name}"] == 1, ["subject_id", "recording_id", "num_fall_windows"]]
        subject_rows = []
        for subject, group in recording.groupby("subject_id", sort=True):
            subject_falls = group[group.true_label == 1]
            total = len(subject_falls)
            subject_missed = int(subject_falls[f"fully_missed_fall_recording_{name}"].sum())
            subject_rows.append({"subject_id": subject, "total_recordings": len(group), "fall_recordings": total, "adl_recordings": int((group.true_label == 0).sum()), f"fall_recordings_missed_{name}": subject_missed, f"fall_recordings_detected_{name}": total - subject_missed, f"subject_recall_{name}": (total - subject_missed) / total if total else np.nan, f"review_below_90_percent_{name}": bool(total and (total - subject_missed) / total < 0.9)})
        current_subjects = pd.DataFrame(subject_rows)
        subject_frame = current_subjects if subject_frame is None else subject_frame.merge(current_subjects, on=["subject_id", "total_recordings", "fall_recordings", "adl_recordings"], how="outer")
    if chosen != DEFAULT_THRESHOLD:
        missed = missed_by_threshold[threshold_name(DEFAULT_THRESHOLD)].merge(missed_by_threshold[threshold_name(chosen)], on=["subject_id", "recording_id", "num_fall_windows"], how="outer", suffixes=("_threshold_0_5", "_chosen_threshold"))
    else:
        missed = missed_by_threshold[threshold_name(DEFAULT_THRESHOLD)]
    comparison = pd.DataFrame([
        {"level": "window", "threshold_0_5_recall": threshold_metrics(test_y, probabilities, DEFAULT_THRESHOLD)["recall"], "chosen_threshold_recall": threshold_metrics(test_y, probabilities, chosen)["recall"]},
        {"level": "recording", "threshold_0_5_recall": summaries[threshold_name(DEFAULT_THRESHOLD)]["recording_level_recall"], "chosen_threshold_recall": summaries[threshold_name(chosen)]["recording_level_recall"]},
        {"level": "subject", "threshold_0_5_recall": float(subject_frame[f"subject_recall_{threshold_name(DEFAULT_THRESHOLD)}"].dropna().mean()), "chosen_threshold_recall": float(subject_frame[f"subject_recall_{threshold_name(chosen)}"].dropna().mean())},
    ])
    impact_rows = []
    impact_summary: dict[str, Any] = {}
    false_trigger_frames = []
    for threshold in thresholds:
        name = threshold_name(threshold)
        falls = recording[recording.true_label == 1].copy()
        predicted = falls[f"predicted_positive_{name}"].astype(bool)
        detected_windows = falls[f"num_detected_fall_windows_{name}"]
        genuine = predicted & (detected_windows > 0)
        false_trigger = predicted & (detected_windows == 0)
        falls[f"detection_type_{name}"] = detection_type_labels(genuine, false_trigger, ~predicted)
        impact_rows.append(falls[["subject_id", "recording_id", "num_fall_windows", f"predicted_positive_{name}", f"num_detected_fall_windows_{name}", f"detection_type_{name}"]])
        selected = falls[false_trigger][["subject_id", "recording_id", "num_fall_windows"]].copy()
        selected.insert(0, "threshold", threshold)
        false_trigger_frames.append(selected)
        total = len(falls)
        impact_summary[name] = {"threshold": threshold, "total_fall_recordings": total, "genuine_impact_detections": int(genuine.sum()), "genuine_impact_detection_rate": float(genuine.mean()), "false_trigger_only_detections": int(false_trigger.sum()), "false_trigger_only_rate": float(false_trigger.mean()), "fully_missed_recordings": int((~predicted).sum()), "original_any_window_recording_recall": float(predicted.mean()), "corrected_true_impact_detection_recall": float(genuine.mean()), "recall_difference_corrected_minus_original": float(genuine.mean() - predicted.mean())}
    impact = impact_rows[0]
    for extra in impact_rows[1:]:
        impact = impact.merge(extra, on=["subject_id", "recording_id", "num_fall_windows"], how="outer")
    false_trigger_output = pd.concat(false_trigger_frames, ignore_index=True) if false_trigger_frames else pd.DataFrame(columns=["threshold", "subject_id", "recording_id", "num_fall_windows"])
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    recording.to_csv(EVAL_DIR / "recording_level_metrics.csv", index=False)
    subject_frame.to_csv(EVAL_DIR / "subject_level_metrics.csv", index=False)
    missed.to_csv(EVAL_DIR / "missed_fall_recordings.csv", index=False)
    comparison.to_csv(EVAL_DIR / "summary_comparison.csv", index=False)
    evaluation_summary = {"thresholds_used": {"default": DEFAULT_THRESHOLD, "chosen": chosen}, "window_level_recall": {threshold_name(DEFAULT_THRESHOLD): comparison.iloc[0].threshold_0_5_recall, threshold_name(chosen): comparison.iloc[0].chosen_threshold_recall}, "recording_level": summaries, "subject_level": {threshold_name(t): {"mean_subject_recall": float(subject_frame[f"subject_recall_{threshold_name(t)}"].dropna().mean()), "min_subject_recall": float(subject_frame[f"subject_recall_{threshold_name(t)}"].dropna().min()), "max_subject_recall": float(subject_frame[f"subject_recall_{threshold_name(t)}"].dropna().max()), "subjects_below_90_percent": subject_frame.loc[subject_frame[f"review_below_90_percent_{threshold_name(t)}"], "subject_id"].tolist()} for t in thresholds}}
    (EVAL_DIR / "evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")
    impact.to_csv(EVAL_DIR / "impact_window_verification.csv", index=False)
    false_trigger_output.to_csv(EVAL_DIR / "false_trigger_only_recordings.csv", index=False)
    (EVAL_DIR / "impact_verification_summary.json").write_text(json.dumps({"thresholds": impact_summary, "source": "recomputed with CNN-LSTM inference"}, indent=2), encoding="utf-8")
    return {"recording": summaries, "impact": impact_summary, "comparison": comparison, "missed": missed}


def compare_with_cnn(cnn_lstm: dict[str, Any], cnn_lstm_time: float, cnn_lstm_epochs: int, cnn_lstm_params: int, cnn_lstm_chosen_threshold: float) -> None:
    baseline_metadata = json.loads((ROOT / "results" / "cnn_baseline" / "run_metadata.json").read_text(encoding="utf-8"))
    baseline_eval = json.loads((ROOT / "results" / "cnn_baseline" / "recording_level_eval" / "evaluation_summary.json").read_text(encoding="utf-8"))
    baseline_impact = json.loads((ROOT / "results" / "cnn_baseline" / "recording_level_eval" / "impact_verification_summary.json").read_text(encoding="utf-8"))
    rows = []
    for threshold_key, label in [("0.5", "0.5"), ("0.2", "chosen")]:
        baseline_metric = baseline_metadata["metrics"]["test_threshold_0.5" if threshold_key == "0.5" else "test_chosen_threshold"]
        lstm_metric = cnn_lstm["test_metrics"][label]
        baseline_name = "threshold_0_5" if label == "0.5" else threshold_name(float(baseline_metadata["chosen_threshold"]["threshold"]))
        lstm_name = "threshold_0_5" if label == "0.5" else threshold_name(cnn_lstm_chosen_threshold)
        rows.append({"model": "CNN", "threshold": label, "parameters": 33729, "window_precision": baseline_metric["precision"], "window_recall": baseline_metric["recall"], "window_f1": baseline_metric["classification_report"]["fall"]["f1-score"], "window_f2": baseline_metric["f2"], "recording_recall": baseline_eval["recording_level"][baseline_name]["recording_level_recall"], "impact_recall": baseline_impact["thresholds"][baseline_name]["corrected_true_impact_detection_recall"], "training_time_seconds": np.nan, "epochs": baseline_metadata["best_epoch"]})
        rows.append({"model": "CNN-LSTM", "threshold": label, "parameters": cnn_lstm_params, "window_precision": lstm_metric["precision"], "window_recall": lstm_metric["recall"], "window_f1": lstm_metric["classification_report"]["fall"]["f1-score"], "window_f2": lstm_metric["f2"], "recording_recall": cnn_lstm["recording"][lstm_name]["recording_level_recall"], "impact_recall": cnn_lstm["impact"][lstm_name]["corrected_true_impact_detection_recall"], "training_time_seconds": cnn_lstm_time, "epochs": cnn_lstm_epochs})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "cnn_vs_cnn_lstm_comparison.csv", index=False)
    cnn_lstm_missed = "SA14:F01_SA14_R02.txt" not in set(cnn_lstm["missed"]["recording_id"].astype(str))
    print("\nCNN vs CNN-LSTM comparison")
    print(comparison.to_string(index=False))
    print(f"SA14 baseline miss detected by CNN-LSTM at 0.5: {cnn_lstm_missed}")
    print("Conclusion: CNN-LSTM comparison is recorded above; interpret improvement using impact-verified recall and false-trigger counts, not window recall alone.")


def load_chosen_threshold() -> dict[str, float]:
    metadata_path = RESULTS_DIR / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chosen = metadata.get("chosen_threshold")
        if isinstance(chosen, dict) and "threshold" in chosen:
            return {key: float(chosen[key]) for key in ["threshold", "precision", "recall", "f1", "f2"] if key in chosen}
    sweep_path = RESULTS_DIR / "threshold_sweep.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(f"Chosen threshold not found in {metadata_path} or {sweep_path}.")
    return choose_threshold(pd.read_csv(sweep_path))


def evaluate_saved_best(tf: Any) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_x, test_y = load_split("test")
    recording_ids, subject_ids = load_test_metadata()
    chosen = load_chosen_threshold()
    model_path = RESULTS_DIR / "cnn_lstm_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved best CNN-LSTM model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    default_test = evaluate_test(model, test_x, test_y, DEFAULT_THRESHOLD)
    chosen_test = evaluate_test(model, test_x, test_y, chosen["threshold"])
    evaluation = run_recording_evaluation(model, test_x, test_y, recording_ids, subject_ids, chosen["threshold"])

    metadata_path = RESULTS_DIR / "run_metadata.json"
    training_seconds = np.nan
    epochs_completed = 0
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("metrics", {})
        metadata["metrics"]["test_threshold_0.5"] = default_test
        metadata["metrics"]["test_chosen_threshold"] = chosen_test
        metadata["post_training_evaluation_completed"] = True
        metadata["post_training_evaluation_model"] = str(model_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        training_seconds = float(metadata.get("training_time_seconds", np.nan))
        epochs_completed = int(metadata.get("epochs_completed", 0))

    compare_with_cnn(
        {"test_metrics": {"0.5": default_test, "chosen": chosen_test}, **evaluation},
        training_seconds,
        epochs_completed,
        model.count_params(),
        chosen["threshold"],
    )
    print(f"Finished evaluation-only run from: {model_path}")
    print(f"Evaluation outputs saved to: {EVAL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true", help="Load cnn_lstm_best.keras and finish post-training evaluation without fitting.")
    args = parser.parse_args()
    set_seeds()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required. Install dependencies with `pip install -r requirements.txt`.") from exc
    tf.keras.utils.set_random_seed(SEED)
    if args.evaluate_only:
        evaluate_saved_best(tf)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    test_x, test_y = load_split("test")
    recording_ids, subject_ids = load_test_metadata()
    if len(test_x) != len(recording_ids) or len(test_x) != len(subject_ids):
        raise ValueError("test.npy, test_labels.npy, test_recording_ids.npy, and test_subject_ids.npy are not aligned.")
    print("Loaded natural, imbalanced splits:")
    for split, features, labels in [("train", train_x, train_y), ("val", val_x, val_y), ("test", test_x, test_y)]:
        print_balance(split, features, labels)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=train_y)
    class_weights = {0: float(weights[0]), 1: float(weights[1])}
    model = build_model(tf)
    model.summary()
    print(f"CNN baseline parameter count: 33,729; CNN-LSTM parameter count: {model.count_params():,}")
    best_path = RESULTS_DIR / "cnn_lstm_best.keras"
    last_path = RESULTS_DIR / "cnn_lstm_last.keras"
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1), tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1), tf.keras.callbacks.ModelCheckpoint(last_path, monitor="val_loss", save_best_only=False, verbose=0)]
    started = time.perf_counter()
    interrupted = False
    try:
        history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weights, callbacks=callbacks, verbose=2)
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted; saving partial history and last model.")
        history = getattr(model, "history", None)
        if history is None:
            raise
    training_seconds = time.perf_counter() - started
    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    model.save(RESULTS_DIR / "cnn_lstm_final.keras")
    val_probabilities = model.predict(val_x, batch_size=BATCH_SIZE, verbose=0).ravel()
    sweep = pd.DataFrame([threshold_metrics(val_y, val_probabilities, threshold) for threshold in THRESHOLDS])
    sweep.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    chosen = choose_threshold(sweep)
    default_test = evaluate_test(model, test_x, test_y, DEFAULT_THRESHOLD)
    chosen_test = evaluate_test(model, test_x, test_y, chosen["threshold"])
    save_plots(history_frame, np.asarray(default_test["confusion_matrix"]))
    metadata = {"seed": SEED, "input_shape": list(INPUT_SHAPE), "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "learning_rate": LEARNING_RATE, "loss": "binary_crossentropy", "optimizer": "Adam", "class_weights": class_weights, "unroll": True, "lstm_layers": 1, "parameter_count": model.count_params(), "cnn_baseline_parameter_count": 33729, "chosen_threshold": chosen, "best_epoch": int(history_frame.loc[history_frame.val_loss.idxmin(), "epoch"]), "epochs_completed": len(history_frame), "training_time_seconds": training_seconds, "interrupted": interrupted, "tensorflow_version": tf.__version__, "metrics": {"test_threshold_0.5": default_test, "test_chosen_threshold": chosen_test}, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(shlex.quote(part) for part in sys.argv)}
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nCNN-LSTM test classification report @ threshold 0.5:")
    print(pd.DataFrame(default_test["classification_report"]).T.to_string())
    evaluation = run_recording_evaluation(model, test_x, test_y, recording_ids, subject_ids, chosen["threshold"])
    compare_with_cnn({"test_metrics": {"0.5": default_test, "chosen": chosen_test}, **evaluation}, training_seconds, len(history_frame), model.count_params(), chosen["threshold"])
    print(f"CNN-LSTM training time: {training_seconds:.2f}s; epochs completed: {len(history_frame)}; best epoch: {metadata['best_epoch']}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
