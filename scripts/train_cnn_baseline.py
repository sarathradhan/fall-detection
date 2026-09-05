"""Train and evaluate a lightweight 1D CNN fall-detection baseline.

This script consumes the existing processed arrays and does not modify the
preprocessing pipeline or any source data.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "cnn_baseline"
SEED = 42
INPUT_SHAPE = (64, 6)
BATCH_SIZE = 64
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
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
        raise ValueError(f"{split}: data and label arrays are not aligned: {features.shape} vs {labels.shape}.")
    if not np.isfinite(features).all():
        raise ValueError(f"{split}: data contains NaN or Inf values.")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{split}: labels must be finite binary values 0/1.")
    return features.astype(np.float32, copy=False), labels.astype(np.int32, copy=False)


def print_split_sanity(split: str, features: np.ndarray, labels: np.ndarray) -> None:
    counts = np.bincount(labels, minlength=2)
    print(
        f"{split:>5}: data={features.shape}, labels={labels.shape}, "
        f"ADL/background={counts[0]:,} ({counts[0] / len(labels) * 100:.2f}%), "
        f"fall={counts[1]:,} ({counts[1] / len(labels) * 100:.2f}%)"
    )


def build_model(tf: Any):
    keras = tf.keras
    layers = keras.layers
    model = keras.Sequential(
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
            layers.Conv1D(96, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_baseline",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def metric_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int32)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
    }


def choose_threshold(table: pd.DataFrame) -> dict[str, float]:
    chosen = table.sort_values(["f2", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
    return {key: float(chosen[key]) for key in ["threshold", "precision", "recall", "f2"]}


def save_history(history: Any, path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    frame.to_csv(path, index=False)
    return frame


def plot_training(history: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [("loss", "Loss"), ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall")]
    for axis, (metric, title) in zip(axes.flat, plots):
        if metric in history:
            axis.plot(history["epoch"], history[metric], label="train")
            if f"val_{metric}" in history:
                axis.plot(history["epoch"], history[f"val_{metric}"], label="validation")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_confusion(matrix: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title="Test confusion matrix at threshold 0.5")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_model(model: Any, features: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    probabilities = model.predict(features, batch_size=BATCH_SIZE, verbose=0).ravel()
    metrics = metric_at_threshold(labels, probabilities, threshold)
    predictions = (probabilities >= threshold).astype(np.int32)
    metrics["confusion_matrix"] = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    metrics["classification_report"] = classification_report(labels, predictions, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0, output_dict=True)
    metrics["probabilities"] = probabilities
    return metrics


def main() -> None:
    set_seeds()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required. Install project dependencies with `pip install -r requirements.txt`.") from exc

    tf.keras.utils.set_random_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    test_x, test_y = load_split("test")
    print("Loaded natural, imbalanced splits:")
    for split, features, labels in [("train", train_x, train_y), ("val", val_x, val_y), ("test", test_x, test_y)]:
        print_split_sanity(split, features, labels)

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_y)
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}
    print(f"Class weights: {class_weights}")

    model = build_model(tf)
    model.summary()
    best_path = RESULTS_DIR / "cnn_baseline_best.keras"
    last_path = RESULTS_DIR / "cnn_baseline_last.keras"
    history_path = RESULTS_DIR / "training_history.csv"
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(last_path, monitor="val_loss", save_best_only=False, verbose=0),
    ]

    interrupted = False
    try:
        history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weights, callbacks=callbacks, verbose=2)
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted. Saving partial history and the last checkpoint.")
        history = getattr(model, "history", None)
        if history is None:
            raise

    history_frame = save_history(history, history_path)
    model.save(RESULTS_DIR / "cnn_baseline_final.keras")
    plot_training(history_frame, RESULTS_DIR / "training_curves.png")

    val_probabilities = model.predict(val_x, batch_size=BATCH_SIZE, verbose=0).ravel()
    threshold_table = pd.DataFrame([metric_at_threshold(val_y, val_probabilities, threshold) for threshold in THRESHOLDS])
    threshold_table.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    chosen = choose_threshold(threshold_table)

    test_default = evaluate_model(model, test_x, test_y, 0.5)
    test_chosen = evaluate_model(model, test_x, test_y, chosen["threshold"])
    plot_confusion(np.asarray(test_default["confusion_matrix"]), RESULTS_DIR / "confusion_matrix_test_0.5.png")
    print("\nTest classification report @ threshold 0.5:")
    print(pd.DataFrame(test_default["classification_report"]).T.to_string())
    print(f"Test confusion matrix @ threshold 0.5:\n{np.asarray(test_default['confusion_matrix'])}")
    print(f"Test classification report @ chosen threshold {chosen['threshold']:.2f}:")
    print(pd.DataFrame(test_chosen["classification_report"]).T.to_string())
    print(f"Test confusion matrix @ chosen threshold:\n{np.asarray(test_chosen['confusion_matrix'])}")
    test_default.pop("probabilities")
    test_chosen.pop("probabilities")
    metadata = {
        "seed": SEED,
        "input_shape": list(INPUT_SHAPE),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "loss": "binary_crossentropy",
        "optimizer": "Adam",
        "class_weights": class_weights,
        "threshold_selection": "maximum validation F2; ties favor recall, then precision, then lower threshold",
        "chosen_threshold": chosen,
        "interrupted": interrupted,
        "best_epoch": int(history_frame.loc[history_frame["val_loss"].idxmin(), "epoch"]),
        "tensorflow_version": tf.__version__,
        "metrics": {"test_threshold_0.5": test_default, "test_chosen_threshold": test_chosen},
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nFinal summary")
    print(f"Best epoch: {metadata['best_epoch']}")
    print(f"Test @ 0.5: precision={test_default['precision']:.4f}, recall={test_default['recall']:.4f}, F2={test_default['f2']:.4f}")
    print(f"Chosen validation threshold: {chosen['threshold']:.2f} (precision={chosen['precision']:.4f}, recall={chosen['recall']:.4f}, F2={chosen['f2']:.4f})")
    print(f"Test @ chosen threshold: precision={test_chosen['precision']:.4f}, recall={test_chosen['recall']:.4f}, F2={test_chosen['f2']:.4f}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()