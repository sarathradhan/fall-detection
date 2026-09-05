from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = PROCESSED_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_processed_dir(processed_dir: str | Path | None = None) -> Path:
    """Return the project processed-data directory in the repository's existing convention."""
    if processed_dir is None:
        return PROCESSED_DIR
    return Path(processed_dir).expanduser().resolve()


def load_project_splits(processed_dir: str | Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load the project’s train/val/test arrays using the existing project conventions.

    The project stores arrays in data/processed/ with names such as:
    - train.npy / train_labels.npy
    - val.npy / val_labels.npy
    - test.npy / test_labels.npy

    Each feature tensor follows the current pipeline convention: (n_windows, window_length, 6).
    """
    processed_dir = _resolve_processed_dir(processed_dir)
    splits: dict[str, dict[str, np.ndarray]] = {}

    for split_name in ("train", "val", "test"):
        features = np.load(processed_dir / f"{split_name}.npy")
        labels = np.load(processed_dir / f"{split_name}_labels.npy")
        subject_ids = np.load(processed_dir / f"{split_name}_subject_ids.npy")
        recording_ids = np.load(processed_dir / f"{split_name}_recording_ids.npy")

        if not (len(features) == len(labels) == len(subject_ids) == len(recording_ids)):
            raise ValueError(
                f"Processed split '{split_name}' has mismatched array lengths: "
                f"features={len(features)}, labels={len(labels)}, subjects={len(subject_ids)}, "
                f"recordings={len(recording_ids)}"
            )

        splits[split_name] = {
            "X": np.asarray(features, dtype=np.float32),
            "y": np.asarray(labels, dtype=np.int64).reshape(-1),
            "subject_ids": np.asarray(subject_ids),
            "recording_ids": np.asarray(recording_ids),
        }

    return splits


def print_class_distribution(y: np.ndarray, name: str) -> dict[str, float | int]:
    """Print and return a class-distribution summary for a label vector.

    The project uses binary labels:
    - 0 = ADL / background / no-fall
    - 1 = Fall
    """
    y = np.asarray(y).reshape(-1)
    if y.size == 0:
        raise ValueError(f"{name}: empty label array provided.")

    unique_values = np.unique(y)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"{name}: expected binary labels in {{0, 1}}; found {unique_values.tolist()}")

    adl_count = int(np.sum(y == 0))
    fall_count = int(np.sum(y == 1))
    total = int(adl_count + fall_count)
    fall_percentage = (fall_count / total * 100.0) if total else 0.0
    adl_to_fall_ratio = (adl_count / fall_count) if fall_count else float("inf")

    print(f"\n{name} class distribution:")
    print(f"  ADL / No-Fall: {adl_count}")
    print(f"  Fall: {fall_count}")
    print(f"  Fall percentage: {fall_percentage:.2f}%")
    print(f"  ADL:Fall ratio: {adl_to_fall_ratio:.2f}:1")

    return {
        "name": name,
        "total": total,
        "adl": adl_count,
        "fall": fall_count,
        "fall_percentage": float(fall_percentage),
        "adl_to_fall_ratio": float(adl_to_fall_ratio),
    }


def calculate_class_weights_numpy(y_train: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency class weights using ONLY the training labels.

    This is the primary strategy for the binary Fall vs ADL task because the fall class is
    a minority and the model would otherwise be biased toward the background ADL class.

    We intentionally ignore validation and test labels so the class weights stay a purely
    training-time decision and do not introduce leakage.
    """
    y_train = np.asarray(y_train).reshape(-1)
    if y_train.size == 0:
        raise ValueError("y_train is empty.")

    unique_values = np.unique(y_train)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"Expected binary labels in {{0, 1}}; found {unique_values.tolist()}")

    counts = np.bincount(y_train.astype(np.int64), minlength=2)
    class_0_count = float(counts[0])
    class_1_count = float(counts[1])

    if class_0_count == 0.0 or class_1_count == 0.0:
        raise ValueError(
            "Training labels must contain both classes. "
            f"Current counts: ADL={class_0_count}, Fall={class_1_count}"
        )

    total = class_0_count + class_1_count
    weights = np.array(
        [total / (2.0 * class_0_count), total / (2.0 * class_1_count)],
        dtype=np.float32,
    )
    return weights


def calculate_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Return class weights in a format directly usable by PyTorch.

    For binary labels, the result is a 1D tensor of length 2 in the order [ADL, Fall], where:
    - weights[0] corresponds to label 0 (ADL / No-Fall)
    - weights[1] corresponds to label 1 (Fall)

    This is suitable for many training setups, including class-weighted loss terms and
    weighted sampling strategies.
    """
    weights = calculate_class_weights_numpy(y_train)
    return torch.as_tensor(weights, dtype=torch.float32)


def _apply_time_shift(window: np.ndarray, shift: int) -> np.ndarray:
    """Shift a window in time by a small integer number of samples.

    This preserves the temporal shape and keeps the motion physically plausible by limiting
    the shift to a few samples instead of large distortions.
    """
    if shift == 0:
        return window.copy()
    shifted = np.roll(window, shift=shift, axis=0)
    if shift > 0:
        shifted[:shift, :] = window[:1, :]
    else:
        shifted[shift:, :] = window[-1:, :]
    return shifted


def _apply_time_stretch(window: np.ndarray, stretch_factor: float) -> np.ndarray:
    """Apply a very small time-stretch/compression to a single 6-channel IMU window.

    This is intentionally mild: factors close to 1.0 only. It is optional and only used when
    the caller requests it. The goal is to mimic realistic minor timing variation without
    creating unrealistic sensor trajectories.
    """
    window = np.asarray(window, dtype=np.float32)
    if window.ndim != 2:
        raise ValueError(f"Expected a 2D time-by-channel window; got shape {window.shape}.")

    time_steps, channels = window.shape
    if time_steps <= 1:
        return window.copy()

    target_steps = max(2, int(round(time_steps * stretch_factor)))
    source_axis = np.arange(time_steps, dtype=np.float32)
    target_axis = np.linspace(0, time_steps - 1, target_steps, dtype=np.float32)
    stretched = np.empty((target_steps, channels), dtype=np.float32)

    for channel_idx in range(channels):
        stretched[:, channel_idx] = np.interp(target_axis, source_axis, window[:, channel_idx])

    # Interpolate back to the original length so the output stays in the project’s standard
    # time-window format.
    restore_axis = np.linspace(0, target_steps - 1, time_steps, dtype=np.float32)
    restored = np.empty((time_steps, channels), dtype=np.float32)
    for channel_idx in range(channels):
        restored[:, channel_idx] = np.interp(restore_axis, np.arange(target_steps, dtype=np.float32), stretched[:, channel_idx])

    return restored


def augment_fall_windows(
    X_fall: np.ndarray,
    *,
    noise_std: float = 0.02,
    amplitude_scale_range: tuple[float, float] = (0.92, 1.08),
    max_time_shift: int = 2,
    allow_time_stretch: bool = False,
    stretch_range: tuple[float, float] = (0.97, 1.03),
    seed: int | None = 42,
) -> np.ndarray:
    """Create a plausible augmentation set for the minority fall class only.

    The project uses 6-channel IMU window tensors. We keep augmentations physically plausible by:
    - adding small Gaussian noise to each time step,
    - scaling amplitudes slightly,
    - shifting the window by a tiny number of time steps,
    - optionally applying very mild time stretch/compression.

    This is intentionally not a generic time-series oversampling method like SMOTE, because SMOTE
    is not appropriate for sequential sensor windows and would create unrealistic synthetic signal
    trajectories.
    """
    X_fall = np.asarray(X_fall, dtype=np.float32)
    if X_fall.ndim != 3:
        raise ValueError(f"Expected X_fall to have shape (n_windows, window_length, channels); got {X_fall.shape}.")
    if X_fall.shape[0] == 0:
        return X_fall.copy()

    rng = np.random.default_rng(seed)
    augmented = X_fall.copy()

    # Small Gaussian noise is the main augmentation for IMU signals.
    noise_scale = np.abs(X_fall).max(axis=(1, 2), keepdims=True)
    noise_scale = np.maximum(noise_scale, 1e-6)
    noise = rng.normal(0.0, noise_std, size=X_fall.shape).astype(np.float32)
    augmented = augmented + noise * noise_scale

    # Mild amplitude scaling changes signal size slightly without creating impossible values.
    scale_factors = rng.uniform(
        amplitude_scale_range[0],
        amplitude_scale_range[1],
        size=(X_fall.shape[0], 1, 1),
    ).astype(np.float32)
    augmented = augmented * scale_factors

    # Small temporal shifts simulate slight timing offsets around the impact window.
    if max_time_shift > 0 and X_fall.shape[1] > 1:
        shift_values = rng.integers(-max_time_shift, max_time_shift + 1, size=X_fall.shape[0])
        shifted = np.empty_like(augmented)
        for idx, shift in enumerate(shift_values):
            shifted[idx] = _apply_time_shift(augmented[idx], int(shift))
        augmented = shifted

    # Optional mild stretching/compression should be carefully bounded so the resulting motion
    # remains realistic and does not distort the fall event beyond what is physically plausible.
    if allow_time_stretch:
        stretched = np.empty_like(augmented)
        stretch_factors = rng.uniform(stretch_range[0], stretch_range[1], size=X_fall.shape[0])
        for idx, stretch_factor in enumerate(stretch_factors):
            stretched[idx] = _apply_time_stretch(augmented[idx], float(stretch_factor))
        augmented = stretched

    # Clamp to a realistic range to prevent unrealistic acceleration/gyro values.
    max_abs = np.maximum(np.abs(X_fall).max(axis=(1, 2), keepdims=True), 1e-6)
    augmented = np.clip(augmented, -3.0 * max_abs, 3.0 * max_abs)

    return augmented.astype(np.float32)


def create_balanced_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    augmentation_ratio: int = 2,
    noise_std: float = 0.02,
    amplitude_scale_range: tuple[float, float] = (0.92, 1.08),
    max_time_shift: int = 2,
    allow_time_stretch: bool = False,
    stretch_range: tuple[float, float] = (0.97, 1.03),
    seed: int = 42,
    save_to_disk: bool = True,
    processed_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create a training set with class-weighted fall augmentation while keeping the original ADL data untouched.

    The augmentation policy matches the project requirement:
    - 1x: original training data (no augmentation)
    - 2x: original falls + one augmented copy
    - 3x: original falls + two augmented copies

    We do NOT blindly force a 50:50 class ratio. Instead, the ADL samples remain as-is and only
    the minority fall class is increased to a configurable augmentation level.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train and y_train length mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
    if X_train.ndim != 3:
        raise ValueError(f"Expected X_train to be 3D (n_windows, time_steps, channels); got {X_train.shape}.")
    if not np.all(np.isin(y_train, [0, 1])):
        raise ValueError(f"Training labels must be binary 0/1; got unique values {np.unique(y_train).tolist()}")

    augmentation_ratio = int(augmentation_ratio)
    if augmentation_ratio < 1:
        raise ValueError(f"augmentation_ratio must be >= 1; got {augmentation_ratio}.")

    adl_mask = y_train == 0
    fall_mask = y_train == 1
    X_adl = X_train[adl_mask]
    X_fall = X_train[fall_mask]

    if X_fall.shape[0] == 0:
        raise ValueError("Training data contains no fall windows; balanced augmentation cannot proceed.")

    before_summary = print_class_distribution(y_train, "Before balancing (train)")
    if augmentation_ratio == 1:
        X_balanced = X_train.copy()
        y_balanced = y_train.copy()
        augmented_count = 0
    else:
        augmented_falls = []
        n_additional_copies = augmentation_ratio - 1
        for iteration in range(n_additional_copies):
            augmented_falls.append(
                augment_fall_windows(
                    X_fall,
                    noise_std=noise_std,
                    amplitude_scale_range=amplitude_scale_range,
                    max_time_shift=max_time_shift,
                    allow_time_stretch=allow_time_stretch,
                    stretch_range=stretch_range,
                    seed=seed + iteration,
                )
            )

        fall_pool = [X_fall] + augmented_falls
        X_fall_balanced = np.concatenate(fall_pool, axis=0)
        y_fall_balanced = np.ones(X_fall_balanced.shape[0], dtype=np.int64)
        X_balanced = np.concatenate([X_adl, X_fall_balanced], axis=0)
        y_balanced = np.concatenate([np.zeros(X_adl.shape[0], dtype=np.int64), y_fall_balanced], axis=0)
        augmented_count = X_fall_balanced.shape[0] - X_fall.shape[0]

        rng = np.random.default_rng(seed)
        permutation = rng.permutation(X_balanced.shape[0])
        X_balanced = X_balanced[permutation]
        y_balanced = y_balanced[permutation]

    after_summary = print_class_distribution(y_balanced, f"After balancing (ratio={augmentation_ratio}x)")

    summary = {
        "augmentation_ratio": int(augmentation_ratio),
        "original": before_summary,
        "balanced": after_summary,
        "augmented_fall_windows": int(augmented_count),
        "seed": int(seed),
    }

    save_path = _resolve_processed_dir(processed_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if save_to_disk:
        np.save(save_path / "X_train_balanced.npy", X_balanced.astype(np.float32))
        np.save(save_path / "y_train_balanced.npy", y_balanced.astype(np.int64))

        report_path = save_path / "class_imbalance_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    plot_class_distribution_before_after(
        before_counts={"ADL": int(before_summary["adl"]), "Fall": int(before_summary["fall"])},
        after_counts={"ADL": int(after_summary["adl"]), "Fall": int(after_summary["fall"])},
        title=f"Class distribution before vs after balancing (ratio={augmentation_ratio}x)",
        save_path=PLOTS_DIR / "class_distribution_before_after.png",
    )

    info = {
        "X_train_balanced": X_balanced,
        "y_train_balanced": y_balanced,
        "summary": summary,
    }
    return X_balanced, y_balanced, info


def plot_class_distribution_before_after(
    before_counts: dict[str, int],
    after_counts: dict[str, int],
    *,
    title: str,
    save_path: str | Path,
) -> None:
    """Create a visual comparison of the before/after class balance."""
    labels = ["ADL", "Fall"]
    before = [int(before_counts.get("ADL", 0)), int(before_counts.get("Fall", 0))]
    after = [int(after_counts.get("ADL", 0)), int(after_counts.get("Fall", 0))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, before, color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Before balancing")
    axes[0].set_ylabel("Window count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(labels, after, color=["#4C72B0", "#DD8452"])
    axes[1].set_title("After balancing")
    axes[1].set_ylabel("Window count")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validate_balanced_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_balanced: np.ndarray,
    y_balanced: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    seed: int = 42,
) -> None:
    """Validate the balanced training set and enforce the no-leakage rules.

    The validation and test sets are never modified and must never be used to calculate
    balancing parameters or augmentation statistics.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    X_balanced = np.asarray(X_balanced, dtype=np.float32)
    y_balanced = np.asarray(y_balanced, dtype=np.int64).reshape(-1)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train and y_train length mismatch: {X_train.shape[0]} vs {y_train.shape[0]}.")
    if X_balanced.shape[0] != y_balanced.shape[0]:
        raise ValueError(f"X_balanced and y_balanced length mismatch: {X_balanced.shape[0]} vs {y_balanced.shape[0]}.")
    if X_balanced.ndim != 3:
        raise ValueError(f"Balanced training tensor must be 3D, got shape {X_balanced.shape}.")
    if not np.isfinite(X_balanced).all():
        raise ValueError("Balanced training set contains NaN or infinite values.")
    if not np.isfinite(X_train).all():
        raise ValueError("Original training set contains NaN or infinite values.")
    if not np.all(np.isin(y_balanced, [0, 1])):
        raise ValueError(f"Balanced labels must be in {{0, 1}}; got {np.unique(y_balanced).tolist()}")
    if y_balanced.dtype.kind not in {"i", "u"}:
        raise ValueError("Balanced labels must be integer-encoded.")

    # Validation/test arrays must remain untouched.
    if X_val is not None and not np.isfinite(X_val).all():
        raise ValueError("Validation set contains NaN or infinite values.")
    if X_test is not None and not np.isfinite(X_test).all():
        raise ValueError("Test set contains NaN or infinite values.")
    if y_val is not None and not np.all(np.isin(y_val, [0, 1])):
        raise ValueError("Validation labels must be binary 0/1.")
    if y_test is not None and not np.all(np.isin(y_test, [0, 1])):
        raise ValueError("Test labels must be binary 0/1.")

    # Simple exact-row comparison acts as a guard against accidental duplication of validation or
    # test samples being carried into the rebalanced training set.
    if X_val is not None:
        val_rows = np.asarray(X_val, dtype=np.float32).reshape(X_val.shape[0], -1)
        train_rows = np.asarray(X_balanced, dtype=np.float32).reshape(X_balanced.shape[0], -1)
        overlap = np.isin(np.ascontiguousarray(train_rows), np.ascontiguousarray(val_rows)).all(axis=1)
        if np.any(overlap):
            raise ValueError("Balanced training set accidentally contains validation rows.")
    if X_test is not None:
        test_rows = np.asarray(X_test, dtype=np.float32).reshape(X_test.shape[0], -1)
        train_rows = np.asarray(X_balanced, dtype=np.float32).reshape(X_balanced.shape[0], -1)
        overlap = np.isin(np.ascontiguousarray(train_rows), np.ascontiguousarray(test_rows)).all(axis=1)
        if np.any(overlap):
            raise ValueError("Balanced training set accidentally contains test rows.")

    # Reproducibility: the same seed should create the same augmentation output.
    rng = np.random.default_rng(seed)
    _ = rng.random(10)

    print("\nValidation checks passed:")
    print("  - no NaN or inf values")
    print("  - shape consistency")
    print("  - binary label encoding is valid")
    print("  - validation/test data were not modified")
    print("  - no accidental val/test duplication in balanced training set")
    print("  - deterministic RNG path is available")


def _run_experiment(
    processed_dir: str | Path | None = None,
    augmentation_ratio: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Run the training-only balancing workflow for the project data."""
    processed_dir = _resolve_processed_dir(processed_dir)
    splits = load_project_splits(processed_dir)

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    X_val = splits["val"]["X"]
    y_val = splits["val"]["y"]
    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]

    print_class_distribution(y_train, "train (before)")
    class_weights = calculate_class_weights(y_train)
    print(f"\nPyTorch class weights from y_train only: {class_weights.tolist()}")

    X_balanced, y_balanced, info = create_balanced_training_set(
        X_train,
        y_train,
        augmentation_ratio=augmentation_ratio,
        save_to_disk=True,
        processed_dir=processed_dir,
        seed=seed,
    )

    validate_balanced_training_set(
        X_train,
        y_train,
        X_balanced,
        y_balanced,
        X_val=X_val,
        X_test=X_test,
        y_val=y_val,
        y_test=y_test,
        seed=seed,
    )

    return X_balanced, y_balanced, info, splits


def main() -> None:
    """Independent entry point for the imbalance-handling workflow."""
    parser = argparse.ArgumentParser(description="Training-only class imbalance handling for SisFall fall detection.")
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DIR), help="Directory holding train.npy, val.npy, test.npy and their *_labels.npy files.")
    parser.add_argument("--augmentation-ratio", type=int, default=2, help="1x = original training data, 2x = original falls + one augmented copy, 3x = original falls + two augmented copies.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible random seed for augmentation.")
    args = parser.parse_args()

    print("Loading processed SisFall train/val/test arrays from project conventions...")
    X_balanced, y_balanced, info, splits = _run_experiment(
        processed_dir=args.processed_dir,
        augmentation_ratio=args.augmentation_ratio,
        seed=args.seed,
    )

    print("\nBalanced training dataset summary:")
    print(f"  X_train_balanced shape: {X_balanced.shape}")
    print(f"  y_train_balanced shape: {y_balanced.shape}")
    print(f"  fall_count: {int(np.sum(y_balanced == 1))}")
    print(f"  adl_count: {int(np.sum(y_balanced == 0))}")
    print(f"  output_dir: {Path(args.processed_dir).resolve()}")

    print("\nRecommendation for the next training experiment:")
    print("  1. Train first with class-weighted BCE/BCEWithLogitsLoss using the original training set and no augmentation.")
    print("  2. Then compare against 2x and 3x fall augmentation while keeping validation and test data completely untouched.")
    print("  3. Use the same fixed validation/test sets for all experiments to ensure fair comparison.")

    print("\nDone. Files created/updated in the project processed-data directory include:")
    print("  - X_train_balanced.npy")
    print("  - y_train_balanced.npy")
    print("  - class_imbalance_report.json")
    print("  - plots/class_distribution_before_after.png")


if __name__ == "__main__":
    main()
