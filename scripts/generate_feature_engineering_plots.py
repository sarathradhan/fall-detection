from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEVERITY_DIR = ROOT / "data" / "processed" / "severity"
FEATURES_DIR = SEVERITY_DIR / "clustering_features"
OUTPUT_DIR = ROOT / "data" / "processed" / "plots" / "feature_engineering"

CHANNELS = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"]
MAGNITUDES = ["acc_mag", "gyro_mag"]
CHANNEL_AGGREGATES = ["mean", "std", "min", "max", "range", "rms", "peak_abs"]
MAGNITUDE_AGGREGATES = ["mean", "std", "min", "max", "range", "rms", "peak", "energy"]

FIGURE_DPI = 300


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_feature_names() -> list[str]:
    with (SEVERITY_DIR / "feature_names.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=FIGURE_DPI)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def plot_feature_count_bar(feature_names: list[str]) -> None:
    channel_count = sum(name.startswith(tuple(f"{channel}_" for channel in CHANNELS)) for name in feature_names)
    magnitude_count = sum(name.startswith(tuple(f"{magnitude}_" for magnitude in MAGNITUDES)) for name in feature_names)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = ["Channel-level\nfeatures", "Magnitude-level\nfeatures"]
    counts = [channel_count, magnitude_count]
    colors = ["#4E79A7", "#8C8C8C"]
    bars = ax.bar(labels, counts, color=colors, width=0.52)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            str(count),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.set_title("Extracted Feature Composition")
    ax.set_ylabel("Number of features")
    ax.set_ylim(0, max(counts) + 8)
    ax.text(
        0.5,
        0.93,
        f"Total extracted features: {len(feature_names)}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "feature_count_bar")


def plot_feature_group_matrix(feature_names: list[str]) -> None:
    rows = CHANNELS + MAGNITUDES
    cols = ["mean", "std", "min", "max", "range", "rms", "peak", "energy"]
    matrix = np.zeros((len(rows), len(cols)), dtype=float)

    for row_idx, source in enumerate(rows):
        for col_idx, aggregate in enumerate(cols):
            if source in CHANNELS and aggregate == "peak":
                feature_name = f"{source}_peak_abs"
            else:
                feature_name = f"{source}_{aggregate}"
            matrix[row_idx, col_idx] = 1.0 if feature_name in feature_names else 0.0

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#FFFFFF", "#4E79A7"]), aspect="auto", vmin=0, vmax=1)

    ax.set_title("Feature Groups Across IMU Channels and Magnitudes")
    ax.set_xlabel("Statistical feature group")
    ax.set_ylabel("Signal source")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(["Mean", "Std", "Min", "Max", "Range", "RMS", "Peak", "Energy"], rotation=0)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z", "Acc Mag.", "Gyro Mag."])

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="#D0D0D0", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for idx in range(len(CHANNELS)):
        ax.text(len(cols) - 1, idx, "not used", ha="center", va="center", fontsize=9, color="#666666")
    for idx in range(len(rows)):
        count = int(matrix[idx].sum())
        ax.text(len(cols) + 0.18, idx, f"{count}", ha="left", va="center", fontsize=10, color="#333333")

    ax.text(len(cols) + 0.18, -0.85, "Features", ha="left", va="center", fontsize=10, color="#333333")
    ax.set_xlim(-0.5, len(cols) + 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    save_figure(fig, "feature_group_matrix")


def plot_scaled_feature_distribution() -> None:
    train_features = np.load(FEATURES_DIR / "train_fall_features.npy")
    with (SEVERITY_DIR / "feature_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)

    scaled = scaler.transform(train_features)
    values = scaled.ravel()
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.hist(values, bins=70, color="#4E79A7", edgecolor="white", linewidth=0.35)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_title("Distribution of Scaled Feature Values Before K-Means")
    ax.set_xlabel("Standardized feature value")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    summary = (
        f"Training fall windows: {scaled.shape[0]}\n"
        f"Features per window: {scaled.shape[1]}\n"
        "Scaler fitted on training fall windows"
    )
    ax.text(
        0.98,
        0.94,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#BFBFBF", "linewidth": 0.7},
    )

    save_figure(fig, "scaled_feature_distribution_before_kmeans")


def main() -> None:
    configure_style()
    feature_names = load_feature_names()
    if len(feature_names) != 58:
        raise ValueError(f"Expected 58 extracted features, found {len(feature_names)}.")

    plot_feature_count_bar(feature_names)
    plot_feature_group_matrix(feature_names)
    plot_scaled_feature_distribution()
    print(f"Saved feature engineering plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
