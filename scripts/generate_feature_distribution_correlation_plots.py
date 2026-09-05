from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
SEVERITY_DIR = ROOT / "data" / "processed" / "severity"
FEATURES_DIR = SEVERITY_DIR / "clustering_features"
OUTPUT_DIR = ROOT / "data" / "processed" / "plots" / "feature_engineering"
SPLITS = ["train", "val", "test"]
FIGURE_DPI = 300


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.8,
            "xtick.color": "#303030",
            "ytick.color": "#303030",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_feature_names() -> list[str]:
    with (SEVERITY_DIR / "feature_names.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_fall_features() -> np.ndarray:
    split_features = [np.load(FEATURES_DIR / f"{split}_fall_features.npy") for split in SPLITS]
    features = np.vstack(split_features).astype(np.float64)
    if features.ndim != 2 or features.shape[1] != 58:
        raise ValueError(f"Expected fall feature matrix with 58 columns, found shape {features.shape}.")
    if not np.isfinite(features).all():
        raise ValueError("Fall feature matrix contains non-finite values.")
    return features


def display_feature_name(name: str) -> str:
    replacements = {
        "acc1_": "acc_",
        "acc_mag": "acc mag",
        "gyro_mag": "gyro mag",
        "peak_abs": "peak abs",
    }
    label = name
    for source, target in replacements.items():
        label = label.replace(source, target)
    return label.replace("_", " ")


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=FIGURE_DPI)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def plot_feature_boxplots(features: np.ndarray, feature_names: list[str]) -> None:
    labels = [display_feature_name(name) for name in feature_names]

    fig, ax = plt.subplots(figsize=(12.5, 14.5))
    box = ax.boxplot(
        [features[:, idx] for idx in range(features.shape[1])],
        orientation="horizontal",
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.62,
        medianprops={"color": "#111111", "linewidth": 0.9},
        boxprops={"facecolor": "#D9D9D9", "edgecolor": "#333333", "linewidth": 0.7},
        whiskerprops={"color": "#333333", "linewidth": 0.7},
        capprops={"color": "#333333", "linewidth": 0.7},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.95)

    ax.set_title("Distributions of Extracted Features from Fall Windows")
    ax.set_xlabel("Feature value (raw extracted scale; symmetric log axis)")
    ax.set_ylabel("Extracted feature")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.subplots_adjust(bottom=0.055)
    fig.text(
        0.985,
        0.018,
        f"Fall windows: {features.shape[0]}  |  Features: {features.shape[1]}  |  Outliers hidden for readability",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#303030",
    )

    save_figure(fig, "fall_window_feature_boxplots")


def plot_feature_correlation_heatmap(features: np.ndarray, feature_names: list[str]) -> None:
    corr = np.corrcoef(features, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    labels = [display_feature_name(name) for name in feature_names]
    signed_gray = LinearSegmentedColormap.from_list(
        "signed_gray",
        [(0.0, "#1A1A1A"), (0.5, "#FFFFFF"), (1.0, "#6F6F6F")],
    )

    fig, ax = plt.subplots(figsize=(13.5, 11.5))
    image = ax.imshow(corr, cmap=signed_gray, vmin=-1, vmax=1, interpolation="nearest", aspect="equal")

    ax.set_title("Feature Correlation Heatmap for Fall Windows")
    ax.set_xlabel("Extracted feature")
    ax.set_ylabel("Extracted feature")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("Pearson correlation coefficient", rotation=270, labelpad=16)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.outline.set_edgecolor("#303030")
    colorbar.outline.set_linewidth(0.6)

    save_figure(fig, "fall_window_feature_correlation_heatmap")


def main() -> None:
    configure_style()
    feature_names = load_feature_names()
    if len(feature_names) != 58:
        raise ValueError(f"Expected 58 extracted features, found {len(feature_names)}.")

    features = load_fall_features()
    plot_feature_boxplots(features, feature_names)
    plot_feature_correlation_heatmap(features, feature_names)
    print(f"Saved feature distribution and correlation plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
