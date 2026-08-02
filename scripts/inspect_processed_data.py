from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = PROCESSED_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"]


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.load(PROCESSED_DIR / f"{name}.npy")
    labels = np.load(PROCESSED_DIR / f"{name}_labels.npy")
    subject_ids = np.load(PROCESSED_DIR / f"{name}_subject_ids.npy")
    recording_ids = np.load(PROCESSED_DIR / f"{name}_recording_ids.npy")
    return features, labels, subject_ids, recording_ids


def summarize_split(name: str, features: np.ndarray, labels: np.ndarray, subject_ids: np.ndarray, recording_ids: np.ndarray) -> pd.DataFrame:
    flat = features.reshape(-1, features.shape[-1])
    stats = []
    for idx, channel in enumerate(CHANNELS):
        values = flat[:, idx]
        stats.append(
            {
                "split": name,
                "channel": channel,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "p25": float(np.quantile(values, 0.25)),
                "p50": float(np.quantile(values, 0.50)),
                "p75": float(np.quantile(values, 0.75)),
            }
        )
    stats_df = pd.DataFrame(stats)
    stats_df["samples"] = int(features.shape[0])
    stats_df["windows"] = int(features.shape[0])
    stats_df["label_0"] = int((labels == 0).sum())
    stats_df["label_1"] = int((labels == 1).sum())
    return stats_df


def save_plots() -> None:
    split_names = ["train", "val", "test"]
    loaded = {name: load_split(name) for name in split_names}
    summary_frames = []
    for name, (features, labels, subject_ids, recording_ids) in loaded.items():
        summary_frames.append(summarize_split(name, features, labels, subject_ids, recording_ids))
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df.to_csv(PROCESSED_DIR / "channel_statistics.csv", index=False)

    label_counts = pd.DataFrame(
        {
            "split": split_names,
            "adl": [int((loaded[name][1] == 0).sum()) for name in split_names],
            "fall": [int((loaded[name][1] == 1).sum()) for name in split_names],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(split_names))
    width = 0.35
    ax.bar(x - width / 2, label_counts["adl"], width, label="ADL")
    ax.bar(x + width / 2, label_counts["fall"], width, label="Fall")
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.set_title("Class balance per split")
    ax.set_ylabel("Number of windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_balance.png", dpi=200)
    plt.close(fig)

    for name, (features, labels, subject_ids, recording_ids) in loaded.items():
        sample_window = features[0]
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for axis, channel in enumerate(CHANNELS[:3]):
            axes[axis].plot(sample_window[:, axis], linewidth=1.2)
            axes[axis].set_ylabel(channel)
            axes[axis].set_title(f"{name}: first window")
        axes[-1].set_xlabel("Time step")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{name}_first_window.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, channel in zip(axes, CHANNELS):
        values = []
        labels = []
        for name in split_names:
            features, _, _, _ = loaded[name]
            flat = features.reshape(-1, features.shape[-1])
            values.append(flat[:, CHANNELS.index(channel)])
            labels.append(name)
        ax.boxplot(values, vert=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_title(channel)
    fig.suptitle("Distribution of channel values by split")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "channel_boxplots.png", dpi=200)
    plt.close(fig)

    print("Processed data inspection complete.")
    print("Saved plots:")
    for path in sorted(PLOTS_DIR.glob("*.png")):
        print(path.relative_to(ROOT))

    print("\nSample rows from the first window of each split:")
    for name, (features, labels, subject_ids, recording_ids) in loaded.items():
        sample = features[0]
        print(f"\n{name} first window:")
        print(pd.DataFrame(sample, columns=CHANNELS).to_string(index=False))

    print("\nSummary statistics:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    save_plots()
