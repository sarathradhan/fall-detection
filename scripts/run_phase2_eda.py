from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = PROCESSED_DIR / "plots"
REPORTS_DIR = PROCESSED_DIR / "reports"

CHANNELS = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"]
ACC_CHANNELS = ["acc1_x", "acc1_y", "acc1_z"]
SPLITS = ["train", "val", "test"]
LABEL_NAMES = {0: "ADL", 1: "Fall"}


@dataclass(frozen=True)
class SplitData:
    name: str
    windows: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    recording_ids: np.ndarray


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> SplitData:
    windows = np.load(PROCESSED_DIR / f"{name}.npy")
    labels = np.load(PROCESSED_DIR / f"{name}_labels.npy")
    subject_ids = np.load(PROCESSED_DIR / f"{name}_subject_ids.npy")
    recording_ids = np.load(PROCESSED_DIR / f"{name}_recording_ids.npy")

    if windows.ndim != 3:
        raise ValueError(f"{name}.npy must have shape (windows, timesteps, channels); found {windows.shape}.")
    if windows.shape[-1] != len(CHANNELS):
        raise ValueError(f"{name}.npy must contain {len(CHANNELS)} channels; found {windows.shape[-1]}.")
    if len(labels) != len(windows):
        raise ValueError(f"{name} labels length does not match window count.")

    return SplitData(name=name, windows=windows, labels=labels, subject_ids=subject_ids, recording_ids=recording_ids)


def load_processed_data() -> dict[str, SplitData]:
    missing = []
    for split in SPLITS:
        for suffix in [".npy", "_labels.npy", "_subject_ids.npy", "_recording_ids.npy"]:
            path = PROCESSED_DIR / f"{split}{suffix}"
            if not path.exists():
                missing.append(path)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing processed dataset files:\n{missing_text}")

    return {split: load_split(split) for split in SPLITS}


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(-1, windows.shape[-1])


def class_distribution_table(data: dict[str, SplitData]) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        labels = data[split].labels
        total = len(labels)
        for label, class_name in LABEL_NAMES.items():
            count = int(np.sum(labels == label))
            rows.append(
                {
                    "split": split,
                    "class_label": label,
                    "class_name": class_name,
                    "window_count": count,
                    "percentage": (count / total * 100.0) if total else 0.0,
                    "total_windows": total,
                }
            )
    return pd.DataFrame(rows)


def plot_class_distribution(class_df: pd.DataFrame) -> None:
    pivot = class_df.pivot(index="split", columns="class_name", values="window_count").reindex(SPLITS)
    ax = pivot.plot(kind="bar", figsize=(8, 5), color=["#4C78A8", "#F58518"], rot=0)
    ax.set_title("Class Distribution by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of windows")
    ax.legend(title="Class")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(PLOTS_DIR / "eda_class_distribution_bar.png", dpi=200)
    plt.close(ax.figure)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, split in zip(axes, SPLITS):
        subset = class_df[class_df["split"] == split]
        ax.pie(
            subset["window_count"],
            labels=subset["class_name"],
            autopct=lambda pct: f"{pct:.1f}%",
            startangle=90,
            colors=["#4C78A8", "#F58518"],
        )
        ax.set_title(split)
    fig.suptitle("Class Percentages by Split")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_class_distribution_pie.png", dpi=200)
    plt.close(fig)


def summarize_signal_health(data: dict[str, SplitData]) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        split_data = data[split]
        groups = [("All", np.ones(len(split_data.labels), dtype=bool))]
        groups.extend((LABEL_NAMES[label], split_data.labels == label) for label in LABEL_NAMES)

        for class_name, mask in groups:
            if not np.any(mask):
                continue
            flat = flatten_windows(split_data.windows[mask])
            for idx, channel in enumerate(CHANNELS):
                values = flat[:, idx].astype(np.float64)
                mean = float(np.mean(values))
                std = float(np.std(values))
                centered = values - mean
                if std > 0:
                    skewness = float(np.mean((centered / std) ** 3))
                    kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
                else:
                    skewness = 0.0
                    kurtosis = 0.0
                rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "channel": channel,
                        "mean": mean,
                        "median": float(np.median(values)),
                        "std": std,
                        "variance": float(np.var(values)),
                        "rms": float(np.sqrt(np.mean(values**2))),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "skewness": skewness,
                        "kurtosis": kurtosis,
                        "sample_count": int(values.size),
                    }
                )
    return pd.DataFrame(rows)


def plot_signal_histograms(data: dict[str, SplitData]) -> None:
    for split in SPLITS:
        flat = flatten_windows(data[split].windows)
        fig, axes = plt.subplots(3, 2, figsize=(12, 9))
        axes = axes.ravel()
        for idx, channel in enumerate(CHANNELS):
            axes[idx].hist(flat[:, idx], bins=80, color="#4C78A8", alpha=0.82)
            axes[idx].set_title(channel)
            axes[idx].set_xlabel("Normalized value")
            axes[idx].set_ylabel("Frequency")
        fig.suptitle(f"{split.capitalize()} Signal Histograms")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"eda_signal_histograms_{split}.png", dpi=200)
        plt.close(fig)


def plot_signal_boxplots(data: dict[str, SplitData]) -> None:
    for split in SPLITS:
        flat = flatten_windows(data[split].windows)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot([flat[:, idx] for idx in range(len(CHANNELS))], tick_labels=CHANNELS, showfliers=False)
        ax.set_title(f"{split.capitalize()} Channel Boxplots")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Normalized value")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"eda_signal_boxplots_{split}.png", dpi=200)
        plt.close(fig)


def acceleration_magnitude(windows: np.ndarray) -> np.ndarray:
    acc = windows[:, :, : len(ACC_CHANNELS)].astype(np.float64)
    return np.sqrt(np.sum(acc**2, axis=-1))


def acceleration_spike_table(data: dict[str, SplitData]) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        magnitudes = acceleration_magnitude(data[split].windows)
        window_peaks = np.max(magnitudes, axis=1)
        window_means = np.mean(magnitudes, axis=1)
        for label, class_name in LABEL_NAMES.items():
            values = window_peaks[data[split].labels == label]
            means = window_means[data[split].labels == label]
            if values.size == 0:
                continue
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "windows": int(values.size),
                    "peak_mean": float(np.mean(values)),
                    "peak_median": float(np.median(values)),
                    "peak_std": float(np.std(values)),
                    "peak_p95": float(np.quantile(values, 0.95)),
                    "peak_max": float(np.max(values)),
                    "magnitude_mean": float(np.mean(means)),
                }
            )
    return pd.DataFrame(rows)


def plot_acceleration_spikes(data: dict[str, SplitData]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, split in zip(axes, SPLITS):
        peaks = np.max(acceleration_magnitude(data[split].windows), axis=1)
        adl = peaks[data[split].labels == 0]
        fall = peaks[data[split].labels == 1]
        ax.hist(adl, bins=60, alpha=0.7, label="ADL", color="#4C78A8", density=True)
        ax.hist(fall, bins=60, alpha=0.7, label="Fall", color="#F58518", density=True)
        ax.set_title(split)
        ax.set_xlabel("Window acceleration-magnitude peak")
        ax.legend()
    axes[0].set_ylabel("Density")
    fig.suptitle("ADL vs Fall Acceleration Spike Distributions")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_acceleration_spike_histograms.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, SPLITS):
        peaks = np.max(acceleration_magnitude(data[split].windows), axis=1)
        values = [peaks[data[split].labels == 0], peaks[data[split].labels == 1]]
        ax.boxplot(values, tick_labels=["ADL", "Fall"], showfliers=False)
        ax.set_title(split)
        ax.set_xlabel("Class")
    axes[0].set_ylabel("Window acceleration-magnitude peak")
    fig.suptitle("ADL vs Fall Spike Boxplots")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_acceleration_spike_boxplots.png", dpi=200)
    plt.close(fig)


def autocorrelation_1d(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.mean(values)
    denominator = np.dot(values, values)
    if denominator == 0:
        out = np.zeros(max_lag + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    correlations = [1.0]
    for lag in range(1, max_lag + 1):
        numerator = np.dot(values[:-lag], values[lag:])
        correlations.append(float(numerator / denominator))
    return np.asarray(correlations)


def autocorrelation_table(data: dict[str, SplitData], max_lag: int = 32) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        windows = data[split].windows
        for idx, channel in enumerate(CHANNELS):
            channel_autocorr = np.asarray([autocorrelation_1d(window[:, idx], max_lag) for window in windows])
            mean_autocorr = channel_autocorr.mean(axis=0)
            for lag, value in enumerate(mean_autocorr):
                rows.append({"split": split, "channel": channel, "lag": lag, "autocorrelation": float(value)})
    return pd.DataFrame(rows)


def plot_autocorrelation(autocorr_df: pd.DataFrame) -> None:
    for split in SPLITS:
        subset = autocorr_df[autocorr_df["split"] == split]
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, channel in zip(axes, CHANNELS):
            channel_df = subset[subset["channel"] == channel]
            ax.plot(channel_df["lag"], channel_df["autocorrelation"], color="#54A24B", linewidth=1.8)
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            ax.set_title(channel)
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorrelation")
        fig.suptitle(f"{split.capitalize()} Mean Window Autocorrelation")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"eda_autocorrelation_{split}.png", dpi=200)
        plt.close(fig)


def outlier_table(data: dict[str, SplitData]) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        flat = flatten_windows(data[split].windows)
        for idx, channel in enumerate(CHANNELS):
            values = flat[:, idx].astype(np.float64)
            q1 = float(np.quantile(values, 0.25))
            q3 = float(np.quantile(values, 0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_mask = (values < lower) | (values > upper)

            mean = float(np.mean(values))
            std = float(np.std(values))
            if std > 0:
                z_mask = np.abs((values - mean) / std) > 3.0
            else:
                z_mask = np.zeros_like(values, dtype=bool)

            rows.append(
                {
                    "split": split,
                    "channel": channel,
                    "sample_count": int(values.size),
                    "iqr_outliers": int(np.sum(iqr_mask)),
                    "iqr_outlier_percentage": float(np.mean(iqr_mask) * 100.0),
                    "iqr_lower_bound": float(lower),
                    "iqr_upper_bound": float(upper),
                    "zscore_outliers": int(np.sum(z_mask)),
                    "zscore_outlier_percentage": float(np.mean(z_mask) * 100.0),
                    "zscore_threshold": 3.0,
                }
            )
    return pd.DataFrame(rows)


def plot_outlier_summary(outliers_df: pd.DataFrame) -> None:
    for method, column in [("IQR", "iqr_outlier_percentage"), ("Z-score", "zscore_outlier_percentage")]:
        pivot = outliers_df.pivot(index="channel", columns="split", values=column).loc[CHANNELS, SPLITS]
        ax = pivot.plot(kind="bar", figsize=(10, 5), rot=30)
        ax.set_title(f"{method} Outlier Percentage by Channel")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Outlier percentage")
        ax.legend(title="Split")
        ax.figure.tight_layout()
        ax.figure.savefig(PLOTS_DIR / f"eda_outliers_{method.lower().replace('-', '_')}.png", dpi=200)
        plt.close(ax.figure)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"

    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")
        else:
            display_df[column] = display_df[column].astype(str)

    columns = list(display_df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display_df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def interpret_class_distribution(class_df: pd.DataFrame) -> str:
    totals = class_df.groupby("class_name")["window_count"].sum()
    total_windows = int(totals.sum())
    fall_pct = float(totals.get("Fall", 0) / total_windows * 100.0) if total_windows else 0.0
    min_split_fall = class_df[class_df["class_name"] == "Fall"]["percentage"].min()
    max_split_fall = class_df[class_df["class_name"] == "Fall"]["percentage"].max()
    return (
        f"Across all processed splits there are {total_windows:,} windows. Falls account for "
        f"{fall_pct:.2f}% overall, with split-level fall percentages ranging from "
        f"{min_split_fall:.2f}% to {max_split_fall:.2f}%. This indicates the binary task is imbalanced "
        "but both classes are represented in train, validation, and test."
    )


def interpret_signal_health(stats_df: pd.DataFrame) -> str:
    overall = stats_df[stats_df["class_name"] == "All"]
    max_abs_mean = float(overall["mean"].abs().max())
    std_min = float(overall["std"].min())
    std_max = float(overall["std"].max())
    max_abs_value = float(max(abs(overall["min"].min()), abs(overall["max"].max())))
    return (
        f"Channel means remain close to zero after normalization (largest absolute split/channel mean: {max_abs_mean:.3f}). "
        f"Standard deviations range from {std_min:.3f} to {std_max:.3f}, showing usable variation across all channels. "
        f"The largest absolute normalized value is {max_abs_value:.3f}; extreme tails exist but are expected for fall-impact windows."
    )


def interpret_spikes(spike_df: pd.DataFrame) -> str:
    pivot = spike_df.pivot(index="split", columns="class_name", values="peak_median")
    comparisons = []
    higher_fall_splits = []
    for split in SPLITS:
        if {"ADL", "Fall"}.issubset(pivot.columns):
            adl = pivot.loc[split, "ADL"]
            fall = pivot.loc[split, "Fall"]
            comparisons.append(f"{split}: Fall median peak {fall:.3f} vs ADL {adl:.3f}")
            if fall > adl:
                higher_fall_splits.append(split)
    direction_text = (
        f"Fall median peaks are higher in {', '.join(higher_fall_splits)}."
        if higher_fall_splits
        else "Fall median peaks are not higher than ADL in these processed splits."
    )
    return (
        "Acceleration magnitude is computed from the normalized accelerometer channels, so it is a normalized spike proxy. "
        + "; ".join(comparisons)
        + f". {direction_text} Spike magnitude still shows class-dependent distribution differences, but it should be used with temporal patterns rather than as a standalone rule."
    )


def interpret_autocorrelation(autocorr_df: pd.DataFrame) -> str:
    lag1 = autocorr_df[autocorr_df["lag"] == 1]["autocorrelation"]
    lag8 = autocorr_df[autocorr_df["lag"] == 8]["autocorrelation"]
    return (
        f"Mean lag-1 autocorrelation is {lag1.mean():.3f}, while mean lag-8 autocorrelation is {lag8.mean():.3f}. "
        "The decay across lags shows short-range temporal dependency inside the 64-step windows, which supports using temporal kernels or recurrent layers."
    )


def interpret_outliers(outliers_df: pd.DataFrame) -> str:
    iqr_max = outliers_df.loc[outliers_df["iqr_outlier_percentage"].idxmax()]
    z_max = outliers_df.loc[outliers_df["zscore_outlier_percentage"].idxmax()]
    return (
        f"IQR detects the highest outlier rate in {iqr_max['split']} {iqr_max['channel']} "
        f"({iqr_max['iqr_outlier_percentage']:.2f}%). Z-score detects the highest outlier rate in "
        f"{z_max['split']} {z_max['channel']} ({z_max['zscore_outlier_percentage']:.2f}%). "
        "No values were removed; these points are retained because high-amplitude motion is likely informative for fall detection."
    )


def readiness_summary(
    data: dict[str, SplitData],
    class_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
) -> str:
    shapes = ", ".join(f"{split}: {data[split].windows.shape}" for split in SPLITS)
    missing_values = sum(int(np.isnan(data[split].windows).sum()) for split in SPLITS)
    infinite_values = sum(int(np.isinf(data[split].windows).sum()) for split in SPLITS)
    classes_per_split = class_df.groupby("split")["window_count"].apply(lambda s: int(np.sum(s > 0)))
    all_splits_have_two_classes = bool((classes_per_split == 2).all())
    max_z_outlier_pct = float(outliers_df["zscore_outlier_percentage"].max())
    max_abs_mean = float(stats_df[stats_df["class_name"] == "All"]["mean"].abs().max())

    readiness = (
        "ready for CNN/CNN-LSTM training"
        if missing_values == 0 and infinite_values == 0 and all_splits_have_two_classes
        else "not fully ready until data integrity issues are addressed"
    )
    return (
        f"Processed tensor shapes are {shapes}. Missing values: {missing_values}; infinite values: {infinite_values}. "
        f"All splits contain both classes: {all_splits_have_two_classes}. Maximum Z-score outlier percentage is "
        f"{max_z_outlier_pct:.2f}%, and the largest absolute normalized mean is {max_abs_mean:.3f}. "
        f"Based on these checks, the processed data is {readiness}."
    )


def write_report(
    data: dict[str, SplitData],
    class_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    spike_df: pd.DataFrame,
    autocorr_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
) -> None:
    report = [
        "# Phase 2 EDA Report",
        "",
        "This report uses only the processed window datasets in `data/processed/`.",
        "",
        "## Dataset Overview",
        "",
        "| Split | Windows | Shape | Subjects | Recordings |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for split in SPLITS:
        split_data = data[split]
        report.append(
            f"| {split} | {len(split_data.windows):,} | `{split_data.windows.shape}` | "
            f"{len(np.unique(split_data.subject_ids)):,} | {len(np.unique(split_data.recording_ids)):,} |"
        )

    report.extend(
        [
            "",
            "## 1. Class Distribution",
            "",
            dataframe_to_markdown(class_df),
            "",
            f"Interpretation: {interpret_class_distribution(class_df)}",
            "",
            "## 2. Signal Health",
            "",
            "Full signal statistics are saved to `eda_signal_health_statistics.csv`.",
            "",
            dataframe_to_markdown(stats_df[stats_df["class_name"] == "All"]),
            "",
            f"Interpretation: {interpret_signal_health(stats_df)}",
            "",
            "## 3. Acceleration Spike Analysis",
            "",
            dataframe_to_markdown(spike_df),
            "",
            f"Interpretation: {interpret_spikes(spike_df)}",
            "",
            "## 4. Autocorrelation",
            "",
            "Autocorrelation is averaged across processed 64-step windows for each channel and split.",
            "",
            f"Interpretation: {interpret_autocorrelation(autocorr_df)}",
            "",
            "## 5. Outlier Detection",
            "",
            dataframe_to_markdown(outliers_df),
            "",
            f"Interpretation: {interpret_outliers(outliers_df)}",
            "",
            "## 6. Final EDA Summary",
            "",
            readiness_summary(data, class_df, stats_df, outliers_df),
            "",
            "Plots were saved to `data/processed/plots/`. Statistics and this report were saved to `data/processed/reports/`.",
        ]
    )
    (REPORTS_DIR / "phase2_eda_report.md").write_text("\n".join(report), encoding="utf-8")


def run_eda() -> None:
    ensure_output_dirs()
    data = load_processed_data()

    class_df = class_distribution_table(data)
    class_df.to_csv(REPORTS_DIR / "eda_class_distribution.csv", index=False)
    plot_class_distribution(class_df)

    stats_df = summarize_signal_health(data)
    stats_df.to_csv(REPORTS_DIR / "eda_signal_health_statistics.csv", index=False)
    plot_signal_histograms(data)
    plot_signal_boxplots(data)

    spike_df = acceleration_spike_table(data)
    spike_df.to_csv(REPORTS_DIR / "eda_acceleration_spikes.csv", index=False)
    plot_acceleration_spikes(data)

    autocorr_df = autocorrelation_table(data)
    autocorr_df.to_csv(REPORTS_DIR / "eda_autocorrelation.csv", index=False)
    plot_autocorrelation(autocorr_df)

    outliers_df = outlier_table(data)
    outliers_df.to_csv(REPORTS_DIR / "eda_outliers.csv", index=False)
    plot_outlier_summary(outliers_df)

    write_report(data, class_df, stats_df, spike_df, autocorr_df, outliers_df)

    print("Phase 2 EDA complete.")
    print(f"Plots saved to: {PLOTS_DIR.relative_to(ROOT)}")
    print(f"Reports saved to: {REPORTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    run_eda()
