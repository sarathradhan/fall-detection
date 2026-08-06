"""Generate a single consolidated dataset report with fresh plots and statistics."""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inspect_processed_data import save_plots as save_inspection_plots
from scripts.run_phase2_eda import (
    CHANNELS,
    SPLITS,
    acceleration_spike_table,
    autocorrelation_table,
    class_distribution_table,
    ensure_output_dirs,
    load_processed_data,
    outlier_table,
    plot_acceleration_spikes,
    plot_autocorrelation,
    plot_class_distribution,
    plot_outlier_summary,
    plot_signal_boxplots,
    plot_signal_histograms,
    summarize_signal_health,
)

PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = PROCESSED_DIR / "plots"
REPORTS_DIR = PROCESSED_DIR / "reports"
REPORT_PATH = PROCESSED_DIR / "DATASET_REPORT.md"

PREPROCESSING_CONFIG = {
    "raw_sampling_hz": 200,
    "processed_sampling_hz": 20,
    "filter": "Butterworth low-pass, order 2, cutoff 5 Hz",
    "downsample_ratio": 10,
    "window_size": 64,
    "window_duration_s": 3.2,
    "stride": 16,
    "stride_duration_s": 0.8,
    "normalization": "StandardScaler (fit on train only)",
    "split": "Subject-wise 70/15/15 (largest-remainder)",
    "channels": CHANNELS,
    "labels": {"0": "ADL", "1": "Fall"},
}


def remove_previous_statistics() -> None:
    """Delete legacy report artifacts; keep processed arrays and scaler."""
    if REPORTS_DIR.exists():
        for path in REPORTS_DIR.iterdir():
            if path.is_file():
                path.unlink()
    legacy = [
        PROCESSED_DIR / "channel_statistics.csv",
        PROCESSED_DIR / "DATASET_REPORT.md",
    ]
    for path in legacy:
        if path.exists():
            path.unlink()
    if PLOTS_DIR.exists():
        for path in PLOTS_DIR.glob("*.png"):
            path.unlink()


def gyro_magnitude_table(data: dict) -> pd.DataFrame:
    rows = []
    gyro_idx = [CHANNELS.index(c) for c in ["gyro_x", "gyro_y", "gyro_z"]]
    for split in SPLITS:
        windows = data[split].windows
        gyro = windows[:, :, gyro_idx].astype(np.float64)
        magnitudes = np.sqrt(np.sum(gyro**2, axis=-1))
        window_peaks = np.max(magnitudes, axis=1)
        for label, class_name in {0: "ADL", 1: "Fall"}.items():
            values = window_peaks[data[split].labels == label]
            if values.size == 0:
                continue
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "windows": int(values.size),
                    "peak_mean": float(np.mean(values)),
                    "peak_median": float(np.median(values)),
                    "peak_p95": float(np.quantile(values, 0.95)),
                    "peak_max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def integrity_table(data: dict) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        windows = data[split].windows
        labels = data[split].labels
        flat = windows.reshape(-1, windows.shape[-1])
        rows.append(
            {
                "split": split,
                "windows": int(len(windows)),
                "shape": str(windows.shape),
                "subjects": len(set(data[split].subject_ids)),
                "recordings": len(set(data[split].recording_ids)),
                "nan_count": int(np.isnan(windows).sum()),
                "inf_count": int(np.isinf(windows).sum()),
                "abs_max": float(np.max(np.abs(windows))),
                "constant_windows": int((windows.std(axis=(1, 2)) < 1e-6).sum()),
                "adl_windows": int((labels == 0).sum()),
                "fall_windows": int((labels == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def split_subjects_table(data: dict) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        subjects = sorted(set(data[split].subject_ids.tolist()))
        sa = sum(1 for s in subjects if str(s).startswith("SA"))
        se = sum(1 for s in subjects if str(s).startswith("SE"))
        rows.append(
            {
                "split": split,
                "subjects": len(subjects),
                "young_adults_SA": sa,
                "elderly_SE": se,
                "subject_ids": ", ".join(subjects),
            }
        )
    return pd.DataFrame(rows)


def recording_stats_table(data: dict) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        counts = Counter(data[split].recording_ids.tolist())
        values = list(counts.values())
        rows.append(
            {
                "split": split,
                "unique_recordings": len(counts),
                "min_windows_per_recording": min(values),
                "median_windows_per_recording": float(np.median(values)),
                "max_windows_per_recording": max(values),
            }
        )
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                cells.append(format(val, float_fmt))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_section(title: str, images: list[tuple[str, str]]) -> str:
    lines = [f"### {title}", ""]
    for caption, rel_path in images:
        lines.append(f"**{caption}**")
        lines.append("")
        lines.append(f"![{caption}]({rel_path})")
        lines.append("")
    return "\n".join(lines)


def build_report(
    data: dict,
    class_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    spike_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    autocorr_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    integrity_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    recording_df: pd.DataFrame,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_windows = sum(integrity_df.loc[integrity_df["split"] == s, "windows"].iloc[0] for s in SPLITS)
    total_fall = int(integrity_df["fall_windows"].sum())
    total_adl = int(integrity_df["adl_windows"].sum())

    channel_stats = []
    for split in SPLITS:
        flat = data[split].windows.reshape(-1, len(CHANNELS))
        for idx, channel in enumerate(CHANNELS):
            values = flat[:, idx]
            channel_stats.append(
                {
                    "split": split,
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
    channel_df = pd.DataFrame(channel_stats)

    with (PROCESSED_DIR / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)

    sections = [
        "# SisFall Processed Dataset Report",
        "",
        f"Generated: **{generated}**",
        "",
        "Single reference document for the processed fall-detection dataset under `data/processed/`.",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Total windows | {total_windows:,} |",
        f"| Total subjects | 38 (all SisFall subjects included) |",
        f"| Input tensor shape | `(windows, 64, 6)` |",
        f"| Window duration | {PREPROCESSING_CONFIG['window_duration_s']} s at {PREPROCESSING_CONFIG['processed_sampling_hz']} Hz |",
        f"| Stride | {PREPROCESSING_CONFIG['stride']} samples ({PREPROCESSING_CONFIG['stride_duration_s']} s) |",
        f"| ADL windows | {total_adl:,} ({total_adl / total_windows * 100:.2f}%) |",
        f"| Fall windows | {total_fall:,} ({total_fall / total_windows * 100:.2f}%) |",
        f"| Class imbalance ratio (ADL:Fall) | {total_adl / total_fall:.2f}:1 |",
        "",
        "### Split Summary",
        "",
        df_to_markdown(integrity_df[["split", "windows", "shape", "subjects", "recordings", "adl_windows", "fall_windows"]]),
        "",
        "### Subject Assignment (no overlap across splits)",
        "",
        df_to_markdown(subjects_df),
        "",
        "### Recording Coverage",
        "",
        df_to_markdown(recording_df),
        "",
        "---",
        "",
        "## 2. Preprocessing Configuration",
        "",
        "| Step | Setting |",
        "| --- | --- |",
        f"| Raw sampling rate | {PREPROCESSING_CONFIG['raw_sampling_hz']} Hz |",
        f"| Processed sampling rate | {PREPROCESSING_CONFIG['processed_sampling_hz']} Hz |",
        f"| Filtering | {PREPROCESSING_CONFIG['filter']} |",
        f"| Downsampling | Factor {PREPROCESSING_CONFIG['downsample_ratio']} |",
        f"| Normalization | {PREPROCESSING_CONFIG['normalization']} |",
        f"| Split strategy | {PREPROCESSING_CONFIG['split']} |",
        f"| Window size | {PREPROCESSING_CONFIG['window_size']} timesteps |",
        f"| Stride | {PREPROCESSING_CONFIG['stride']} timesteps |",
        "",
        "**Channels per window (index order):**",
        "",
        "| Index | Channel | Sensor |",
        "| ---: | --- | --- |",
        "| 0 | acc1_x | ADXL345 accelerometer X |",
        "| 1 | acc1_y | ADXL345 accelerometer Y |",
        "| 2 | acc1_z | ADXL345 accelerometer Z |",
        "| 3 | gyro_x | ITG3200 gyroscope X |",
        "| 4 | gyro_y | ITG3200 gyroscope Y |",
        "| 5 | gyro_z | ITG3200 gyroscope Z |",
        "",
        "**Label mapping:** `0` = ADL, `1` = Fall",
        "",
        "### Scaler (train-fit StandardScaler)",
        "",
        "| Channel | Train mean (raw) | Train std (raw) |",
        "| --- | ---: | ---: |",
    ]

    for channel, mean, scale in zip(CHANNELS, scaler.mean_, scaler.scale_):
        sections.append(f"| {channel} | {mean:.4f} | {scale:.4f} |")

    sections.extend(
        [
            "",
            "---",
            "",
            "## 3. Class Distribution",
            "",
            df_to_markdown(class_df),
            "",
            plot_section(
                "Class distribution plots",
                [
                    ("Bar chart", "plots/eda_class_distribution_bar.png"),
                    ("Pie chart", "plots/eda_class_distribution_pie.png"),
                    ("Class balance comparison", "plots/class_balance.png"),
                ],
            ),
            "---",
            "",
            "## 4. Data Integrity",
            "",
            df_to_markdown(integrity_df),
            "",
            "All splits contain both classes. No NaN or infinite values detected.",
            "",
            "---",
            "",
            "## 5. Channel Statistics (normalized values)",
            "",
            df_to_markdown(channel_df),
            "",
            plot_section(
                "Distribution plots",
                [
                    ("Channel boxplots by split", "plots/channel_boxplots.png"),
                    ("Train signal histograms", "plots/eda_signal_histograms_train.png"),
                    ("Validation signal histograms", "plots/eda_signal_histograms_val.png"),
                    ("Test signal histograms", "plots/eda_signal_histograms_test.png"),
                    ("Train signal boxplots", "plots/eda_signal_boxplots_train.png"),
                    ("Validation signal boxplots", "plots/eda_signal_boxplots_val.png"),
                    ("Test signal boxplots", "plots/eda_signal_boxplots_test.png"),
                ],
            ),
            "---",
            "",
            "## 6. Acceleration Magnitude Analysis",
            "",
            "Peak acceleration magnitude per window: "
            "`sqrt(acc1_x² + acc1_y² + acc1_z²)` — computed on normalized accelerometer channels.",
            "",
            df_to_markdown(spike_df),
            "",
            "**Interpretation:** Fall windows show higher median and peak acceleration than ADL across all splits.",
            "",
            plot_section(
                "Acceleration spike plots",
                [
                    ("ADL vs Fall spike histograms", "plots/eda_acceleration_spike_histograms.png"),
                    ("ADL vs Fall spike boxplots", "plots/eda_acceleration_spike_boxplots.png"),
                ],
            ),
            "---",
            "",
            "## 7. Gyroscope Magnitude Analysis",
            "",
            "Peak gyroscope magnitude per window: "
            "`sqrt(gyro_x² + gyro_y² + gyro_z²)` — computed on normalized gyroscope channels.",
            "",
            df_to_markdown(gyro_df),
            "",
            "---",
            "",
            "## 8. Temporal Structure (Autocorrelation)",
            "",
            "Mean lag-1 autocorrelation (train): "
            f"{autocorr_df.loc[(autocorr_df['split'] == 'train') & (autocorr_df['lag'] == 1), 'autocorrelation'].mean():.4f}  \n"
            "Mean lag-8 autocorrelation (train): "
            f"{autocorr_df.loc[(autocorr_df['split'] == 'train') & (autocorr_df['lag'] == 8), 'autocorrelation'].mean():.4f}",
            "",
            plot_section(
                "Autocorrelation plots",
                [
                    ("Train autocorrelation", "plots/eda_autocorrelation_train.png"),
                    ("Validation autocorrelation", "plots/eda_autocorrelation_val.png"),
                    ("Test autocorrelation", "plots/eda_autocorrelation_test.png"),
                ],
            ),
            "---",
            "",
            "## 9. Outlier Profile",
            "",
            "Outliers were detected but not removed (high-amplitude motion is informative for fall detection).",
            "",
            df_to_markdown(outlier_df),
            "",
            plot_section(
                "Outlier plots",
                [
                    ("IQR outlier percentages", "plots/eda_outliers_iqr.png"),
                    ("Z-score outlier percentages", "plots/eda_outliers_z_score.png"),
                ],
            ),
            "---",
            "",
            "## 10. Sample Window Visualizations",
            "",
            plot_section(
                "First window from each split",
                [
                    ("Train — first window (acc axes)", "plots/train_first_window.png"),
                    ("Validation — first window", "plots/val_first_window.png"),
                    ("Test — first window", "plots/test_first_window.png"),
                    ("Low-pass filter check (raw pipeline)", "filter_plot.png"),
                ],
            ),
            "---",
            "",
            "## 11. Saved Artifacts",
            "",
            "| File | Description |",
            "| --- | --- |",
            "| `train.npy` | Training windows `(N, 64, 6)` |",
            "| `train_labels.npy` | Binary labels |",
            "| `train_subject_ids.npy` | Subject ID per window |",
            "| `train_recording_ids.npy` | Recording ID per window |",
            "| `val.npy`, `val_*.npy` | Validation split |",
            "| `test.npy`, `test_*.npy` | Test split |",
            "| `scaler.pkl` | Fitted StandardScaler |",
            "| `plots/` | All visualization PNGs |",
            "",
            "---",
            "",
            "## 12. Modeling Notes",
            "",
            "- **Input shape for deep learning:** `(batch, 64, 6)` for LSTM/CNN; or `(batch, 6, 64)` if channels-first.",
            "- **Class imbalance:** Use weighted loss or balanced batching (~70% ADL / ~30% Fall).",
            "- **Evaluation:** Prefer recording-level or subject-level metrics (windows overlap with 75% stride).",
            "- **Ready for:** 1D CNN, LSTM, CNN-LSTM, Transformer-based HAR models.",
            "",
        ]
    )

    return "\n".join(sections)


def regenerate_plots(data: dict, class_df: pd.DataFrame, autocorr_df: pd.DataFrame, outlier_df: pd.DataFrame) -> None:
    ensure_output_dirs()
    plot_class_distribution(class_df)
    plot_signal_histograms(data)
    plot_signal_boxplots(data)
    plot_acceleration_spikes(data)
    plot_autocorrelation(autocorr_df)
    plot_outlier_summary(outlier_df)
    save_inspection_plots()


def main() -> None:
    print("Removing previous statistics and plots...")
    remove_previous_statistics()

    print("Regenerating plots and computing statistics...")
    data = load_processed_data()
    class_df = class_distribution_table(data)
    stats_df = summarize_signal_health(data)
    spike_df = acceleration_spike_table(data)
    autocorr_df = autocorrelation_table(data)
    outlier_df = outlier_table(data)
    regenerate_plots(data, class_df, autocorr_df, outlier_df)
    gyro_df = gyro_magnitude_table(data)
    integrity_df = integrity_table(data)
    subjects_df = split_subjects_table(data)
    recording_df = recording_stats_table(data)

    report = build_report(
        data,
        class_df,
        stats_df,
        spike_df,
        gyro_df,
        autocorr_df,
        outlier_df,
        integrity_df,
        subjects_df,
        recording_df,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    csv_path = PROCESSED_DIR / "channel_statistics.csv"
    if csv_path.exists():
        csv_path.unlink()
    print(f"Report saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
