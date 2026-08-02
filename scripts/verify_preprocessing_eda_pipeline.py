from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sisfall_preprocessing import (
    IMU_CHANNELS,
    build_master_dataframe,
    generate_timestamps,
    map_activities,
    select_imu_channels,
    split_by_subjects,
)


DATASET_ROOT = ROOT / "SisFall_dataset"
PROCESSED_DIR = ROOT / "data" / "processed"
VERIFY_DIR = PROCESSED_DIR / "verification"
PLOTS_DIR = VERIFY_DIR / "plots"
TABLES_DIR = VERIFY_DIR / "tables"
REPORT_PATH = VERIFY_DIR / "preprocessing_eda_verification_report.md"

SPLITS = ["train", "val", "test"]
LABEL_NAMES = {0: "ADL", 1: "Fall"}
ACC_CHANNELS = ["acc1_x", "acc1_y", "acc1_z"]
WINDOW_SIZE = 64
STRIDE = 16
FS_RAW = 200.0
FS_DOWN = 20.0
RANDOM_SEED = 42


@dataclass(frozen=True)
class WindowSet:
    name: str
    windows: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    recording_ids: np.ndarray
    activity_codes: np.ndarray
    window_indices: np.ndarray
    start_indices: np.ndarray


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def activity_from_recording_id(recording_id: str) -> str:
    filename = recording_id.split(":", 1)[-1]
    return Path(filename).stem.split("_")[0]


def label_name(label: int) -> str:
    return LABEL_NAMES[int(label)]


def magnitude(samples: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(samples[..., :3].astype(np.float64) ** 2, axis=-1))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))


def summarize_window_magnitude(window_set: WindowSet, stage: str) -> pd.DataFrame:
    mag = magnitude(window_set.windows)
    peaks = mag.max(axis=1)
    means = mag.mean(axis=1)
    medians = np.median(mag, axis=1)
    rows = []
    for label in [0, 1]:
        mask = window_set.labels == label
        class_mag = mag[mask].reshape(-1)
        class_peaks = peaks[mask]
        class_means = means[mask]
        class_medians = medians[mask]
        rows.append(
            {
                "stage": stage,
                "class_label": label,
                "class_name": label_name(label),
                "windows": int(mask.sum()),
                "mean_magnitude": float(np.mean(class_means)),
                "median_magnitude": float(np.median(class_medians)),
                "rms_magnitude": rms(class_mag),
                "mean_peak_magnitude": float(np.mean(class_peaks)),
                "median_peak_magnitude": float(np.median(class_peaks)),
                "maximum_peak": float(np.max(class_peaks)),
                "p95_peak": float(np.quantile(class_peaks, 0.95)),
                "std_peak": float(np.std(class_peaks)),
            }
        )
    return pd.DataFrame(rows)


def summarize_activity_magnitude(window_set: WindowSet, stage: str) -> pd.DataFrame:
    mag = magnitude(window_set.windows)
    peaks = mag.max(axis=1)
    rows = []
    for activity in sorted(np.unique(window_set.activity_codes)):
        mask = window_set.activity_codes == activity
        activity_mag = mag[mask].reshape(-1)
        activity_peaks = peaks[mask]
        label = 1 if activity.startswith("F") else 0
        rows.append(
            {
                "stage": stage,
                "activity_code": activity,
                "class_name": label_name(label),
                "windows": int(mask.sum()),
                "mean_magnitude": float(np.mean(activity_mag)),
                "median_magnitude": float(np.median(activity_mag)),
                "mean_peak": float(np.mean(activity_peaks)),
                "median_peak": float(np.median(activity_peaks)),
                "maximum_peak": float(np.max(activity_peaks)),
                "rms": rms(activity_mag),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("mean_peak", ascending=False).reset_index(drop=True)


def build_correct_windows(split_name: str, frame: pd.DataFrame, features: np.ndarray) -> WindowSet:
    frame = frame.reset_index(drop=True)
    windows = []
    labels = []
    subject_ids = []
    recording_ids = []
    activity_codes = []
    window_indices = []
    start_indices = []

    for recording_id in sorted(frame["recording_id"].unique()):
        indices = frame.index[frame["recording_id"] == recording_id].to_numpy()
        if len(indices) < WINDOW_SIZE:
            continue
        group = frame.loc[indices].reset_index(drop=True)
        recording_window_index = 0
        for start in range(0, len(indices) - WINDOW_SIZE + 1, STRIDE):
            selected = indices[start : start + WINDOW_SIZE]
            windows.append(features[selected])
            labels.append(int(group.loc[0, "binary_label"]))
            subject_ids.append(str(group.loc[0, "subject_id"]))
            recording_ids.append(str(recording_id))
            activity_codes.append(str(group.loc[0, "activity_code"]))
            window_indices.append(recording_window_index)
            start_indices.append(start)
            recording_window_index += 1

    return WindowSet(
        name=split_name,
        windows=np.stack(windows, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        recording_ids=np.asarray(recording_ids),
        activity_codes=np.asarray(activity_codes),
        window_indices=np.asarray(window_indices, dtype=np.int64),
        start_indices=np.asarray(start_indices, dtype=np.int64),
    )


def build_bug_compatible_windows(split_name: str, frame: pd.DataFrame, features: np.ndarray) -> WindowSet:
    """Replicate the current generate_sliding_windows indexing behavior for verification only."""
    frame = frame.reset_index(drop=True)
    windows = []
    labels = []
    subject_ids = []
    recording_ids = []
    activity_codes = []
    window_indices = []
    start_indices = []

    for recording_id in sorted(frame["recording_id"].unique()):
        group = frame.loc[frame["recording_id"] == recording_id].reset_index(drop=True)
        if len(group) < WINDOW_SIZE:
            continue
        group_indices = group.index.to_numpy()
        feature_group = features[group_indices]
        recording_window_index = 0
        for start in range(0, len(group_indices) - WINDOW_SIZE + 1, STRIDE):
            windows.append(feature_group[start : start + WINDOW_SIZE])
            labels.append(int(group.loc[0, "binary_label"]))
            subject_ids.append(str(group.loc[0, "subject_id"]))
            recording_ids.append(str(recording_id))
            activity_codes.append(str(group.loc[0, "activity_code"]))
            window_indices.append(recording_window_index)
            start_indices.append(start)
            recording_window_index += 1

    return WindowSet(
        name=split_name,
        windows=np.stack(windows, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        recording_ids=np.asarray(recording_ids),
        activity_codes=np.asarray(activity_codes),
        window_indices=np.asarray(window_indices, dtype=np.int64),
        start_indices=np.asarray(start_indices, dtype=np.int64),
    )


def load_saved_processed_windows() -> dict[str, WindowSet]:
    data = {}
    for split in SPLITS:
        windows = np.load(PROCESSED_DIR / f"{split}.npy")
        labels = np.load(PROCESSED_DIR / f"{split}_labels.npy")
        subject_ids = np.load(PROCESSED_DIR / f"{split}_subject_ids.npy")
        recording_ids = np.load(PROCESSED_DIR / f"{split}_recording_ids.npy")
        activity_codes = np.asarray([activity_from_recording_id(str(rid)) for rid in recording_ids])
        window_indices = np.zeros(len(labels), dtype=np.int64)
        counters: dict[str, int] = {}
        for idx, recording_id in enumerate(recording_ids.astype(str)):
            window_indices[idx] = counters.get(recording_id, 0)
            counters[recording_id] = counters.get(recording_id, 0) + 1
        start_indices = window_indices * STRIDE
        data[split] = WindowSet(
            name=split,
            windows=windows,
            labels=labels,
            subject_ids=subject_ids.astype(str),
            recording_ids=recording_ids.astype(str),
            activity_codes=activity_codes,
            window_indices=window_indices,
            start_indices=start_indices,
        )
    return data


def fast_low_pass_filter(
    raw_df: pd.DataFrame,
    sampling_frequency_hz: float = FS_RAW,
    cutoff_frequency_hz: float = 5.0,
    filter_order: int = 2,
) -> pd.DataFrame:
    """Equivalent filtering for verification, using groupby to avoid repeated full-frame scans."""
    nyquist = sampling_frequency_hz / 2.0
    normalized_cutoff = cutoff_frequency_hz / nyquist
    sos = signal.butter(filter_order, normalized_cutoff, btype="low", output="sos")
    frames = []
    for _, group in raw_df.groupby("recording_id", sort=True):
        group = group.sort_values("timestamp").copy()
        values = group[IMU_CHANNELS].to_numpy(dtype=np.float64)
        if len(values) < 3:
            filtered_values = values
        else:
            try:
                filtered_values = signal.sosfiltfilt(sos, values, axis=0, padtype=None)
            except ValueError:
                filtered_values = signal.sosfilt(sos, values, axis=0)
        group.loc[:, IMU_CHANNELS] = filtered_values
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def fast_downsample(
    filtered_df: pd.DataFrame,
    target_sampling_frequency_hz: float = FS_DOWN,
    original_sampling_frequency_hz: float = FS_RAW,
) -> pd.DataFrame:
    step = int(round(original_sampling_frequency_hz / target_sampling_frequency_hz))
    frames = []
    for _, group in filtered_df.groupby("recording_id", sort=True):
        sampled = group.iloc[np.arange(0, len(group), step)].copy().reset_index(drop=True)
        sampled.loc[:, "timestamp"] = np.arange(len(sampled)) / target_sampling_frequency_hz
        frames.append(sampled)
    return pd.concat(frames, ignore_index=True)


def reconstruct_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    master_df = build_master_dataframe(dataset_root=DATASET_ROOT, show_progress=False)
    activity_df = map_activities(master_df)
    raw_df = select_imu_channels(generate_timestamps(activity_df))
    filtered_df = fast_low_pass_filter(
        raw_df,
        sampling_frequency_hz=FS_RAW,
        cutoff_frequency_hz=5.0,
        filter_order=2,
    )
    downsampled_df = fast_downsample(
        filtered_df,
        target_sampling_frequency_hz=FS_DOWN,
        original_sampling_frequency_hz=FS_RAW,
    )
    split_frames = split_by_subjects(downsampled_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)

    with (PROCESSED_DIR / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    normalized_features = {}
    for split, frame in split_frames.items():
        normalized_features[split] = scaler.transform(frame[IMU_CHANNELS].to_numpy(dtype=np.float64)).astype(np.float32)
    return raw_df, filtered_df, downsampled_df, split_frames, normalized_features


def compare_window_integrity(
    saved: dict[str, WindowSet],
    correct: dict[str, WindowSet],
    bug_compatible: dict[str, WindowSet],
) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        saved_ws = saved[split]
        correct_ws = correct[split]
        bug_ws = bug_compatible[split]
        n = min(len(saved_ws.windows), len(correct_ws.windows), len(bug_ws.windows))
        rows.append(
            {
                "split": split,
                "saved_windows": len(saved_ws.windows),
                "correct_reconstructed_windows": len(correct_ws.windows),
                "bug_compatible_windows": len(bug_ws.windows),
                "saved_vs_correct_shape_match": bool(saved_ws.windows.shape == correct_ws.windows.shape),
                "saved_vs_bug_shape_match": bool(saved_ws.windows.shape == bug_ws.windows.shape),
                "saved_vs_correct_max_abs_diff": float(np.max(np.abs(saved_ws.windows[:n] - correct_ws.windows[:n]))),
                "saved_vs_bug_max_abs_diff": float(np.max(np.abs(saved_ws.windows[:n] - bug_ws.windows[:n]))),
                "saved_labels_match_correct": bool(np.array_equal(saved_ws.labels[:n], correct_ws.labels[:n])),
                "saved_recordings_match_correct": bool(np.array_equal(saved_ws.recording_ids[:n], correct_ws.recording_ids[:n])),
                "saved_labels_match_bug": bool(np.array_equal(saved_ws.labels[:n], bug_ws.labels[:n])),
                "saved_recordings_match_bug": bool(np.array_equal(saved_ws.recording_ids[:n], bug_ws.recording_ids[:n])),
            }
        )
    return pd.DataFrame(rows)


def combine_summary(window_sets: dict[str, WindowSet], stage: str) -> pd.DataFrame:
    return pd.concat([summarize_window_magnitude(window_sets[split], stage) for split in SPLITS], ignore_index=True)


def combine_activity(window_sets: dict[str, WindowSet], stage: str) -> pd.DataFrame:
    frames = [summarize_activity_magnitude(window_sets[split], stage) for split in SPLITS]
    split_df = pd.concat(frames, ignore_index=True)
    rows = []
    for activity in sorted(split_df["activity_code"].unique()):
        activity_rows = split_df[split_df["activity_code"] == activity]
        total_windows = int(activity_rows["windows"].sum())
        rows.append(
            {
                "stage": stage,
                "activity_code": activity,
                "class_name": activity_rows["class_name"].iloc[0],
                "windows": total_windows,
                "mean_magnitude": float(np.average(activity_rows["mean_magnitude"], weights=activity_rows["windows"])),
                "median_magnitude_mean_by_split": float(np.average(activity_rows["median_magnitude"], weights=activity_rows["windows"])),
                "mean_peak": float(np.average(activity_rows["mean_peak"], weights=activity_rows["windows"])),
                "median_peak_mean_by_split": float(np.average(activity_rows["median_peak"], weights=activity_rows["windows"])),
                "maximum_peak": float(activity_rows["maximum_peak"].max()),
                "rms_mean_by_split": float(np.average(activity_rows["rms"], weights=activity_rows["windows"])),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_peak", ascending=False).reset_index(drop=True)


def plot_activity_ranking(activity_df: pd.DataFrame, stage: str) -> None:
    plot_df = activity_df.sort_values("mean_peak", ascending=True)
    colors = ["#F58518" if code.startswith("F") else "#4C78A8" for code in plot_df["activity_code"]]
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(plot_df["activity_code"], plot_df["mean_peak"], color=colors)
    ax.set_title(f"Activity Ranking by Mean Peak Magnitude ({stage})")
    ax.set_xlabel("Mean peak magnitude")
    ax.set_ylabel("Activity")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"activity_ranking_{stage}.png", dpi=200)
    plt.close(fig)


def plot_distribution_comparisons(window_sets: dict[str, WindowSet], stage: str) -> pd.DataFrame:
    rows = []
    all_peaks = []
    all_labels = []
    for split in SPLITS:
        peaks = magnitude(window_sets[split].windows).max(axis=1)
        all_peaks.append(peaks)
        all_labels.append(window_sets[split].labels)
    peaks = np.concatenate(all_peaks)
    labels = np.concatenate(all_labels)
    adl = peaks[labels == 0]
    fall = peaks[labels == 1]

    bins = np.linspace(float(peaks.min()), float(peaks.max()), 80)
    adl_hist, edges = np.histogram(adl, bins=bins, density=True)
    fall_hist, _ = np.histogram(fall, bins=bins, density=True)
    width = np.diff(edges)
    overlap = float(np.sum(np.minimum(adl_hist, fall_hist) * width) * 100.0)
    rows.append({"stage": stage, "overlap_percentage_histogram": overlap, "adl_windows": len(adl), "fall_windows": len(fall)})

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(adl, bins=bins, alpha=0.55, density=True, label="ADL", color="#4C78A8")
    ax.hist(fall, bins=bins, alpha=0.55, density=True, label="Fall", color="#F58518")
    ax.set_title(f"Peak Magnitude Histogram ({stage})")
    ax.set_xlabel("Peak magnitude")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"peak_histogram_{stage}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for values, name, color in [(adl, "ADL", "#4C78A8"), (fall, "Fall", "#F58518")]:
        sample = values
        if len(sample) > 50000:
            rng = np.random.default_rng(RANDOM_SEED)
            sample = rng.choice(sample, size=50000, replace=False)
        kde = stats.gaussian_kde(sample)
        xs = np.linspace(float(peaks.min()), float(peaks.max()), 300)
        ax.plot(xs, kde(xs), label=name, color=color, linewidth=2)
    ax.set_title(f"Peak Magnitude KDE ({stage})")
    ax.set_xlabel("Peak magnitude")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"peak_kde_{stage}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.violinplot([adl, fall], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2], ["ADL", "Fall"])
    ax.set_title(f"Peak Magnitude Violin Plot ({stage})")
    ax.set_ylabel("Peak magnitude")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"peak_violin_{stage}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([adl, fall], tick_labels=["ADL", "Fall"], showfliers=False)
    ax.set_title(f"Peak Magnitude Boxplot ({stage})")
    ax.set_ylabel("Peak magnitude")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"peak_boxplot_{stage}.png", dpi=200)
    plt.close(fig)

    return pd.DataFrame(rows)


def statistical_tests(window_sets: dict[str, WindowSet], stage: str) -> pd.DataFrame:
    peaks = []
    labels = []
    for split in SPLITS:
        peaks.append(magnitude(window_sets[split].windows).max(axis=1))
        labels.append(window_sets[split].labels)
    peaks = np.concatenate(peaks)
    labels = np.concatenate(labels)
    adl = peaks[labels == 0].astype(np.float64)
    fall = peaks[labels == 1].astype(np.float64)

    rng = np.random.default_rng(RANDOM_SEED)
    adl_sample = rng.choice(adl, size=min(5000, len(adl)), replace=False)
    fall_sample = rng.choice(fall, size=min(5000, len(fall)), replace=False)
    shapiro_adl = stats.shapiro(adl_sample)
    shapiro_fall = stats.shapiro(fall_sample)
    mann = stats.mannwhitneyu(adl, fall, alternative="two-sided")
    welch = stats.ttest_ind(adl, fall, equal_var=False)
    pooled_sd = np.sqrt(((len(adl) - 1) * np.var(adl, ddof=1) + (len(fall) - 1) * np.var(fall, ddof=1)) / (len(adl) + len(fall) - 2))
    cohens_d = float((np.mean(fall) - np.mean(adl)) / pooled_sd) if pooled_sd > 0 else 0.0
    common_language = float(stats.mannwhitneyu(fall, adl, alternative="greater").statistic / (len(fall) * len(adl)))
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "adl_n": len(adl),
                "fall_n": len(fall),
                "adl_mean_peak": float(np.mean(adl)),
                "fall_mean_peak": float(np.mean(fall)),
                "adl_median_peak": float(np.median(adl)),
                "fall_median_peak": float(np.median(fall)),
                "shapiro_adl_statistic_sample_5000": float(shapiro_adl.statistic),
                "shapiro_adl_p_value_sample_5000": float(shapiro_adl.pvalue),
                "shapiro_fall_statistic_sample_5000": float(shapiro_fall.statistic),
                "shapiro_fall_p_value_sample_5000": float(shapiro_fall.pvalue),
                "primary_test": "Mann-Whitney U",
                "mann_whitney_u_statistic": float(mann.statistic),
                "mann_whitney_p_value": float(mann.pvalue),
                "welch_t_statistic_reference": float(welch.statistic),
                "welch_p_value_reference": float(welch.pvalue),
                "cohens_d_fall_minus_adl": cohens_d,
                "probability_fall_peak_greater_than_adl": common_language,
            }
        ]
    )


def select_sample_windows(correct_norm: dict[str, WindowSet]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    combined = []
    for split in SPLITS:
        ws = correct_norm[split]
        for idx in range(len(ws.windows)):
            combined.append((split, idx, int(ws.labels[idx])))
    rows = []
    for label in [1, 0]:
        candidates = [(split, idx) for split, idx, item_label in combined if item_label == label]
        selected = rng.choice(len(candidates), size=10, replace=False)
        for selection in selected:
            split, idx = candidates[int(selection)]
            ws = correct_norm[split]
            mag = magnitude(ws.windows[idx : idx + 1])[0]
            max_offset = int(np.argmax(mag))
            rows.append(
                {
                    "split": split,
                    "class_name": label_name(label),
                    "array_index": int(idx),
                    "recording_id": str(ws.recording_ids[idx]),
                    "subject_id": str(ws.subject_ids[idx]),
                    "activity_code": str(ws.activity_codes[idx]),
                    "window_index_within_recording": int(ws.window_indices[idx]),
                    "window_start_sample_downsampled": int(ws.start_indices[idx]),
                    "max_offset_in_window": max_offset,
                    "timestamp_of_max_seconds": float((ws.start_indices[idx] + max_offset) / FS_DOWN),
                    "max_acceleration_magnitude": float(mag[max_offset]),
                    "max_inside_window": bool(0 <= max_offset < WINDOW_SIZE),
                }
            )
            plot_sample_window(ws, idx, split, label_name(label))
    return pd.DataFrame(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_sample_window(ws: WindowSet, idx: int, split: str, class_name: str) -> None:
    window = ws.windows[idx]
    mag = magnitude(window[None, :, :])[0]
    max_offset = int(np.argmax(mag))
    times = np.arange(WINDOW_SIZE) / FS_DOWN
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for channel_idx, channel in enumerate(ACC_CHANNELS):
        axes[0].plot(times, window[:, channel_idx], label=channel)
    axes[0].axvline(times[max_offset], color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"{class_name} {split} {ws.recording_ids[idx]} window {ws.window_indices[idx]}")
    axes[0].set_ylabel("Normalized channel value")
    axes[0].legend()
    axes[1].plot(times, mag, color="#F58518", label="magnitude")
    axes[1].scatter([times[max_offset]], [mag[max_offset]], color="black", zorder=3)
    axes[1].set_xlabel("Seconds within window")
    axes[1].set_ylabel("Normalized magnitude")
    axes[1].legend()
    fig.tight_layout()
    filename = f"window_{class_name}_{split}_{idx}_{safe_name(str(ws.recording_ids[idx]))}.png"
    fig.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close(fig)


def stage_peak_for_recording(frame: pd.DataFrame, recording_id: str) -> float:
    values = frame.loc[frame["recording_id"] == recording_id, ACC_CHANNELS].to_numpy(dtype=np.float64)
    return float(np.max(magnitude(values)))


def filtering_verification(
    raw_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    downsampled_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    normalized_features: dict[str, np.ndarray],
) -> pd.DataFrame:
    fall_recordings = sorted(downsampled_df.loc[downsampled_df["binary_label"] == 1, "recording_id"].unique())[:6]
    rows = []
    normalized_frames = {}
    for split, frame in split_frames.items():
        norm_frame = frame.copy().reset_index(drop=True)
        norm_frame.loc[:, IMU_CHANNELS] = normalized_features[split]
        normalized_frames[split] = norm_frame

    for recording_id in fall_recordings:
        raw_peak = stage_peak_for_recording(raw_df, recording_id)
        filtered_peak = stage_peak_for_recording(filtered_df, recording_id)
        down_peak = stage_peak_for_recording(downsampled_df, recording_id)
        split_name = next(split for split, frame in split_frames.items() if recording_id in set(frame["recording_id"].unique()))
        norm_peak = stage_peak_for_recording(normalized_frames[split_name], recording_id)

        rows.append(
            {
                "recording_id": recording_id,
                "activity_code": activity_from_recording_id(recording_id),
                "split": split_name,
                "raw_peak": raw_peak,
                "filtered_peak": filtered_peak,
                "downsampled_peak": down_peak,
                "normalized_peak_unitless": norm_peak,
                "raw_to_filtered_pct_change": pct_change(raw_peak, filtered_peak),
                "filtered_to_downsampled_pct_change": pct_change(filtered_peak, down_peak),
                "downsampled_to_normalized_pct_change_unitless": pct_change(down_peak, norm_peak),
            }
        )
        plot_filtering_stages(recording_id, raw_df, filtered_df, downsampled_df, normalized_frames[split_name])
    return pd.DataFrame(rows)


def pct_change(before: float, after: float) -> float:
    return float((after - before) / before * 100.0) if before != 0 else np.nan


def plot_filtering_stages(
    recording_id: str,
    raw_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    downsampled_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
) -> None:
    stages = [
        ("Raw", raw_df, FS_RAW),
        ("Filtered", filtered_df, FS_RAW),
        ("Downsampled", downsampled_df, FS_DOWN),
        ("Normalized", normalized_df, FS_DOWN),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    for ax, (stage_name, frame, fs) in zip(axes, stages):
        group = frame.loc[frame["recording_id"] == recording_id].reset_index(drop=True)
        values = group[ACC_CHANNELS].to_numpy(dtype=np.float64)
        mag = magnitude(values)
        times = np.arange(len(group)) / fs
        ax.plot(times, mag, color="#F58518", linewidth=1.0, label="magnitude")
        ax.set_title(f"{stage_name} peak={np.max(mag):.4f}")
        ax.set_ylabel("Magnitude")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Seconds")
    fig.suptitle(f"Filtering Stage Verification: {recording_id}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"filtering_stages_{safe_name(recording_id)}.png", dpi=200)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")
        else:
            display[column] = display[column].astype(str)
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def write_report(
    integrity_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    activity_pre_df: pd.DataFrame,
    activity_saved_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    filtering_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> None:
    highest_adl_pre = activity_pre_df[activity_pre_df["class_name"] == "ADL"].iloc[0]
    highest_fall_pre = activity_pre_df[activity_pre_df["class_name"] == "Fall"].iloc[0]
    highest_adl_saved = activity_saved_df[activity_saved_df["class_name"] == "ADL"].iloc[0]
    highest_fall_saved = activity_saved_df[activity_saved_df["class_name"] == "Fall"].iloc[0]
    report = [
        "# Preprocessing and EDA Verification Report",
        "",
        "This report reconstructs intermediate preprocessing stages and compares them with the saved processed arrays. No dataset files were modified and no model was trained.",
        "",
        "## Pipeline Integrity Check",
        "",
        dataframe_to_markdown(integrity_df),
        "",
        "Interpretation: `saved_vs_correct_max_abs_diff` compares saved processed windows with correctly reconstructed normalized windows. `saved_vs_bug_max_abs_diff` compares saved processed windows with a reconstruction of the current window-indexing behavior.",
        "",
        "## Part 1 - Peak Acceleration Summary",
        "",
        dataframe_to_markdown(peak_df),
        "",
        "## Part 2 - Activity-wise Ranking",
        "",
        f"Highest ADL before normalization: `{highest_adl_pre['activity_code']}` with mean peak `{highest_adl_pre['mean_peak']:.6g}`.",
        f"Highest Fall before normalization: `{highest_fall_pre['activity_code']}` with mean peak `{highest_fall_pre['mean_peak']:.6g}`.",
        f"Highest ADL in saved normalized processed data: `{highest_adl_saved['activity_code']}` with mean peak `{highest_adl_saved['mean_peak']:.6g}`.",
        f"Highest Fall in saved normalized processed data: `{highest_fall_saved['activity_code']}` with mean peak `{highest_fall_saved['mean_peak']:.6g}`.",
        "",
        "Complete ranking before normalization:",
        "",
        dataframe_to_markdown(activity_pre_df),
        "",
        "Complete ranking in saved normalized processed data:",
        "",
        dataframe_to_markdown(activity_saved_df),
        "",
        "## Part 3 - Window Verification Sample",
        "",
        dataframe_to_markdown(sample_df),
        "",
        "Window plots were saved as `window_*.png` in the verification plots folder. The `max_inside_window` column verifies whether the maximum magnitude sample is contained in the selected 64-step window.",
        "",
        "## Part 4 - Filtering Verification",
        "",
        dataframe_to_markdown(filtering_df),
        "",
        "The normalized peak is unitless and is not directly comparable to raw physical units; its percentage change is included only as a numerical transform check.",
        "",
        "## Part 5 - Distribution Overlap",
        "",
        dataframe_to_markdown(overlap_df),
        "",
        "Overlap percentage is estimated from density histograms of ADL and Fall peak magnitudes. Higher values indicate stronger class overlap.",
        "",
        "## Part 6 - Statistical Significance",
        "",
        dataframe_to_markdown(stats_df),
        "",
        "Shapiro-Wilk tests use deterministic samples of up to 5,000 windows per class because the full sample size is very large. Mann-Whitney U is used as the primary non-parametric test.",
        "",
        "## Part 7 - Evidence-based Answers",
        "",
        "1. Do Fall windows actually have higher acceleration than ADL?",
        "",
        "Use the Part 1 and Part 6 tables. The answer depends on the stage. The report gives class means, medians, p-values, effect sizes, and the probability that a random Fall peak exceeds a random ADL peak.",
        "",
        "2. If not, why?",
        "",
        "Use the activity ranking and distribution overlap tables. High-motion ADL activities can overlap with or exceed some fall activities, so binary labels alone do not guarantee higher acceleration peaks for every fall window.",
        "",
        "3. Is normalization responsible?",
        "",
        "Compare the before-normalization and saved-normalized rows in Part 1. Normalization changes units and relative axis scaling, so it can change magnitude ordering when magnitude is computed after scaling.",
        "",
        "4. Is Butterworth filtering responsible?",
        "",
        "Use Part 4. Raw-to-filtered percentage changes quantify how much peak magnitude changed after filtering for selected fall recordings.",
        "",
        "5. Is downsampling responsible?",
        "",
        "Use Part 4. Filtered-to-downsampled percentage changes quantify peak loss or preservation after downsampling.",
        "",
        "6. Are high-impact ADL activities causing overlap?",
        "",
        "Use Part 2. If the highest-ranked ADL activities appear above fall activities, then ADL motion contributes directly to overlap.",
        "",
        "7. Are the sliding windows missing the impact?",
        "",
        "Use Part 3. The sampled windows report the maximum offset and timestamp within each window, and plots show whether the local peak is visually contained.",
        "",
        "8. Is the preprocessing pipeline correct?",
        "",
        "Use the Pipeline Integrity Check. Any large difference between saved processed windows and correctly reconstructed windows requires fixing before training.",
        "",
        "9. Is this behaviour expected in the SisFall dataset?",
        "",
        "It is plausible for SisFall because ADL activities include vigorous motions such as jogging, jumping, stumbling, and fast sitting/standing, while some falls may be lower amplitude depending on fall type and sensor orientation. The activity ranking provides dataset-specific evidence.",
        "",
        "10. Should anything be changed before model training?",
        "",
        "If the integrity check shows saved processed windows do not match correctly reconstructed windows, regenerate the processed datasets after fixing window indexing. If the integrity check passes, keep the preprocessing but avoid using peak acceleration as a standalone decision rule.",
        "",
        "## Output Files",
        "",
        f"- Tables: `{TABLES_DIR.relative_to(ROOT)}`",
        f"- Plots: `{PLOTS_DIR.relative_to(ROOT)}`",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def run_verification() -> None:
    ensure_dirs()
    raw_df, filtered_df, downsampled_df, split_frames, normalized_features = reconstruct_pipeline()

    saved_norm = load_saved_processed_windows()
    pre_norm_correct = {
        split: build_correct_windows(split, split_frames[split], split_frames[split][IMU_CHANNELS].to_numpy(dtype=np.float32))
        for split in SPLITS
    }
    norm_correct = {split: build_correct_windows(split, split_frames[split], normalized_features[split]) for split in SPLITS}
    norm_bug = {split: build_bug_compatible_windows(split, split_frames[split], normalized_features[split]) for split in SPLITS}

    integrity_df = compare_window_integrity(saved_norm, norm_correct, norm_bug)
    integrity_df.to_csv(TABLES_DIR / "pipeline_integrity_check.csv", index=False)

    peak_df = pd.concat(
        [
            combine_summary(pre_norm_correct, "filtered_downsampled_before_normalization"),
            combine_summary(saved_norm, "saved_processed_after_normalization"),
            combine_summary(norm_correct, "correct_reconstructed_after_normalization"),
        ],
        ignore_index=True,
    )
    peak_df.to_csv(TABLES_DIR / "peak_acceleration_summary.csv", index=False)

    activity_pre_df = combine_activity(pre_norm_correct, "filtered_downsampled_before_normalization")
    activity_saved_df = combine_activity(saved_norm, "saved_processed_after_normalization")
    activity_correct_norm_df = combine_activity(norm_correct, "correct_reconstructed_after_normalization")
    activity_pre_df.to_csv(TABLES_DIR / "activity_ranking_before_normalization.csv", index=False)
    activity_saved_df.to_csv(TABLES_DIR / "activity_ranking_saved_processed_after_normalization.csv", index=False)
    activity_correct_norm_df.to_csv(TABLES_DIR / "activity_ranking_correct_reconstructed_after_normalization.csv", index=False)
    plot_activity_ranking(activity_pre_df, "before_normalization")
    plot_activity_ranking(activity_saved_df, "saved_after_normalization")
    plot_activity_ranking(activity_correct_norm_df, "correct_after_normalization")

    sample_df = select_sample_windows(norm_correct)
    sample_df.to_csv(TABLES_DIR / "sample_window_verification.csv", index=False)

    filtering_df = filtering_verification(raw_df, filtered_df, downsampled_df, split_frames, normalized_features)
    filtering_df.to_csv(TABLES_DIR / "filtering_stage_peak_reduction.csv", index=False)

    overlap_df = pd.concat(
        [
            plot_distribution_comparisons(pre_norm_correct, "before_normalization"),
            plot_distribution_comparisons(saved_norm, "saved_after_normalization"),
            plot_distribution_comparisons(norm_correct, "correct_after_normalization"),
        ],
        ignore_index=True,
    )
    overlap_df.to_csv(TABLES_DIR / "distribution_overlap.csv", index=False)

    stats_df = pd.concat(
        [
            statistical_tests(pre_norm_correct, "before_normalization"),
            statistical_tests(saved_norm, "saved_after_normalization"),
            statistical_tests(norm_correct, "correct_after_normalization"),
        ],
        ignore_index=True,
    )
    stats_df.to_csv(TABLES_DIR / "statistical_tests.csv", index=False)

    write_report(integrity_df, peak_df, activity_pre_df, activity_saved_df, sample_df, filtering_df, overlap_df, stats_df)
    print("Verification complete.")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Tables: {TABLES_DIR.relative_to(ROOT)}")
    print(f"Plots: {PLOTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    run_verification()
