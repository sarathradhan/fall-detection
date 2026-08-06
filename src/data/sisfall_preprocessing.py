from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm


SENSOR_COLUMNS = [
    "acc1_x",
    "acc1_y",
    "acc1_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc2_x",
    "acc2_y",
    "acc2_z",
]

IMU_CHANNELS = [
    "acc1_x",
    "acc1_y",
    "acc1_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

ACC_CHANNEL_INDICES = (0, 1, 2)

METADATA_COLUMNS = [
    "subject_id",
    "activity_code",
    "activity_name",
    "recording_id",
    "binary_label",
    "timestamp",
    "source_file",
    "source_path",
]

REMOVED_CHANNELS = [
    "acc2_x",
    "acc2_y",
    "acc2_z",
]

ACTIVITY_MAPPING = {
    "D01": "Walking slowly",
    "D02": "Walking quickly",
    "D03": "Jogging slowly",
    "D04": "Jogging quickly",
    "D05": "Walking upstairs and downstairs slowly",
    "D06": "Walking upstairs and downstairs quickly",
    "D07": "Slowly sit in a half height chair, wait a moment, and up slowly",
    "D08": "Quickly sit in a half height chair, wait a moment, and up quickly",
    "D09": "Slowly sit in a low height chair, wait a moment, and up slowly",
    "D10": "Quickly sit in a low height chair, wait a moment, and up quickly",
    "D11": "Sitting a moment, trying to get up, and collapse into a chair",
    "D12": "Sitting a moment, lying slowly, wait a moment, and sit again",
    "D13": "Sitting a moment, lying quickly, wait a moment, and sit again",
    "D14": "Being on one’s back change to lateral position, wait a moment, and change to one’s back",
    "D15": "Standing, slowly bending at knees, and getting up",
    "D16": "Standing, slowly bending without bending knees, and getting up",
    "D17": "Standing, get into a car, remain seated and get out of the car",
    "D18": "Stumble while walking",
    "D19": "Gently jump without falling (trying to reach a high object)",
    "F01": "Fall forward while walking caused by a slip",
    "F02": "Fall backward while walking caused by a slip",
    "F03": "Lateral fall while walking caused by a slip",
    "F04": "Fall forward while walking caused by a trip",
    "F05": "Fall forward while jogging caused by a trip",
    "F06": "Vertical fall while walking caused by fainting",
    "F07": "Fall while walking, with use of hands in a table to dampen fall, caused by fainting",
    "F08": "Fall forward when trying to get up",
    "F09": "Lateral fall when trying to get up",
    "F10": "Fall forward when trying to sit down",
    "F11": "Fall backward when trying to sit down",
    "F12": "Lateral fall when trying to sit down",
    "F13": "Fall forward while sitting, caused by fainting or falling asleep",
    "F14": "Fall backward while sitting, caused by fainting or falling asleep",
    "F15": "Lateral fall while sitting, caused by fainting or falling asleep",
}


def _extract_subject_id(path: Path) -> str:
    """Return the subject ID for a subject directory."""
    return path.name


def _extract_activity_code(file_name: str) -> str:
    """Return the activity code from the filename.

    SisFall files are typically named as ``<activity_code>_<subject>_<trial>.txt``.
    This helper also accepts a simplified form such as ``D01.txt`` so that the
    preprocessing logic remains robust for small test fixtures and edge cases.
    """

    stem = Path(file_name).stem
    parts = stem.split("_")
    if not parts:
        raise ValueError(f"Unexpected filename format: {file_name}")

    candidate = parts[0]
    if candidate.startswith(("D", "F")) and len(candidate) > 1 and candidate[1:].isdigit():
        return candidate

    if len(parts) >= 2:
        return parts[0]

    raise ValueError(f"Unexpected filename format: {file_name}")


def _load_text_file(file_path: Path) -> pd.DataFrame:
    """Load a single SisFall activity file into a DataFrame."""
    rows: list[list[float]] = []
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            cleaned = line.strip().rstrip(";").strip()
            if not cleaned:
                continue
            values = [float(part.strip()) for part in cleaned.split(",")]
            if len(values) != 9:
                raise ValueError(
                    f"Expected 9 sensor values in {file_path.name}, but found {len(values)} on line {line_number}."
                )
            rows.append(values)

    if not rows:
        raise ValueError(f"No usable rows found in {file_path.name}.")

    return pd.DataFrame(rows, columns=SENSOR_COLUMNS)


def build_master_dataframe(
    dataset_root: str | Path | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Merge every SisFall activity file into one master DataFrame."""

    if dataset_root is None:
        dataset_root = Path(__file__).resolve().parents[2] / "SisFall_dataset"
    else:
        dataset_root = Path(dataset_root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root was not found: {dataset_root}")

    subject_dirs = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir() and (path.name.startswith("SA") or path.name.startswith("SE"))]
    )

    if not subject_dirs:
        raise ValueError(f"No subject directories were found in {dataset_root}.")

    records: list[pd.DataFrame] = []
    missing_files: list[dict[str, str]] = []
    corrupted_files: list[dict[str, str]] = []
    expected_file_count = 0

    for subject_dir in tqdm(subject_dirs, disable=not show_progress, desc="Building master dataframe"):
        subject_id = _extract_subject_id(subject_dir)
        activity_files = sorted(subject_dir.glob("*.txt"))
        if not activity_files:
            missing_files.append({"subject_id": subject_id, "reason": "No activity files found."})
            continue

        expected_file_count += len(activity_files)
        for file_path in activity_files:
            try:
                dataframe = _load_text_file(file_path)
            except Exception as exc:  # pragma: no cover - defensive programming
                corrupted_files.append({"subject_id": subject_id, "filename": file_path.name, "reason": str(exc)})
                continue

            activity_code = _extract_activity_code(file_path.name)
            dataframe = dataframe.copy()
            dataframe["subject_id"] = subject_id
            dataframe["activity_code"] = activity_code
            dataframe["source_file"] = file_path.name
            dataframe["source_path"] = str(file_path)
            dataframe["recording_id"] = f"{subject_id}:{file_path.name}"
            records.append(dataframe)

    if not records:
        raise ValueError("No valid activity files could be loaded from the dataset.")

    master_df = pd.concat(records, ignore_index=True)

    print("\nTask 1: Merge Dataset")
    print("=" * 40)
    print(f"Number of subjects discovered: {len(subject_dirs)}")
    print(f"Number of expected activity files: {expected_file_count}")
    print(f"Number of valid activity files loaded: {len(records)}")
    print(f"Number of rows in master DataFrame: {len(master_df):,}")
    print(f"Missing activity files: {expected_file_count - len(records) - len(corrupted_files)}")
    print(f"Missing or empty subject folders: {len(missing_files)}")
    print(f"Corrupted or unreadable files: {len(corrupted_files)}")

    if missing_files:
        print("Missing folders:")
        for entry in missing_files[:10]:
            print(f"- {entry['subject_id']}: {entry['reason']}")

    if corrupted_files:
        print("Corrupted files:")
        for entry in corrupted_files[:10]:
            print(f"- {entry['subject_id']}/{entry['filename']}: {entry['reason']}")

    print("\nFirst 5 rows:")
    print(master_df.head().to_string(index=False))
    print("\nLast 5 rows:")
    print(master_df.tail().to_string(index=False))

    return master_df


def map_activities(master_df: pd.DataFrame) -> pd.DataFrame:
    """Add activity names and binary labels to the master DataFrame."""

    activity_mapping = ACTIVITY_MAPPING

    required_columns = {"activity_code", "subject_id", "source_file", "source_path", "recording_id"}
    missing_columns = required_columns.difference(master_df.columns)
    if missing_columns:
        raise ValueError(f"The input DataFrame is missing required columns: {sorted(missing_columns)}")

    mapped_df = master_df.copy()
    mapped_df["activity_name"] = mapped_df["activity_code"].map(activity_mapping)

    unknown_codes = sorted(set(mapped_df["activity_code"]) - set(activity_mapping.keys()))
    if unknown_codes:
        raise ValueError(f"Unmapped activity codes found: {unknown_codes}")

    mapped_df["binary_label"] = mapped_df["activity_code"].str.startswith("F").astype(int)

    class_counts = mapped_df["binary_label"].value_counts().sort_index()
    class_distribution = mapped_df["binary_label"].value_counts(normalize=True).sort_index()

    adl_codes = sorted([code for code in activity_mapping if code.startswith("D")])
    fall_codes = sorted([code for code in activity_mapping if code.startswith("F")])

    print("\nTask 2: Map Activities")
    print("=" * 40)
    print("Class counts:")
    print(class_counts)
    print("\nClass distribution:")
    print(class_distribution)
    print(f"\nADL activity codes: {len(adl_codes)}")
    print(f"Fall activity codes: {len(fall_codes)}")
    print(f"Activity codes mapped: {mapped_df['activity_code'].nunique()}")

    return mapped_df


def generate_timestamps(
    master_df: pd.DataFrame,
    sampling_frequency_hz: float = 200.0,
) -> pd.DataFrame:
    """Generate timestamps for each sample and reset them per activity file."""

    if sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be positive.")

    required_columns = {"subject_id", "source_file", "source_path", "recording_id"}
    missing_columns = required_columns.difference(master_df.columns)
    if missing_columns:
        raise ValueError(f"The input DataFrame is missing required columns: {sorted(missing_columns)}")

    timestamped_df = master_df.copy().sort_values(["recording_id", "source_file", "subject_id"]).reset_index(drop=True)
    timestamps: list[float] = []

    current_recording: str | None = None
    current_counter = 0

    for recording_id in timestamped_df["recording_id"]:
        if recording_id != current_recording:
            current_recording = recording_id
            current_counter = 0
        timestamps.append(current_counter / sampling_frequency_hz)
        current_counter += 1

    timestamped_df["timestamp"] = timestamps

    print("\nTask 3: Generate Timestamps")
    print("=" * 40)
    print(f"Sampling frequency: {sampling_frequency_hz} Hz")
    print("First 10 timestamps for the first few files:")
    sample_files = timestamped_df["recording_id"].drop_duplicates().head(3)
    for recording_id in sample_files:
        subset = timestamped_df[timestamped_df["recording_id"] == recording_id].head(10)
        print(f"\n{recording_id}:")
        print(subset[["subject_id", "activity_code", "timestamp"]].to_string(index=False))

    return timestamped_df


def select_imu_channels(
    preprocessed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the wearable IMU channels and preserve all metadata columns."""

    missing_columns = set(IMU_CHANNELS + REMOVED_CHANNELS + METADATA_COLUMNS).difference(preprocessed_df.columns)
    if missing_columns:
        raise ValueError(f"The input DataFrame is missing required columns: {sorted(missing_columns)}")

    before_shape = preprocessed_df.shape
    selected_df = preprocessed_df.loc[:, IMU_CHANNELS + METADATA_COLUMNS]
    after_shape = selected_df.shape

    print("\nStage 2 - Task 1: Channel Selection")
    print("=" * 40)
    print("Removed columns:")
    print(REMOVED_CHANNELS)
    print("\nRemaining columns:")
    print(selected_df.columns.tolist())
    print(f"\nShape before: {before_shape}")
    print(f"Shape after: {after_shape}")

    return selected_df


def apply_low_pass_filter(
    preprocessed_df: pd.DataFrame,
    sampling_frequency_hz: float = 200.0,
    cutoff_frequency_hz: float = 5.0,
    filter_order: int = 4,
    plot_path: str | Path | None = None,
) -> pd.DataFrame:
    """Apply a Butterworth low-pass filter independently to each recording."""

    if sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be positive.")
    if cutoff_frequency_hz <= 0:
        raise ValueError("cutoff_frequency_hz must be positive.")
    if filter_order <= 0:
        raise ValueError("filter_order must be positive.")

    required_columns = set(IMU_CHANNELS + METADATA_COLUMNS)
    missing_columns = required_columns.difference(preprocessed_df.columns)
    if missing_columns:
        raise ValueError(f"The input DataFrame is missing required columns: {sorted(missing_columns)}")

    filtered_rows: list[pd.DataFrame] = []
    recording_groups = preprocessed_df.groupby("recording_id", sort=True)

    print("\nStage 2 - Task 2: Noise Filtering")
    print("=" * 40)
    print("Recommended parameters:")
    print("- Butterworth low-pass filter")
    print(f"- Order: {filter_order}")
    print(f"- Cutoff: {cutoff_frequency_hz:g} Hz")
    print("- Reason: preserves activity-relevant motion while removing high-frequency noise")

    for recording_id, group in tqdm(
        recording_groups,
        total=preprocessed_df["recording_id"].nunique(),
        desc="Filtering recordings",
    ):
        group = group.copy()
        group = group.sort_values("timestamp").reset_index(drop=True)
        values = group[IMU_CHANNELS].to_numpy(dtype=np.float64)

        if len(values) < 3:
            filtered_values = values
        else:
            nyquist = sampling_frequency_hz / 2.0
            normalized_cutoff = cutoff_frequency_hz / nyquist
            sos = signal.butter(filter_order, normalized_cutoff, btype="low", output="sos")
            try:
                filtered_values = signal.sosfiltfilt(sos, values, axis=0, padtype=None)
            except ValueError:
                filtered_values = signal.sosfilt(sos, values, axis=0)

        filtered_group = group.copy()
        filtered_group.loc[:, IMU_CHANNELS] = filtered_values
        filtered_rows.append(filtered_group)

    filtered_df = pd.concat(filtered_rows, ignore_index=True)

    if plot_path is not None:
        plot_path = Path(plot_path).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        sample_recording = filtered_df["recording_id"].iloc[0]
        before_group = preprocessed_df.loc[preprocessed_df["recording_id"] == sample_recording].copy()
        after_group = filtered_df.loc[filtered_df["recording_id"] == sample_recording].copy()
        if len(before_group) > 10 and len(after_group) > 10:
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            for axis, channel in enumerate(["acc1_x", "acc1_y", "acc1_z"]):
                axes[axis].plot(before_group["timestamp"], before_group[channel], label="before", alpha=0.7)
                axes[axis].plot(after_group["timestamp"], after_group[channel], label="after", alpha=0.9)
                axes[axis].set_ylabel(channel)
                axes[axis].legend(loc="upper right")
            axes[-1].set_xlabel("Timestamp (s)")
            fig.suptitle(f"Filtering check for {sample_recording}")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Filtering plot saved to: {plot_path}")
        else:
            print("Plot skipped because the sample recording was too short.")

    return filtered_df


def downsample_recordings(
    filtered_df: pd.DataFrame,
    target_sampling_frequency_hz: float = 20.0,
    original_sampling_frequency_hz: float = 200.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Downsample each recording independently after filtering."""

    if target_sampling_frequency_hz <= 0:
        raise ValueError("target_sampling_frequency_hz must be positive.")
    if original_sampling_frequency_hz <= 0:
        raise ValueError("original_sampling_frequency_hz must be positive.")
    if target_sampling_frequency_hz >= original_sampling_frequency_hz:
        raise ValueError("target_sampling_frequency_hz must be lower than the original sampling rate.")

    required_columns = set(IMU_CHANNELS + METADATA_COLUMNS)
    missing_columns = required_columns.difference(filtered_df.columns)
    if missing_columns:
        raise ValueError(f"The input DataFrame is missing required columns: {sorted(missing_columns)}")

    step = int(round(original_sampling_frequency_hz / target_sampling_frequency_hz))
    if step <= 1:
        raise ValueError("The computed downsampling step is invalid.")

    downsampled_rows: list[pd.DataFrame] = []
    original_counts: list[int] = []
    downsampled_counts: list[int] = []

    print("\nStage 2 - Task 3: Downsampling")
    print("=" * 40)
    print(f"Original sampling frequency: {original_sampling_frequency_hz} Hz")
    print(f"Target sampling frequency: {target_sampling_frequency_hz} Hz")
    print(f"Downsampling ratio: {step}")

    recording_groups = filtered_df.groupby("recording_id", sort=True)
    for recording_id, group in tqdm(
        recording_groups,
        total=filtered_df["recording_id"].nunique(),
        desc="Downsampling recordings",
    ):
        group = group.copy()
        if len(group) == 0:
            continue

        indices = np.arange(0, len(group), step)
        sampled_group = group.iloc[indices].copy()
        if sampled_group.empty:
            continue

        sampled_group = sampled_group.reset_index(drop=True)
        sampled_group.loc[:, "timestamp"] = np.arange(len(sampled_group)) / target_sampling_frequency_hz
        downsampled_rows.append(sampled_group)
        original_counts.append(len(group))
        downsampled_counts.append(len(sampled_group))

    if not downsampled_rows:
        raise ValueError("No recordings were retained after downsampling.")

    downsampled_df = pd.concat(downsampled_rows, ignore_index=True)
    summary = {
        "original_sample_count": int(sum(original_counts)),
        "downsampled_sample_count": int(sum(downsampled_counts)),
        "downsampling_ratio": step,
        "new_sampling_frequency_hz": target_sampling_frequency_hz,
    }

    print("Downsampling summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    return downsampled_df, summary


def split_by_subjects(
    preprocessed_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split the dataset by subject so no subject appears in more than one split."""

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    required_columns = set(METADATA_COLUMNS).union(IMU_CHANNELS)
    missing_columns = required_columns.difference(preprocessed_df.columns)
    if missing_columns:
        raise ValueError(f"The input DataFrame is missing required columns: {sorted(missing_columns)}")

    subjects = sorted(preprocessed_df["subject_id"].unique())
    rng = np.random.default_rng(random_state)
    shuffled_subjects = rng.permutation(subjects)

    if len(shuffled_subjects) == 0:
        raise ValueError("Cannot split an empty dataset by subject.")

    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    exact_counts = ratios * len(shuffled_subjects)
    target_counts = np.floor(exact_counts).astype(int).tolist()
    remainder = len(shuffled_subjects) - sum(target_counts)
    if remainder > 0:
        fractional_parts = exact_counts - np.floor(exact_counts)
        for split_index in np.argsort(fractional_parts)[::-1][:remainder]:
            target_counts[int(split_index)] += 1

    if len(shuffled_subjects) >= 3:
        for split_index, count in enumerate(target_counts):
            if count == 0:
                donor_index = int(np.argmax(target_counts))
                if target_counts[donor_index] > 1:
                    target_counts[donor_index] -= 1
                    target_counts[split_index] += 1

    if sum(target_counts) != len(shuffled_subjects):
        raise ValueError(
            "Could not assign every subject to a split. "
            f"Requested counts {target_counts} for {len(shuffled_subjects)} subjects."
        )

    n_train, n_val, n_test = target_counts

    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train : n_train + n_val]
    test_subjects = shuffled_subjects[n_train + n_val : n_train + n_val + n_test]

    assigned_subjects = set(train_subjects) | set(val_subjects) | set(test_subjects)
    if assigned_subjects != set(subjects):
        missing = sorted(set(subjects) - assigned_subjects)
        raise ValueError(f"Subject split did not include all subjects. Missing: {missing}")

    if len(set(train_subjects).intersection(set(val_subjects))) > 0:
        raise ValueError("Subject overlap detected between train and validation splits.")
    if len(set(train_subjects).intersection(set(test_subjects))) > 0:
        raise ValueError("Subject overlap detected between train and test splits.")
    if len(set(val_subjects).intersection(set(test_subjects))) > 0:
        raise ValueError("Subject overlap detected between validation and test splits.")

    split_frames = {
        "train": preprocessed_df.loc[preprocessed_df["subject_id"].isin(train_subjects)].copy(),
        "val": preprocessed_df.loc[preprocessed_df["subject_id"].isin(val_subjects)].copy(),
        "test": preprocessed_df.loc[preprocessed_df["subject_id"].isin(test_subjects)].copy(),
    }

    print("\nStage 3: Subject-wise Dataset Split")
    print("=" * 40)
    for split_name, frame in split_frames.items():
        print(f"{split_name.upper()} subjects:")
        print(sorted(frame["subject_id"].unique()))
        print(f"Recordings: {frame['recording_id'].nunique()}")
        print(f"Samples: {len(frame)}")
        print("Class distribution:")
        print(frame["binary_label"].value_counts().sort_index().to_string())
        print()

    print("Subject overlap check:")
    print(f"- Train/Val overlap: {len(set(train_subjects).intersection(set(val_subjects)))}")
    print(f"- Train/Test overlap: {len(set(train_subjects).intersection(set(test_subjects)))}")
    print(f"- Val/Test overlap: {len(set(val_subjects).intersection(set(test_subjects)))}")
    print(f"- Subjects assigned: {len(assigned_subjects)} / {len(subjects)}")
    print(f"- Split sizes (subjects): train={n_train}, val={n_val}, test={n_test}")

    return split_frames


def compute_raw_acceleration_magnitude(raw_features: np.ndarray) -> np.ndarray:
    """Per-timestep acceleration magnitude from raw (pre-normalization) acc channels."""

    acc = raw_features[:, ACC_CHANNEL_INDICES].astype(np.float64)
    return np.sqrt(np.sum(acc**2, axis=1))


def detect_impact_index_peak(magnitudes: np.ndarray) -> int:
    """Return the timestep index of peak raw acceleration magnitude."""

    if magnitudes.size == 0:
        raise ValueError("Cannot detect impact in an empty recording.")
    return int(np.argmax(magnitudes))


def detect_impact_index_threshold(
    magnitudes: np.ndarray,
    threshold_multiplier: float = 3.0,
    baseline_fraction: float = 0.2,
) -> tuple[int, bool]:
    """Return the first threshold crossing and whether peak fallback was used."""

    if magnitudes.size == 0:
        raise ValueError("Cannot detect impact in an empty recording.")

    baseline_len = max(3, int(len(magnitudes) * baseline_fraction))
    baseline = magnitudes[:baseline_len]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))
    if baseline_std < 1e-9:
        baseline_std = 1e-9

    threshold = baseline_mean + threshold_multiplier * baseline_std
    crossings = np.where(magnitudes > threshold)[0]
    if crossings.size == 0:
        return detect_impact_index_peak(magnitudes), True
    return int(crossings[0]), False


def detect_impact_index(
    raw_features: np.ndarray,
    method: str = "peak",
    threshold_multiplier: float = 3.0,
    baseline_fraction: float = 0.2,
) -> tuple[int, bool]:
    """Detect impact index on raw accelerometer data for one recording."""

    magnitudes = compute_raw_acceleration_magnitude(raw_features)
    if method == "peak":
        return detect_impact_index_peak(magnitudes), False
    if method == "threshold":
        return detect_impact_index_threshold(
            magnitudes,
            threshold_multiplier=threshold_multiplier,
            baseline_fraction=baseline_fraction,
        )
    raise ValueError("method must be either 'peak' or 'threshold'.")


def window_contains_impact(start: int, window_size: int, impact_index: int) -> bool:
    """True when the impact timestep lies inside [start, start + window_size)."""

    return start <= impact_index < start + window_size


def normalize_splits(
    split_frames: dict[str, pd.DataFrame],
    scaler_type: str = "standard",
    output_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Normalize the IMU channels using only the training split."""

    if scaler_type not in {"standard", "minmax"}:
        raise ValueError("scaler_type must be either 'standard' or 'minmax'.")

    for split_name, frame in split_frames.items():
        missing_columns = set(IMU_CHANNELS).difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Split {split_name} is missing IMU columns: {sorted(missing_columns)}")

    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "scaler.pkl"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    train_features = split_frames["train"][IMU_CHANNELS].to_numpy(dtype=np.float64)
    scaler.fit(train_features)

    normalized_splits: dict[str, dict[str, Any]] = {}
    for split_name, frame in split_frames.items():
        features = frame[IMU_CHANNELS].to_numpy(dtype=np.float64)
        if len(frame) == 0:
            transformed = np.empty((0, features.shape[1]), dtype=np.float32)
            raw_features = np.empty((0, features.shape[1]), dtype=np.float64)
        else:
            raw_features = features.copy()
            transformed = scaler.transform(features).astype(np.float32)
        normalized_splits[split_name] = {
            "dataframe": frame.reset_index(drop=True),
            "features": transformed,
            "raw_features": raw_features,
            "labels": frame["binary_label"].to_numpy(dtype=np.int64),
            "subject_ids": frame["subject_id"].to_numpy(),
            "recording_ids": frame["recording_id"].to_numpy(),
        }

    with output_path.open("wb") as handle:
        pickle.dump(scaler, handle)

    print("\nStage 4: Normalization")
    print("=" * 40)
    print(f"Scaler type: {scaler_type}")
    print(f"Scaler saved to: {output_path}")
    for split_name, payload in normalized_splits.items():
        features = payload["features"]
        if len(features) == 0:
            print(f"{split_name} has no samples; skipping summary statistics.")
            continue
        print(f"{split_name} mean: {features.mean(axis=0)}")
        print(f"{split_name} std: {features.std(axis=0)}")
        print(f"{split_name} min: {features.min(axis=0)}")
        print(f"{split_name} max: {features.max(axis=0)}")

    return normalized_splits


def generate_sliding_windows(
    normalized_splits: dict[str, dict[str, Any]],
    window_size: int = 64,
    stride: int = 16,
    impact_method: str = "peak",
    threshold_multiplier: float = 3.0,
    baseline_fraction: float = 0.2,
) -> dict[str, dict[str, Any]]:
    """Generate sliding windows with impact-centered labels for fall recordings."""

    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if stride > window_size:
        raise ValueError("stride must be less than or equal to window_size.")
    if impact_method not in {"peak", "threshold"}:
        raise ValueError("impact_method must be either 'peak' or 'threshold'.")

    windowed_splits: dict[str, dict[str, Any]] = {}
    impact_summary: dict[str, Any] = {
        "impact_method": impact_method,
        "fall_recordings_processed": 0,
        "windows_relabeled_1_to_0": 0,
        "impact_fallback_to_peak": 0,
        "fall_recordings_without_fall_window": [],
        "impact_indices_peak_vs_threshold": [],
    }

    print("\nStage 5: Sliding Window Generation (impact-centered fall labels)")
    print("=" * 40)
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")
    print(f"Impact detection method: {impact_method}")

    for split_name, payload in tqdm(normalized_splits.items(), desc="Generating windows"):
        frame = payload["dataframe"].copy()
        features = payload["features"]
        raw_features = payload.get("raw_features")
        if raw_features is None:
            raise ValueError(
                f"Split {split_name} is missing raw_features required for impact detection."
            )

        windows: list[np.ndarray] = []
        labels: list[int] = []
        subject_ids: list[str] = []
        recording_ids: list[str] = []
        window_starts: list[int] = []

        for recording_id in sorted(frame["recording_id"].unique()):
            group = frame.loc[frame["recording_id"] == recording_id].copy()
            if len(group) < window_size:
                continue

            group_indices = group.index.to_numpy()
            feature_group = features[group_indices]
            raw_group = raw_features[group_indices]
            group = group.reset_index(drop=True)
            recording_label = int(group.iloc[0]["binary_label"])

            impact_index: int | None = None
            used_peak_fallback = False
            if recording_label == 1:
                impact_summary["fall_recordings_processed"] += 1
                magnitudes = compute_raw_acceleration_magnitude(raw_group)
                peak_index = detect_impact_index_peak(magnitudes)
                threshold_index, used_peak_fallback = detect_impact_index_threshold(
                    magnitudes,
                    threshold_multiplier=threshold_multiplier,
                    baseline_fraction=baseline_fraction,
                )
                impact_summary["impact_indices_peak_vs_threshold"].append(
                    {
                        "split": split_name,
                        "recording_id": recording_id,
                        "peak_index": peak_index,
                        "threshold_index": threshold_index,
                        "delta": abs(peak_index - threshold_index),
                    }
                )
                if used_peak_fallback:
                    impact_summary["impact_fallback_to_peak"] += 1

                if impact_method == "peak":
                    impact_index = peak_index
                else:
                    impact_index = threshold_index

            recording_window_labels: list[int] = []
            recording_window_starts: list[int] = []

            for start in range(0, len(group_indices) - window_size + 1, stride):
                window = feature_group[start : start + window_size]
                windows.append(window)

                if recording_label == 0:
                    window_label = 0
                else:
                    assert impact_index is not None
                    window_label = 1 if window_contains_impact(start, window_size, impact_index) else 0

                labels.append(window_label)
                recording_window_labels.append(window_label)
                recording_window_starts.append(start)
                subject_ids.append(group.iloc[0]["subject_id"])
                recording_ids.append(recording_id)
                window_starts.append(start)

            if recording_label == 1:
                old_fall_windows = len(recording_window_labels)
                new_fall_windows = int(sum(recording_window_labels))
                impact_summary["windows_relabeled_1_to_0"] += old_fall_windows - new_fall_windows

                if new_fall_windows == 0:
                    best_idx = int(
                        np.argmin(
                            [
                                0
                                if window_contains_impact(start, window_size, impact_index)
                                else min(
                                    abs(impact_index - start),
                                    abs(impact_index - (start + window_size - 1)),
                                )
                                for start in recording_window_starts
                            ]
                        )
                    )
                    relabel_index = len(labels) - len(recording_window_labels) + best_idx
                    labels[relabel_index] = 1
                    recording_window_labels[best_idx] = 1
                    impact_summary["fall_recordings_without_fall_window"].append(
                        {
                            "split": split_name,
                            "recording_id": recording_id,
                            "impact_index": impact_index,
                            "forced_window_start": recording_window_starts[best_idx],
                            "reason": "impact_not_contained_in_stride_grid",
                        }
                    )

                assert sum(recording_window_labels) >= 1, (
                    f"Fall recording {recording_id} produced zero impact-centered windows."
                )

        if not windows:
            windowed_splits[split_name] = {
                "windows": np.empty((0, window_size, features.shape[1]), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int64),
                "subject_ids": np.empty((0,), dtype=object),
                "recording_ids": np.empty((0,), dtype=object),
            }
            print(f"{split_name}: no windows generated; created empty array payload.")
            continue

        windowed_splits[split_name] = {
            "windows": np.stack(windows, axis=0).astype(np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "subject_ids": np.asarray(subject_ids),
            "recording_ids": np.asarray(recording_ids),
        }

        print(f"{split_name}:")
        print(f"- Number of windows: {len(windowed_splits[split_name]['windows'])}")
        print(f"- Window shape: {windowed_splits[split_name]['windows'].shape}")
        print("- Label distribution:")
        print(pd.Series(windowed_splits[split_name]["labels"]).value_counts().sort_index().to_string())
        print(f"- Average windows per recording: {len(windowed_splits[split_name]['windows']) / frame['recording_id'].nunique():.2f}")

    print("\nImpact-centered labeling summary:")
    print(f"- Fall recordings processed: {impact_summary['fall_recordings_processed']}")
    print(f"- Windows relabeled 1->0: {impact_summary['windows_relabeled_1_to_0']}")
    print(f"- Threshold detections that fell back to peak: {impact_summary['impact_fallback_to_peak']}")
    flagged = impact_summary["fall_recordings_without_fall_window"]
    print(f"- Fall recordings force-corrected (impact outside stride grid): {len(flagged)}")
    if flagged:
        for entry in flagged[:10]:
            print(f"  * {entry['recording_id']} ({entry['reason']})")
        if len(flagged) > 10:
            print(f"  ... and {len(flagged) - 10} more")

    peak_threshold_deltas = [
        item["delta"] for item in impact_summary["impact_indices_peak_vs_threshold"]
    ]
    if peak_threshold_deltas:
        print(
            "- Peak vs threshold index delta: "
            f"mean={np.mean(peak_threshold_deltas):.2f}, "
            f"median={np.median(peak_threshold_deltas):.1f}, "
            f"max={max(peak_threshold_deltas)}"
        )

    windowed_splits["_impact_summary"] = impact_summary
    return windowed_splits


def save_processed_datasets(
    windowed_splits: dict[str, dict[str, Any]],
    output_dir: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> dict[str, Path]:
    """Save processed arrays and the fitted scaler to disk."""

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if scaler_path is None:
        scaler_path = output_dir / "scaler.pkl"
    else:
        scaler_path = Path(scaler_path).expanduser().resolve()

    saved_files: dict[str, Path] = {}
    print("\nStage 6: Save Processed Dataset")
    print("=" * 40)
    print(f"Output directory: {output_dir}")

    for split_name in ["train", "val", "test"]:
        if split_name not in windowed_splits:
            continue
        payload = windowed_splits[split_name]
        feature_path = output_dir / f"{split_name}.npy"
        label_path = output_dir / f"{split_name}_labels.npy"
        subject_path = output_dir / f"{split_name}_subject_ids.npy"
        recording_path = output_dir / f"{split_name}_recording_ids.npy"
        np.save(feature_path, payload["windows"])
        np.save(label_path, payload["labels"])
        np.save(subject_path, payload["subject_ids"])
        np.save(recording_path, payload["recording_ids"])
        saved_files[split_name] = feature_path
        saved_files[f"{split_name}_labels"] = label_path
        saved_files[f"{split_name}_subject_ids"] = subject_path
        saved_files[f"{split_name}_recording_ids"] = recording_path
        print(f"Saved {split_name}: {feature_path} -> {payload['windows'].shape}")
        print(f"Saved {split_name} labels: {label_path} -> {payload['labels'].shape}")

    if scaler_path.exists():
        scaler_path = scaler_path
    else:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
    if scaler_path.exists():
        print(f"Scaler already exists at: {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    saved_files["scaler"] = scaler_path
    print(f"Saved scaler: {scaler_path}")

    return saved_files


def validate_preprocessed_dataframe(preprocessed_df: pd.DataFrame) -> dict[str, Any]:
    """Validate the final DataFrame and return a summary of quality checks."""

    if preprocessed_df.empty:
        raise ValueError("The preprocessed DataFrame is empty.")

    required_columns = {
        "subject_id",
        "activity_code",
        "source_file",
        "source_path",
        "recording_id",
        "activity_name",
        "binary_label",
        "timestamp",
    }
    missing_columns = required_columns.difference(preprocessed_df.columns)
    if missing_columns:
        raise ValueError(f"The DataFrame is missing required columns: {sorted(missing_columns)}")

    validation_summary = {
        "number_of_subjects": preprocessed_df["subject_id"].nunique(),
        "number_of_activity_files": preprocessed_df["recording_id"].nunique(),
        "number_of_adl_files": preprocessed_df.loc[preprocessed_df["binary_label"] == 0, "recording_id"].nunique(),
        "number_of_fall_files": preprocessed_df.loc[preprocessed_df["binary_label"] == 1, "recording_id"].nunique(),
        "number_of_samples": int(len(preprocessed_df)),
        "rows_per_activity": preprocessed_df.groupby("activity_code").size().sort_values(ascending=False).to_dict(),
        "rows_per_subject": preprocessed_df.groupby("subject_id").size().sort_values(ascending=False).to_dict(),
        "missing_values": int(preprocessed_df.isna().sum().sum()),
        "duplicate_rows": int(preprocessed_df.duplicated().sum()),
        "duplicate_recordings": int(preprocessed_df.duplicated(subset=["recording_id", "subject_id", "source_file"]).sum()),
        "unknown_activity_codes": int(preprocessed_df["activity_name"].isna().sum()),
        "rows_with_nan": int(preprocessed_df.isna().any(axis=1).sum()),
        "rows_with_inf": int(np.isinf(preprocessed_df.select_dtypes(include=[np.number]).to_numpy()).sum()),
    }

    print("\nDataset Validation")
    print("=" * 40)
    for key, value in validation_summary.items():
        print(f"{key}: {value}")

    return validation_summary


def inspect_preprocessed_dataframe(preprocessed_df: pd.DataFrame) -> None:
    """Print a concise inspection summary for the final DataFrame."""

    print("\nFinal DataFrame Inspection")
    print("=" * 40)
    print(f"Shape: {preprocessed_df.shape}")
    print("Columns:")
    print(preprocessed_df.columns.tolist())
    print("\nData types:")
    print(preprocessed_df.dtypes)
    print("\nMemory usage:")
    print(preprocessed_df.memory_usage(deep=True))
    print("\nHead:")
    print(preprocessed_df.head().to_string(index=False))
    print("\nTail:")
    print(preprocessed_df.tail().to_string(index=False))
    print("\nDescribe:")
    print(preprocessed_df.describe(include="all").to_string())
    print("\nClass distribution:")
    print(preprocessed_df["binary_label"].value_counts().sort_index().to_string())
    print("\nActivity distribution:")
    print(preprocessed_df["activity_code"].value_counts().sort_values(ascending=False).to_string())
    print("\nSubject distribution:")
    print(preprocessed_df["subject_id"].value_counts().sort_values(ascending=False).to_string())


def run_full_preprocessing_pipeline(
    dataset_root: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full preprocessing pipeline up to timestamp generation and validation."""

    master_df = build_master_dataframe(dataset_root=dataset_root, show_progress=show_progress)
    mapped_df = map_activities(master_df)
    timestamped_df = generate_timestamps(mapped_df)
    validation_summary = validate_preprocessed_dataframe(timestamped_df)
    inspect_preprocessed_dataframe(timestamped_df)
    return timestamped_df, validation_summary
