from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sisfall_preprocessing import (
    apply_low_pass_filter,
    downsample_recordings,
    generate_sliding_windows,
    normalize_splits,
    select_imu_channels,
    split_by_subjects,
)


def test_pipeline_components(tmp_path: Path) -> None:
    """Exercise the new preprocessing stages on a small synthetic DataFrame."""

    num_rows = 40
    time = np.arange(num_rows, dtype=np.float64)
    base = pd.DataFrame(
        {
            "acc1_x": np.sin(time / 5.0),
            "acc1_y": np.cos(time / 6.0),
            "acc1_z": np.sin(time / 7.0) + 0.3,
            "gyro_x": np.sin(time / 8.0) * 0.5,
            "gyro_y": np.cos(time / 9.0) * 0.5,
            "gyro_z": np.sin(time / 10.0) * 0.25,
            "acc2_x": np.zeros(num_rows, dtype=np.float64),
            "acc2_y": np.zeros(num_rows, dtype=np.float64),
            "acc2_z": np.zeros(num_rows, dtype=np.float64),
            "subject_id": ["SA01"] * num_rows,
            "activity_code": ["D01"] * num_rows,
            "activity_name": ["Walking slowly"] * num_rows,
            "recording_id": ["SA01:demo.txt"] * num_rows,
            "binary_label": [0] * num_rows,
            "timestamp": np.arange(num_rows, dtype=np.float64) * 0.005,
            "source_file": ["demo.txt"] * num_rows,
            "source_path": ["demo.txt"] * num_rows,
        }
    )

    selected = select_imu_channels(base)
    assert selected.shape[1] == 14

    filtered = apply_low_pass_filter(selected, sampling_frequency_hz=200.0, cutoff_frequency_hz=5.0, filter_order=2, plot_path=tmp_path / "filter_plot.png")
    assert filtered.shape[0] == selected.shape[0]

    downsampled, summary = downsample_recordings(filtered, target_sampling_frequency_hz=20.0, original_sampling_frequency_hz=200.0)
    assert summary["downsampled_sample_count"] < summary["original_sample_count"]

    split_frames = split_by_subjects(pd.concat([downsampled, downsampled], ignore_index=True), train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    assert set(split_frames.keys()) == {"train", "val", "test"}

    normalized = normalize_splits(split_frames, scaler_type="standard", output_path=tmp_path / "scaler.pkl")
    assert normalized["train"]["features"].shape[0] == len(split_frames["train"])

    windowed = generate_sliding_windows(normalized, window_size=4, stride=2)
    assert windowed["train"]["windows"].shape[0] > 0


def test_sliding_windows_use_recording_row_positions() -> None:
    """Ensure each recording window reads its own feature rows after grouping."""

    frame = pd.DataFrame(
        {
            "recording_id": ["rec_a"] * 4 + ["rec_b"] * 4,
            "binary_label": [0] * 4 + [1] * 4,
            "subject_id": ["SA01"] * 4 + ["SA02"] * 4,
        }
    )
    features = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
            [100, 100, 100, 100, 100, 100],
            [101, 101, 101, 101, 101, 101],
            [102, 102, 102, 102, 102, 102],
            [103, 103, 103, 103, 103, 103],
        ],
        dtype=np.float32,
    )
    raw_features = features.astype(np.float64).copy()
    raw_features[6, 0] = 500.0

    windowed = generate_sliding_windows(
        {
            "train": {
                "dataframe": frame,
                "features": features,
                "raw_features": raw_features,
            }
        },
        window_size=4,
        stride=4,
        impact_method="peak",
    )

    windows = windowed["train"]["windows"]
    assert windows.shape == (2, 4, 6)
    assert np.array_equal(windows[0, :, 0], np.asarray([0, 1, 2, 3], dtype=np.float32))
    assert np.array_equal(windows[1, :, 0], np.asarray([100, 101, 102, 103], dtype=np.float32))
    assert windowed["train"]["labels"].tolist() == [0, 1]
    assert windowed["train"]["recording_ids"].tolist() == ["rec_a", "rec_b"]


def test_impact_centered_fall_labeling() -> None:
    """Fall recordings label only windows that contain the detected impact index."""

    num_rows = 20
    frame = pd.DataFrame(
        {
            "recording_id": ["fall_rec"] * num_rows,
            "binary_label": [1] * num_rows,
            "subject_id": ["SA01"] * num_rows,
        }
    )
    features = np.zeros((num_rows, 6), dtype=np.float32)
    raw_features = np.zeros((num_rows, 6), dtype=np.float64)
    raw_features[10, 0] = 800.0

    windowed = generate_sliding_windows(
        {
            "train": {
                "dataframe": frame,
                "features": features,
                "raw_features": raw_features,
            }
        },
        window_size=4,
        stride=2,
        impact_method="peak",
    )

    labels = windowed["train"]["labels"]
    assert (labels == 1).sum() >= 1
    assert (labels == 0).sum() >= 1
    assert len(labels) == 9


def test_split_by_subjects_includes_every_subject() -> None:
    """Ensure remainder subjects are assigned when ratios do not divide evenly."""

    subjects = [f"SA{i:02d}" for i in range(1, 24)] + [f"SE{i:02d}" for i in range(1, 16)]
    rows = []
    for subject_id in subjects:
        rows.append(
            {
                "acc1_x": 0.0,
                "acc1_y": 0.0,
                "acc1_z": 0.0,
                "gyro_x": 0.0,
                "gyro_y": 0.0,
                "gyro_z": 0.0,
                "subject_id": subject_id,
                "activity_code": "D01",
                "activity_name": "Walking slowly",
                "recording_id": f"{subject_id}:demo.txt",
                "binary_label": 0,
                "timestamp": 0.0,
                "source_file": "demo.txt",
                "source_path": "demo.txt",
            }
        )

    frame = pd.DataFrame(rows)
    splits = split_by_subjects(frame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)

    assigned = set()
    for split_name, split_frame in splits.items():
        assigned.update(split_frame["subject_id"].unique())

    assert assigned == set(subjects)
    assert "SA09" in assigned
    assert "SA14" in assigned
    assert sum(len(split_frame["subject_id"].unique()) for split_frame in splits.values()) == len(subjects)
