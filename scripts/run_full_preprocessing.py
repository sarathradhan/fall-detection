from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sisfall_preprocessing import (
    apply_low_pass_filter,
    build_master_dataframe,
    downsample_recordings,
    generate_sliding_windows,
    generate_timestamps,
    map_activities,
    normalize_splits,
    save_processed_datasets,
    select_imu_channels,
    split_by_subjects,
)

if __name__ == "__main__":
    import gc

    root = Path("SisFall_dataset")
    master_df = build_master_dataframe(dataset_root=root, show_progress=False)
    print("master shape", master_df.shape)

    activity_df = map_activities(master_df)
    del master_df
    gc.collect()

    timestamped_df = generate_timestamps(activity_df)
    del activity_df
    gc.collect()
    print("timestamped shape", timestamped_df.shape)

    selected_df = select_imu_channels(timestamped_df)
    del timestamped_df
    gc.collect()
    filtered_df = apply_low_pass_filter(
        selected_df,
        sampling_frequency_hz=200.0,
        cutoff_frequency_hz=5.0,
        filter_order=2,
        plot_path=Path("data/processed/filter_plot.png"),
    )
    del selected_df
    gc.collect()

    downsampled_df, summary = downsample_recordings(
        filtered_df,
        target_sampling_frequency_hz=20.0,
        original_sampling_frequency_hz=200.0,
    )
    del filtered_df
    gc.collect()
    print("downsampled shape", downsampled_df.shape)
    print("summary", summary)

    splits = split_by_subjects(
        downsampled_df,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    normalized = normalize_splits(
        splits,
        scaler_type="standard",
        output_path=Path("data/processed/scaler.pkl"),
    )

    windowed = generate_sliding_windows(normalized, window_size=64, stride=16, impact_method="peak")
    saved = save_processed_datasets(
        windowed,
        output_dir=Path("data/processed"),
        scaler_path=Path("data/processed/scaler.pkl"),
    )

    print("saved files", sorted(saved))
    print("train windows", windowed["train"]["windows"].shape)
    print("val windows", windowed["val"]["windows"].shape)
    print("test windows", windowed["test"]["windows"].shape)
