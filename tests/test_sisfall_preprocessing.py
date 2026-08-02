from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sisfall_preprocessing import (
    build_master_dataframe,
    generate_timestamps,
    map_activities,
)


def test_build_master_dataframe_mapping_and_timestamps(tmp_path: Path) -> None:
    """Verify that the preprocessing pipeline builds, labels, and timestamps data correctly."""

    dataset_dir = tmp_path / "SisFall_dataset"
    subject_dir = dataset_dir / "SA01"
    subject_dir.mkdir(parents=True)
    file_path = subject_dir / "D01_SA01_R01.txt"
    file_path.write_text(
        "1,2,3,4,5,6,7,8,9;\n10,11,12,13,14,15,16,17,18;\n",
        encoding="utf-8",
    )

    subject_dir_two = dataset_dir / "SA02"
    subject_dir_two.mkdir(parents=True)
    file_path_two = subject_dir_two / "D01.txt"
    file_path_two.write_text(
        "1,2,3,4,5,6,7,8,9;\n",
        encoding="utf-8",
    )

    master_df = build_master_dataframe(dataset_root=dataset_dir, show_progress=False)

    assert "subject_id" in master_df.columns
    assert "activity_code" in master_df.columns
    assert master_df.iloc[0]["subject_id"] == "SA01"
    assert master_df.iloc[0]["activity_code"] == "D01"
    assert len(master_df) == 3

    mapped_df = map_activities(master_df)
    assert mapped_df.iloc[0]["activity_name"] == "Walking slowly"
    assert mapped_df.iloc[0]["binary_label"] == 0

    timestamped_df = generate_timestamps(mapped_df, sampling_frequency_hz=200.0)
    assert timestamped_df.iloc[0]["timestamp"] == 0.0
    assert timestamped_df.iloc[1]["timestamp"] == 1 / 200.0
    assert timestamped_df.iloc[2]["timestamp"] == 0.0
