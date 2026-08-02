from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


@dataclass
class ActivityRecord:
    """Container for one activity file and its metadata."""

    subject_id: str
    filename: str
    file_path: Path
    dataframe: pd.DataFrame


def _resolve_dataset_root(dataset_root: str | Path | None = None) -> Path:
    """Return the SisFall dataset root path.

    The loader first checks the provided path. If none is given, it looks for a
    folder named ``SisFall_dataset`` in the project root or in the current
    workspace.
    """

    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve()
        if root.exists():
            return root
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    project_root = Path(__file__).resolve().parents[2]
    workspace_root = project_root.parent
    candidates = [
        workspace_root / "SisFall_dataset",
        project_root / "SisFall_dataset",
        project_root / "data" / "raw" / "SisFall_dataset",
        workspace_root / "fall detection project" / "SisFall_dataset",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the SisFall dataset. Expected one of: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def _parse_sensor_line(line: str) -> list[float]:
    """Convert one text line into a list of floats.

    SisFall files contain comma-separated values and each line ends with a
    semicolon. This helper removes the terminal semicolon and converts the
    values into Python floats.
    """

    cleaned = line.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("Encountered an empty line.")

    parts = [part.strip() for part in cleaned.split(",")]
    return [float(part) for part in parts]


def load_activity_file(file_path: Path, subject_id: str) -> ActivityRecord:
    """Load one activity text file into a pandas DataFrame."""

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    rows: list[list[float]] = []
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values = _parse_sensor_line(line)
            except ValueError as exc:
                raise ValueError(f"Invalid data in {file_path.name} line {line_number}: {exc}") from exc

            if len(values) != 9:
                raise ValueError(
                    f"Expected 9 sensor columns in {file_path.name}, but found {len(values)}."
                )

            rows.append(values)

    if not rows:
        raise ValueError(f"No usable rows found in {file_path.name}.")

    columns = [
        "adxl345_x",
        "adxl345_y",
        "adxl345_z",
        "itg3200_x",
        "itg3200_y",
        "itg3200_z",
        "mma8451q_x",
        "mma8451q_y",
        "mma8451q_z",
    ]

    dataframe = pd.DataFrame(rows, columns=columns)

    return ActivityRecord(
        subject_id=subject_id,
        filename=file_path.name,
        file_path=file_path,
        dataframe=dataframe,
    )


def discover_subject_directories(dataset_root: Path) -> list[Path]:
    """Return the subject directories that look like SisFall subjects."""

    subject_dirs = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir() and (path.name.startswith("SA") or path.name.startswith("SE"))
    ]
    return subject_dirs


def load_sisfall_dataset(
    dataset_root: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Load the complete SisFall dataset into memory.

    The returned structure is organized by subject ID and activity file name.
    It also includes a summary dictionary with loading statistics.
    """

    root = _resolve_dataset_root(dataset_root)
    subject_dirs = discover_subject_directories(root)

    loaded_subjects: dict[str, dict[str, ActivityRecord]] = {}
    missing_files: list[dict[str, str]] = []
    failed_files: list[dict[str, str]] = []
    rows_per_file: dict[str, int] = {}

    for subject_dir in tqdm(subject_dirs, disable=not show_progress, desc="Loading SisFall subjects"):
        activity_files = sorted(subject_dir.glob("*.txt"))
        if not activity_files:
            missing_files.append(
                {
                    "subject_id": subject_dir.name,
                    "reason": "No activity files were found in the subject directory.",
                }
            )
            continue

        subject_records: dict[str, ActivityRecord] = {}
        for activity_file in activity_files:
            try:
                record = load_activity_file(activity_file, subject_dir.name)
            except Exception as exc:  # pragma: no cover - defensive programming
                failed_files.append(
                    {
                        "subject_id": subject_dir.name,
                        "filename": activity_file.name,
                        "reason": str(exc),
                    }
                )
                continue

            subject_records[activity_file.name] = record
            rows_per_file[activity_file.name] = int(record.dataframe.shape[0])

        loaded_subjects[subject_dir.name] = subject_records

    summary = {
        "dataset_root": str(root),
        "number_of_subjects": len(subject_dirs),
        "number_of_activity_files": sum(len(records) for records in loaded_subjects.values())
        + len(failed_files)
        + len(missing_files),
        "successfully_loaded_files": sum(len(records) for records in loaded_subjects.values()),
        "failed_files": len(failed_files),
        "missing_files": len(missing_files),
        "missing_file_details": missing_files,
        "failed_file_details": failed_files,
        "rows_per_file": rows_per_file,
    }

    return {
        "dataset_root": root,
        "subjects": loaded_subjects,
        "summary": summary,
    }


def print_summary(dataset_payload: dict[str, Any]) -> None:
    """Print a concise summary of the loaded dataset."""

    summary = dataset_payload["summary"]
    print("\nSisFall dataset loading summary")
    print("=" * 40)
    print(f"Dataset root: {summary['dataset_root']}")
    print(f"Number of subjects: {summary['number_of_subjects']}")
    print(f"Number of activity files discovered: {summary['number_of_activity_files']}")
    print(f"Successfully loaded files: {summary['successfully_loaded_files']}")
    print(f"Failed files: {summary['failed_files']}")
    print(f"Missing files: {summary['missing_files']}")

    print("\nRows per file (first 20 shown):")
    rows_items = sorted(summary["rows_per_file"].items(), key=lambda item: item[0])
    for filename, row_count in rows_items[:20]:
        print(f"- {filename}: {row_count} rows")

    if len(rows_items) > 20:
        print(f"... and {len(rows_items) - 20} more files")

    if summary["failed_file_details"]:
        print("\nFailed files:")
        for entry in summary["failed_file_details"][:10]:
            print(f"- {entry['subject_id']}/{entry['filename']}: {entry['reason']}")

    if summary["missing_file_details"]:
        print("\nMissing subject directories:")
        for entry in summary["missing_file_details"]:
            print(f"- {entry['subject_id']}: {entry['reason']}")
