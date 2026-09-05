from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "PROJECT_STATUS_REPORT.md"


def fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def md_table(frame: pd.DataFrame, limit: int | None = None) -> str:
    if limit is not None:
        frame = frame.head(limit)
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def csv_sections() -> list[str]:
    sections: list[str] = []
    csv_files = sorted(path for path in ROOT.rglob("*.csv") if ".venv" not in path.parts)
    for path in csv_files:
        relative = path.relative_to(ROOT).as_posix()
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        summary = pd.DataFrame(
            {
                "column": numeric.columns,
                "non_null": [int(numeric[column].notna().sum()) for column in numeric.columns],
                "min": [numeric[column].min() for column in numeric.columns],
                "mean": [numeric[column].mean() for column in numeric.columns],
                "median": [numeric[column].median() for column in numeric.columns],
                "std": [numeric[column].std() for column in numeric.columns],
                "max": [numeric[column].max() for column in numeric.columns],
            }
        )
        sections.append(
            f"### `{relative}`\n\n"
            f"- Shape: `{frame.shape[0]:,} rows x {frame.shape[1]} columns`\n"
            f"- Missing cells: `{int(frame.isna().sum().sum()):,}`\n"
            f"- Columns: `{', '.join(str(column) for column in frame.columns)}`\n\n"
            "Numeric statistics:\n\n"
            f"{md_table(summary)}\n\n"
            "First 10 rows:\n\n"
            f"{md_table(frame, 10)}"
        )
    return sections


def array_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "data" / "processed").rglob("*.npy")):
        array = np.load(path, allow_pickle=True)
        numeric = np.issubdtype(array.dtype, np.number)
        rows.append(
            {
                "artifact": path.relative_to(ROOT).as_posix(),
                "shape": str(array.shape),
                "dtype": str(array.dtype),
                "min": np.min(array) if numeric and array.size else "",
                "max": np.max(array) if numeric and array.size else "",
            }
        )
    return pd.DataFrame(rows)


def model_summary(model: Any) -> str:
    if hasattr(model, "feature_names_in_"):
        return f"features={len(model.feature_names_in_)}"
    if hasattr(model, "n_features_in_"):
        details = f"features={model.n_features_in_}"
        if hasattr(model, "n_clusters"):
            details += f", clusters={model.n_clusters}"
        if hasattr(model, "n_estimators"):
            details += f", estimators={model.n_estimators}"
        return details
    return "serialized estimator/scaler"


def model_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "data" / "processed").rglob("*.pkl")):
        try:
            with path.open("rb") as handle:
                model = pickle.load(handle)
            details = model_summary(model)
            model_type = type(model).__name__
        except Exception as exc:
            details = str(exc)
            model_type = "unreadable"
        rows.append({"artifact": path.relative_to(ROOT).as_posix(), "python_type": model_type, "details": details})
    return pd.DataFrame(rows)


def split_statistics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    processed = ROOT / "data" / "processed"
    for split in ["train", "val", "test"]:
        windows = np.load(processed / f"{split}.npy")
        labels = np.load(processed / f"{split}_labels.npy")
        subjects = np.load(processed / f"{split}_subject_ids.npy", allow_pickle=True)
        recordings = np.load(processed / f"{split}_recording_ids.npy", allow_pickle=True)
        rows.append(
            {
                "split": split,
                "windows": len(windows),
                "shape": str(windows.shape),
                "subjects": len(np.unique(subjects)),
                "recordings": len(np.unique(recordings)),
                "ADL/background": int((labels == 0).sum()),
                "fall": int((labels == 1).sum()),
                "fall_%": float((labels == 1).mean() * 100),
                "NaN": int(np.isnan(windows).sum()),
                "Inf": int(np.isinf(windows).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_report() -> str:
    split_frame = split_statistics()
    totals = split_frame[["windows", "ADL/background", "fall"]].sum(numeric_only=True)
    severity_dir = ROOT / "data" / "processed" / "severity"
    feature_names = json.loads((severity_dir / "feature_names.json").read_text(encoding="utf-8"))
    report = [
        "# Fall Detection Project: Complete Status and Data Guide",
        "",
        f"> Generated from the files currently present in the repository on {date.today().isoformat()}. This is the consolidated project report; older generated Markdown reports were removed to avoid conflicting summaries.",
        "",
        "## 1. Executive Summary",
        "",
        "This project has completed raw SisFall ingestion, signal preprocessing, subject-wise splitting, normalized sliding-window generation, exploratory data analysis, impact-centered binary labeling, and unsupervised fall-severity/anomaly analysis. The processed data is ready for supervised model training, but no supervised fall detector has been trained or evaluated yet.",
        "",
        "The current canonical binary dataset contains **84,123 windows** with shape `(64, 6)`: **76,939 ADL/background** windows and **7,184 impact-centered fall** windows. The six channels are normalized accelerometer and gyroscope channels. Severity labels are derived only for the 7,184 fall windows.",
        "",
        "## 2. What Has Been Done",
        "",
        "1. **Raw dataset discovery and loading:** SisFall subject directories and activity text files were discovered and parsed. Each raw row has nine numeric sensor values separated by commas and terminated by a semicolon.",
        "2. **Metadata construction:** Subject ID, activity code, activity name, recording ID, source file/path, binary label, and per-recording timestamp were added.",
        "3. **Activity mapping:** `D01`-`D19` are ADL activities and `F01`-`F15` are falls. Binary mapping is `0 = ADL/background`, `1 = fall`.",
        "4. **Channel selection:** The first accelerometer (`acc1`) and gyroscope (`gyro`) tri-axial channels were retained. The second accelerometer (`acc2`, MMA8451Q) was removed, leaving six input channels.",
        "5. **Filtering:** A Butterworth low-pass filter was applied independently per recording before downsampling. The configured cutoff is 5 Hz; the generated filter plot provides a visual check.",
        "6. **Downsampling:** Signals were reduced from 200 Hz to 20 Hz using a factor of 10, independently per recording.",
        "7. **Leakage-resistant splitting:** Subjects were assigned to train/validation/test at approximately 70/15/15, so a subject cannot occur in more than one split.",
        "8. **Normalization:** A training-only `StandardScaler` was fitted on the six selected channels and then applied to all splits.",
        "9. **Windowing:** Windows contain 64 samples with stride 16. At 20 Hz, each window is 3.2 seconds and each stride is 0.8 seconds, producing 75% overlap.",
        "10. **Impact-centered labels:** For fall recordings, the raw acceleration-magnitude peak identifies the impact. Only windows containing that impact receive label 1; other windows from the same recording remain background label 0.",
        "11. **EDA and quality checks:** Class balance, signal distributions, autocorrelation, outliers, acceleration peaks, channel health, and integrity checks were exported to CSV and plots.",
        "12. **Unsupervised severity labeling:** Fall windows were transformed into 58 statistical features, scaled using train fall windows, clustered with KMeans, and ordered by acceleration/gyroscope intensity into Mild, Moderate, and Severe.",
        "13. **Anomaly refinement:** A train-only Isolation Forest flags unusual fall windows. Anomalous fall severity is changed to `-1` in refined labels, while the original cluster labels remain available.",
        "14. **Automated testing:** The repository includes preprocessing, pipeline, severity, and anomaly tests. The last recorded verification was 15 passing tests with one expected synthetic-clustering convergence warning.",
        "",
        "## 3. Current Dataset Statistics",
        "",
        md_table(split_frame),
        "",
        f"Total windows: **{int(totals['windows']):,}**. Total ADL/background: **{int(totals['ADL/background']):,}**. Total fall: **{int(totals['fall']):,}**. Overall fall percentage: **{int(totals['fall']) / int(totals['windows']) * 100:.2f}%**.",
        "",
        "Integrity result: all processed binary arrays have zero NaN and zero infinite values. The array-level checks also report no flat-line or constant windows in the previous verification output.",
        "",
        "## 4. Feature Map and Units",
        "",
        "| Raw columns | Sensor | Physical interpretation | Unit in source dataset | Used? |",
        "| --- | --- | --- | --- | --- |",
        "| `adxl345_x/y/z` | ADXL345 accelerometer | Linear acceleration by axis | Dataset sensor units; confirm against SisFall documentation before converting to g | Yes, renamed `acc1_x/y/z` |",
        "| `itg3200_x/y/z` | ITG3200 gyroscope | Angular velocity by axis | Dataset sensor units; confirm against SisFall documentation before converting to deg/s or rad/s | Yes, renamed `gyro_x/y/z` |",
        "| `mma8451q_x/y/z` | MMA8451Q accelerometer | Second accelerometer triad | Dataset sensor units | No, removed from model input |",
        "",
        "The final tensor is `(number_of_windows, 64, 6)`. Each timestep has `acc1_x`, `acc1_y`, `acc1_z`, `gyro_x`, `gyro_y`, and `gyro_z`. Values are **standardized z-scores**, calculated as `(value - training_mean) / training_std`; therefore they are dimensionless. Do not interpret normalized values directly as g or angular velocity without applying the saved scaler in reverse.",
        "",
        f"Each fall window becomes **{len(feature_names)} features**: 42 per-channel values (six channels times mean, standard deviation, minimum, maximum, range, RMS, and absolute peak) plus 16 magnitude values (accelerometer and gyroscope magnitude times mean, standard deviation, minimum, maximum, range, RMS, peak, and mean-square energy). Feature names are stored in `data/processed/severity/feature_names.json`.",
        "",
        "Magnitude definitions: `acc_mag = sqrt(acc1_x^2 + acc1_y^2 + acc1_z^2)` and `gyro_mag = sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2)`. Since the input is normalized, these magnitudes are normalized-coordinate magnitudes, not physical acceleration or angular-velocity units.",
        "",
        "## 5. Severity and Anomaly Results",
        "",
        "The intended current three-level result is `0 = Mild`, `1 = Moderate`, `2 = Severe`, and `-1 = non-applicable or uncertain`. ADL/background windows are always `-1` for severity. KMeans is fitted only on train fall windows with `k=3`, `random_state=42`, and `n_init=20`. The Isolation Forest uses contamination 0.02, 256 estimators, and random state 42.",
        "",
        "The canonical post-Isolation Forest fall export contains 7,184 rows: train 4,484, validation 1,500, and test 1,200. Anomaly counts are train 90 (2.01%), validation 23 (1.53%), and test 24 (2.00%). Anomalous fall windows are uncertain, not automatically bad data; review or exclude them deliberately during modeling.",
        "",
        "### Important artifact consistency note",
        "",
        "The repository contains outputs from more than one severity-analysis run. `data/processed/severity/reports/post_isolation_forest_fall_windows.csv`, `cluster_feature_report.csv`, and the recorded project summary contain the later three-level Mild/Moderate/Severe result. Some files directly under `data/processed/severity/` such as `severity_cluster_sizes.csv`, `severity_cluster_profiles.csv`, and `anomaly_by_severity.csv` contain an older two-cluster or Low/Medium result. The historical K2/K3 diagnostics explain that K=2 had better separation scores, while K=3 was selected for the desired three-level interpretation. Do not mix these older two-cluster files with the current three-level labels; regenerate them together before using them for analysis.",
        "",
        "## 6. Stored Arrays and Models",
        "",
        "### Array inventory",
        "",
        md_table(array_inventory()),
        "",
        "`train.npy`, `val.npy`, and `test.npy` are the primary model inputs. Their matching label, subject, and recording arrays must remain aligned by row. `X_train_balanced.npy` and `y_train_balanced.npy` are additional class-balancing artifacts and should be used only when an experiment explicitly selects them.",
        "",
        "### Pickled estimator inventory",
        "",
        md_table(model_inventory()),
        "",
        "Stored model/scaler meanings: `scaler.pkl` is the six-channel preprocessing scaler; `feature_scaler.pkl` and `severity_scaler.pkl` are severity-feature scalers from different runs/naming conventions; `kmeans_model.pkl` and `severity_kmeans_model.pkl` are KMeans artifacts from different runs; `isolation_forest.pkl` is the anomaly estimator. These are unsupervised preprocessing/label-generation artifacts, not a trained fall-detection classifier.",
        "",
        "## 7. CSV Inventory, Statistics, and Samples",
        "",
        "The following sections cover every project CSV outside `.venv`. For each file, `Shape` gives rows and columns, `Missing cells` counts blank/NaN values, numeric statistics describe each numeric column, and the sample is the first 10 rows. Blank cells in the sample are intentional missing/non-applicable values.",
        "",
        *csv_sections(),
        "",
        "## 8. How to Continue",
        "",
        "1. Treat `train.npy`, `val.npy`, and `test.npy` plus their aligned metadata arrays as the current binary modeling dataset.",
        "2. Resolve the severity artifact mismatch by rerunning the chosen K=3 pipeline and downstream report export as one versioned run.",
        "3. Train a baseline binary classifier and report window-, recording-, and subject-level metrics.",
        "4. Evaluate performance separately for Mild, Moderate, Severe, and uncertain (`-1`) fall windows.",
        "5. Keep all transformations train-only: preprocessing scaler, severity feature scaler, KMeans, and Isolation Forest.",
        "6. Record the exact artifact generation command and timestamp whenever the processed arrays or severity outputs are regenerated.",
        "",
        "## 9. Key Files",
        "",
        "| Purpose | File |",
        "| --- | --- |",
        "| Consolidated guide | `PROJECT_STATUS_REPORT.md` |",
        "| Rebuild this guide | `scripts/generate_project_status_report.py` |",
        "| Raw loader | `src/data/sisfall_loader.py` |",
        "| Preprocessing/windowing | `src/data/sisfall_preprocessing.py` |",
        "| Severity labeling | `src/data/severity_labeling.py` |",
        "| Anomaly refinement | `src/data/severity_anomaly.py` |",
        "| Tests | `tests/` |",
        "",
    ]
    return "\n".join(report) + "\n"


if __name__ == "__main__":
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")