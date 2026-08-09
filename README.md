# Fall Detection

AI-powered wearable fall detection research using the [SisFall](https://www.sisfall.com/) dataset. This repository contains data loading, preprocessing, exploratory analysis, and validation tooling to produce model-ready windowed sensor arrays.

## Features

- SisFall dataset discovery and loading from raw text files
- Preprocessing pipeline: filtering, downsampling, subject-based splits, normalization, sliding windows
- EDA scripts and verification reports
- Pytest coverage for preprocessing logic

## Project structure

```
fall-detection/
├── data/
│   ├── raw/              # Place SisFall dataset here (not tracked in git)
│   └── processed/        # Generated .npy artifacts (not tracked; reports/plots included)
├── src/
│   └── data/             # Loader and preprocessing modules
│       └── severity_labeling.py  # KMeans severity + Isolation Forest anomaly refinement
├── scripts/              # CLI entry points for preprocessing and EDA
│   └── run_severity_labeling.py  # Execute the severity/anomaly pipeline
├── tests/                # Unit and integration tests
├── notebooks/            # Exploratory notebooks
├── requirements.txt
└── PROJECT_SUMMARY.md    # Detailed project status and results
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sarathradhan/fall-detection.git
cd fall-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the SisFall dataset

Obtain the SisFall dataset from the official source and extract it into the project root:

```
SisFall_dataset/
├── SA01/
├── SA02/
├── SE01/
└── ...
```

Alternatively, place it under `data/raw/SisFall_dataset/` and update script paths as needed.

## Usage

### Load and inspect raw data

```bash
python scripts/load_sisfall_dataset.py
```

### Run full preprocessing

Generates train/validation/test window arrays under `data/processed/`:

```bash
python scripts/run_full_preprocessing.py
```

### Run EDA and verification

```bash
python scripts/run_phase2_eda.py
python scripts/verify_preprocessing_eda_pipeline.py
python scripts/inspect_processed_data.py
```

### Generate fall severity labels

This project can derive data-driven fall severity labels for fall windows only, using train-only K-means clustering on interpretable IMU window features.

```bash
python scripts/run_severity_labeling.py
```

To generate summary plots and CSV reports from the severity/anomaly outputs:

```bash
python scripts/run_severity_anomaly_analysis.py
```

The scripts write severity artifacts to `data/processed/severity/`, including:

- `train_severity_labels.npy`, `val_severity_labels.npy`, `test_severity_labels.npy`
- `train_refined_severity_labels.npy`, `val_refined_severity_labels.npy`, `test_refined_severity_labels.npy`
- `train_anomaly_scores.npy`, `val_anomaly_scores.npy`, `test_anomaly_scores.npy`
- `train_anomaly_flags.npy`, `val_anomaly_flags.npy`, `test_anomaly_flags.npy`
- `feature_scaler.pkl`
- `kmeans_model.pkl`
- `isolation_forest.pkl`
- `isolation_forest_config.json`
- `cluster_statistics.csv`
- `anomaly_summary.csv`
- `anomaly_by_severity.csv`
- `anomaly_by_subject.csv`
- `severity_mapping.json`
- `severity_summary.json`
- `feature_names.json`
- `clustering_features/` with per-split fall-window features and fall-window indices

### Run tests

```bash
pytest
```

## Processed data

Large binary artifacts (`*.npy`, `*.pkl`) are excluded from git. After cloning, run the preprocessing script to regenerate them locally. Summary reports and plots under `data/processed/reports/` and `data/processed/plots/` are included for reference.

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for dataset statistics, class distribution, and pipeline details.

## License

Research and educational use. SisFall dataset usage is subject to its own terms from the original publishers.
