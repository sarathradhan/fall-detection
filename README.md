# Fall Detection

AI-powered wearable fall-detection research using the [SisFall](https://www.sisfall.com/) dataset. The project covers raw data ingestion, preprocessing, exploratory analysis, fall-severity labeling, anomaly detection, and supervised CNN/CNN-LSTM fall-detection baselines.

The current processed dataset contains 84,123 windows of shape `(64, 6)`: 76,939 ADL/background windows and 7,184 impact-centered fall windows. Each window represents 3.2 seconds sampled at 20 Hz with six standardized IMU channels: `acc1_x`, `acc1_y`, `acc1_z`, `gyro_x`, `gyro_y`, and `gyro_z`.

## Current Status

- Raw SisFall parsing, metadata construction, filtering, downsampling, subject-wise splitting, normalization, and sliding-window generation are implemented.
- Fall labels are impact-centered: only windows containing the peak fall impact are labeled as fall.
- Severity labels are generated for fall windows using unsupervised feature extraction, KMeans clustering, and Isolation Forest anomaly refinement.
- Two supervised baselines are available: a compact 1D CNN and a compact CNN-LSTM.
- The current recommended operating point is CNN-LSTM at threshold `0.50`, with 100% recording-level and impact-verified fall recall in the stored evaluation outputs.

For the full generated report, metrics, artifact inventory, and CSV samples, see [PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md).

## Repository Layout

```text
.
├── data/
│   ├── raw/                         # Optional location for raw SisFall files
│   └── processed/                   # Generated arrays, reports, plots, severity outputs
├── reports/                         # Severity comparison reports and figures
├── results/
│   ├── cnn_baseline/                # CNN checkpoints, metrics, plots, recording evaluation
│   └── cnn_lstm_baseline/           # CNN-LSTM checkpoints, metrics, plots, diagnostics
├── scripts/                         # CLI entry points for data, analysis, and model runs
├── src/
│   └── data/                        # SisFall loader, preprocessing, severity, anomaly modules
├── tests/                           # Pytest coverage for preprocessing and severity logic
├── PROJECT_STATUS_REPORT.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python dependencies include NumPy, pandas, SciPy, scikit-learn, matplotlib, tqdm, pytest, and TensorFlow.

## Dataset

Download the SisFall dataset from the official source and extract it as:

```text
SisFall_dataset/
├── SA01/
├── SA02/
├── SE01/
└── ...
```

The default preprocessing script expects `SisFall_dataset/` in the repository root. Raw dataset files are not tracked in git.

## Pipeline

Run the full preprocessing pipeline:

```bash
python scripts/run_full_preprocessing.py
```

This generates the primary model inputs under `data/processed/`:

- `train.npy`, `val.npy`, `test.npy`
- `train_labels.npy`, `val_labels.npy`, `test_labels.npy`
- `train_subject_ids.npy`, `val_subject_ids.npy`, `test_subject_ids.npy`
- `train_recording_ids.npy`, `val_recording_ids.npy`, `test_recording_ids.npy`
- `scaler.pkl`

Run EDA and verification:

```bash
python scripts/run_phase2_eda.py
python scripts/verify_preprocessing_eda_pipeline.py
python scripts/inspect_processed_data.py
```

Generate severity labels and anomaly outputs:

```bash
python scripts/run_severity_labeling.py
python scripts/run_severity_anomaly_analysis.py
python scripts/validate_severity_clusters.py
python scripts/run_cluster_plots.py
```

Severity artifacts are written to `data/processed/severity/`, including refined severity labels, anomaly scores/flags, clustering features, KMeans/Isolation Forest models, CSV reports, and plots.

## Model Training

Train the compact CNN baseline:

```bash
python scripts/train_cnn_baseline.py
```

Train the compact CNN-LSTM baseline:

```bash
python scripts/train_cnn_lstm_baseline.py
```

Evaluate recording-level behavior for the CNN baseline:

```bash
python scripts/evaluate_recording_level.py
```

Model outputs are saved under `results/cnn_baseline/` and `results/cnn_lstm_baseline/`.

## Current Results Snapshot

CNN-LSTM numbers reflect the model weights currently on disk (`results/cnn_lstm_baseline/run_metadata.json`, training commit `6e8b4a96`, 2026-09-13).

| Model | Threshold | Fall precision | Fall recall | Fall F1 | Fall F2 | Recording recall | Impact-verified recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.50 | 0.9761 | 0.9867 | 0.9814 | 0.9845 | 0.9967 | 0.9967 |
| CNN | 0.20 | 0.9683 | 0.9925 | 0.9802 | 0.9876 | 1.0000 | 1.0000 |
| CNN-LSTM | 0.50 | 0.9770 | 0.9925 | 0.9847 | 0.9894 | 1.0000 | 1.0000 |
| CNN-LSTM | 0.25 | 0.9606 | 0.9950 | 0.9775 | 0.9879 | 1.0000 | 1.0000 |

Although the validation-selected CNN-LSTM threshold is `0.25`, the project report recommends `0.50` for the current operating point because it keeps 100% recording-level and impact-verified recall while reducing ADL false-trigger recordings.

## Tests

```bash
pytest
```

The latest recorded verification in the project report was 16 passing tests with one expected synthetic-clustering convergence warning.

## Notes

- Large generated binary artifacts such as `*.npy`, `*.pkl`, and trained model checkpoints may be excluded from git depending on local ignore rules.
- Do not mix severity artifacts from different runs without regenerating them together; the status report notes that some stored severity files may reflect older K2/K3 comparison runs.
- SisFall dataset usage is subject to the original dataset terms.

## License

Research and educational use.
