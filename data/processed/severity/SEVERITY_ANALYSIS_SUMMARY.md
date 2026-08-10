# Severity Analysis Using Clustering — Summary Report

**Generated:** August 10, 2026  
**Dataset:** SisFall (preprocessed IMU windows)  
**Pipeline entry point:** `scripts/run_severity_labeling.py`

---

## Executive Summary

This project derives **unsupervised fall severity labels** (Mild / Moderate / Severe) from preprocessed IMU window data using **K-Means clustering** (k=3) on hand-crafted features, followed by **Isolation Forest** anomaly refinement. Severity is assigned **only to fall windows** (`binary_label == 1`); ADL (non-fall) windows are marked as **Non-applicable (-1)**.

The pipeline correctly consumes preprocessed data from `data/processed/` and enforces **train-only fitting** for all models (feature scaler, KMeans, Isolation Forest) and for cluster-to-severity mapping.

---

## Pipeline Overview

```
Raw SisFall CSV
    │
    ▼
run_full_preprocessing.py  ──►  sisfall_preprocessing.py
    │                              • 200 Hz → 20 Hz downsample
    │                              • Subject split 70/15/15
    │                              • StandardScaler (train only, 6 IMU channels)
    │                              • Sliding windows (64 samples, stride 16)
    │                              • Impact-centered fall labeling
    ▼
data/processed/
    train.npy, val.npy, test.npy          (n_windows, 64, 6)
    *_labels.npy                          (0=ADL, 1=Fall)
    *_subject_ids.npy, *_recording_ids.npy
    │
    ▼
run_severity_labeling.py  ──►  severity_labeling.py
    │                              • Filter fall windows only
    │                              • Extract 58 IMU window features
    │                              • StandardScaler on 58-D features (train fall only)
    │                              • KMeans k=3 (train fall only)
    │                              • Isolation Forest (train fall only)
    │                              • Map clusters → severity by peak intensity
    ▼
data/processed/severity/
    *_severity_labels.npy, models, reports, plots
```

---

## Preprocessed Data Usage — Verification

### Input Contract

The severity pipeline loads four arrays per split via `load_processed_split()`:

| File | Shape (train example) | Role |
|------|----------------------|------|
| `{split}.npy` | `(55423, 64, 6)` | Normalized IMU windows |
| `{split}_labels.npy` | `(55423,)` | Binary label: 0=ADL, 1=Fall |
| `{split}_subject_ids.npy` | `(55423,)` | Subject identifier |
| `{split}_recording_ids.npy` | `(55423,)` | Recording identifier |

**Verified:** All four arrays have matching lengths for train, val, and test splits. Window shape is `(64, 6)` as expected.

### Correct Usage Patterns

| Check | Status | Details |
|-------|--------|---------|
| Loads from `data/processed/` | ✅ Pass | `run_severity_pipeline(processed_dir=...)` |
| Fall-only clustering | ✅ Pass | `_fall_window_indices(labels)` filters `labels == 1` |
| ADL windows excluded from clustering | ✅ Pass | Only `windows[fall_indices]` passed to feature extraction |
| ADL severity = -1 | ✅ Pass | All 50,939 train ADL windows have severity -1 |
| Fall severity ∈ {0, 1, 2} | ✅ Pass | All fall windows assigned valid severity labels |
| Train-only model fitting | ✅ Pass | Scaler, KMeans, Isolation Forest fit on train fall features only |
| Train-only severity mapping | ✅ Pass | `map_clusters_to_severity()` uses train cluster statistics |
| No data leakage to val/test | ✅ Pass | Val/test only transformed/predicted, never used for fit |
| Binary labels preserved | ✅ Pass | Severity assignment does not modify `{split}_labels.npy` |
| `fall_indices` alignment | ✅ Pass | Saved indices match `np.where(labels == 1)` exactly |

### Dataset Counts

| Split | Total Windows | Fall Windows | Mild | Moderate | Severe |
|-------|--------------|--------------|------|----------|--------|
| Train | 55,423 | 4,484 | 1,659 | 1,267 | 1,558 |
| Val | 15,073 | 1,500 | 561 | 421 | 518 |
| Test | 13,627 | 1,200 | 457 | 304 | 439 |

---

## Clustering Methodology

### Algorithm

- **Clustering:** K-Means (`sklearn.cluster.KMeans`)
- **Clusters:** 3 (Mild, Moderate, Severe)
- **Hyperparameters:** `random_state=42`, `n_init=20`
- **Feature scaling:** Separate `StandardScaler` on 58-D feature vectors (train fall windows only)

> **Note:** DBSCAN and other density-based methods are **not** used. Clustering is exclusively K-Means.

### Feature Extraction (58 Features)

Features are extracted from each **fall window** of shape `(64, 6)`:

**Per-channel statistics (6 channels × 7 stats = 42 features):**
- mean, std, min, max, range, RMS, peak_abs
- Channels: `acc1_x/y/z`, `gyro_x/y/z`

**Magnitude statistics (2 magnitudes × 8 stats = 16 features):**
- Accelerometer magnitude: `‖acc‖`
- Gyroscope magnitude: `‖gyro‖`
- Stats: mean, std, min, max, range, RMS, peak, energy

### Cluster → Severity Mapping

Clusters are ordered by **impact intensity** using train-set cluster centroids:

1. Sort clusters by `acc_mag_peak_mean` (ascending)
2. Tie-break with `gyro_mag_peak_mean` (ascending)
3. Assign: lowest intensity → Mild (0), middle → Moderate (1), highest → Severe (2)

**Current mapping (from `severity_mapping.json`):**

| Cluster ID | Severity | Severity Name | Train Count | Acc Peak Mean | Gyro Peak Mean | Intensity Score |
|------------|----------|---------------|-------------|---------------|----------------|-----------------|
| 0 | 0 | Mild | 1,659 | 6.43 | 10.14 | 16.57 |
| 2 | 1 | Moderate | 1,267 | 7.28 | 12.31 | 19.60 |
| 1 | 2 | Severe | 1,558 | 7.86 | 12.98 | 20.83 |

Intensity increases monotonically across severity levels, confirming the mapping is physically interpretable.

---

## Anomaly Detection (Refinement Layer)

After clustering, an **Isolation Forest** flags atypical fall windows:

| Parameter | Value |
|-----------|-------|
| `contamination` | 0.02 (2%) |
| `n_estimators` | 256 |
| `random_state` | 42 |

**Behavior:**
- Trained on scaled train fall features (same space as KMeans)
- `anomaly_flag == 1` → window flagged as suspicious
- **Refined severity:** anomalous fall windows set to `-1` (uncertain) in `*_refined_severity_labels.npy`
- Raw cluster-based severity in `*_severity_labels.npy` is unchanged

### Anomaly Rates

| Split | Fall Windows | Anomalous | Rate |
|-------|-------------|-----------|------|
| Train | 4,484 | 90 | 2.01% |
| Val | 1,500 | 23 | 1.53% |
| Test | 1,200 | 24 | 2.00% |

### Anomaly Rate by Severity (Train)

| Severity | Fall Windows | Anomalous | Rate |
|----------|-------------|-----------|------|
| Mild | 1,659 | 42 | 2.53% |
| Moderate | 1,267 | 10 | 0.79% |
| Severe | 1,558 | 38 | 2.44% |

Moderate falls show the lowest anomaly rate, suggesting they form the most cohesive cluster in feature space.

---

## Output Artifacts

All outputs are written to `data/processed/severity/`:

### Label Arrays

| File | Length | Description |
|------|--------|-------------|
| `{split}_severity_labels.npy` | All windows | Cluster-based severity; ADL = -1 |
| `{split}_refined_severity_labels.npy` | All windows | Anomalous falls also set to -1 |
| `{split}_anomaly_scores.npy` | **Fall windows only** | Isolation Forest decision function |
| `{split}_anomaly_flags.npy` | **Fall windows only** | 1 = anomalous, 0 = normal |

### Models

| File | Description |
|------|-------------|
| `kmeans_model.pkl` | Fitted KMeans (3 clusters) |
| `feature_scaler.pkl` | StandardScaler on 58-D features |
| `isolation_forest.pkl` | Fitted Isolation Forest |

### Reports & Metadata

| File | Description |
|------|-------------|
| `severity_mapping.json` | Cluster → severity mapping |
| `severity_summary.json` | Split counts and configuration |
| `cluster_statistics.csv` | Per-cluster intensity statistics |
| `anomaly_summary.csv` | Overall anomaly rates per split |
| `anomaly_by_severity.csv` | Anomaly breakdown by severity |
| `anomaly_by_subject.csv` | Anomaly breakdown by subject |
| `feature_names.json` | 58 feature names |
| `clustering_features/{split}_fall_features.npy` | Raw 58-D features per fall window |
| `clustering_features/{split}_fall_indices.npy` | Index into full window array |

### Plots (via `run_severity_anomaly_analysis.py`)

- `plots/anomaly_score_distribution.png`
- `plots/anomaly_rate_by_severity.png`
- `plots/anomaly_rate_by_subject_{split}.png`

---

## Important Design Notes

### Two Independent Scalers

| Scaler | Location | Applied To | Purpose |
|--------|----------|-----------|---------|
| Preprocessing scaler | `data/processed/scaler.pkl` | 6 raw IMU channels | Normalize sensor readings before windowing |
| Severity feature scaler | `data/processed/severity/feature_scaler.pkl` | 58 extracted features | Normalize features for clustering |

This is intentional. The severity pipeline operates on **already-normalized windows** and applies a second scaler to the derived statistical features.

### Features Are in Normalized Space

Because preprocessing applies `StandardScaler` to raw IMU values before windowing, severity features (peaks, RMS, etc.) are computed from **normalized sensor data**, not physical g/deg/s units. Cluster intensity values (~6–13 for peaks) reflect normalized feature space, not raw sensor magnitudes.

### Array Length Convention

- **Full-length arrays** (`*_severity_labels.npy`): same length as `{split}.npy` (all windows)
- **Fall-only arrays** (`*_anomaly_scores.npy`, `*_anomaly_flags.npy`): length = number of fall windows

To align fall-only arrays with full arrays, use `clustering_features/{split}_fall_indices.npy`:

```python
fall_indices = np.load("data/processed/severity/clustering_features/train_fall_indices.npy")
severity = np.load("data/processed/severity/train_severity_labels.npy")
anomaly_flags = np.load("data/processed/severity/train_anomaly_flags.npy")

# anomaly_flags[i] corresponds to severity[fall_indices[i]]
```

---

## How to Reproduce

```bash
# Step 1: Generate preprocessed data (required first)
python scripts/run_full_preprocessing.py

# Step 2: Run severity clustering pipeline
python scripts/run_severity_labeling.py

# Step 3 (optional): Generate plots and CSV reports
python scripts/run_severity_anomaly_analysis.py
```

### Programmatic API

```python
from src.data.severity_labeling import run_severity_pipeline

result = run_severity_pipeline(
    processed_dir="data/processed",
    output_dir="data/processed/severity",
    n_clusters=3,
    random_state=42,
    n_init=20,
)
```

---

## Key Source Files

| File | Role |
|------|------|
| `src/data/severity_labeling.py` | Core pipeline: feature extraction, clustering, mapping, artifact saving |
| `src/data/severity_anomaly.py` | Isolation Forest helpers |
| `src/data/sisfall_preprocessing.py` | Upstream preprocessing producing window arrays |
| `scripts/run_severity_labeling.py` | CLI entry point |
| `scripts/run_severity_anomaly_analysis.py` | Post-hoc analysis and plotting |
| `tests/test_severity_labeling.py` | Unit tests for clustering and label assignment |
| `tests/test_severity_anomaly.py` | Unit tests for anomaly detection |

---

## Limitations & Considerations

1. **Unsupervised, not ground-truth severity.** Labels are data-driven cluster assignments ordered by IMU intensity — there is no clinical severity ground truth in SisFall.

2. **No supervised classifier.** The workflow does not train a model to predict severity on new data; it assigns labels via clustering + heuristic ordering.

3. **K=3 is fixed.** The number of severity levels is hard-coded. No elbow/silhouette analysis is automated.

4. **No downstream consumer yet.** Severity labels are generated and saved but no fall-detection model currently uses them for multi-class training.

5. **Normalized feature space.** Intensity thresholds are relative to the training distribution, not absolute physical impact forces.

---

## Conclusion

The severity analysis pipeline **correctly uses preprocessed data** for fall severity classification:

- Reads the expected window arrays, labels, and metadata from `data/processed/`
- Restricts clustering to fall windows while preserving ADL windows as non-applicable
- Fits all models exclusively on training fall data with no leakage to validation or test splits
- Produces aligned, consistent label arrays verified against source preprocessing outputs
- Maps clusters to interpretable severity levels with monotonically increasing IMU intensity

The pipeline is ready for downstream use (e.g., severity-stratified model evaluation or multi-class fall severity training) using the artifacts in `data/processed/severity/`.
