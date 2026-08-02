# SisFall Processed Dataset Summary

## Purpose

This document summarizes the processed SisFall dataset that will be used for binary fall detection model development. It describes the final dataset artifacts, preprocessing choices, split structure, class balance, signal characteristics, outlier profile, and readiness for CNN or CNN-LSTM training.

The summary is based only on the processed datasets and reports stored under `data/processed/`.

## Processed Dataset Location

The final processed arrays are stored in:

```text
data/processed/
```

Main dataset files:

| File | Description |
| --- | --- |
| `train.npy` | Training windows with shape `(windows, timesteps, channels)` |
| `train_labels.npy` | Binary labels for training windows |
| `train_subject_ids.npy` | Subject ID for each training window |
| `train_recording_ids.npy` | Source recording ID for each training window |
| `val.npy` | Validation windows |
| `val_labels.npy` | Binary labels for validation windows |
| `val_subject_ids.npy` | Subject ID for each validation window |
| `val_recording_ids.npy` | Source recording ID for each validation window |
| `test.npy` | Test windows |
| `test_labels.npy` | Binary labels for test windows |
| `test_subject_ids.npy` | Subject ID for each test window |
| `test_recording_ids.npy` | Source recording ID for each test window |
| `scaler.pkl` | Fitted scaler used during normalization |

EDA outputs are stored in:

```text
data/processed/plots/
data/processed/reports/
```

## Label Definition

The task is binary fall detection.

| Label | Class | Meaning |
| ---: | --- | --- |
| `0` | ADL | Activities of daily living, non-fall movement |
| `1` | Fall | Fall events |

## Selected Sensor Channels

Each window contains 6 IMU channels:

| Channel | Sensor Type |
| --- | --- |
| `acc1_x` | Accelerometer x-axis |
| `acc1_y` | Accelerometer y-axis |
| `acc1_z` | Accelerometer z-axis |
| `gyro_x` | Gyroscope x-axis |
| `gyro_y` | Gyroscope y-axis |
| `gyro_z` | Gyroscope z-axis |

The second accelerometer channels were excluded from the final model-ready tensors.

## Preprocessing Pipeline

The processed dataset reflects the following completed preprocessing steps:

1. Loaded and merged all SisFall recordings.
2. Added recording metadata and binary fall labels.
3. Generated timestamps from the original sampling rate.
4. Selected 6 IMU channels.
5. Applied Butterworth low-pass filtering.
6. Downsampled from 200 Hz to 20 Hz.
7. Split the data subject-wise into train, validation, and test sets.
8. Normalized the IMU channels using the saved scaler.
9. Generated sliding windows.
10. Saved final arrays, labels, subject IDs, recording IDs, and scaler.

The final window size is `64 x 6`, meaning each example contains 64 time steps and 6 channels. At 20 Hz, each window covers approximately 3.2 seconds of motion.

## Dataset Shape

| Split | Windows | Tensor Shape | Subjects | Recordings |
| --- | ---: | --- | ---: | ---: |
| Train | 55,423 | `(55423, 64, 6)` | 26 | 2,942 |
| Validation | 12,320 | `(12320, 64, 6)` | 5 | 675 |
| Test | 10,874 | `(10874, 64, 6)` | 5 | 580 |
| Total | 78,617 | mixed by split | 36 | 4,197 |

The split is subject-wise, so the same subject does not appear in more than one split. This is important because it makes validation and test performance more realistic: the model must generalize to unseen subjects rather than memorizing subject-specific motion patterns.

## Class Distribution

| Split | ADL Windows | ADL % | Fall Windows | Fall % | Total Windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 38,578 | 69.61% | 16,845 | 30.39% | 55,423 |
| Validation | 7,820 | 63.47% | 4,500 | 36.53% | 12,320 |
| Test | 7,499 | 68.96% | 3,375 | 31.04% | 10,874 |
| Total | 53,897 | 68.56% | 24,720 | 31.44% | 78,617 |

The dataset is moderately imbalanced toward ADL windows. Falls make up about 31.44% of all processed windows. Both classes are present in every split, so the dataset is usable for supervised binary classification.

Recommended training consideration: use class weighting, balanced sampling, or threshold tuning during model development so that the model does not become biased toward the ADL class.

## Signal Health Summary

Signal health statistics were computed from the processed normalized windows. The full table is available at:

```text
data/processed/reports/eda_signal_health_statistics.csv
```

Summary for all classes combined:

| Split | Channel | Mean | Median | Std | Variance | RMS | Min | Max | Skewness | Kurtosis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | `acc1_x` | 0.0832 | 0.0659 | 0.2603 | 0.0677 | 0.2733 | -0.9746 | 1.1175 | 0.2297 | 0.0365 |
| Train | `acc1_y` | -0.6213 | -0.6092 | 0.2352 | 0.0553 | 0.6643 | -1.5981 | 0.3383 | -0.5151 | -0.4707 |
| Train | `acc1_z` | 0.1134 | 0.0975 | 0.3012 | 0.0907 | 0.3219 | -0.8340 | 0.7844 | -0.1102 | -1.2583 |
| Train | `gyro_x` | -0.0946 | -0.0879 | 0.4193 | 0.1758 | 0.4299 | -1.3155 | 1.6227 | -0.0326 | -0.2660 |
| Train | `gyro_y` | -0.0804 | -0.1068 | 0.8576 | 0.7355 | 0.8614 | -3.9494 | 3.0866 | 0.0492 | -0.3594 |
| Train | `gyro_z` | 0.0101 | 0.0960 | 1.1262 | 1.2684 | 1.1263 | -4.6849 | 3.5602 | -0.2422 | -0.5772 |
| Validation | `acc1_x` | -0.0695 | -0.0682 | 0.2122 | 0.0450 | 0.2232 | -0.7881 | 0.6315 | 0.0382 | 0.1422 |
| Validation | `acc1_y` | -0.5012 | -0.4398 | 0.1698 | 0.0288 | 0.5291 | -1.0489 | -0.1822 | -1.0058 | 0.0190 |
| Validation | `acc1_z` | 0.0216 | 0.0606 | 0.2652 | 0.0703 | 0.2661 | -0.6962 | 0.7347 | -0.3741 | -0.7131 |
| Validation | `gyro_x` | 0.1518 | 0.1283 | 0.3557 | 0.1265 | 0.3867 | -1.2363 | 1.9218 | -0.0857 | 0.0430 |
| Validation | `gyro_y` | -0.0692 | -0.0368 | 0.5673 | 0.3219 | 0.5715 | -2.2738 | 1.8372 | -0.2034 | -0.2346 |
| Validation | `gyro_z` | -0.0147 | -0.0074 | 0.6359 | 0.4044 | 0.6361 | -1.9603 | 1.7536 | -0.0368 | -0.5285 |
| Test | `acc1_x` | 0.0095 | 0.0245 | 0.3341 | 0.1116 | 0.3342 | -0.9111 | 0.8790 | -0.0122 | -0.5328 |
| Test | `acc1_y` | -0.5406 | -0.5278 | 0.3583 | 0.1283 | 0.6486 | -1.3885 | 0.1174 | -0.1494 | -1.2222 |
| Test | `acc1_z` | 0.2804 | 0.3271 | 0.3210 | 0.1030 | 0.4262 | -0.5349 | 0.8674 | -0.4170 | -1.0195 |
| Test | `gyro_x` | 0.1764 | 0.0000 | 0.5928 | 0.3515 | 0.6185 | -1.1549 | 1.6282 | 0.4529 | -0.9145 |
| Test | `gyro_y` | -0.0972 | 0.0212 | 1.1595 | 1.3443 | 1.1635 | -4.0539 | 2.5632 | -0.0023 | -0.9906 |
| Test | `gyro_z` | -0.0165 | -0.0158 | 1.0547 | 1.1123 | 1.0548 | -2.1816 | 2.0989 | -0.0495 | -0.9399 |

Interpretation:

- The processed signals contain no reported missing or infinite values.
- The channels retain meaningful variance after normalization.
- Gyroscope channels, especially `gyro_y` and `gyro_z`, show larger spread than accelerometer channels.
- Extreme values are present, but their magnitude is plausible for fall and high-motion activity windows.
- The signal quality is suitable for model training.

## Acceleration Spike Analysis

Acceleration magnitude was computed as:

```text
sqrt(acc1_x^2 + acc1_y^2 + acc1_z^2)
```

Because the data is normalized, this is a normalized magnitude proxy rather than a physical acceleration in g.

| Split | Class | Windows | Peak Mean | Peak Median | Peak Std | Peak P95 | Peak Max | Mean Magnitude |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | ADL | 38,578 | 1.2761 | 1.2627 | 0.0907 | 1.4352 | 1.8807 | 0.7494 |
| Train | Fall | 16,845 | 1.2307 | 1.1994 | 0.0552 | 1.3368 | 1.3368 | 0.7384 |
| Validation | ADL | 7,820 | 1.0816 | 1.0570 | 0.0694 | 1.2102 | 1.2784 | 0.6050 |
| Validation | Fall | 4,500 | 1.0666 | 1.0227 | 0.0694 | 1.1795 | 1.1795 | 0.5944 |
| Test | ADL | 7,499 | 1.4539 | 1.4343 | 0.0822 | 1.5765 | 1.5765 | 0.7743 |
| Test | Fall | 3,375 | 1.4831 | 1.5220 | 0.0756 | 1.5765 | 1.5765 | 0.7747 |

Interpretation:

- Fall windows have higher median acceleration-magnitude peaks than ADL windows in the test split.
- Train and validation splits show slightly higher ADL median peaks than fall median peaks, which means spike magnitude alone should not be treated as a rule-based fall detector.
- The class distributions still differ, especially when considered with temporal patterns across the full 64-step window.
- CNN and CNN-LSTM models should learn from the full multichannel sequence rather than from only acceleration magnitude.

## Temporal Dependency

Autocorrelation was computed across each 64-step window and averaged by split and channel.

Key result:

- Mean lag-1 autocorrelation: `0.775`
- Mean lag-8 autocorrelation: `-0.174`

Interpretation:

- Consecutive samples are strongly related, which is expected for filtered and downsampled IMU signals.
- Autocorrelation decays with lag, indicating that short-range temporal structure is present inside each window.
- This supports using temporal convolution kernels, CNN-LSTM models, or other sequence-aware architectures.

## Outlier Summary

Outliers were detected for reporting only. No samples or windows were removed.

Methods:

- IQR rule: values below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR`
- Z-score rule: values with absolute Z-score greater than `3.0`

Highest observed outlier rates:

| Method | Split | Channel | Outlier Rate |
| --- | --- | --- | ---: |
| IQR | Validation | `acc1_x` | 1.95% |
| Z-score | Validation | `gyro_x` | 0.45% |

Interpretation:

- Outlier rates are low overall.
- The highest IQR outlier rate is below 2%.
- The highest Z-score outlier rate is below 0.5%.
- Outliers are likely meaningful high-motion or impact-related values, so they should remain in the dataset for fall detection modeling.

## Generated EDA Plot Files

The following EDA plots are available in `data/processed/plots/`:

| Plot | Purpose |
| --- | --- |
| `eda_class_distribution_bar.png` | Bar chart of ADL vs Fall windows by split |
| `eda_class_distribution_pie.png` | Class percentages by split |
| `eda_signal_histograms_train.png` | Training signal histograms |
| `eda_signal_histograms_val.png` | Validation signal histograms |
| `eda_signal_histograms_test.png` | Test signal histograms |
| `eda_signal_boxplots_train.png` | Training channel boxplots |
| `eda_signal_boxplots_val.png` | Validation channel boxplots |
| `eda_signal_boxplots_test.png` | Test channel boxplots |
| `eda_acceleration_spike_histograms.png` | ADL vs Fall acceleration peak distributions |
| `eda_acceleration_spike_boxplots.png` | ADL vs Fall acceleration peak boxplots |
| `eda_autocorrelation_train.png` | Training autocorrelation plots |
| `eda_autocorrelation_val.png` | Validation autocorrelation plots |
| `eda_autocorrelation_test.png` | Test autocorrelation plots |
| `eda_outliers_iqr.png` | IQR outlier percentages |
| `eda_outliers_z_score.png` | Z-score outlier percentages |

## Generated Report Files

The following EDA report files are available in `data/processed/reports/`:

| File | Description |
| --- | --- |
| `phase2_eda_report.md` | Full Phase 2 EDA report |
| `dataset_summary.md` | This dataset summary |
| `eda_class_distribution.csv` | Class counts and percentages |
| `eda_signal_health_statistics.csv` | Per-channel descriptive statistics |
| `eda_acceleration_spikes.csv` | Acceleration magnitude spike statistics |
| `eda_autocorrelation.csv` | Mean autocorrelation values by split/channel/lag |
| `eda_outliers.csv` | IQR and Z-score outlier summaries |

## Modeling Readiness

The processed dataset is ready for CNN or CNN-LSTM training.

Reasons:

- Final tensors have the expected shape: `(windows, 64, 6)`.
- Train, validation, and test sets are already separated.
- The split is subject-wise, reducing leakage across splits.
- Both classes are present in all splits.
- Channels are filtered, downsampled, and normalized.
- No missing or infinite values were reported.
- Outlier rates are low and outliers were retained because they may contain fall-relevant motion.
- Temporal dependencies are present, making sequence models appropriate.

Recommended next steps:

1. Build a baseline 1D CNN using input shape `(64, 6)`.
2. Compare against a CNN-LSTM model for temporal modeling.
3. Use class weights or balanced sampling because falls represent about 31.44% of the windows.
4. Track fall-specific metrics such as recall, precision, F1-score, ROC-AUC, and PR-AUC.
5. Evaluate final performance only once on the held-out test split.

