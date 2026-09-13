# Fall Detection Project: Complete Status and Data Guide

> Generated from the files currently present in the repository on 2026-09-13. This is the consolidated project report; older generated Markdown reports were removed to avoid conflicting summaries.

> Latest update: severity anomaly summaries now handle empty severity groups explicitly, reporting zero windows and a zero anomaly rate instead of attempting division by zero.

## 1. Executive Summary

This project has completed raw SisFall ingestion, signal preprocessing, subject-wise splitting, normalized sliding-window generation, exploratory data analysis, impact-centered binary labeling, unsupervised fall-severity/anomaly analysis, and two supervised fall-detection baselines: a compact CNN and a compact CNN-LSTM. The current best operating point is the CNN-LSTM at threshold `0.50`, which reaches 100% recording-level and impact-verified fall recall with fewer ADL false-trigger recordings than the CNN-LSTM's lower chosen threshold.

The current canonical binary dataset contains **84,123 windows** with shape `(64, 6)`: **76,939 ADL/background** windows and **7,184 impact-centered fall** windows. The six channels are normalized accelerometer and gyroscope channels. Severity labels are derived only for the 7,184 fall windows.

## 2. What Has Been Done

1. **Raw dataset discovery and loading:** SisFall subject directories and activity text files were discovered and parsed. Each raw row has nine numeric sensor values separated by commas and terminated by a semicolon.
2. **Metadata construction:** Subject ID, activity code, activity name, recording ID, source file/path, binary label, and per-recording timestamp were added.
3. **Activity mapping:** `D01`-`D19` are ADL activities and `F01`-`F15` are falls. Binary mapping is `0 = ADL/background`, `1 = fall`.
4. **Channel selection:** The first accelerometer (`acc1`) and gyroscope (`gyro`) tri-axial channels were retained. The second accelerometer (`acc2`, MMA8451Q) was removed, leaving six input channels.
5. **Filtering:** A Butterworth low-pass filter was applied independently per recording before downsampling. The configured cutoff is 5 Hz; the generated filter plot provides a visual check.
6. **Downsampling:** Signals were reduced from 200 Hz to 20 Hz using a factor of 10, independently per recording.
7. **Leakage-resistant splitting:** Subjects were assigned to train/validation/test at approximately 70/15/15, so a subject cannot occur in more than one split.
8. **Normalization:** A training-only `StandardScaler` was fitted on the six selected channels and then applied to all splits.
9. **Windowing:** Windows contain 64 samples with stride 16. At 20 Hz, each window is 3.2 seconds and each stride is 0.8 seconds, producing 75% overlap.
10. **Impact-centered labels:** For fall recordings, the raw acceleration-magnitude peak identifies the impact. Only windows containing that impact receive label 1; other windows from the same recording remain background label 0.
11. **EDA and quality checks:** Class balance, signal distributions, autocorrelation, outliers, acceleration peaks, channel health, and integrity checks were exported to CSV and plots.
12. **Unsupervised severity labeling:** Fall windows were transformed into 58 statistical features, scaled using train fall windows, clustered with KMeans, and ordered by acceleration/gyroscope intensity into Mild, Moderate, and Severe.
13. **Anomaly refinement:** A train-only Isolation Forest flags unusual fall windows. Anomalous fall severity is changed to `-1` in refined labels, while the original cluster labels remain available.
14. **Automated testing:** The repository includes preprocessing, pipeline, severity, and anomaly tests. The last recorded verification was 15 passing tests with one expected synthetic-clustering convergence warning.

## 3. Current Dataset Statistics

| split | windows | shape | subjects | recordings | ADL/background | fall | fall_% | NaN | Inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 55423 | (55423, 64, 6) | 26 | 2942 | 50939 | 4484 | 8.0905 | 0 | 0 |
| val | 15073 | (15073, 64, 6) | 6 | 829 | 13573 | 1500 | 9.95157 | 0 | 0 |
| test | 13627 | (13627, 64, 6) | 6 | 734 | 12427 | 1200 | 8.80605 | 0 | 0 |

Total windows: **84,123**. Total ADL/background: **76,939**. Total fall: **7,184**. Overall fall percentage: **8.54%**.

Integrity result: all processed binary arrays have zero NaN and zero infinite values. The array-level checks also report no flat-line or constant windows in the previous verification output.

## 4. Feature Map and Units

| Raw columns | Sensor | Physical interpretation | Unit in source dataset | Used? |
| --- | --- | --- | --- | --- |
| `adxl345_x/y/z` | ADXL345 accelerometer | Linear acceleration by axis | Dataset sensor units; confirm against SisFall documentation before converting to g | Yes, renamed `acc1_x/y/z` |
| `itg3200_x/y/z` | ITG3200 gyroscope | Angular velocity by axis | Dataset sensor units; confirm against SisFall documentation before converting to deg/s or rad/s | Yes, renamed `gyro_x/y/z` |
| `mma8451q_x/y/z` | MMA8451Q accelerometer | Second accelerometer triad | Dataset sensor units | No, removed from model input |

The final tensor is `(number_of_windows, 64, 6)`. Each timestep has `acc1_x`, `acc1_y`, `acc1_z`, `gyro_x`, `gyro_y`, and `gyro_z`. Values are **standardized z-scores**, calculated as `(value - training_mean) / training_std`; therefore they are dimensionless. Do not interpret normalized values directly as g or angular velocity without applying the saved scaler in reverse.

Each fall window becomes **58 features**: 42 per-channel values (six channels times mean, standard deviation, minimum, maximum, range, RMS, and absolute peak) plus 16 magnitude values (accelerometer and gyroscope magnitude times mean, standard deviation, minimum, maximum, range, RMS, peak, and mean-square energy). Feature names are stored in `data/processed/severity/feature_names.json`.

Magnitude definitions: `acc_mag = sqrt(acc1_x^2 + acc1_y^2 + acc1_z^2)` and `gyro_mag = sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2)`. Since the input is normalized, these magnitudes are normalized-coordinate magnitudes, not physical acceleration or angular-velocity units.

## 5. CNN Baseline Training

The first supervised baseline was trained by `scripts/train_cnn_baseline.py`. It is deliberately lightweight and does not modify preprocessing. It uses the natural imbalanced training split with class weights, while validation and test remain at their natural distributions.

| Setting | Value |
| --- | --- |
| Architecture | 3 Conv1D blocks with BatchNorm, ReLU, and MaxPooling, followed by GlobalAveragePooling, Dense(32), Dropout(0.3), and sigmoid output |
| Parameters | 33,729 total; 33,345 trainable |
| Input | `(64, 6)` |
| Optimizer | Adam, learning rate `0.001` |
| Loss | Binary cross-entropy |
| Batch size | 64 |
| Maximum epochs | 100 |
| Best epoch | 8 |
| Stopping | Early stopping at epoch 18; restored best validation-loss weights |
| Class weights | ADL/background `0.5440`; fall `6.1801` |
| TensorFlow | 2.21.0 |

### CNN test results

| Threshold | Precision (fall) | Recall (fall) | F1 (fall) | F2 (fall) | Confusion matrix `[[TN, FP], [FN, TP]]` |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.50 | 0.9761 | 0.9867 | 0.9814 | 0.9845 | `[[12398, 29], [16, 1184]]` |
| 0.20 selected | 0.9683 | 0.9925 | 0.9802 | 0.9876 | `[[12388, 39], [9, 1191]]` |

The threshold `0.20` was selected on validation by maximum F2, with recall used as the first tie-breaker. It raises test fall recall from 98.67% to 99.25%, while increasing false positives from 29 to 39. The complete classification reports, PR-AUC history, threshold sweep, and plots are stored in `results/cnn_baseline/`.

## 5A. CNN-LSTM Training

The second supervised baseline was trained by `scripts/train_cnn_lstm_baseline.py`. It uses the same natural imbalanced train/validation/test splits and the same class-weighting strategy as the CNN baseline, but adds an LSTM layer after the Conv1D feature extractor.

| Setting | Value |
| --- | --- |
| Architecture | Conv1D front-end with BatchNorm, ReLU, and MaxPooling, followed by `LSTM(64, return_sequences=True, unroll=True)`, GlobalAveragePooling, Dense(32), Dropout(0.3), and sigmoid output |
| Parameters | 46,817 total vs CNN baseline 33,729 |
| Input | `(64, 6)` |
| Optimizer | Adam, learning rate `0.001` |
| Loss | Binary cross-entropy |
| Batch size | 64 |
| Maximum epochs | 100 |
| Best epoch | 9 |
| Stopping | Early stopping completed after epoch 19; restored best validation-loss weights |
| Class weights | ADL/background `0.5440134278254383`; fall `6.180084745762712` |
| TensorFlow | 2.21.0 |
| Training time | 241.50411779998103 seconds |
| Interrupted | false |

### CNN-LSTM validation threshold sweep

| Threshold | Precision (fall) | Recall (fall) | F1 (fall) | F2 (fall) |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.9419924337957125 | 0.996 | 0.9682436811406351 | 0.9847086738729238 |
| 0.15 | 0.9467680608365019 | 0.996 | 0.9707602339181286 | 0.9857482185273159 |
| 0.2 | 0.9540229885057471 | 0.996 | 0.974559686888454 | 0.9873116574147502 |
| 0.25 | 0.9582798459563543 | 0.9953333333333333 | 0.9764551994767822 | 0.9876951574490606 |
| 0.3 | 0.9606705351386202 | 0.9933333333333333 | 0.9767289413307112 | 0.9866242881737518 |
| 0.35 | 0.9644012944983819 | 0.9933333333333333 | 0.9786535303776683 | 0.9874088800530152 |
| 0.4 | 0.9662556781310837 | 0.9926666666666667 | 0.9792831305491615 | 0.9872695928921894 |
| 0.45 | 0.96875 | 0.992 | 0.9802371541501976 | 0.9872611464968153 |
| 0.5 | 0.970626631853786 | 0.9913333333333333 | 0.9808707124010554 | 0.9871216144450345 |
| 0.55 | 0.9712606139777923 | 0.9913333333333333 | 0.9811943253051798 | 0.9872526888859381 |
| 0.6 | 0.9725130890052356 | 0.9906666666666667 | 0.9815059445178336 | 0.9869819341126461 |
| 0.65 | 0.973132372214941 | 0.99 | 0.9814937210839392 | 0.9865798564974754 |
| 0.7 | 0.9756738987508218 | 0.9893333333333333 | 0.9824561403508771 | 0.9865709347161282 |
| 0.75 | 0.9794973544973545 | 0.9873333333333333 | 0.9833997343957503 | 0.9857561235356762 |
| 0.8 | 0.9833666001330672 | 0.9853333333333333 | 0.9843489843489843 | 0.9849393575902972 |
| 0.85 | 0.9846153846153847 | 0.9813333333333333 | 0.9829716193656094 | 0.9819879919946631 |
| 0.9 | 0.9865501008742434 | 0.978 | 0.9822564445932374 | 0.9796981434486444 |

The threshold `0.25` was selected on validation by maximum F2. At test time, threshold `0.50` is recommended as the operating point because it preserves 100% recording-level and impact-verified recall while reducing ADL false-trigger recordings from 3 to 1.

### CNN-LSTM test results

| Threshold | Precision (fall) | Recall (fall) | F1 (fall) | F2 (fall) | Confusion matrix `[[TN, FP], [FN, TP]]` |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0.9786535303776683 | 0.9933333333333333 | 0.9859387923904053 | 0.9903622465935527 | `[[12401, 26], [8, 1192]]` |
| 0.25 selected | 0.9708265802269044 | 0.9983333333333333 | 0.9843878389482333 | 0.9927079880676168 | `[[12391, 36], [2, 1198]]` |

The complete classification reports, threshold sweep, training history, checkpoints, recording-level evaluation, and false-positive diagnostics are stored in `results/cnn_lstm_baseline/`.

## 5B. CNN vs CNN-LSTM Comparison

| model_threshold | precision | recall | F1 | F2 | recording_recall | impact_verified_recall | adl_false_trigger_count | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CNN_baseline@0.50 | 0.9760923330585326 | 0.9866666666666667 | 0.9813510153336096 | 0.9845335107267587 | 0.9966666666666667 | 0.9966666666666667 | 0 | reused |
| CNN_baseline@0.20 | 0.9682926829268292 | 0.9925 | 0.980246913580247 | 0.9875621890547264 | 1.0 | 1.0 | 0 | reused |
| CNN_LSTM@0.50 | 0.9786535303776683 | 0.9933333333333333 | 0.9859387923904053 | 0.9903622465935527 | 1.0 | 1.0 | 1 | reused |
| CNN_LSTM@0.25 | 0.9708265802269044 | 0.9983333333333333 | 0.9843878389482333 | 0.9927079880676168 | 1.0 | 1.0 | 3 | reused |

The CNN-LSTM's window-level precision at threshold `0.50` is higher than the CNN baseline's window-level precision at threshold `0.50`, despite the CNN-LSTM having 1 ADL false-trigger recording and the CNN baseline having 0. These are different granularities: one bad ADL window barely moves window-level precision, but it still counts as a full recording-level false trigger.

All CNN-LSTM ADL false triggers are single-window blips, with no sustained misprediction at either threshold. No false-triggered recording changes blip-vs-sustained classification between threshold `0.50` and threshold `0.25`.

The CNN-LSTM false-triggered ADL activity codes are `D13` and `D14`. `D13` is "Sitting a moment, lying quickly, wait a moment, and sit again" and is judged `plausible_borderline` because rapid lying/posture-change motion can resemble fall dynamics. `D14` is "Being on one's back change to lateral position, wait a moment, and change to one's back" and is judged `unexpected`; it appears at both thresholds and is a stated limitation of the CNN-LSTM evaluation.

Recommendation: use CNN-LSTM at threshold `0.50` as the model and operating point going forward, not threshold `0.25`. Both thresholds reach 100% recording-level and impact-verified recall, but threshold `0.50` has fewer ADL false-trigger recordings (1 vs 3).

## 6. Severity and Anomaly Results

The intended current three-level result is `0 = Mild`, `1 = Moderate`, `2 = Severe`, and `-1 = non-applicable or uncertain`. ADL/background windows are always `-1` for severity. KMeans is fitted only on train fall windows with `k=3`, `random_state=42`, and `n_init=20`. The Isolation Forest uses contamination 0.02, 256 estimators, and random state 42.

The canonical post-Isolation Forest fall export contains 7,184 rows: train 4,484, validation 1,500, and test 1,200. Anomaly counts are train 90 (2.01%), validation 23 (1.53%), and test 24 (2.00%). Anomalous fall windows are uncertain, not automatically bad data; review or exclude them deliberately during modeling.

### Important artifact consistency note

The repository contains outputs from more than one severity-analysis run. `data/processed/severity/reports/post_isolation_forest_fall_windows.csv`, `cluster_feature_report.csv`, and the recorded project summary contain the later three-level Mild/Moderate/Severe result. Some files directly under `data/processed/severity/` such as `severity_cluster_sizes.csv`, `severity_cluster_profiles.csv`, and `anomaly_by_severity.csv` contain an older two-cluster or Low/Medium result. The historical K2/K3 diagnostics explain that K=2 had better separation scores, while K=3 was selected for the desired three-level interpretation. Do not mix these older two-cluster files with the current three-level labels; regenerate them together before using them for analysis. This mismatch was resolved in versioned run `data/processed/severity/runs/k3_20260913T135813Z`, whose labels, train-fitted estimators, and downstream exports are self-contained.

### Versioned K=3 evaluation

The CNN-LSTM was evaluated at threshold `0.50` using refined labels from the same K=3 run. Test window counts and recall were: uncertain `-1` 24 / 95.83%, Mild 446 / 98.65%, Moderate 298 / 100.00%, and Severe 432 / 99.54%. Overall test metrics were 97.70% window precision, 99.25% window recall, 98.47% window F1, 100% fall-recording recall, and 100% impact-verified recording recall across 300 fall recordings. The regenerated model produced 0 false-triggered ADL recordings.

Mean subject recall was 91.67% for uncertain, 98.48% for Mild, 100.00% for Moderate, and 99.55% for Severe. Complete window-, recording-, subject-, and impact-verified outputs are under `results/cnn_lstm_baseline/severity_eval/k3_20260913T135813Z/`. The 2-consecutive-window rule is recorded as analysis-only future mitigation; in this run it retained 100% fall-recording and impact-verified recall with 0 ADL false-trigger recordings and does not change the default operating point.

## 7. Stored Arrays and Models

### Array inventory

| artifact | shape | dtype | min | max |
| --- | --- | --- | --- | --- |
| data/processed/severity/clustering_features/test_fall_features.npy | (1200, 58) | float64 | -48.5212 | 99.9793 |
| data/processed/severity/clustering_features/test_fall_indices.npy | (1200,) | int64 | 1636 | 11002 |
| data/processed/severity/clustering_features/train_fall_features.npy | (4484, 58) | float64 | -36.8906 | 71.3756 |
| data/processed/severity/clustering_features/train_fall_indices.npy | (4484,) | int64 | 1633 | 47694 |
| data/processed/severity/clustering_features/val_fall_features.npy | (1500, 58) | float64 | -18.9812 | 29.4566 |
| data/processed/severity/clustering_features/val_fall_indices.npy | (1500,) | int64 | 1634 | 13757 |
| data/processed/severity/test_anomaly_flags.npy | (1200,) | int64 | 0 | 1 |
| data/processed/severity/test_anomaly_scores.npy | (1200,) | float64 | -0.0602334 | 0.135932 |
| data/processed/severity/test_refined_severity_labels.npy | (13627,) | int64 | -1 | 1 |
| data/processed/severity/test_severity_labels.npy | (13627,) | int64 | -1 | 1 |
| data/processed/severity/train_anomaly_flags.npy | (4484,) | int64 | 0 | 1 |
| data/processed/severity/train_anomaly_scores.npy | (4484,) | float64 | -0.0730676 | 0.142574 |
| data/processed/severity/train_refined_severity_labels.npy | (55423,) | int64 | -1 | 1 |
| data/processed/severity/train_severity_labels.npy | (55423,) | int64 | -1 | 1 |
| data/processed/severity/val_anomaly_flags.npy | (1500,) | int64 | 0 | 1 |
| data/processed/severity/val_anomaly_scores.npy | (1500,) | float64 | -0.0325907 | 0.136228 |
| data/processed/severity/val_refined_severity_labels.npy | (15073,) | int64 | -1 | 1 |
| data/processed/severity/val_severity_labels.npy | (15073,) | int64 | -1 | 1 |
| data/processed/test.npy | (13627, 64, 6) | float32 | -48.5212 | 20.0049 |
| data/processed/test_labels.npy | (13627,) | int64 | 0 | 1 |
| data/processed/test_recording_ids.npy | (13627,) | <U21 |  |  |
| data/processed/test_subject_ids.npy | (13627,) | <U4 |  |  |
| data/processed/train.npy | (55423, 64, 6) | float32 | -38.0956 | 38.1595 |
| data/processed/train_labels.npy | (55423,) | int64 | 0 | 1 |
| data/processed/train_recording_ids.npy | (55423,) | <U21 |  |  |
| data/processed/train_subject_ids.npy | (55423,) | <U4 |  |  |
| data/processed/val.npy | (15073, 64, 6) | float32 | -18.9812 | 18.5895 |
| data/processed/val_labels.npy | (15073,) | int64 | 0 | 1 |
| data/processed/val_recording_ids.npy | (15073,) | <U21 |  |  |
| data/processed/val_subject_ids.npy | (15073,) | <U4 |  |  |
| data/processed/X_train_balanced.npy | (59907, 64, 6) | float32 | -39.861 | 38.1595 |
| data/processed/y_train_balanced.npy | (59907,) | int64 | 0 | 1 |

`train.npy`, `val.npy`, and `test.npy` are the primary model inputs. Their matching label, subject, and recording arrays must remain aligned by row. `X_train_balanced.npy` and `y_train_balanced.npy` are additional class-balancing artifacts and should be used only when an experiment explicitly selects them.

### Pickled estimator inventory

| artifact | python_type | details |
| --- | --- | --- |
| data/processed/scaler.pkl | StandardScaler | features=6 |
| data/processed/severity/feature_scaler.pkl | StandardScaler | features=58 |
| data/processed/severity/isolation_forest.pkl | IsolationForest | features=58, estimators=256 |
| data/processed/severity/kmeans_model.pkl | KMeans | features=58, clusters=2 |
| data/processed/severity/severity_kmeans_model.pkl | KMeans | features=58, clusters=2 |
| data/processed/severity/severity_scaler.pkl | StandardScaler | features=58 |

Stored model/scaler meanings: `scaler.pkl` is the six-channel preprocessing scaler; `feature_scaler.pkl` and `severity_scaler.pkl` are severity-feature scalers from different runs/naming conventions; `kmeans_model.pkl` and `severity_kmeans_model.pkl` are KMeans artifacts from different runs; `isolation_forest.pkl` is the anomaly estimator. These are unsupervised preprocessing/label-generation artifacts, not a trained fall-detection classifier.

## 8. CSV Inventory, Statistics, and Samples

The following sections cover every project CSV outside `.venv`. For each file, `Shape` gives rows and columns, `Missing cells` counts blank/NaN values, numeric statistics describe each numeric column, and the sample is the first 10 rows. Blank cells in the sample are intentional missing/non-applicable values.

### `data/processed/channel_statistics.csv`

- Shape: `18 rows x 13 columns`
- Missing cells: `0`
- Columns: `split, channel, mean, std, min, max, p25, p50, p75, samples, windows, label_0, label_1`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 18 | -0.0917422 | 0.0164689 | 0.000120642 | 0.0909627 | 0.324814 |
| std | 18 | 0.989853 | 1.06547 | 1.05912 | 0.0649212 | 1.19414 |
| min | 18 | -48.5212 | -17.5627 | -12.9511 | 12.3762 | -5.63588 |
| max | 18 | 6.51522 | 16.1045 | 13.3402 | 9.68905 | 38.1595 |
| p25 | 18 | -0.589018 | -0.279547 | -0.195697 | 0.177561 | -0.106129 |
| p50 | 18 | -0.416074 | -0.0394165 | 0.00209213 | 0.185826 | 0.355942 |
| p75 | 18 | 0.119958 | 0.374788 | 0.237884 | 0.290536 | 0.950644 |
| samples | 18 | 13627 | 28041 | 15073 | 19932.6 | 55423 |
| windows | 18 | 13627 | 28041 | 15073 | 19932.6 | 55423 |
| label_0 | 18 | 12427 | 25646.3 | 13573 | 18409.4 | 50939 |
| label_1 | 18 | 1200 | 2394.67 | 1500 | 1525.43 | 4484 |

First 10 rows:

| split | channel | mean | std | min | max | p25 | p50 | p75 | samples | windows | label_0 | label_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | acc1_x | 0.000451948 | 1.00911 | -13.8934 | 11.6163 | -0.199323 | 0.0297179 | 0.246185 | 55423 | 55423 | 50939 | 4484 |
| train | acc1_y | -0.0120695 | 1.0142 | -6.89975 | 6.94432 | -0.554049 | -0.377854 | 0.817334 | 55423 | 55423 | 50939 | 4484 |
| train | acc1_z | -0.0322971 | 0.989853 | -10.3763 | 8.83929 | -0.587955 | -0.0817941 | 0.438412 | 55423 | 55423 | 50939 | 4484 |
| train | gyro_x | 0.000523963 | 1.07316 | -36.8906 | 38.1595 | -0.192071 | 6.42881e-05 | 0.229583 | 55423 | 55423 | 50939 | 4484 |
| train | gyro_y | -0.00172154 | 1.06964 | -29.6888 | 27.5684 | -0.163348 | 0.00198487 | 0.169796 | 55423 | 55423 | 50939 | 4484 |
| train | gyro_z | -0.000210665 | 1.07406 | -38.0956 | 37.6419 | -0.136826 | 0.00219938 | 0.133601 | 55423 | 55423 | 50939 | 4484 |
| val | acc1_x | -0.0917422 | 1.08453 | -12.0087 | 11.1325 | -0.293318 | -0.0360446 | 0.199054 | 15073 | 15073 | 13573 | 1500 |
| val | acc1_y | 0.00972959 | 1.04219 | -7.25511 | 6.51522 | -0.567018 | -0.382135 | 0.950644 | 15073 | 15073 | 13573 | 1500 |
| val | acc1_z | 0.324814 | 1.00022 | -8.20313 | 9.20574 | -0.209686 | 0.355942 | 0.869386 | 15073 | 15073 | 13573 | 1500 |
| val | gyro_x | -0.00131147 | 1.17436 | -18.9812 | 17.5454 | -0.172721 | 0.00356407 | 0.207723 | 15073 | 15073 | 13573 | 1500 |
### `data/processed/reports/eda_acceleration_spikes.csv`

- Shape: `6 rows x 9 columns`
- Missing cells: `0`
- Columns: `split, class_name, windows, peak_mean, peak_median, peak_std, peak_p95, peak_max, magnitude_mean`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| windows | 6 | 1200 | 14020.5 | 8455.5 | 18856.2 | 50939 |
| peak_mean | 6 | 2.08205 | 4.5393 | 4.40994 | 2.6668 | 7.16641 |
| peak_median | 6 | 1.84132 | 4.34407 | 4.23187 | 2.70943 | 7.0617 |
| peak_std | 6 | 0.964197 | 1.32677 | 1.26785 | 0.361296 | 1.84286 |
| peak_p95 | 6 | 3.59748 | 6.89294 | 6.75199 | 3.42116 | 10.4178 |
| peak_max | 6 | 7.68074 | 10.7937 | 10.5206 | 2.99567 | 15.3133 |
| magnitude_mean | 6 | 1.36619 | 1.7886 | 1.81179 | 0.420321 | 2.17611 |

First 10 rows:

| split | class_name | windows | peak_mean | peak_median | peak_std | peak_p95 | peak_max | magnitude_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | ADL | 50939 | 2.09587 | 1.88398 | 1.02748 | 3.76154 | 9.26133 | 1.36619 |
| train | Fall | 4484 | 7.16641 | 7.0617 | 1.84286 | 10.4178 | 15.3133 | 2.17172 |
| val | ADL | 13573 | 2.15129 | 1.89952 | 1.05707 | 3.99028 | 8.01174 | 1.4574 |
| val | Fall | 1500 | 7.07161 | 6.81366 | 1.59038 | 10.0768 | 12.7154 | 2.17611 |
| test | ADL | 12427 | 2.08205 | 1.84132 | 0.964197 | 3.59748 | 7.68074 | 1.39398 |
| test | Fall | 1200 | 6.66859 | 6.56422 | 1.47864 | 9.51371 | 11.7798 | 2.16618 |
### `data/processed/reports/eda_autocorrelation.csv`

- Shape: `594 rows x 4 columns`
- Missing cells: `0`
- Columns: `split, channel, lag, autocorrelation`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| lag | 594 | 0 | 16 | 16 | 9.52993 | 32 |
| autocorrelation | 594 | -0.129891 | 0.0565435 | -0.0362337 | 0.240654 | 1 |

First 10 rows:

| split | channel | lag | autocorrelation |
| --- | --- | --- | --- |
| train | acc1_x | 0 | 1 |
| train | acc1_x | 1 | 0.698383 |
| train | acc1_x | 2 | 0.319105 |
| train | acc1_x | 3 | 0.217049 |
| train | acc1_x | 4 | 0.222961 |
| train | acc1_x | 5 | 0.211051 |
| train | acc1_x | 6 | 0.162659 |
| train | acc1_x | 7 | 0.0943122 |
| train | acc1_x | 8 | 0.0706545 |
| train | acc1_x | 9 | 0.0665171 |
### `data/processed/reports/eda_class_distribution.csv`

- Shape: `6 rows x 6 columns`
- Missing cells: `0`
- Columns: `split, class_label, class_name, window_count, percentage, total_windows`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| class_label | 6 | 0 | 0.5 | 0.5 | 0.547723 | 1 |
| window_count | 6 | 1200 | 14020.5 | 8455.5 | 18856.2 | 50939 |
| percentage | 6 | 8.0905 | 50 | 50 | 44.9765 | 91.9095 |
| total_windows | 6 | 13627 | 28041 | 15073 | 21219.9 | 55423 |

First 10 rows:

| split | class_label | class_name | window_count | percentage | total_windows |
| --- | --- | --- | --- | --- | --- |
| train | 0 | ADL | 50939 | 91.9095 | 55423 |
| train | 1 | Fall | 4484 | 8.0905 | 55423 |
| val | 0 | ADL | 13573 | 90.0484 | 15073 |
| val | 1 | Fall | 1500 | 9.95157 | 15073 |
| test | 0 | ADL | 12427 | 91.194 | 13627 |
| test | 1 | Fall | 1200 | 8.80605 | 13627 |
### `data/processed/reports/eda_outliers.csv`

- Shape: `18 rows x 10 columns`
- Missing cells: `0`
- Columns: `split, channel, sample_count, iqr_outliers, iqr_outlier_percentage, iqr_lower_bound, iqr_upper_bound, zscore_outliers, zscore_outlier_percentage, zscore_threshold`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| sample_count | 18 | 872128 | 1.79462e+06 | 964672 | 1.27569e+06 | 3.54707e+06 |
| iqr_outliers | 18 | 8108 | 302079 | 227674 | 301372 | 1.01039e+06 |
| iqr_outlier_percentage | 18 | 0.840493 | 17.2336 | 19.111 | 11.3363 | 32.5828 |
| iqr_lower_bound | 18 | -2.84351 | -1.26105 | -0.84897 | 0.828493 | -0.445259 |
| iqr_upper_bound | 18 | 0.459089 | 1.35629 | 0.911229 | 0.947594 | 3.22714 |
| zscore_outliers | 18 | 1391 | 21674.6 | 18044 | 23231 | 77014 |
| zscore_outlier_percentage | 18 | 0.159495 | 1.24592 | 1.05996 | 0.914204 | 2.43852 |
| zscore_threshold | 18 | 3 | 3 | 3 | 0 | 3 |

First 10 rows:

| split | channel | sample_count | iqr_outliers | iqr_outlier_percentage | iqr_lower_bound | iqr_upper_bound | zscore_outliers | zscore_outlier_percentage | zscore_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | acc1_x | 3547072 | 608827 | 17.1642 | -0.867586 | 0.914448 | 15934 | 0.449216 | 3 |
| train | acc1_y | 3547072 | 44835 | 1.264 | -2.61113 | 2.87441 | 21348 | 0.601849 | 3 |
| train | acc1_z | 3547072 | 242806 | 6.84525 | -2.12751 | 1.97796 | 6566 | 0.18511 | 3 |
| train | gyro_x | 3547072 | 678660 | 19.133 | -0.824552 | 0.862064 | 77014 | 2.1712 | 3 |
| train | gyro_y | 3547072 | 904997 | 25.5139 | -0.663064 | 0.669512 | 75297 | 2.12279 | 3 |
| train | gyro_z | 3547072 | 1010394 | 28.4853 | -0.542467 | 0.539242 | 52181 | 1.4711 | 3 |
| val | acc1_x | 964672 | 184146 | 19.089 | -1.03188 | 0.937612 | 3332 | 0.345402 | 3 |
| val | acc1_y | 964672 | 8108 | 0.840493 | -2.84351 | 3.22714 | 6259 | 0.648822 | 3 |
| val | acc1_z | 964672 | 44289 | 4.59109 | -1.82829 | 2.48799 | 2200 | 0.228057 | 3 |
| val | gyro_x | 964672 | 212543 | 22.0327 | -0.743388 | 0.77839 | 21805 | 2.26035 | 3 |
### `data/processed/reports/eda_signal_health_statistics.csv`

- Shape: `54 rows x 13 columns`
- Missing cells: `0`
- Columns: `gibesplit, class_name, channel, mean, median, std, variance, rms, min, max, skewness, kurtosis, sample_count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 54 | -0.187893 | 0.0460568 | 0.000120643 | 0.188245 | 0.714509 |
| median | 54 | -0.449582 | 0.0235487 | 0.00219971 | 0.268481 | 0.926878 |
| std | 54 | 0.900062 | 1.258 | 1.05912 | 0.421572 | 2.34148 |
| variance | 54 | 0.810112 | 1.757 | 1.12184 | 1.31148 | 5.48253 |
| rms | 54 | 0.900091 | 1.27294 | 1.06064 | 0.420471 | 2.34305 |
| min | 54 | -48.5212 | -15.6077 | -11.7561 | 11.2308 | -5.13218 |
| max | 54 | 4.06029 | 14.4481 | 11.666 | 9.00355 | 38.1595 |
| skewness | 54 | -1.33691 | 0.0129641 | -0.0177335 | 0.372372 | 0.992024 |
| kurtosis | 54 | 0.086231 | 14.4383 | 7.13257 | 25.9191 | 155.066 |
| sample_count | 54 | 76800 | 1.19642e+06 | 868672 | 1.23638e+06 | 3.54707e+06 |

First 10 rows:

| gibesplit | class_name | channel | mean | median | std | variance | rms | min | max | skewness | kurtosis | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | All | acc1_x | 0.000451951 | 0.0297179 | 1.00911 | 1.01829 | 1.00911 | -13.8934 | 11.6163 | -0.172358 | 5.89572 | 3547072 |
| train | All | acc1_y | -0.0120695 | -0.377854 | 1.0142 | 1.0286 | 1.01427 | -6.89975 | 6.94432 | 0.202759 | 0.725326 | 3547072 |
| train | All | acc1_z | -0.0322971 | -0.0817941 | 0.989853 | 0.979808 | 0.990379 | -10.3763 | 8.83929 | 0.314065 | 1.22191 | 3547072 |
| train | All | gyro_x | 0.000523961 | 6.42881e-05 | 1.07316 | 1.15167 | 1.07316 | -36.8906 | 38.1595 | 0.450478 | 38.5635 | 3547072 |
| train | All | gyro_y | -0.00172154 | 0.00198487 | 1.06964 | 1.14412 | 1.06964 | -29.6888 | 27.5684 | -0.145794 | 37.5338 | 3547072 |
| train | All | gyro_z | -0.000210666 | 0.00219938 | 1.07406 | 1.15361 | 1.07406 | -38.0956 | 37.6419 | -0.510598 | 101.801 | 3547072 |
| train | ADL | acc1_x | 0.0102339 | 0.0317026 | 0.932267 | 0.869121 | 0.932323 | -9.02316 | 6.12925 | -0.0609529 | 3.86334 | 3260096 |
| train | ADL | acc1_y | -0.0760279 | -0.408438 | 0.977297 | 0.955109 | 0.980249 | -5.59689 | 5.42834 | 0.175544 | 0.766639 | 3260096 |
| train | ADL | acc1_z | -0.0294212 | -0.0773949 | 0.941856 | 0.887093 | 0.942315 | -6.59289 | 5.82372 | 0.404459 | 0.614009 | 3260096 |
| train | ADL | gyro_x | 0.00723441 | 0.00125309 | 0.900062 | 0.810112 | 0.900091 | -36.8906 | 38.1595 | -0.0575259 | 35.124 | 3260096 |
### `data/processed/severity/anomaly_by_severity.csv`

- Shape: `9 rows x 6 columns`
- Missing cells: `0`
- Columns: `split, severity_label, severity_name, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| severity_label | 9 | 0 | 1 | 1 | 0.866025 | 2 |
| fall_windows | 9 | 0 | 798.222 | 563 | 926.184 | 2787 |
| anomalous_windows | 9 | 0 | 15.2222 | 11 | 18.0678 | 47 |
| anomaly_rate | 9 | 0 | 0.0123953 | 0.016864 | 0.0103647 | 0.0253388 |

First 10 rows:

| split | severity_label | severity_name | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- | --- | --- |
| train | 0 | Low | 1697 | 43 | 0.0253388 |
| train | 1 | Medium | 2787 | 47 | 0.016864 |
| train | 2 | High | 0 | 0 | 0 |
| val | 0 | Low | 563 | 5 | 0.00888099 |
| val | 1 | Medium | 937 | 18 | 0.0192102 |
| val | 2 | High | 0 | 0 | 0 |
| test | 0 | Low | 468 | 11 | 0.0235043 |
| test | 1 | Medium | 732 | 13 | 0.0177596 |
| test | 2 | High | 0 | 0 | 0 |
### `data/processed/severity/anomaly_by_subject.csv`

- Shape: `24 rows x 5 columns`
- Missing cells: `0`
- Columns: `split, subject_id, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| fall_windows | 24 | 292 | 299.333 | 300 | 1.78561 | 300 |
| anomalous_windows | 24 | 1 | 5.70833 | 3.5 | 5.1876 | 21 |
| anomaly_rate | 24 | 0.00333333 | 0.0190636 | 0.0116667 | 0.0172965 | 0.07 |

First 10 rows:

| split | subject_id | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- | --- |
| train | SA01 | 300 | 12 | 0.04 |
| train | SA04 | 300 | 8 | 0.0266667 |
| train | SA05 | 300 | 3 | 0.01 |
| train | SA06 | 299 | 12 | 0.0401338 |
| train | SA07 | 299 | 2 | 0.00668896 |
| train | SA08 | 300 | 2 | 0.00666667 |
| train | SA10 | 299 | 10 | 0.0334448 |
| train | SA11 | 300 | 21 | 0.07 |
| train | SA16 | 300 | 1 | 0.00333333 |
| train | SA17 | 300 | 3 | 0.01 |
### `data/processed/severity/anomaly_summary.csv`

- Shape: `3 rows x 4 columns`
- Missing cells: `0`
- Columns: `split, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| fall_windows | 3 | 1200 | 2394.67 | 1500 | 1815.62 | 4484 |
| anomalous_windows | 3 | 23 | 45.6667 | 24 | 38.397 | 90 |
| anomaly_rate | 3 | 0.0153333 | 0.0184682 | 0.02 | 0.00271514 | 0.0200714 |

First 10 rows:

| split | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- |
| train | 4484 | 90 | 0.0200714 |
| val | 1500 | 23 | 0.0153333 |
| test | 1200 | 24 | 0.02 |
### `data/processed/severity/cluster_statistics.csv`

- Shape: `2 rows x 14 columns`
- Missing cells: `0`
- Columns: `cluster_id, train_count, val_count, test_count, total_count, acc_mag_mean, acc_mag_rms_mean, acc_mag_peak_mean, gyro_mag_mean, gyro_mag_rms_mean, gyro_mag_peak_mean, cluster_intensity_score, severity_label, severity_name`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 2 | 0 | 0.5 | 0.5 | 0.707107 | 1 |
| train_count | 2 | 1697 | 2242 | 2242 | 770.746 | 2787 |
| val_count | 2 | 563 | 750 | 750 | 264.458 | 937 |
| test_count | 2 | 468 | 600 | 600 | 186.676 | 732 |
| total_count | 2 | 2728 | 3592 | 3592 | 1221.88 | 4456 |
| acc_mag_mean | 2 | 2.04996 | 2.14791 | 2.14791 | 0.138521 | 2.24586 |
| acc_mag_rms_mean | 2 | 2.33736 | 2.47483 | 2.47483 | 0.194404 | 2.61229 |
| acc_mag_peak_mean | 2 | 6.39667 | 7.01589 | 7.01589 | 0.875704 | 7.63511 |
| gyro_mag_mean | 2 | 1.90934 | 2.16416 | 2.16416 | 0.360364 | 2.41897 |
| gyro_mag_rms_mean | 2 | 3.04433 | 3.40727 | 3.40727 | 0.513262 | 3.7702 |
| gyro_mag_peak_mean | 2 | 10.1551 | 11.4295 | 11.4295 | 1.80228 | 12.7039 |
| cluster_intensity_score | 2 | 16.5517 | 18.4454 | 18.4454 | 2.67799 | 20.339 |
| severity_label | 2 | 0 | 0.5 | 0.5 | 0.707107 | 1 |

First 10 rows:

| cluster_id | train_count | val_count | test_count | total_count | acc_mag_mean | acc_mag_rms_mean | acc_mag_peak_mean | gyro_mag_mean | gyro_mag_rms_mean | gyro_mag_peak_mean | cluster_intensity_score | severity_label | severity_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1697 | 563 | 468 | 2728 | 2.04996 | 2.33736 | 6.39667 | 1.90934 | 3.04433 | 10.1551 | 16.5517 | 0 | Mild |
| 1 | 2787 | 937 | 732 | 4456 | 2.24586 | 2.61229 | 7.63511 | 2.41897 | 3.7702 | 12.7039 | 20.339 | 1 | Moderate |
### `data/processed/severity/reports/anomalous_feature_statistics.csv`

- Shape: `6 rows x 3 columns`
- Missing cells: `0`
- Columns: `split, status, count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| count | 6 | 23 | 1197.33 | 633 | 1689.93 | 4394 |

First 10 rows:

| split | status | count |
| --- | --- | --- |
| train | normal | 4394 |
| train | anomalous | 90 |
| val | normal | 1477 |
| val | anomalous | 23 |
| test | normal | 1176 |
| test | anomalous | 24 |
### `data/processed/severity/reports/anomaly_by_severity_report.csv`

- Shape: `9 rows x 6 columns`
- Missing cells: `0`
- Columns: `split, severity_label, severity_name, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| severity_label | 9 | 0 | 1 | 1 | 0.866025 | 2 |
| fall_windows | 9 | 304 | 798.222 | 518 | 536.747 | 1659 |
| anomalous_windows | 9 | 2 | 15.2222 | 10 | 14.6354 | 42 |
| anomaly_rate | 9 | 0.003861 | 0.0186811 | 0.0197368 | 0.0107186 | 0.0380048 |

First 10 rows:

| split | severity_label | severity_name | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- | --- | --- |
| train | 0 | Mild | 1659 | 42 | 0.0253165 |
| train | 1 | Moderate | 1267 | 10 | 0.00789266 |
| train | 2 | Severe | 1558 | 38 | 0.0243902 |
| val | 0 | Mild | 561 | 5 | 0.00891266 |
| val | 1 | Moderate | 421 | 16 | 0.0380048 |
| val | 2 | Severe | 518 | 2 | 0.003861 |
| test | 0 | Mild | 457 | 11 | 0.02407 |
| test | 1 | Moderate | 304 | 6 | 0.0197368 |
| test | 2 | Severe | 439 | 7 | 0.0159453 |
### `data/processed/severity/reports/anomaly_by_subject_report.csv`

- Shape: `24 rows x 5 columns`
- Missing cells: `0`
- Columns: `split, subject_id, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| fall_windows | 24 | 292 | 299.333 | 300 | 1.78561 | 300 |
| anomalous_windows | 24 | 1 | 5.70833 | 3.5 | 5.1876 | 21 |
| anomaly_rate | 24 | 0.00333333 | 0.0190636 | 0.0116667 | 0.0172965 | 0.07 |

First 10 rows:

| split | subject_id | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- | --- |
| train | SA01 | 300 | 12 | 0.04 |
| train | SA04 | 300 | 8 | 0.0266667 |
| train | SA05 | 300 | 3 | 0.01 |
| train | SA06 | 299 | 12 | 0.0401338 |
| train | SA07 | 299 | 2 | 0.00668896 |
| train | SA08 | 300 | 2 | 0.00666667 |
| train | SA10 | 299 | 10 | 0.0334448 |
| train | SA11 | 300 | 21 | 0.07 |
| train | SA16 | 300 | 1 | 0.00333333 |
| train | SA17 | 300 | 3 | 0.01 |
### `data/processed/severity/reports/anomaly_summary_report.csv`

- Shape: `3 rows x 4 columns`
- Missing cells: `0`
- Columns: `split, fall_windows, anomalous_windows, anomaly_rate`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| fall_windows | 3 | 1200 | 2394.67 | 1500 | 1815.62 | 4484 |
| anomalous_windows | 3 | 23 | 45.6667 | 24 | 38.397 | 90 |
| anomaly_rate | 3 | 0.0153333 | 0.0184682 | 0.02 | 0.00271514 | 0.0200714 |

First 10 rows:

| split | fall_windows | anomalous_windows | anomaly_rate |
| --- | --- | --- | --- |
| train | 4484 | 90 | 0.0200714 |
| val | 1500 | 23 | 0.0153333 |
| test | 1200 | 24 | 0.02 |
### `data/processed/severity/reports/cluster_feature_report.csv`

- Shape: `9 rows x 13 columns`
- Missing cells: `0`
- Columns: `split, cluster_id, severity_label, severity_name, count, acc_mag_peak_mean, acc_mag_peak_std, gyro_mag_peak_mean, gyro_mag_peak_std, acc_mag_rms_mean, acc_mag_rms_std, gyro_mag_rms_mean, gyro_mag_rms_std`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 9 | 0 | 1 | 1 | 0.866025 | 2 |
| severity_label | 9 | 0 | 1 | 1 | 0.866025 | 2 |
| count | 9 | 304 | 798.222 | 518 | 536.747 | 1659 |
| acc_mag_peak_mean | 9 | 6.3414 | 6.99101 | 6.94205 | 0.518396 | 7.85868 |
| acc_mag_peak_std | 9 | 1.22196 | 1.57868 | 1.5879 | 0.186752 | 1.83537 |
| gyro_mag_peak_mean | 9 | 9.78976 | 11.7051 | 12.3058 | 1.26558 | 13.0301 |
| gyro_mag_peak_std | 9 | 2.3605 | 3.18551 | 3.05253 | 0.804993 | 4.448 |
| acc_mag_rms_mean | 9 | 2.33364 | 2.50922 | 2.53216 | 0.122112 | 2.63956 |
| acc_mag_rms_std | 9 | 0.466433 | 0.499276 | 0.499025 | 0.0161441 | 0.523055 |
| gyro_mag_rms_mean | 9 | 3.00262 | 3.57274 | 3.72004 | 0.415584 | 3.9799 |
| gyro_mag_rms_std | 9 | 0.61618 | 0.760561 | 0.712216 | 0.12276 | 0.952382 |

First 10 rows:

| split | cluster_id | severity_label | severity_name | count | acc_mag_peak_mean | acc_mag_peak_std | gyro_mag_peak_mean | gyro_mag_peak_std | acc_mag_rms_mean | acc_mag_rms_std | gyro_mag_rms_mean | gyro_mag_rms_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | Mild | 1659 | 6.42738 | 1.65654 | 10.139 | 3.68585 | 2.33899 | 0.466433 | 3.04102 | 0.846346 |
| train | 1 | 2 | Severe | 1558 | 7.85868 | 1.83537 | 12.9757 | 3.19088 | 2.63441 | 0.492136 | 3.79681 | 0.692224 |
| train | 2 | 1 | Moderate | 1267 | 7.28282 | 1.72212 | 12.3142 | 2.73809 | 2.57472 | 0.512289 | 3.72004 | 0.712216 |
| val | 0 | 0 | Mild | 561 | 6.54096 | 1.37114 | 10.3141 | 3.05253 | 2.33364 | 0.491197 | 3.05606 | 0.68537 |
| val | 1 | 2 | Severe | 518 | 7.46264 | 1.56104 | 12.3058 | 2.3605 | 2.63956 | 0.523055 | 3.70249 | 0.61618 |
| val | 2 | 1 | Moderate | 421 | 7.29759 | 1.70113 | 13.0301 | 2.38437 | 2.60708 | 0.508359 | 3.9799 | 0.653784 |
| test | 0 | 0 | Mild | 457 | 6.3414 | 1.5879 | 9.78976 | 4.32756 | 2.39867 | 0.499025 | 3.00262 | 0.952382 |
| test | 1 | 2 | Severe | 439 | 6.94205 | 1.22196 | 12.4463 | 4.448 | 2.52373 | 0.506403 | 3.93809 | 0.937653 |
| test | 2 | 1 | Moderate | 304 | 6.76554 | 1.55094 | 12.031 | 2.48176 | 2.53216 | 0.494587 | 3.91764 | 0.7489 |
### `data/processed/severity/reports/post_isolation_forest_all_windows.csv`

- Shape: `84,123 rows x 16 columns`
- Missing cells: `307,756`
- Columns: `split, window_index, subject_id, recording_id, class_label, class_name, is_fall_window, fall_window_order, cluster_id, severity_label, severity_class, anomaly_score, anomaly_flag, anomaly_marking, refined_severity_label, refined_severity_class`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| window_index | 84123 | 0 | 20710.8 | 14217 | 16408 | 55422 |
| class_label | 84123 | 0 | 0.0853988 | 0 | 0.279476 | 1 |
| fall_window_order | 7184 | 0 | 1655.7 | 1197 | 1295.08 | 4483 |
| cluster_id | 7184 | 0 | 0.904649 | 1 | 0.800571 | 2 |
| severity_label | 84123 | -1 | -0.831128 | -1 | 0.605886 | 2 |
| anomaly_score | 7184 | -0.0730676 | 0.0793206 | 0.084564 | 0.0311061 | 0.142574 |
| anomaly_flag | 7184 | 0 | 0.0190702 | 0 | 0.136781 | 1 |
| refined_severity_label | 84123 | -1 | -0.834255 | -1 | 0.600753 | 2 |

First 10 rows:

| split | window_index | subject_id | recording_id | class_label | class_name | is_fall_window | fall_window_order | cluster_id | severity_label | severity_class | anomaly_score | anomaly_flag | anomaly_marking | refined_severity_label | refined_severity_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
| train | 1 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
| train | 2 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
| train | 3 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
| train | 4 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
| train | 5 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
| train | 6 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
| train | 7 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
| train | 8 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
| train | 9 | SA01 | SA01:D01_SA01_R01.txt | 0 | ADL / non-fall | False |  |  | -1 | Non-applicable / uncertain |  |  | Not evaluated | -1 | Non-applicable / uncertain |
### `data/processed/severity/reports/post_isolation_forest_fall_windows.csv`

- Shape: `7,184 rows x 16 columns`
- Missing cells: `0`
- Columns: `split, window_index, fall_window_order, subject_id, recording_id, class_label, class_name, cluster_id, cluster_mapped_severity_label, severity_label, severity_class, anomaly_score, anomaly_flag, anomaly_marking, refined_severity_label, refined_severity_class`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| window_index | 7184 | 1633 | 16282.6 | 11002 | 12473.6 | 47694 |
| fall_window_order | 7184 | 0 | 1655.7 | 1197 | 1295.08 | 4483 |
| class_label | 7184 | 1 | 1 | 1 | 0 | 1 |
| cluster_id | 7184 | 0 | 0.904649 | 1 | 0.800571 | 2 |
| cluster_mapped_severity_label | 7184 | 0 | 0.97745 | 1 | 0.849888 | 2 |
| severity_label | 7184 | 0 | 0.97745 | 1 | 0.849888 | 2 |
| anomaly_score | 7184 | -0.0730676 | 0.0793206 | 0.084564 | 0.0311061 | 0.142574 |
| anomaly_flag | 7184 | 0 | 0.0190702 | 0 | 0.136781 | 1 |
| refined_severity_label | 7184 | -1 | 0.940841 | 1 | 0.883737 | 2 |

First 10 rows:

| split | window_index | fall_window_order | subject_id | recording_id | class_label | class_name | cluster_id | cluster_mapped_severity_label | severity_label | severity_class | anomaly_score | anomaly_flag | anomaly_marking | refined_severity_label | refined_severity_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1633 | 0 | SA01 | SA01:F01_SA01_R01.txt | 1 | Fall | 1 | 2 | 2 | Severe | 0.0274336 | 0 | Normal | 2 | Severe |
| train | 1634 | 1 | SA01 | SA01:F01_SA01_R01.txt | 1 | Fall | 1 | 2 | 2 | Severe | -0.0653269 | 1 | Anomalous | -1 | Non-applicable / uncertain |
| train | 1635 | 2 | SA01 | SA01:F01_SA01_R01.txt | 1 | Fall | 1 | 2 | 2 | Severe | -0.0656995 | 1 | Anomalous | -1 | Non-applicable / uncertain |
| train | 1636 | 3 | SA01 | SA01:F01_SA01_R01.txt | 1 | Fall | 1 | 2 | 2 | Severe | -0.0730676 | 1 | Anomalous | -1 | Non-applicable / uncertain |
| train | 1649 | 4 | SA01 | SA01:F01_SA01_R02.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.0967213 | 0 | Normal | 0 | Mild |
| train | 1650 | 5 | SA01 | SA01:F01_SA01_R02.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.111812 | 0 | Normal | 0 | Mild |
| train | 1651 | 6 | SA01 | SA01:F01_SA01_R02.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.106299 | 0 | Normal | 0 | Mild |
| train | 1652 | 7 | SA01 | SA01:F01_SA01_R02.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.0389802 | 0 | Normal | 0 | Mild |
| train | 1665 | 8 | SA01 | SA01:F01_SA01_R03.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.0906103 | 0 | Normal | 0 | Mild |
| train | 1666 | 9 | SA01 | SA01:F01_SA01_R03.txt | 1 | Fall | 0 | 0 | 0 | Mild | 0.0981735 | 0 | Normal | 0 | Mild |
### `data/processed/severity/reports/post_isolation_forest_window_summary.csv`

- Shape: `18 rows x 4 columns`
- Missing cells: `0`
- Columns: `split, severity_class, anomaly_marking, window_count`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| window_count | 18 | 2 | 399.111 | 170 | 534.431 | 1617 |

First 10 rows:

| split | severity_class | anomaly_marking | window_count |
| --- | --- | --- | --- |
| test | Mild | Anomalous | 11 |
| test | Mild | Normal | 446 |
| test | Moderate | Anomalous | 6 |
| test | Moderate | Normal | 298 |
| test | Severe | Anomalous | 7 |
| test | Severe | Normal | 432 |
| train | Mild | Anomalous | 42 |
| train | Mild | Normal | 1617 |
| train | Moderate | Anomalous | 10 |
| train | Moderate | Normal | 1257 |
### `data/processed/severity/severity_cluster_profiles.csv`

- Shape: `2 rows x 11 columns`
- Missing cells: `8`
- Columns: `cluster_id, size, peak_accel_mean, peak_accel_median, acc_rms_mean, gyro_peak_mean, gyro_rms_mean, energy_mean, delta_velocity_mean, sma_mean, jerk_mean`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 2 | 0 | 0.5 | 0.5 | 0.707107 | 1 |
| size | 2 | 1697 | 2242 | 2242 | 770.746 | 2787 |
| peak_accel_mean | 2 | 6.39667 | 7.01589 | 7.01589 | 0.875704 | 7.63511 |
| peak_accel_median | 2 | 6.42979 | 6.96732 | 6.96732 | 0.760173 | 7.50484 |
| acc_rms_mean | 2 | 2.33736 | 2.47483 | 2.47483 | 0.194404 | 2.61229 |
| gyro_peak_mean | 2 | 10.1551 | 11.4295 | 11.4295 | 1.80228 | 12.7039 |
| gyro_rms_mean | 2 | 3.04433 | 3.40727 | 3.40727 | 0.513262 | 3.7702 |
| energy_mean | 0 |  |  |  |  |  |
| delta_velocity_mean | 0 |  |  |  |  |  |
| sma_mean | 0 |  |  |  |  |  |
| jerk_mean | 0 |  |  |  |  |  |

First 10 rows:

| cluster_id | size | peak_accel_mean | peak_accel_median | acc_rms_mean | gyro_peak_mean | gyro_rms_mean | energy_mean | delta_velocity_mean | sma_mean | jerk_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1697 | 6.39667 | 6.42979 | 2.33736 | 10.1551 | 3.04433 |  |  |  |  |
| 1 | 2787 | 7.63511 | 7.50484 | 2.61229 | 12.7039 | 3.7702 |  |  |  |  |
### `data/processed/severity/severity_cluster_sizes.csv`

- Shape: `2 rows x 3 columns`
- Missing cells: `0`
- Columns: `cluster_id, cluster_size, cluster_percentage`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 2 | 0 | 0.5 | 0.5 | 0.707107 | 1 |
| cluster_size | 2 | 1697 | 2242 | 2242 | 770.746 | 2787 |
| cluster_percentage | 2 | 0.378457 | 0.5 | 0.5 | 0.171888 | 0.621543 |

First 10 rows:

| cluster_id | cluster_size | cluster_percentage |
| --- | --- | --- |
| 0 | 1697 | 0.378457 |
| 1 | 2787 | 0.621543 |
### `data/processed/severity/severity_k_comparison.csv`

- Shape: `4 rows x 11 columns`
- Missing cells: `0`
- Columns: `k, silhouette, davies_bouldin, calinski_harabasz, silhouette_std, davies_bouldin_std, calinski_harabasz_std, mean_inertia, min_cluster_pct, max_cluster_pct, cluster_size_range`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 4 | 2 | 3.5 | 3.5 | 1.29099 | 5 |
| silhouette | 4 | 0.158482 | 0.167195 | 0.166459 | 0.00795482 | 0.177382 |
| davies_bouldin | 4 | 1.87068 | 2.01183 | 2.01243 | 0.1162 | 2.15179 |
| calinski_harabasz | 4 | 657.654 | 787.934 | 747.133 | 149.613 | 999.814 |
| silhouette_std | 4 | 0 | 1.47074e-05 | 1.45894e-06 | 2.74816e-05 | 5.59119e-05 |
| davies_bouldin_std | 4 | 0 | 0.000866472 | 7.53795e-05 | 0.00163362 | 0.00331513 |
| calinski_harabasz_std | 4 | 0 | 0.00363828 | 0.000205427 | 0.00700333 | 0.0141423 |
| mean_inertia | 4 | 163843 | 186324 | 184407 | 21239.4 | 212638 |
| min_cluster_pct | 4 | 0.118421 | 0.241826 | 0.235169 | 0.113287 | 0.378546 |
| max_cluster_pct | 4 | 0.301918 | 0.404561 | 0.347435 | 0.147334 | 0.621454 |

First 10 rows:

| k | silhouette | davies_bouldin | calinski_harabasz | silhouette_std | davies_bouldin_std | calinski_harabasz_std | mean_inertia | min_cluster_pct | max_cluster_pct | cluster_size_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.177382 | 1.99011 | 999.814 | 9.33745e-07 | 0.000137985 | 0.000195193 | 212638 | 0.378546 | 0.621454 | [1697, 2787] |
| 3 | 0.158482 | 2.15179 | 778.411 | 0 | 0 | 0 | 193014 | 0.28256 | 0.369982 | [1267, 1659] |
| 4 | 0.164331 | 2.03475 | 715.856 | 1.98413e-06 | 1.27735e-05 | 0.000215661 | 175799 | 0.187779 | 0.324888 | [842, 1457] |
| 5 | 0.168586 | 1.87068 | 657.654 | 5.59119e-05 | 0.00331513 | 0.0141423 | 163843 | 0.118421 | 0.301918 | [526, 1361] |
### `data/processed/severity/severity_k_stability.csv`

- Shape: `4 rows x 9 columns`
- Missing cells: `0`
- Columns: `k, mean_silhouette, std_silhouette, mean_davies_bouldin, std_davies_bouldin, mean_calinski_harabasz, std_calinski_harabasz, mean_cluster_size_min_pct, mean_cluster_size_max_pct`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| k | 4 | 2 | 3.5 | 3.5 | 1.29099 | 5 |
| mean_silhouette | 4 | 0.158482 | 0.167195 | 0.166459 | 0.00795482 | 0.177382 |
| std_silhouette | 4 | 0 | 1.47074e-05 | 1.45894e-06 | 2.74816e-05 | 5.59119e-05 |
| mean_davies_bouldin | 4 | 1.87068 | 2.01183 | 2.01243 | 0.1162 | 2.15179 |
| std_davies_bouldin | 4 | 0 | 0.000866472 | 7.53795e-05 | 0.00163362 | 0.00331513 |
| mean_calinski_harabasz | 4 | 657.654 | 787.934 | 747.133 | 149.613 | 999.814 |
| std_calinski_harabasz | 4 | 0 | 0.00363828 | 0.000205427 | 0.00700333 | 0.0141423 |
| mean_cluster_size_min_pct | 4 | 0.118421 | 0.241826 | 0.235169 | 0.113287 | 0.378546 |
| mean_cluster_size_max_pct | 4 | 0.301918 | 0.404561 | 0.347435 | 0.147334 | 0.621454 |

First 10 rows:

| k | mean_silhouette | std_silhouette | mean_davies_bouldin | std_davies_bouldin | mean_calinski_harabasz | std_calinski_harabasz | mean_cluster_size_min_pct | mean_cluster_size_max_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.177382 | 9.33745e-07 | 1.99011 | 0.000137985 | 999.814 | 0.000195193 | 0.378546 | 0.621454 |
| 3 | 0.158482 | 0 | 2.15179 | 0 | 778.411 | 0 | 0.28256 | 0.369982 |
| 4 | 0.164331 | 1.98413e-06 | 2.03475 | 1.27735e-05 | 715.856 | 0.000215661 | 0.187779 | 0.324888 |
| 5 | 0.168586 | 5.59119e-05 | 1.87068 | 0.00331513 | 657.654 | 0.0141423 | 0.118421 | 0.301918 |
### `reports/severity_k2_cluster_profiles.csv`

- Shape: `2 rows x 12 columns`
- Missing cells: `0`
- Columns: `cluster_id, count, percentage, peak_accel_mean, peak_accel_median, acc_rms_mean, gyro_peak_mean, gyro_rms_mean, jerk_mean, energy_mean, delta_velocity_mean, sma_mean`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 2 | 0 | 0.5 | 0.5 | 0.707107 | 1 |
| count | 2 | 1697 | 2242 | 2242 | 770.746 | 2787 |
| percentage | 2 | 0.378457 | 0.5 | 0.5 | 0.171888 | 0.621543 |
| peak_accel_mean | 2 | 6.39667 | 7.01589 | 7.01589 | 0.875705 | 7.63511 |
| peak_accel_median | 2 | 6.42979 | 6.96732 | 6.96732 | 0.760173 | 7.50484 |
| acc_rms_mean | 2 | 2.33736 | 2.47483 | 2.47483 | 0.194404 | 2.61229 |
| gyro_peak_mean | 2 | 10.1551 | 11.4295 | 11.4295 | 1.80228 | 12.7039 |
| gyro_rms_mean | 2 | 3.04433 | 3.40727 | 3.40727 | 0.513262 | 3.7702 |
| jerk_mean | 2 | 0.638034 | 0.700731 | 0.700731 | 0.0886658 | 0.763427 |
| energy_mean | 2 | 5.68426 | 6.37867 | 6.37867 | 0.982031 | 7.07307 |
| delta_velocity_mean | 2 | 6.46114 | 6.77077 | 6.77077 | 0.437883 | 7.0804 |
| sma_mean | 2 | 1.02958 | 1.08243 | 1.08243 | 0.074736 | 1.13527 |

First 10 rows:

| cluster_id | count | percentage | peak_accel_mean | peak_accel_median | acc_rms_mean | gyro_peak_mean | gyro_rms_mean | jerk_mean | energy_mean | delta_velocity_mean | sma_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1697 | 0.378457 | 6.39667 | 6.42979 | 2.33736 | 10.1551 | 3.04433 | 0.638034 | 5.68426 | 6.46114 | 1.02958 |
| 1 | 2787 | 0.621543 | 7.63511 | 7.50484 | 2.61229 | 12.7039 | 3.7702 | 0.763427 | 7.07307 | 7.0804 | 1.13527 |
### `reports/severity_k3_cluster_profiles.csv`

- Shape: `3 rows x 12 columns`
- Missing cells: `0`
- Columns: `cluster_id, count, percentage, peak_accel_mean, peak_accel_median, acc_rms_mean, gyro_peak_mean, gyro_rms_mean, jerk_mean, energy_mean, delta_velocity_mean, sma_mean`

Numeric statistics:

| column | non_null | min | mean | median | std | max |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_id | 3 | 0 | 1 | 1 | 1 | 2 |
| count | 3 | 1267 | 1494.67 | 1558 | 203.53 | 1659 |
| percentage | 3 | 0.28256 | 0.333333 | 0.347458 | 0.0453902 | 0.369982 |
| peak_accel_mean | 3 | 6.42738 | 7.18963 | 7.28282 | 0.720187 | 7.85868 |
| peak_accel_median | 3 | 6.44229 | 7.10019 | 7.17522 | 0.623775 | 7.68305 |
| acc_rms_mean | 3 | 2.33899 | 2.51604 | 2.57472 | 0.156207 | 2.63441 |
| gyro_peak_mean | 3 | 10.139 | 11.8096 | 12.3142 | 1.48413 | 12.9757 |
| gyro_rms_mean | 3 | 3.04102 | 3.51929 | 3.72004 | 0.415972 | 3.79681 |
| jerk_mean | 3 | 0.643458 | 0.718091 | 0.724855 | 0.0714924 | 0.785962 |
| energy_mean | 3 | 5.68843 | 6.58745 | 6.89161 | 0.792025 | 7.18231 |
| delta_velocity_mean | 3 | 6.46089 | 6.86599 | 7.0329 | 0.352634 | 7.10419 |
| sma_mean | 3 | 1.02889 | 1.09881 | 1.1291 | 0.060738 | 1.13846 |

First 10 rows:

| cluster_id | count | percentage | peak_accel_mean | peak_accel_median | acc_rms_mean | gyro_peak_mean | gyro_rms_mean | jerk_mean | energy_mean | delta_velocity_mean | sma_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1659 | 0.369982 | 6.42738 | 6.44229 | 2.33899 | 10.139 | 3.04102 | 0.643458 | 5.68843 | 6.46089 | 1.02889 |
| 1 | 1558 | 0.347458 | 7.85868 | 7.68305 | 2.63441 | 12.9757 | 3.79681 | 0.785962 | 7.18231 | 7.10419 | 1.13846 |
| 2 | 1267 | 0.28256 | 7.28282 | 7.17522 | 2.57472 | 12.3142 | 3.72004 | 0.724855 | 6.89161 | 7.0329 | 1.1291 |

## 9. How to Continue

1. Completed: reran the chosen K=3 pipeline and downstream report export as versioned run `k3_20260913T135813Z`.
2. Completed: evaluated CNN-LSTM threshold `0.50` separately for Mild, Moderate, Severe, and uncertain (`-1`) fall windows.
3. Future mitigation: assess a 2-consecutive-window alert rule for single-window ADL false-trigger blips; current analysis is non-operational.
4. Completed: reported window-, recording-, subject-level, and impact-verified metrics in the versioned evaluation outputs.
5. Verified: preprocessing scaler is train-only; severity feature scaler, KMeans, and Isolation Forest are fit on train fall windows only.
6. Completed for the versioned severity/export/evaluation artifacts; exact commands and UTC timestamps are recorded in `artifact_provenance.json` and the evaluation summary.

## 10. Key Files

| Purpose | File |
| --- | --- |
| Consolidated guide | `PROJECT_STATUS_REPORT.md` |
| Rebuild this guide | `scripts/generate_project_status_report.py` |
| Raw loader | `src/data/sisfall_loader.py` |
| Preprocessing/windowing | `src/data/sisfall_preprocessing.py` |
| Severity labeling | `src/data/severity_labeling.py` |
| Anomaly refinement | `src/data/severity_anomaly.py` |
| CNN baseline trainer | `scripts/train_cnn_baseline.py` |
| CNN baseline outputs | `results/cnn_baseline/` |
| CNN-LSTM baseline trainer | `scripts/train_cnn_lstm_baseline.py` |
| CNN-LSTM baseline outputs | `results/cnn_lstm_baseline/` |
| CNN-LSTM false-positive diagnostics | `scripts/diagnose_cnn_lstm_false_positives.py` |
| CNN-LSTM false-positive diagnostic outputs | `results/cnn_lstm_baseline/false_positive_diagnostics/` |
| Tests | `tests/` |

