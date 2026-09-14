# Fall Detection Project: Current Verification and Evaluation Report

Generated on 2026-09-14 from the files currently present in this workspace.

## 1. Executive Conclusion

The project is substantially complete through data ingestion, preprocessing, EDA, impact-centered fall labeling, class-imbalance handling, severity clustering, anomaly refinement, supervised CNN and CNN-LSTM baseline training, recording-level evaluation, impact-window verification, false-positive diagnostics, and severity-wise model evaluation.

The strongest current operating point is the CNN-LSTM at threshold 0.50. In the current stored evaluation outputs it gives:

| Metric | Value |
| --- | ---: |
| Test window precision | 0.9770 |
| Test window recall | 0.9925 |
| Test window F1 | 0.9847 |
| Test window F2 | 0.9894 |
| Recording-level fall recall | 1.0000 |
| Impact-verified fall recall | 1.0000 |
| ADL false-trigger recordings in recording-level summary | 0 |

The model selected threshold from validation is 0.25, but threshold 0.50 is the better practical default because it keeps perfect recording-level and impact-verified fall recall while reducing false-trigger risk in the stored diagnostic comparison.

## 2. Verification Performed Now

| Check | Result |
| --- | --- |
| Git worktree | Many modified tracked files are present; report treats current disk state as source of truth. |
| Python system environment | `python3` exists but lacks numpy, pandas, and pytest. |
| Project virtual environment | `.venv/Scripts/python.exe` exists and was executed through `cmd.exe`. |
| Test command | `.venv\Scripts\python.exe -m pytest` |
| Test result | 16 passed, 1 warning, 37.77 seconds |
| Warning | `ConvergenceWarning` in synthetic severity-labeling test because duplicate synthetic points produce fewer distinct clusters than requested. |

Conclusion: the repository test suite passes in the project virtual environment.

## 3. Pipeline Status by Step

| Step | Status | Main Artifacts | Evaluation Metrics / Evidence |
| --- | --- | --- | --- |
| Raw SisFall loading | Done | `src/data/sisfall_loader.py`, `scripts/load_sisfall_dataset.py` | Raw rows parsed from 9 comma-separated sensor values ending with semicolon. |
| Metadata construction | Done | processed metadata arrays | Subject ID, activity code/name, recording ID, source, label, timestamp are generated. |
| Binary activity mapping | Done | labels arrays | `D01-D19 = ADL/background`, `F01-F15 = fall`, binary labels `0/1`. |
| Channel selection | Done | `train.npy`, `val.npy`, `test.npy` | Input shape is `(64, 6)` using `acc1_x/y/z` and `gyro_x/y/z`; second accelerometer removed. |
| Filtering | Done | `data/processed/filter_plot.png` | Butterworth low-pass filtering at 5 Hz before downsampling. |
| Downsampling | Done | processed arrays | 200 Hz to 20 Hz, factor 10. |
| Subject-wise splitting | Done | subject ID arrays | Subject overlap = 0; recording overlap = 0. |
| Normalization | Done | `data/processed/scaler.pkl` | Train-only `StandardScaler`; model inputs are standardized z-scores. |
| Sliding windows | Done | `train.npy`, `val.npy`, `test.npy` | Window length 64 samples, stride 16, duration 3.2 s, stride duration 0.8 s, 75% overlap. |
| Impact-centered fall labeling | Done | label arrays, impact verification reports | Fall windows are only windows containing peak impact; model impact recall is evaluated separately. |
| EDA and quality checks | Done | `data/processed/reports/*.csv`, plots | No NaN or Inf in processed arrays; outlier and signal-health reports generated. |
| Class imbalance handling | Done | `X_train_balanced.npy`, `y_train_balanced.npy`, `class_imbalance_report.json` | Train fall share improved from 8.09% to 14.97% in balanced artifact. |
| Severity clustering | Done | `data/processed/severity/runs/k3_20260913T135813Z/*` | K=3 selected for Mild/Moderate/Severe interpretation; full metrics below. |
| Anomaly refinement | Done | Isolation Forest outputs | Train 2.01%, val 1.53%, test 2.00% fall windows flagged anomalous. |
| CNN baseline | Done | `results/cnn_baseline/*` | Test fall F1 0.9814 at threshold 0.50; 0.9802 at selected threshold 0.20. |
| CNN-LSTM baseline | Done | `results/cnn_lstm_baseline/*` | Test fall F1 0.9847 at threshold 0.50; recording recall 1.0000. |
| Recording-level evaluation | Done | `recording_level_eval/*` | CNN-LSTM threshold 0.50 detects 300/300 fall recordings. |
| Impact verification | Done | `impact_verification_summary.json` | CNN-LSTM threshold 0.50 impact-verified recall 1.0000. |
| Severity-wise model evaluation | Done | `severity_eval/k3_20260913T135813Z/*` | Mild recall 0.9865, Moderate 1.0000, Severe 0.9954 at window level. |

## 4. Dataset Metrics

| Split | Windows | Shape | Subjects | Recordings | ADL/background | Fall | Fall % | NaN | Inf |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 55,423 | `(55423, 64, 6)` | 26 | 2,942 | 50,939 | 4,484 | 8.09 | 0 | 0 |
| val | 15,073 | `(15073, 64, 6)` | 6 | 829 | 13,573 | 1,500 | 9.95 | 0 | 0 |
| test | 13,627 | `(13627, 64, 6)` | 6 | 734 | 12,427 | 1,200 | 8.81 | 0 | 0 |
| total | 84,123 | - | 38 unique split assignments | 4,505 | 76,939 | 7,184 | 8.54 | 0 | 0 |

Dataset contract verification:

| Metric | Value |
| --- | ---: |
| `is_valid` | true |
| Subject overlap | 0 |
| Recording overlap | 0 |
| Missing values | 0 |
| Infinite values | 0 |
| Scaler present | true |

## 5. Class Imbalance Metrics

| Dataset | Total | ADL | Fall | Fall % | ADL:Fall Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original train | 55,423 | 50,939 | 4,484 | 8.09 | 11.36 |
| Balanced train artifact | 59,907 | 50,939 | 8,968 | 14.97 | 5.68 |

Class balancing used augmentation ratio 2 with seed 42, adding 4,484 augmented fall windows.

## 6. EDA Metrics

Class distribution:

| Split | ADL Windows | ADL % | Fall Windows | Fall % |
| --- | ---: | ---: | ---: | ---: |
| train | 50,939 | 91.91 | 4,484 | 8.09 |
| val | 13,573 | 90.05 | 1,500 | 9.95 |
| test | 12,427 | 91.19 | 1,200 | 8.81 |

Acceleration spike summary:

| Split | Class | Windows | Peak Mean | Peak Median | Peak P95 | Peak Max | Magnitude Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | ADL | 50,939 | 2.0959 | 1.8840 | 3.7615 | 9.2613 | 1.3662 |
| train | Fall | 4,484 | 7.1664 | 7.0617 | 10.4178 | 15.3133 | 2.1717 |
| val | ADL | 13,573 | 2.1513 | 1.8995 | 3.9903 | 8.0117 | 1.4574 |
| val | Fall | 1,500 | 7.0716 | 6.8137 | 10.0768 | 12.7154 | 2.1761 |
| test | ADL | 12,427 | 2.0820 | 1.8413 | 3.5975 | 7.6807 | 1.3940 |
| test | Fall | 1,200 | 6.6686 | 6.5642 | 9.5137 | 11.7798 | 2.1662 |

Outlier summary:

| Metric | Highest Observed Case | Value |
| --- | --- | ---: |
| Highest z-score outlier percentage | test `gyro_x` | 2.44% |
| Highest IQR outlier percentage | val `gyro_z` | 32.58% |
| Highest signal kurtosis | train ADL `gyro_z` | 155.07 |
| Highest signal std | test Fall `gyro_x` | 2.34 |

Interpretation: fall windows have much higher acceleration peak values than ADL windows, supporting impact-centered labels. Gyroscope tails are heavy, especially `gyro_z`; these are documented in the outlier reports and should be handled carefully rather than blindly clipped.

## 7. Severity Labeling Metrics

Canonical versioned run used for current severity conclusions:

`data/processed/severity/runs/k3_20260913T135813Z`

Run provenance:

| Metric | Value |
| --- | --- |
| Run ID | `k3_20260913T135813Z` |
| Generated at UTC | 2026-09-13T13:58:13.566997+00:00 |
| Completed at UTC | 2026-09-13T13:58:56.052038+00:00 |
| Selected K | 3 |
| Fit scope | severity scaler, KMeans, and Isolation Forest fitted on train fall windows only |
| Uncertain policy | Isolation Forest anomalies become refined severity `-1` |

K selection metrics:

| K | Silhouette | Davies-Bouldin | Calinski-Harabasz | Min Cluster % | Max Cluster % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.1774 | 1.9901 | 999.8138 | 37.85 | 62.15 |
| 3 | 0.1585 | 2.1518 | 778.4105 | 28.26 | 37.00 |
| 4 | 0.1643 | 2.0347 | 715.8563 | 18.78 | 32.49 |
| 5 | 0.1686 | 1.8707 | 657.6541 | 11.84 | 30.19 |

Interpretation: K=2 has the best silhouette and Calinski-Harabasz score, while K=5 has the best Davies-Bouldin score. K=3 was selected because it supports the required three-level Mild/Moderate/Severe interpretation and keeps cluster sizes balanced enough for downstream severity analysis.

Train cluster sizes:

| Cluster | Size | Share |
| ---: | ---: | ---: |
| 0 | 1,659 | 36.998% |
| 1 | 1,558 | 34.746% |
| 2 | 1,267 | 28.256% |

Cluster intensity profiles:

| Cluster | Severity Mapping | Size | Peak Accel Mean | Acc RMS Mean | Gyro Peak Mean | Gyro RMS Mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Mild | 1,659 | 6.4274 | 2.3390 | 10.1390 | 3.0410 |
| 2 | Moderate | 1,267 | 7.2828 | 2.5747 | 12.3142 | 3.7200 |
| 1 | Severe | 1,558 | 7.8587 | 2.6344 | 12.9757 | 3.7968 |

Severity split counts before anomaly refinement:

| Split | Fall Windows | Mild | Moderate | Severe |
| --- | ---: | ---: | ---: | ---: |
| train | 4,484 | 1,659 | 1,267 | 1,558 |
| val | 1,500 | 561 | 421 | 518 |
| test | 1,200 | 457 | 304 | 439 |

## 8. Anomaly Refinement Metrics

Isolation Forest configuration:

| Metric | Value |
| --- | ---: |
| Contamination | 0.02 |
| Estimators | 256 |
| Random state | 42 |
| Score meaning | Higher decision-function score means more normal |
| Anomaly flag | `IsolationForest.predict == -1` |

Anomaly counts by split:

| Split | Fall Windows | Anomalous Windows | Anomaly Rate |
| --- | ---: | ---: | ---: |
| train | 4,484 | 90 | 2.01% |
| val | 1,500 | 23 | 1.53% |
| test | 1,200 | 24 | 2.00% |

Anomaly counts by severity:

| Split | Severity | Fall Windows | Anomalous Windows | Anomaly Rate |
| --- | --- | ---: | ---: | ---: |
| train | Mild | 1,659 | 42 | 2.53% |
| train | Moderate | 1,267 | 10 | 0.79% |
| train | Severe | 1,558 | 38 | 2.44% |
| val | Mild | 561 | 5 | 0.89% |
| val | Moderate | 421 | 16 | 3.80% |
| val | Severe | 518 | 2 | 0.39% |
| test | Mild | 457 | 11 | 2.41% |
| test | Moderate | 304 | 6 | 1.97% |
| test | Severe | 439 | 7 | 1.59% |

Conclusion: anomaly rates are close to the configured 2% contamination overall. Validation Moderate is the highest per-severity anomaly cell at 3.80%, which deserves review if using severity labels for training a severity head.

## 9. CNN Baseline Metrics

Training configuration:

| Metric | Value |
| --- | ---: |
| Parameters | 33,729 |
| Input shape | `(64, 6)` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | binary cross-entropy |
| Batch size | 64 |
| Max epochs | 100 |
| Epochs completed | 18 |
| Best epoch | 8 |
| Class weight ADL | 0.5440 |
| Class weight fall | 6.1801 |

Training-history checkpoints:

| Epoch | Train Accuracy | Train Loss | Train PR-AUC | Train Precision | Train Recall | Val Accuracy | Val Loss | Val PR-AUC | Val Precision | Val Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9719 | 0.0820 | 0.9531 | 0.7506 | 0.9781 | 0.9685 | 0.1021 | 0.9724 | 0.7603 | 0.9980 |
| 8 best val loss | 0.9955 | 0.0132 | 0.9899 | 0.9497 | 0.9978 | 0.9960 | 0.0125 | 0.9945 | 0.9669 | 0.9933 |
| 18 last | 0.9991 | 0.0023 | 0.9983 | 0.9894 | 1.0000 | 0.9965 | 0.0180 | 0.9887 | 0.9820 | 0.9827 |

Validation-selected threshold:

| Threshold | Validation Precision | Validation Recall | Validation F2 |
| ---: | ---: | ---: | ---: |
| 0.20 | 0.9560 | 0.9987 | 0.9898 |

Test metrics:

| Threshold | Fall Precision | Fall Recall | Fall F1 | Fall F2 | Accuracy | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.50 | 0.9761 | 0.9867 | 0.9814 | 0.9845 | 0.9967 | `[[12398, 29], [16, 1184]]` |
| 0.20 selected | 0.9683 | 0.9925 | 0.9802 | 0.9876 | 0.9965 | `[[12388, 39], [9, 1191]]` |

Recording and impact metrics:

| Threshold | Fall Recordings | Detected Fall Recordings | Fully Missed | Recording Recall | ADL Recordings | ADL False Triggers | Impact-Verified Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 300 | 299 | 1 | 0.9967 | 434 | 0 | 0.9967 |
| 0.20 selected | 300 | 300 | 0 | 1.0000 | 434 | 0 | 1.0000 |

Conclusion: the CNN baseline is strong. The selected 0.20 threshold improves recall and removes the single missed fall recording at the cost of more window-level false positives.

## 10. CNN-LSTM Baseline Metrics

Training configuration:

| Metric | Value |
| --- | ---: |
| Parameters | 46,817 |
| CNN parameter baseline | 33,729 |
| Input shape | `(64, 6)` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | binary cross-entropy |
| Batch size | 64 |
| Max epochs | 100 |
| Epochs completed | 14 |
| Best epoch | 4 |
| Training time seconds | 466.1174 |
| LSTM layers | 1 |
| LSTM unroll | true |
| Class weight ADL | 0.5440 |
| Class weight fall | 6.1801 |

Training-history checkpoints:

| Epoch | Train Accuracy | Train Loss | Train PR-AUC | Train Precision | Train Recall | Val Accuracy | Val Loss | Val PR-AUC | Val Precision | Val Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9673 | 0.0993 | 0.9384 | 0.7226 | 0.9668 | 0.9933 | 0.0200 | 0.9958 | 0.9539 | 0.9800 |
| 4 best val loss | 0.9934 | 0.0205 | 0.9898 | 0.9275 | 0.9964 | 0.9957 | 0.0132 | 0.9944 | 0.9693 | 0.9880 |
| 14 last | 0.9989 | 0.0028 | 0.9983 | 0.9864 | 0.9998 | 0.9969 | 0.0157 | 0.9892 | 0.9821 | 0.9873 |

Validation-selected threshold:

| Threshold | Validation Precision | Validation Recall | Validation F1 | Validation F2 |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.9539 | 0.9927 | 0.9729 | 0.9847 |

Test metrics:

| Threshold | Fall Precision | Fall Recall | Fall F1 | Fall F2 | Accuracy | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.50 | 0.9770 | 0.9925 | 0.9847 | 0.9894 | 0.9973 | `[[12399, 28], [9, 1191]]` |
| 0.25 selected | 0.9606 | 0.9950 | 0.9775 | 0.9879 | 0.9960 | `[[12378, 49], [6, 1194]]` |

Recording and impact metrics:

| Threshold | Fall Recordings | Detected Fall Recordings | Fully Missed | Recording Recall | ADL Recordings | ADL False Triggers | Impact-Verified Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 300 | 300 | 0 | 1.0000 | 434 | 0 | 1.0000 |
| 0.25 selected | 300 | 300 | 0 | 1.0000 | 434 | 0 | 1.0000 |

False-positive diagnostics:

| Threshold | False-Triggered ADL Recordings in Diagnostic File | Notes |
| ---: | ---: | --- |
| 0.50 | 1 | Single-window blip, activity `D14`, judged unexpected. |
| 0.25 | 3 | Single-window blips, activities `D13` and `D14`; two plausible-borderline and one unexpected recording in listed rows. |

Note: the recording-level summary currently reports 0 ADL false-trigger recordings for CNN-LSTM at both thresholds, while the false-positive diagnostics file lists 1 at threshold 0.50 and 3 at threshold 0.25. Treat this as a report-consistency issue to resolve before publication. The practical conclusion remains that threshold 0.50 is less risky than 0.25.

## 11. CNN vs CNN-LSTM Comparison

| Model | Threshold | Parameters | Window Precision | Window Recall | Window F1 | Window F2 | Recording Recall | Impact Recall | Training Time Seconds | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.50 | 33,729 | 0.9761 | 0.9867 | 0.9814 | 0.9845 | 0.9967 | 0.9967 | - | 8 best |
| CNN-LSTM | 0.50 | 46,817 | 0.9770 | 0.9925 | 0.9847 | 0.9894 | 1.0000 | 1.0000 | 466.1174 | 14 completed |
| CNN | selected 0.20 | 33,729 | 0.9683 | 0.9925 | 0.9802 | 0.9876 | 1.0000 | 1.0000 | - | 8 best |
| CNN-LSTM | selected 0.25 | 46,817 | 0.9606 | 0.9950 | 0.9775 | 0.9879 | 1.0000 | 1.0000 | 466.1174 | 14 completed |

Conclusion: CNN-LSTM at 0.50 is currently the best balanced operating point because it improves recall over CNN at 0.50 and preserves perfect recording/impact recall without using the lower, noisier selected threshold.

## 12. Severity-Wise CNN-LSTM Evaluation

Evaluation source:

`results/cnn_lstm_baseline/severity_eval/k3_20260913T135813Z`

Overall metrics at threshold 0.50:

| Metric | Value |
| --- | ---: |
| TP | 1,191 |
| FP | 28 |
| FN | 9 |
| TN | 12,399 |
| Window precision | 0.9770 |
| Window recall | 0.9925 |
| Window F1 | 0.9847 |
| Fall recordings | 300 |
| Detected fall recordings | 300 |
| Recording recall | 1.0000 |
| ADL recordings | 434 |
| ADL false-trigger recordings | 0 |
| ADL false-trigger rate | 0.0000 |
| Impact-verified recordings | 300 |
| Test subjects | 6 |

Window metrics by refined severity:

| Severity | Windows | Detected Windows | Window Recall | Window Precision | Window F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncertain (-1) | 24 | 23 | 0.9583 | 0.0189 | 0.0370 |
| Mild | 446 | 440 | 0.9865 | 0.3610 | 0.5285 |
| Moderate | 298 | 298 | 1.0000 | 0.2445 | 0.3929 |
| Severe | 432 | 430 | 0.9954 | 0.3527 | 0.5209 |

Important interpretation: severity-wise precision is low because the binary detector outputs fall/not-fall, not a severity class. The precision values are calculated against each severity subset, so positives from other severities count against a single severity's precision. Use severity-wise recall as the main question here: "are falls of this severity detected?"

Recording metrics by severity:

| Severity | Recordings | Detected Recordings | Impact-Verified Recordings | Two-Consecutive-Window Recordings |
| --- | ---: | ---: | ---: | ---: |
| Uncertain (-1) | 13 | 12 | 12 | 4 |
| Mild | 121 | 121 | 121 | 112 |
| Moderate | 79 | 79 | 79 | 76 |
| Severe | 111 | 111 | 111 | 109 |

Subject metrics by severity:

| Severity | Subjects | Mean Subject Recall | Min Subject Recall | Impact-Verified Subjects |
| --- | ---: | ---: | ---: | ---: |
| Uncertain (-1) | 4 | 0.9167 | 0.6667 | 4 |
| Mild | 4 | 0.9848 | 0.9651 | 4 |
| Moderate | 4 | 1.0000 | 1.0000 | 4 |
| Severe | 4 | 0.9955 | 0.9818 | 4 |

Future two-consecutive-window policy:

| Metric | Value |
| --- | ---: |
| Status | analysis only |
| Fall recordings detected | 300/300 |
| Recording recall | 1.0000 |
| ADL false-trigger recordings | 0 |
| ADL false-trigger rate | 0.0000 |

Conclusion: severity-wise recall is strong for Mild, Moderate, and Severe. Uncertain/anomalous windows remain the weakest group, which is expected because they are explicitly flagged as unusual.

## 13. Known Issues and Caveats

1. `PROJECT_STATUS_REPORT.md` and README snapshots do not fully agree with every current stored metric. For example, README lists older CNN-LSTM test precision/recall values than the current `run_metadata.json`.
2. Some files directly under `data/processed/severity/` appear to contain older K=2 or mixed-run severity outputs. Prefer the versioned K=3 run `data/processed/severity/runs/k3_20260913T135813Z` for current conclusions.
3. CNN-LSTM recording-level summary and false-positive diagnostics disagree on ADL false-trigger recording counts. Re-run `scripts/evaluate_recording_level.py` and `scripts/diagnose_cnn_lstm_false_positives.py` together before final publication.
4. The project has a large dirty worktree. Before starting new experiments, commit or intentionally archive the current state so future metrics are traceable.
5. The raw SisFall dataset is present locally but not tracked, which is correct. Any collaborator must download/extract it separately.
6. Severity labels are unsupervised proxy labels, not clinically validated fall-injury severity.
7. Standardized channel values and derived magnitudes are dimensionless z-score-space values unless inverse transformed through the scaler.

## 14. Recommended Next Tasks

1. Freeze a canonical run: regenerate preprocessing, severity K=3, anomaly export, CNN-LSTM evaluation, and false-positive diagnostics in one timestamped output folder.
2. Update `README.md` and `PROJECT_STATUS_REPORT.md` from that same canonical run so there is only one metric story.
3. Add a small script that validates cross-file consistency: threshold values, TP/FP/FN/TN, recording-level false triggers, severity label counts, and anomaly counts.
4. Decide whether the final application should use threshold 0.50 alone or threshold 0.50 plus the two-consecutive-window alert rule.
5. For further modeling, train a multi-task or two-stage model: binary fall detector first, severity classifier only on detected fall/impact windows second.
6. Evaluate robustness by subject group and activity type, especially ADL activities `D13` and `D14`.
7. If writing a thesis or report, clearly separate "binary fall detection performance" from "unsupervised severity-cluster interpretation."

## 15. Files to Use for Future Work

| Purpose | Recommended File/Folder |
| --- | --- |
| Main binary train data | `data/processed/train.npy`, `data/processed/train_labels.npy` |
| Main binary validation data | `data/processed/val.npy`, `data/processed/val_labels.npy` |
| Main binary test data | `data/processed/test.npy`, `data/processed/test_labels.npy` |
| Subject metadata | `data/processed/*_subject_ids.npy` |
| Recording metadata | `data/processed/*_recording_ids.npy` |
| Preprocessing scaler | `data/processed/scaler.pkl` |
| Canonical severity run | `data/processed/severity/runs/k3_20260913T135813Z/` |
| CNN baseline results | `results/cnn_baseline/` |
| CNN-LSTM baseline results | `results/cnn_lstm_baseline/` |
| Severity-wise CNN-LSTM evaluation | `results/cnn_lstm_baseline/severity_eval/k3_20260913T135813Z/` |
| Tests | `tests/` |

## 16. Final Decision Support

Use this conclusion for now:

| Decision | Recommendation |
| --- | --- |
| Main detector | CNN-LSTM |
| Main threshold | 0.50 |
| Main success metric | Impact-verified recording-level fall recall |
| Current success result | 300/300 fall recordings detected, impact-verified recall 1.0000 |
| Main risk to report | Artifact/report consistency around false-positive counts and mixed severity runs |
| Most important cleanup before publication | Regenerate all final metrics from one canonical timestamped run |

