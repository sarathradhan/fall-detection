# Requested Severity Plots

Generated on 2026-09-14 with:

```bash
.venv\Scripts\python.exe scripts\generate_requested_plots.py
```

## KMeans Severity Test Plots

| Plot | Purpose |
| --- | --- |
| `kmeans_severity_test_cluster_counts.png` | Shows the number of test fall windows assigned to each KMeans severity cluster. |
| `kmeans_severity_test_feature_scatter.png` | Shows test fall-window cluster separation using `acc_mag_peak` and `gyro_mag_peak`. |
| `kmeans_severity_test_pca.png` | Shows test fall-window cluster separation in PCA space. PCA is fit on train fall features and applied to test features. |

Test fall-window cluster counts:

| Severity | Test Windows |
| --- | ---: |
| Mild | 457 |
| Moderate | 304 |
| Severe | 439 |

## CNN-LSTM Severity Evaluation Plot

| Plot | Purpose |
| --- | --- |
| `cnn_lstm_severity_evaluation_metrics.png` | Summarizes CNN-LSTM severity-wise window recall, window precision/F1, recording detection, impact verification, and subject recall at threshold 0.50. |

Core CNN-LSTM severity metrics:

| Severity | Windows | Detected Windows | Window Recall | Recording Detection | Impact Verified |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncertain | 24 | 23 | 0.9583 | 12/13 | 12/13 |
| Mild | 446 | 440 | 0.9865 | 121/121 | 121/121 |
| Moderate | 298 | 298 | 1.0000 | 79/79 | 79/79 |
| Severe | 432 | 430 | 0.9954 | 111/111 | 111/111 |

Overall CNN-LSTM threshold 0.50 metrics:

| Metric | Value |
| --- | ---: |
| Window precision | 0.9770 |
| Window recall | 0.9925 |
| Window F1 | 0.9847 |
| Fall recording recall | 1.0000 |
| Impact-verified recordings | 300/300 |
| ADL false-trigger rate | 0.0000 |

