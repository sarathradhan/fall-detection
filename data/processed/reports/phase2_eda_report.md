# Phase 2 EDA Report

This report uses only the processed window datasets in `data/processed/`.

## Dataset Overview

| Split | Windows | Shape | Subjects | Recordings |
| --- | ---: | --- | ---: | ---: |
| train | 55,423 | `(55423, 64, 6)` | 26 | 2,942 |
| val | 12,320 | `(12320, 64, 6)` | 5 | 675 |
| test | 10,874 | `(10874, 64, 6)` | 5 | 580 |

## 1. Class Distribution

| split | class_label | class_name | window_count | percentage | total_windows |
| --- | --- | --- | --- | --- | --- |
| train | 0 | ADL | 38578 | 69.6065 | 55423 |
| train | 1 | Fall | 16845 | 30.3935 | 55423 |
| val | 0 | ADL | 7820 | 63.4740 | 12320 |
| val | 1 | Fall | 4500 | 36.5260 | 12320 |
| test | 0 | ADL | 7499 | 68.9627 | 10874 |
| test | 1 | Fall | 3375 | 31.0373 | 10874 |

Interpretation: Across all processed splits there are 78,617 windows. Falls account for 31.44% overall, with split-level fall percentages ranging from 30.39% to 36.53%. This indicates the binary task is imbalanced but both classes are represented in train, validation, and test.

## 2. Signal Health

Full signal statistics are saved to `eda_signal_health_statistics.csv`.

| split | class_name | channel | mean | median | std | variance | rms | min | max | skewness | kurtosis | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | All | acc1_x | 0.0005 | 0.0297 | 1.0091 | 1.0183 | 1.0091 | -13.8934 | 11.6163 | -0.1724 | 5.8957 | 3547072 |
| train | All | acc1_y | -0.0121 | -0.3779 | 1.0142 | 1.0286 | 1.0143 | -6.8997 | 6.9443 | 0.2028 | 0.7253 | 3547072 |
| train | All | acc1_z | -0.0323 | -0.0818 | 0.9899 | 0.9798 | 0.9904 | -10.3763 | 8.8393 | 0.3141 | 1.2219 | 3547072 |
| train | All | gyro_x | 0.0005 | 0.0001 | 1.0732 | 1.1517 | 1.0732 | -36.8906 | 38.1595 | 0.4505 | 38.5635 | 3547072 |
| train | All | gyro_y | -0.0017 | 0.0020 | 1.0696 | 1.1441 | 1.0696 | -29.6888 | 27.5684 | -0.1458 | 37.5338 | 3547072 |
| train | All | gyro_z | -0.0002 | 0.0022 | 1.0741 | 1.1536 | 1.0741 | -38.0956 | 37.6419 | -0.5106 | 101.8007 | 3547072 |
| val | All | acc1_x | -0.1033 | -0.0164 | 1.0538 | 1.1105 | 1.0588 | -12.0087 | 11.0391 | -0.3783 | 4.5942 | 788480 |
| val | All | acc1_y | 0.0180 | -0.3793 | 1.0489 | 1.1002 | 1.0491 | -7.2551 | 6.5152 | 0.1488 | 0.8078 | 788480 |
| val | All | acc1_z | 0.2206 | 0.2479 | 1.0063 | 1.0127 | 1.0303 | -8.2031 | 9.2057 | -0.1517 | 1.3656 | 788480 |
| val | All | gyro_x | 0.0030 | 0.0031 | 1.2076 | 1.4583 | 1.2076 | -18.9812 | 17.5454 | 0.1285 | 20.1957 | 788480 |
| val | All | gyro_y | -0.0112 | 0.0147 | 1.0781 | 1.1623 | 1.0782 | -11.7561 | 13.0458 | -0.2715 | 14.0866 | 788480 |
| val | All | gyro_z | 0.0087 | 0.0026 | 1.1122 | 1.2369 | 1.1122 | -16.5250 | 15.0641 | 0.1347 | 23.7464 | 788480 |
| test | All | acc1_x | -0.0303 | -0.0134 | 1.0075 | 1.0151 | 1.0080 | -11.2061 | 11.1325 | -0.1168 | 5.2588 | 695936 |
| test | All | acc1_y | -0.0856 | -0.4112 | 1.0026 | 1.0052 | 1.0062 | -5.6359 | 6.7121 | 0.3877 | 0.5910 | 695936 |
| test | All | acc1_z | 0.1878 | 0.1457 | 1.0176 | 1.0354 | 1.0348 | -8.2918 | 11.0164 | 0.0553 | 0.6703 | 695936 |
| test | All | gyro_x | 0.0037 | 0.0029 | 1.2297 | 1.5122 | 1.2297 | -48.5212 | 17.1683 | -0.9180 | 38.4374 | 695936 |
| test | All | gyro_y | 0.0043 | 0.0244 | 1.0791 | 1.1645 | 1.0791 | -15.2681 | 18.5895 | 0.5423 | 15.4205 | 695936 |
| test | All | gyro_z | 0.0123 | 0.0031 | 1.1600 | 1.3456 | 1.1601 | -15.4053 | 14.5097 | 0.3009 | 14.9796 | 695936 |

Interpretation: Channel means remain close to zero after normalization (largest absolute split/channel mean: 0.221). Standard deviations range from 0.990 to 1.230, showing usable variation across all channels. The largest absolute normalized value is 48.521; extreme tails exist but are expected for fall-impact windows.

## 3. Acceleration Spike Analysis

| split | class_name | windows | peak_mean | peak_median | peak_std | peak_p95 | peak_max | magnitude_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | ADL | 38578 | 2.0161 | 1.7589 | 1.0327 | 3.8400 | 5.8997 | 1.1636 |
| train | Fall | 16845 | 3.6283 | 2.9302 | 2.4769 | 8.7598 | 15.3133 | 2.0447 |
| val | ADL | 7820 | 2.0490 | 1.7084 | 1.1043 | 4.4202 | 5.9814 | 1.1777 |
| val | Fall | 4500 | 3.6039 | 2.9758 | 2.3825 | 8.2802 | 12.7154 | 2.0447 |
| test | ADL | 7499 | 2.0932 | 1.8824 | 0.9660 | 3.7808 | 6.0781 | 1.2685 |
| test | Fall | 3375 | 3.5242 | 2.9211 | 2.4085 | 7.9730 | 11.8657 | 1.9634 |

Interpretation: Acceleration magnitude is computed from the normalized accelerometer channels, so it is a normalized spike proxy. train: Fall median peak 2.930 vs ADL 1.759; val: Fall median peak 2.976 vs ADL 1.708; test: Fall median peak 2.921 vs ADL 1.882. Fall median peaks are higher in train, val, test. Spike magnitude still shows class-dependent distribution differences, but it should be used with temporal patterns rather than as a standalone rule.

## 4. Autocorrelation

Autocorrelation is averaged across processed 64-step windows for each channel and split.

Interpretation: Mean lag-1 autocorrelation is 0.768, while mean lag-8 autocorrelation is 0.063. The decay across lags shows short-range temporal dependency inside the 64-step windows, which supports using temporal kernels or recurrent layers.

## 5. Outlier Detection

| split | channel | sample_count | iqr_outliers | iqr_outlier_percentage | iqr_lower_bound | iqr_upper_bound | zscore_outliers | zscore_outlier_percentage | zscore_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | acc1_x | 3547072 | 608827 | 17.1642 | -0.8676 | 0.9144 | 15934 | 0.4492 | 3.0000 |
| train | acc1_y | 3547072 | 44835 | 1.2640 | -2.6111 | 2.8744 | 21348 | 0.6018 | 3.0000 |
| train | acc1_z | 3547072 | 242806 | 6.8453 | -2.1275 | 1.9780 | 6566 | 0.1851 | 3.0000 |
| train | gyro_x | 3547072 | 678660 | 19.1330 | -0.8246 | 0.8621 | 77014 | 2.1712 | 3.0000 |
| train | gyro_y | 3547072 | 904997 | 25.5139 | -0.6631 | 0.6695 | 75297 | 2.1228 | 3.0000 |
| train | gyro_z | 3547072 | 1010394 | 28.4853 | -0.5425 | 0.5392 | 52181 | 1.4711 | 3.0000 |
| val | acc1_x | 788480 | 144254 | 18.2952 | -1.0163 | 0.9385 | 2900 | 0.3678 | 3.0000 |
| val | acc1_y | 788480 | 7656 | 0.9710 | -2.8417 | 3.2623 | 5814 | 0.7374 | 3.0000 |
| val | acc1_z | 788480 | 61543 | 7.8053 | -1.8594 | 2.3445 | 1880 | 0.2384 | 3.0000 |
| val | gyro_x | 788480 | 169815 | 21.5370 | -0.7798 | 0.8212 | 18025 | 2.2860 | 3.0000 |
| val | gyro_y | 788480 | 216684 | 27.4812 | -0.6339 | 0.6406 | 18157 | 2.3028 | 3.0000 |
| val | gyro_z | 788480 | 249675 | 31.6654 | -0.4777 | 0.4795 | 16705 | 2.1186 | 3.0000 |
| test | acc1_x | 695936 | 118890 | 17.0835 | -0.9516 | 0.9766 | 2635 | 0.3786 | 3.0000 |
| test | acc1_y | 695936 | 17040 | 2.4485 | -2.3079 | 2.2545 | 3290 | 0.4727 | 3.0000 |
| test | acc1_z | 695936 | 2588 | 0.3719 | -2.3315 | 2.6743 | 1071 | 0.1539 | 3.0000 |
| test | gyro_x | 695936 | 147707 | 21.2242 | -0.8737 | 0.8968 | 17247 | 2.4782 | 3.0000 |
| test | gyro_y | 695936 | 197567 | 28.3887 | -0.6910 | 0.7050 | 13200 | 1.8967 | 3.0000 |
| test | gyro_z | 695936 | 216601 | 31.1237 | -0.5899 | 0.6096 | 14303 | 2.0552 | 3.0000 |

Interpretation: IQR detects the highest outlier rate in val gyro_z (31.67%). Z-score detects the highest outlier rate in test gyro_x (2.48%). No values were removed; these points are retained because high-amplitude motion is likely informative for fall detection.

## 6. Final EDA Summary

Processed tensor shapes are train: (55423, 64, 6), val: (12320, 64, 6), test: (10874, 64, 6). Missing values: 0; infinite values: 0. All splits contain both classes: True. Maximum Z-score outlier percentage is 2.48%, and the largest absolute normalized mean is 0.221. Based on these checks, the processed data is ready for CNN/CNN-LSTM training.

Plots were saved to `data/processed/plots/`. Statistics and this report were saved to `data/processed/reports/`.