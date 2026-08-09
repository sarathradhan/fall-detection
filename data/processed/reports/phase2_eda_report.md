# Phase 2 EDA Report

This report uses only the processed window datasets in `data/processed/`.

## Dataset Overview

| Split | Windows | Shape | Subjects | Recordings |
| --- | ---: | --- | ---: | ---: |
| train | 55,423 | `(55423, 64, 6)` | 26 | 2,942 |
| val | 15,073 | `(15073, 64, 6)` | 6 | 829 |
| test | 13,627 | `(13627, 64, 6)` | 6 | 734 |

## 1. Class Distribution

| split | class_label | class_name | window_count | percentage | total_windows |
| --- | --- | --- | --- | --- | --- |
| train | 0 | ADL | 50939 | 91.9095 | 55423 |
| train | 1 | Fall | 4484 | 8.0905 | 55423 |
| val | 0 | ADL | 13573 | 90.0484 | 15073 |
| val | 1 | Fall | 1500 | 9.9516 | 15073 |
| test | 0 | ADL | 12427 | 91.1940 | 13627 |
| test | 1 | Fall | 1200 | 8.8060 | 13627 |

Interpretation: Across all processed splits there are 84,123 windows. Falls account for 8.54% overall, with split-level fall percentages ranging from 8.09% to 9.95%. This indicates the binary task is imbalanced but both classes are represented in train, validation, and test.

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
| val | All | acc1_x | -0.0917 | -0.0360 | 1.0845 | 1.1762 | 1.0884 | -12.0087 | 11.1325 | -0.1827 | 4.4387 | 964672 |
| val | All | acc1_y | 0.0097 | -0.3821 | 1.0422 | 1.0862 | 1.0422 | -7.2551 | 6.5152 | 0.2150 | 0.6372 | 964672 |
| val | All | acc1_z | 0.3248 | 0.3559 | 1.0002 | 1.0004 | 1.0516 | -8.2031 | 9.2057 | -0.2058 | 1.2643 | 964672 |
| val | All | gyro_x | -0.0013 | 0.0036 | 1.1744 | 1.3791 | 1.1744 | -18.9812 | 17.5454 | 0.2538 | 21.3425 | 964672 |
| val | All | gyro_y | 0.0113 | 0.0445 | 1.0933 | 1.1953 | 1.0933 | -11.7561 | 18.5895 | 0.2090 | 16.5398 | 964672 |
| val | All | gyro_z | 0.0147 | 0.0044 | 1.1315 | 1.2804 | 1.1316 | -16.5250 | 15.0641 | 0.2203 | 21.9776 | 964672 |
| test | All | acc1_x | -0.0408 | 0.0186 | 1.0060 | 1.0121 | 1.0069 | -11.2061 | 9.3332 | -0.3529 | 4.5712 | 872128 |
| test | All | acc1_y | -0.0574 | -0.4161 | 1.0091 | 1.0183 | 1.0108 | -5.6359 | 6.7121 | 0.3524 | 0.4029 | 872128 |
| test | All | acc1_z | 0.1501 | 0.1188 | 0.9994 | 0.9988 | 1.0106 | -9.0726 | 11.0164 | 0.1224 | 0.9501 | 872128 |
| test | All | gyro_x | 0.0414 | 0.0251 | 1.1941 | 1.4260 | 1.1949 | -48.5212 | 18.8698 | -0.5217 | 40.2934 | 872128 |
| test | All | gyro_y | -0.0270 | -0.0200 | 1.0486 | 1.0996 | 1.0489 | -15.2681 | 15.1227 | 0.0128 | 13.2533 | 872128 |
| test | All | gyro_z | 0.0081 | -0.0005 | 1.1650 | 1.3572 | 1.1650 | -15.8501 | 20.0049 | 0.6195 | 20.6260 | 872128 |

Interpretation: Channel means remain close to zero after normalization (largest absolute split/channel mean: 0.325). Standard deviations range from 0.990 to 1.194, showing usable variation across all channels. The largest absolute normalized value is 48.521; extreme tails exist but are expected for fall-impact windows.

## 3. Acceleration Spike Analysis

| split | class_name | windows | peak_mean | peak_median | peak_std | peak_p95 | peak_max | magnitude_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | ADL | 50939 | 2.0959 | 1.8840 | 1.0275 | 3.7615 | 9.2613 | 1.3662 |
| train | Fall | 4484 | 7.1664 | 7.0617 | 1.8429 | 10.4178 | 15.3133 | 2.1717 |
| val | ADL | 13573 | 2.1513 | 1.8995 | 1.0571 | 3.9903 | 8.0117 | 1.4574 |
| val | Fall | 1500 | 7.0716 | 6.8137 | 1.5904 | 10.0768 | 12.7154 | 2.1761 |
| test | ADL | 12427 | 2.0820 | 1.8413 | 0.9642 | 3.5975 | 7.6807 | 1.3940 |
| test | Fall | 1200 | 6.6686 | 6.5642 | 1.4786 | 9.5137 | 11.7798 | 2.1662 |

Interpretation: Acceleration magnitude is computed from the normalized accelerometer channels, so it is a normalized spike proxy. train: Fall median peak 7.062 vs ADL 1.884; val: Fall median peak 6.814 vs ADL 1.900; test: Fall median peak 6.564 vs ADL 1.841. Fall median peaks are higher in train, val, test. Spike magnitude still shows class-dependent distribution differences, but it should be used with temporal patterns rather than as a standalone rule.

## 4. Autocorrelation

Autocorrelation is averaged across processed 64-step windows for each channel and split.

Interpretation: Mean lag-1 autocorrelation is 0.771, while mean lag-8 autocorrelation is 0.063. The decay across lags shows short-range temporal dependency inside the 64-step windows, which supports using temporal kernels or recurrent layers.

## 5. Outlier Detection

| split | channel | sample_count | iqr_outliers | iqr_outlier_percentage | iqr_lower_bound | iqr_upper_bound | zscore_outliers | zscore_outlier_percentage | zscore_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | acc1_x | 3547072 | 608827 | 17.1642 | -0.8676 | 0.9144 | 15934 | 0.4492 | 3.0000 |
| train | acc1_y | 3547072 | 44835 | 1.2640 | -2.6111 | 2.8744 | 21348 | 0.6018 | 3.0000 |
| train | acc1_z | 3547072 | 242806 | 6.8453 | -2.1275 | 1.9780 | 6566 | 0.1851 | 3.0000 |
| train | gyro_x | 3547072 | 678660 | 19.1330 | -0.8246 | 0.8621 | 77014 | 2.1712 | 3.0000 |
| train | gyro_y | 3547072 | 904997 | 25.5139 | -0.6631 | 0.6695 | 75297 | 2.1228 | 3.0000 |
| train | gyro_z | 3547072 | 1010394 | 28.4853 | -0.5425 | 0.5392 | 52181 | 1.4711 | 3.0000 |
| val | acc1_x | 964672 | 184146 | 19.0890 | -1.0319 | 0.9376 | 3332 | 0.3454 | 3.0000 |
| val | acc1_y | 964672 | 8108 | 0.8405 | -2.8435 | 3.2271 | 6259 | 0.6488 | 3.0000 |
| val | acc1_z | 964672 | 44289 | 4.5911 | -1.8283 | 2.4880 | 2200 | 0.2281 | 3.0000 |
| val | gyro_x | 964672 | 212543 | 22.0327 | -0.7434 | 0.7784 | 21805 | 2.2604 | 3.0000 |
| val | gyro_y | 964672 | 276334 | 28.6454 | -0.5774 | 0.6087 | 21370 | 2.2153 | 3.0000 |
| val | gyro_z | 964672 | 314317 | 32.5828 | -0.4453 | 0.4591 | 20859 | 2.1623 | 3.0000 |
| test | acc1_x | 872128 | 158809 | 18.2094 | -0.9026 | 0.9484 | 3431 | 0.3934 | 3.0000 |
| test | acc1_y | 872128 | 10914 | 1.2514 | -2.5685 | 2.7101 | 3800 | 0.4357 | 3.0000 |
| test | acc1_z | 872128 | 30792 | 3.5307 | -2.0903 | 2.3329 | 1391 | 0.1595 | 3.0000 |
| test | gyro_x | 872128 | 178406 | 20.4564 | -0.8304 | 0.9080 | 21267 | 2.4385 | 3.0000 |
| test | gyro_y | 872128 | 247944 | 28.4298 | -0.6821 | 0.6542 | 17613 | 2.0195 | 3.0000 |
| test | gyro_z | 872128 | 280301 | 32.1399 | -0.5190 | 0.5230 | 18475 | 2.1184 | 3.0000 |

Interpretation: IQR detects the highest outlier rate in val gyro_z (32.58%). Z-score detects the highest outlier rate in test gyro_x (2.44%). No values were removed; these points are retained because high-amplitude motion is likely informative for fall detection.

## 6. Final EDA Summary

Processed tensor shapes are train: (55423, 64, 6), val: (15073, 64, 6), test: (13627, 64, 6). Missing values: 0; infinite values: 0. All splits contain both classes: True. Maximum Z-score outlier percentage is 2.44%, and the largest absolute normalized mean is 0.325. Based on these checks, the processed data is ready for CNN/CNN-LSTM training.

Plots were saved to `data/processed/plots/`. Statistics and this report were saved to `data/processed/reports/`.