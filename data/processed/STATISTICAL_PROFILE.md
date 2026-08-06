# SisFall Windowed Dataset — Full Statistical Profile

Generated from saved arrays in `data/processed/` and SisFall source files.
Impact-centered labeling (`impact_method=peak`). Window size=64, stride=16, sampling rate=20 Hz.

## Computation Reference

| Section | Source | Method |
| --- | --- | --- |
| Overview counts | `*.npy`, `*_recording_ids.npy`, `*_subject_ids.npy` | `np.load`, unique counts |
| Demographics | `SisFall_dataset/Readme.txt` subject table | Joined by `subject_id` per split |
| Recording duration | SisFall `*.txt` files | Line count / 10 downsampling / 20 Hz |
| Signal stats (raw) | Normalized windows + `scaler.pkl` | `scaler.inverse_transform` on flattened timesteps |
| Magnitude stats | Normalized windows | `max_t sqrt(sum(channel^2))` per window |
| Normalization check | Normalized windows | Per-channel mean/std across all timesteps in split |
| Overlap analysis | Labels + recording IDs + raw SisFall acc | Impact index on raw acc; count label=1 vs 0 per fall recording |

---

## 1. Dataset Overview

### 1.1 Split Summary

| split | subjects | recordings | windows | window_shape |
| --- | --- | --- | --- | --- |
| train | 26 | 2942 | 55423 | (55423, 64, 6) |
| val | 6 | 829 | 15073 | (15073, 64, 6) |
| test | 6 | 734 | 13627 | (13627, 64, 6) |

### 1.2 Subject Demographics per Split

Demographics from official SisFall README; joined to subjects present in each split.

| split | subjects | SA_count | SE_count | age_mean | age_min | age_max | height_cm_mean | weight_kg_mean | female | male |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 26 | 14 | 12 | 43.15 | 19 | 75 | 164.62 | 64.87 | 11 | 15 |
| val | 6 | 5 | 1 | 29.67 | 19 | 64 | 160.83 | 57.25 | 5 | 1 |
| test | 6 | 4 | 2 | 37.67 | 19 | 67 | 164.83 | 56.25 | 3 | 3 |

### 1.3 Recording Duration by Activity Code (20 Hz downsampled)

Computed as `(raw_lines // 10) / 20` from each SisFall `.txt` file.

| activity_code | recordings | min | mean | median | max |
| --- | --- | --- | --- | --- | --- |
| D01 | 38 | 99.950 | 102.104 | 100.000 | 180.000 |
| D02 | 38 | 100.000 | 102.108 | 100.000 | 180.000 |
| D03 | 38 | 99.950 | 102.105 | 100.000 | 180.000 |
| D04 | 36 | 100.000 | 102.501 | 100.000 | 180.000 |
| D05 | 190 | 24.950 | 25.001 | 25.000 | 25.050 |
| D06 | 118 | 25.000 | 25.001 | 25.000 | 25.050 |
| D07 | 190 | 12.000 | 12.001 | 12.000 | 12.050 |
| D08 | 190 | 12.000 | 12.001 | 12.000 | 12.050 |
| D09 | 185 | 12.000 | 12.000 | 12.000 | 12.050 |
| D10 | 185 | 12.000 | 12.000 | 12.000 | 12.050 |
| D11 | 190 | 10.000 | 11.959 | 12.000 | 12.050 |
| D12 | 190 | 12.000 | 12.000 | 12.000 | 12.050 |
| D13 | 120 | 12.000 | 12.000 | 12.000 | 12.000 |
| D14 | 190 | 12.000 | 12.000 | 12.000 | 12.000 |
| D15 | 190 | 10.000 | 11.947 | 12.000 | 12.000 |
| D16 | 190 | 10.000 | 11.948 | 12.000 | 12.050 |
| D17 | 189 | 16.050 | 24.372 | 25.000 | 27.050 |
| D18 | 120 | 12.000 | 12.000 | 12.000 | 12.000 |
| D19 | 120 | 12.000 | 12.000 | 12.000 | 12.000 |
| F01 | 119 | 14.950 | 15.000 | 15.000 | 15.000 |
| F02 | 120 | 15.000 | 15.000 | 15.000 | 15.000 |
| F03 | 120 | 15.000 | 15.000 | 15.000 | 15.050 |
| F04 | 120 | 15.000 | 15.000 | 15.000 | 15.000 |
| F05 | 120 | 15.000 | 15.000 | 15.000 | 15.000 |
| F06 | 120 | 14.950 | 15.000 | 15.000 | 15.000 |
| F07 | 120 | 15.000 | 15.000 | 15.000 | 15.000 |
| F08 | 120 | 15.000 | 15.000 | 15.000 | 15.000 |
| F09 | 120 | 15.000 | 15.000 | 15.000 | 15.000 |
| F10 | 119 | 15.000 | 15.000 | 15.000 | 15.000 |
| F11 | 120 | 14.950 | 15.000 | 15.000 | 15.000 |
| F12 | 120 | 15.000 | 15.000 | 15.000 | 15.000 |
| F13 | 120 | 15.000 | 15.000 | 15.000 | 15.000 |
| F14 | 120 | 15.000 | 15.000 | 15.000 | 15.000 |
| F15 | 120 | 14.500 | 14.996 | 15.000 | 15.050 |

### 1.4 Window Count by Activity Code and Split

| split | activity_code | windows_total | windows_fall_label | windows_adl_label |
| --- | --- | --- | --- | --- |
| train | D01 | 3271 | 0 | 3271 |
| train | D02 | 3272 | 0 | 3272 |
| train | D03 | 3272 | 0 | 3272 |
| train | D04 | 3040 | 0 | 3040 |
| train | D05 | 3640 | 0 | 3640 |
| train | D06 | 2044 | 0 | 2044 |
| train | D07 | 1560 | 0 | 1560 |
| train | D08 | 1560 | 0 | 1560 |
| train | D09 | 1500 | 0 | 1500 |
| train | D10 | 1500 | 0 | 1500 |
| train | D11 | 1548 | 0 | 1548 |
| train | D12 | 1560 | 0 | 1560 |
| train | D13 | 900 | 0 | 900 |
| train | D14 | 1560 | 0 | 1560 |
| train | D15 | 1545 | 0 | 1545 |
| train | D16 | 1545 | 0 | 1545 |
| train | D17 | 3461 | 0 | 3461 |
| train | D18 | 900 | 0 | 900 |
| train | D19 | 900 | 0 | 900 |
| train | F01 | 1110 | 295 | 815 |
| train | F02 | 1125 | 300 | 825 |
| train | F03 | 1125 | 300 | 825 |
| train | F04 | 1125 | 299 | 826 |
| train | F05 | 1125 | 299 | 826 |
| train | F06 | 1125 | 299 | 826 |
| train | F07 | 1125 | 300 | 825 |
| train | F08 | 1125 | 300 | 825 |
| train | F09 | 1125 | 300 | 825 |
| train | F10 | 1110 | 296 | 814 |
| train | F11 | 1125 | 300 | 825 |
| train | F12 | 1125 | 299 | 826 |
| train | F13 | 1125 | 300 | 825 |
| train | F14 | 1125 | 298 | 827 |
| train | F15 | 1125 | 299 | 826 |
| val | D01 | 732 | 0 | 732 |
| val | D02 | 732 | 0 | 732 |
| val | D03 | 732 | 0 | 732 |
| val | D04 | 732 | 0 | 732 |
| val | D05 | 840 | 0 | 840 |
| val | D06 | 700 | 0 | 700 |
| val | D07 | 360 | 0 | 360 |
| val | D08 | 360 | 0 | 360 |
| val | D09 | 360 | 0 | 360 |
| val | D10 | 360 | 0 | 360 |
| val | D11 | 360 | 0 | 360 |
| val | D12 | 360 | 0 | 360 |
| val | D13 | 300 | 0 | 300 |
| val | D14 | 360 | 0 | 360 |
| val | D15 | 360 | 0 | 360 |
| val | D16 | 360 | 0 | 360 |
| val | D17 | 840 | 0 | 840 |
| val | D18 | 300 | 0 | 300 |
| val | D19 | 300 | 0 | 300 |
| val | F01 | 375 | 100 | 275 |
| val | F02 | 375 | 100 | 275 |
| val | F03 | 375 | 100 | 275 |
| val | F04 | 375 | 100 | 275 |
| val | F05 | 375 | 100 | 275 |
| val | F06 | 375 | 100 | 275 |
| val | F07 | 375 | 100 | 275 |
| val | F08 | 375 | 100 | 275 |
| val | F09 | 375 | 100 | 275 |
| val | F10 | 375 | 100 | 275 |
| val | F11 | 375 | 100 | 275 |
| val | F12 | 375 | 100 | 275 |
| val | F13 | 375 | 100 | 275 |
| val | F14 | 375 | 100 | 275 |
| val | F15 | 375 | 100 | 275 |
| test | D01 | 732 | 0 | 732 |
| test | D02 | 732 | 0 | 732 |
| test | D03 | 731 | 0 | 731 |
| test | D04 | 732 | 0 | 732 |
| test | D05 | 840 | 0 | 840 |
| test | D06 | 560 | 0 | 560 |
| test | D07 | 360 | 0 | 360 |
| test | D08 | 360 | 0 | 360 |
| test | D09 | 360 | 0 | 360 |
| test | D10 | 360 | 0 | 360 |
| test | D11 | 360 | 0 | 360 |
| test | D12 | 360 | 0 | 360 |
| test | D13 | 240 | 0 | 240 |
| test | D14 | 360 | 0 | 360 |
| test | D15 | 360 | 0 | 360 |
| test | D16 | 360 | 0 | 360 |
| test | D17 | 840 | 0 | 840 |
| test | D18 | 240 | 0 | 240 |
| test | D19 | 240 | 0 | 240 |
| test | F01 | 300 | 80 | 220 |
| test | F02 | 300 | 80 | 220 |
| test | F03 | 300 | 80 | 220 |
| test | F04 | 300 | 80 | 220 |
| test | F05 | 300 | 80 | 220 |
| test | F06 | 300 | 80 | 220 |
| test | F07 | 300 | 80 | 220 |
| test | F08 | 300 | 80 | 220 |
| test | F09 | 300 | 80 | 220 |
| test | F10 | 300 | 80 | 220 |
| test | F11 | 300 | 80 | 220 |
| test | F12 | 300 | 80 | 220 |
| test | F13 | 300 | 80 | 220 |
| test | F14 | 300 | 80 | 220 |
| test | F15 | 300 | 80 | 220 |

---

## 2. Class Distribution

### 2.1 Fall vs ADL Windows

| split | adl_windows | fall_windows | total | fall_pct | adl_pct |
| --- | --- | --- | --- | --- | --- |
| train | 50939 | 4484 | 55423 | 8.09 | 91.91 |
| val | 13573 | 1500 | 15073 | 9.95 | 90.05 |
| test | 12427 | 1200 | 13627 | 8.81 | 91.19 |

### 2.2 Fall Windows by Fall Code (label=1 only)

| split | fall_code | fall_windows |
| --- | --- | --- |
| train | F01 | 295 |
| train | F02 | 300 |
| train | F03 | 300 |
| train | F04 | 299 |
| train | F05 | 299 |
| train | F06 | 299 |
| train | F07 | 300 |
| train | F08 | 300 |
| train | F09 | 300 |
| train | F10 | 296 |
| train | F11 | 300 |
| train | F12 | 299 |
| train | F13 | 300 |
| train | F14 | 298 |
| train | F15 | 299 |
| val | F01 | 100 |
| val | F02 | 100 |
| val | F03 | 100 |
| val | F04 | 100 |
| val | F05 | 100 |
| val | F06 | 100 |
| val | F07 | 100 |
| val | F08 | 100 |
| val | F09 | 100 |
| val | F10 | 100 |
| val | F11 | 100 |
| val | F12 | 100 |
| val | F13 | 100 |
| val | F14 | 100 |
| val | F15 | 100 |
| test | F01 | 80 |
| test | F02 | 80 |
| test | F03 | 80 |
| test | F04 | 80 |
| test | F05 | 80 |
| test | F06 | 80 |
| test | F07 | 80 |
| test | F08 | 80 |
| test | F09 | 80 |
| test | F10 | 80 |
| test | F11 | 80 |
| test | F12 | 80 |
| test | F13 | 80 |
| test | F14 | 80 |
| test | F15 | 80 |

### 2.3 ADL Recording Windows by Activity Code (all windows from D01–D19 recordings)

| split | adl_code | windows |
| --- | --- | --- |
| train | D01 | 3271 |
| train | D02 | 3272 |
| train | D03 | 3272 |
| train | D04 | 3040 |
| train | D05 | 3640 |
| train | D06 | 2044 |
| train | D07 | 1560 |
| train | D08 | 1560 |
| train | D09 | 1500 |
| train | D10 | 1500 |
| train | D11 | 1548 |
| train | D12 | 1560 |
| train | D13 | 900 |
| train | D14 | 1560 |
| train | D15 | 1545 |
| train | D16 | 1545 |
| train | D17 | 3461 |
| train | D18 | 900 |
| train | D19 | 900 |
| val | D01 | 732 |
| val | D02 | 732 |
| val | D03 | 732 |
| val | D04 | 732 |
| val | D05 | 840 |
| val | D06 | 700 |
| val | D07 | 360 |
| val | D08 | 360 |
| val | D09 | 360 |
| val | D10 | 360 |
| val | D11 | 360 |
| val | D12 | 360 |
| val | D13 | 300 |
| val | D14 | 360 |
| val | D15 | 360 |
| val | D16 | 360 |
| val | D17 | 840 |
| val | D18 | 300 |
| val | D19 | 300 |
| test | D01 | 732 |
| test | D02 | 732 |
| test | D03 | 731 |
| test | D04 | 732 |
| test | D05 | 840 |
| test | D06 | 560 |
| test | D07 | 360 |
| test | D08 | 360 |
| test | D09 | 360 |
| test | D10 | 360 |
| test | D11 | 360 |
| test | D12 | 360 |
| test | D13 | 240 |
| test | D14 | 360 |
| test | D15 | 360 |
| test | D16 | 360 |
| test | D17 | 840 |
| test | D18 | 240 |
| test | D19 | 240 |

### 2.4 Background Windows from Fall Recordings (label=0, F-code recordings)

| split | background_windows_from_fall_recordings |
| --- | --- |
| train | 12361 |
| val | 4125 |
| test | 3300 |

### 2.5 Windows per Subject

| split | subjects | min | mean | median | max |
| --- | --- | --- | --- | --- | --- |
| train | 26 | 1013 | 2131.65 | 2739.00 | 2753 |
| val | 6 | 1308 | 2512.17 | 2753.00 | 2753 |
| test | 6 | 1307 | 2271.17 | 2753.00 | 2753 |


Subjects with >2× median window count:

| split | subject_id | windows | ratio_to_median |
| --- | --- | --- | --- |

---

## 3. Signal Statistics (Raw, Pre-Normalization)

Computed by applying `scaler.inverse_transform()` to all timesteps in saved normalized windows.

### 3.1 Sampling Rate and Windowed Timestep Duration

| split | timesteps_in_windows | duration_s_at_20hz | sampling_rate_hz |
| --- | --- | --- | --- |
| train | 3547072 | 177353.6 | 20.0 |
| val | 964672 | 48233.6 | 20.0 |
| test | 872128 | 43606.4 | 20.0 |

### 3.2 Raw Channel Stats — train

| split | class | channel | mean | std | min | p5 | p25 | p50 | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | ADL | acc1_x | 1.916 | 86.862 | -839.755 | -196.209 | -16.128 | 3.917 | 23.185 | 181.250 | 260.552 | 572.046 |
| train | ADL | acc1_y | -188.949 | 130.689 | -927.226 | -361.208 | -253.753 | -233.400 | -101.359 | 40.936 | 96.496 | 547.123 |
| train | ADL | acc1_z | -41.577 | 108.360 | -796.697 | -229.107 | -102.806 | -47.096 | 10.620 | 196.680 | 240.876 | 631.821 |
| train | ADL | gyro_x | -9.315 | 324.097 | -13295.619 | -473.946 | -74.563 | -11.468 | 68.880 | 454.363 | 949.882 | 13728.685 |
| train | ADL | gyro_y | 37.323 | 420.609 | -12933.272 | -583.141 | -29.146 | 37.386 | 105.757 | 645.742 | 1312.646 | 12078.562 |
| train | ADL | gyro_z | -6.724 | 298.670 | -12142.513 | -436.689 | -45.333 | -4.791 | 32.946 | 418.485 | 810.983 | 11986.978 |
| train | Fall | acc1_x | -9.349 | 153.085 | -1293.534 | -253.710 | -60.768 | 0.064 | 39.859 | 253.060 | 355.995 | 1083.288 |
| train | Fall | acc1_y | -83.234 | 151.828 | -1101.451 | -285.762 | -216.213 | -58.799 | 28.896 | 133.443 | 252.456 | 749.847 |
| train | Fall | acc1_z | -45.666 | 163.997 | -1231.977 | -264.745 | -158.574 | -56.681 | 55.427 | 231.751 | 276.389 | 978.760 |
| train | Fall | gyro_x | -39.181 | 807.218 | -13295.619 | -1241.882 | -234.511 | -21.519 | 110.604 | 1140.058 | 2845.657 | 11035.623 |
| train | Fall | gyro_y | 8.878 | 829.478 | -7844.202 | -1299.303 | -147.421 | 27.954 | 176.470 | 1260.564 | 2659.976 | 6978.537 |
| train | Fall | gyro_z | 7.675 | 658.588 | -5900.186 | -778.809 | -100.349 | -4.798 | 93.086 | 848.015 | 2523.493 | 6136.123 |

### 3.2 Raw Channel Stats — val

| split | class | channel | mean | std | min | p5 | p25 | p50 | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | ADL | acc1_x | -6.695 | 93.333 | -721.370 | -234.974 | -24.294 | -2.296 | 18.809 | 196.602 | 254.312 | 620.577 |
| val | ADL | acc1_y | -187.560 | 133.727 | -920.873 | -356.293 | -255.548 | -234.171 | -86.577 | 46.597 | 105.472 | 479.105 |
| val | ADL | acc1_z | 0.709 | 108.591 | -659.982 | -210.936 | -59.364 | 3.874 | 60.424 | 210.322 | 244.678 | 632.615 |
| val | ADL | gyro_x | -10.108 | 352.899 | -5215.025 | -546.329 | -67.070 | -10.475 | 59.950 | 496.426 | 1114.743 | 4206.713 |
| val | ADL | gyro_y | 41.615 | 421.004 | -4498.893 | -614.773 | -14.913 | 55.784 | 101.282 | 671.176 | 1341.183 | 7268.773 |
| val | ADL | gyro_z | -3.624 | 309.527 | -4910.082 | -495.483 | -34.871 | -4.090 | 26.487 | 485.444 | 989.077 | 4407.907 |
| val | Fall | acc1_x | -15.635 | 153.980 | -1117.932 | -256.176 | -64.434 | -3.709 | 31.980 | 246.655 | 353.126 | 1038.215 |
| val | Fall | acc1_y | -86.275 | 155.313 | -1148.971 | -298.553 | -221.401 | -54.835 | 26.872 | 135.004 | 252.178 | 692.465 |
| val | Fall | acc1_z | -14.679 | 161.707 | -981.954 | -267.543 | -115.491 | -9.829 | 83.282 | 238.874 | 286.164 | 1020.919 |
| val | Fall | gyro_x | -33.055 | 818.228 | -6846.724 | -1271.106 | -204.531 | -13.821 | 113.734 | 1170.331 | 2843.159 | 6305.901 |
| val | Fall | gyro_y | 32.606 | 829.482 | -5099.660 | -1224.803 | -121.634 | 46.896 | 185.463 | 1216.375 | 2963.364 | 8156.285 |
| val | Fall | gyro_z | 24.508 | 662.020 | -5270.270 | -712.527 | -86.800 | -4.150 | 101.188 | 887.123 | 2620.634 | 4793.853 |

### 3.2 Raw Channel Stats — test

| split | class | channel | mean | std | min | p5 | p25 | p50 | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | ADL | acc1_x | -1.516 | 86.388 | -542.514 | -214.775 | -16.433 | 2.792 | 23.960 | 139.315 | 256.216 | 627.930 |
| test | ADL | acc1_y | -196.079 | 128.669 | -932.440 | -367.656 | -258.940 | -238.902 | -124.067 | 35.697 | 85.046 | 536.934 |
| test | ADL | acc1_z | -20.679 | 109.639 | -699.953 | -213.752 | -85.066 | -24.100 | 36.494 | 192.134 | 243.801 | 428.940 |
| test | ADL | gyro_x | 1.459 | 366.161 | -6575.531 | -521.331 | -70.624 | -2.719 | 76.424 | 541.674 | 1182.320 | 5139.677 |
| test | ADL | gyro_y | 23.910 | 408.834 | -4987.748 | -667.763 | -35.001 | 28.056 | 98.281 | 678.249 | 1203.470 | 5529.344 |
| test | ADL | gyro_z | -8.558 | 321.017 | -3362.865 | -526.420 | -43.040 | -5.721 | 29.769 | 498.579 | 966.906 | 4662.609 |
| test | Fall | acc1_x | -16.544 | 149.283 | -1043.149 | -253.678 | -82.460 | 0.992 | 40.330 | 236.216 | 340.938 | 870.568 |
| test | Fall | acc1_y | -86.810 | 156.390 | -865.083 | -300.003 | -225.718 | -61.215 | 32.766 | 141.328 | 254.247 | 718.794 |
| test | Fall | acc1_z | -23.486 | 160.109 | -1081.990 | -272.882 | -128.439 | -30.488 | 91.993 | 233.215 | 269.343 | 1229.231 |
| test | Fall | gyro_x | 18.905 | 843.128 | -17483.588 | -1095.263 | -176.171 | -6.207 | 145.143 | 1401.865 | 2853.290 | 6782.769 |
| test | Fall | gyro_y | 24.492 | 807.311 | -6633.823 | -1222.410 | -142.497 | 18.190 | 179.212 | 1235.804 | 2820.127 | 6641.884 |
| test | Fall | gyro_z | 55.401 | 702.483 | -5055.245 | -744.933 | -87.280 | -4.369 | 135.394 | 1141.946 | 2883.211 | 6367.964 |

---

## 4. Magnitude Statistics (Normalized Windows)

### 4.1 Peak Magnitude — train

| split | class | metric | mean | std | min | p5 | p25 | p50 | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | ADL | peak_accel | 2.096 | 1.027 | 0.503 | 0.679 | 1.284 | 1.884 | 2.918 | 3.762 | 4.973 | 9.261 |
| train | ADL | peak_gyro | 2.803 | 2.432 | 0.054 | 0.157 | 1.180 | 2.417 | 4.005 | 6.687 | 9.376 | 49.147 |
| train | Fall | peak_accel | 7.166 | 1.843 | 2.562 | 4.341 | 5.851 | 7.062 | 8.391 | 10.418 | 11.865 | 15.313 |
| train | Fall | peak_gyro | 11.739 | 3.501 | 1.999 | 5.890 | 9.448 | 11.752 | 14.090 | 16.959 | 19.912 | 37.110 |

### 4.1 Peak Magnitude — val

| split | class | metric | mean | std | min | p5 | p25 | p50 | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | ADL | peak_accel | 2.151 | 1.057 | 0.509 | 0.694 | 1.355 | 1.900 | 2.980 | 3.990 | 5.117 | 8.012 |
| val | ADL | peak_gyro | 2.821 | 2.323 | 0.062 | 0.126 | 0.701 | 2.467 | 4.088 | 7.250 | 9.304 | 17.762 |
| val | Fall | peak_accel | 7.072 | 1.590 | 2.540 | 4.868 | 6.041 | 6.814 | 7.944 | 10.077 | 11.866 | 12.715 |
| val | Fall | peak_gyro | 11.764 | 2.888 | 4.287 | 6.710 | 9.913 | 11.927 | 13.579 | 16.500 | 18.242 | 21.862 |

### 4.1 Peak Magnitude — test

| split | class | metric | mean | std | min | p5 | p25 | p50 | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | ADL | peak_accel | 2.082 | 0.964 | 0.501 | 0.702 | 1.359 | 1.841 | 2.897 | 3.597 | 4.555 | 7.681 |
| test | ADL | peak_gyro | 2.984 | 2.268 | 0.049 | 0.155 | 1.023 | 2.778 | 4.311 | 7.140 | 9.332 | 19.089 |
| test | Fall | peak_accel | 6.669 | 1.479 | 3.755 | 4.618 | 5.682 | 6.564 | 7.418 | 9.514 | 11.007 | 11.780 |
| test | Fall | peak_gyro | 11.329 | 4.173 | 2.504 | 5.812 | 8.618 | 11.393 | 13.159 | 17.076 | 20.125 | 48.682 |

### 4.2 Time-to-Peak Within Window (samples from window start)

| split | class | metric | mean | std | min | p5 | p25 | p50 | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | ADL | time_to_peak_accel_samples | 30.79 | 21.09 | 0.00 | 0.00 | 11.00 | 31.00 | 51.00 | 63.00 | 63.00 | 63.00 |
| train | ADL | time_to_peak_gyro_samples | 30.58 | 21.22 | 0.00 | 0.00 | 10.00 | 30.00 | 50.00 | 63.00 | 63.00 | 63.00 |
| train | Fall | time_to_peak_accel_samples | 31.98 | 18.47 | 0.00 | 3.00 | 16.00 | 32.00 | 48.00 | 61.00 | 63.00 | 63.00 |
| train | Fall | time_to_peak_gyro_samples | 31.03 | 18.45 | 0.00 | 2.00 | 15.00 | 31.00 | 47.00 | 60.00 | 63.00 | 63.00 |
| val | ADL | time_to_peak_accel_samples | 30.42 | 21.16 | 0.00 | 0.00 | 10.00 | 30.00 | 50.00 | 63.00 | 63.00 | 63.00 |
| val | ADL | time_to_peak_gyro_samples | 31.17 | 21.54 | 0.00 | 0.00 | 10.00 | 31.00 | 52.00 | 63.00 | 63.00 | 63.00 |
| val | Fall | time_to_peak_accel_samples | 31.77 | 18.52 | 0.00 | 2.00 | 16.00 | 32.00 | 48.00 | 61.00 | 63.00 | 63.00 |
| val | Fall | time_to_peak_gyro_samples | 31.04 | 18.52 | 0.00 | 2.00 | 15.00 | 31.00 | 47.00 | 60.00 | 63.00 | 63.00 |
| test | ADL | time_to_peak_accel_samples | 30.40 | 21.18 | 0.00 | 0.00 | 10.00 | 30.00 | 50.00 | 63.00 | 63.00 | 63.00 |
| test | ADL | time_to_peak_gyro_samples | 30.34 | 21.42 | 0.00 | 0.00 | 10.00 | 30.00 | 50.00 | 63.00 | 63.00 | 63.00 |
| test | Fall | time_to_peak_accel_samples | 31.55 | 18.48 | 0.00 | 3.00 | 16.00 | 32.00 | 48.00 | 60.00 | 63.00 | 63.00 |
| test | Fall | time_to_peak_gyro_samples | 30.16 | 18.46 | 0.00 | 2.00 | 14.00 | 30.00 | 46.00 | 59.00 | 63.00 | 63.00 |

---

## 5. Normalization Validation

Post-normalization mean/std across all timesteps in each split's windows.

| split | channel | mean | std | mean_abs_drift_from_0 | std_drift_from_1 |
| --- | --- | --- | --- | --- | --- |
| train | acc1_x | 0.0005 | 1.0091 | 0.0005 | 0.0091 |
| train | acc1_y | -0.0121 | 1.0142 | 0.0121 | 0.0142 |
| train | acc1_z | -0.0323 | 0.9899 | 0.0323 | 0.0101 |
| train | gyro_x | 0.0005 | 1.0732 | 0.0005 | 0.0732 |
| train | gyro_y | -0.0017 | 1.0696 | 0.0017 | 0.0696 |
| train | gyro_z | -0.0002 | 1.0741 | 0.0002 | 0.0741 |
| val | acc1_x | -0.0917 | 1.0845 | 0.0917 | 0.0845 |
| val | acc1_y | 0.0097 | 1.0422 | 0.0097 | 0.0422 |
| val | acc1_z | 0.3248 | 1.0002 | 0.3248 | 0.0002 |
| val | gyro_x | -0.0013 | 1.1744 | 0.0013 | 0.1744 |
| val | gyro_y | 0.0113 | 1.0933 | 0.0113 | 0.0933 |
| val | gyro_z | 0.0147 | 1.1315 | 0.0147 | 0.1315 |
| test | acc1_x | -0.0408 | 1.0060 | 0.0408 | 0.0060 |
| test | acc1_y | -0.0574 | 1.0091 | 0.0574 | 0.0091 |
| test | acc1_z | 0.1501 | 0.9994 | 0.1501 | 0.0006 |
| test | gyro_x | 0.0414 | 1.1941 | 0.0414 | 0.1941 |
| test | gyro_y | -0.0270 | 1.0486 | 0.0270 | 0.0486 |
| test | gyro_z | 0.0081 | 1.1650 | 0.0081 | 0.1650 |

---

## 6. Window Overlap / Redundancy (Fall Recordings)

### 6.1 Impact vs Background Windows per Fall Recording

| split | fall_recordings | impact_windows_total | background_windows_total | impact_windows_per_recording_mean | impact_windows_per_recording_median | impact_windows_per_recording_max | background_windows_per_recording_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1123 | 4484 | 12361 | 3.99 | 4.00 | 4 | 11.01 |
| val | 375 | 1500 | 4125 | 4.00 | 4.00 | 4 | 11.00 |
| test | 300 | 1200 | 3300 | 4.00 | 4.00 | 4 | 11.00 |

### 6.2 Effective Unique Fall Events

Each fall recording contributes 1 physical fall event; overlapping impact windows map to the same event.

| split | unique_fall_recordings_with_windows | recordings_with_ge_1_impact_window |
| --- | --- | --- |
| test | 300 | 300 |
| train | 1123 | 1123 |
| val | 375 | 375 |

---

## 7. Missing / Anomaly Checks

### 7.1 NaN / Inf Counts

| split | nan_total | inf_total | nan_acc1_x | nan_acc1_y | nan_acc1_z | nan_gyro_x | nan_gyro_y | nan_gyro_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 7.2 Recordings Excluded (< 64 downsampled samples)

Count: **0** (none appear in window arrays if excluded).

### 7.3 Flat-Line Windows (std < 1e-6 across all channels/times)

| split | flatline_windows |
| --- | --- |
| train | 0 |
| val | 0 |
| test | 0 |

---

## Anomalies

- No major anomalies flagged beyond known class imbalance.