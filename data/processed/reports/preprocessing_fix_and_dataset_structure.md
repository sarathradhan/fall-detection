# Preprocessing Fix and Final Dataset Structure

Generated: `2026-08-02T17:14:28Z`

## Purpose

This document records the preprocessing issue that was found before model training, the code changes made to fix it, the datasets that were regenerated, and the final structure of the corrected processed SisFall dataset.

No model training was performed.

## Summary of the Issue

A bug was found in the sliding-window generation step.

The old logic reset each recording group index and then used those reset indices to select rows from the full split-level feature array. Because every recording group started again at index `0`, later recordings could incorrectly reuse feature rows from the beginning of the split.

This meant saved windows could have had:

- Correct label
- Correct subject ID
- Correct recording ID
- Incorrect feature values inside the window

This issue explained why the earlier acceleration spike EDA looked suspicious: Fall windows did not consistently show higher acceleration peaks than ADL windows.

## Root Cause

Affected function:

```text
src/data/sisfall_preprocessing.py
```

Affected function:

```python
generate_sliding_windows()
```

Problematic behavior:

```python
group = frame.loc[frame["recording_id"] == recording_id].reset_index(drop=True)
group_indices = group.index.to_numpy()
feature_group = features[group_indices]
```

After `reset_index(drop=True)`, `group_indices` became local indices such as:

```text
0, 1, 2, 3, ...
```

Those local indices were then applied to `features`, which is indexed at the split level. That caused feature-window and metadata alignment to become unreliable.

## Fix Applied

The sliding-window function was changed so that feature lookup uses the original split-level row positions.

Corrected behavior:

```python
group = frame.loc[frame["recording_id"] == recording_id].copy()
group_indices = group.index.to_numpy()
feature_group = features[group_indices]
group = group.reset_index(drop=True)
```

This keeps the real row positions for `features[group_indices]`, while still allowing clean metadata access through `group.iloc[0]`.

## Additional Code Changes

### 1. Regression Test Added

A regression test was added to ensure each recording pulls its own feature rows during windowing.

File:

```text
tests/test_sisfall_pipeline.py
```

Test:

```python
test_sliding_windows_use_recording_row_positions()
```

The test creates two recordings with clearly different feature values:

- `rec_a`: values around `0, 1, 2, 3`
- `rec_b`: values around `100, 101, 102, 103`

It verifies that the second window contains the `rec_b` values, not repeated rows from `rec_a`.

### 2. Preprocessing Script Import Path Fixed

The full preprocessing script was updated so it can be run reliably from the Windows virtual environment.

File:

```text
scripts/run_full_preprocessing.py
```

Change:

```python
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

### 3. Filtering and Downsampling Runtime Improved

The filtering and downsampling loops were updated to use `groupby("recording_id", sort=True)` instead of repeatedly scanning the full dataframe for every recording.

This did not change the intended preprocessing math. It made full preprocessing practical to rerun.

Affected functions:

```python
apply_low_pass_filter()
downsample_recordings()
```

### 4. Filter Parameter Printout Corrected

The filtering log now prints the actual filter order and cutoff passed to the function.

The full preprocessing script uses:

```text
Butterworth low-pass filter
Order: 2
Cutoff: 5 Hz
```

## Verification Performed

### Unit Tests

Command run:

```text
.venv\Scripts\python.exe -m pytest tests
```

Result:

```text
3 passed
```

### Full Preprocessing Rerun

Command run:

```text
.venv\Scripts\python.exe scripts\run_full_preprocessing.py
```

Log:

```text
data/processed/preprocessing_fix_run.log
```

Result:

```text
Processed datasets regenerated successfully.
```

### EDA Rerun

Command run:

```text
.venv\Scripts\python.exe scripts\run_phase2_eda.py
```

Updated report:

```text
data/processed/reports/phase2_eda_report.md
```

After the fix, acceleration magnitude results became consistent with expectations:

| Split | ADL Median Peak | Fall Median Peak | Result |
| --- | ---: | ---: | --- |
| Train | 1.759 | 2.930 | Fall higher |
| Validation | 1.708 | 2.976 | Fall higher |
| Test | 1.882 | 2.921 | Fall higher |

This strongly supports that the original suspicious EDA result was caused by the window-alignment bug.

## Final Preprocessing Pipeline

The final processed dataset was generated using this sequence:

1. Load and merge all SisFall recordings.
2. Map activity codes to activity names.
3. Convert SisFall activity codes into binary labels.
4. Generate timestamps at the original sampling rate.
5. Select 6 IMU channels.
6. Apply Butterworth low-pass filtering.
7. Downsample from 200 Hz to 20 Hz.
8. Split subject-wise into train, validation, and test.
9. Normalize IMU channels using the training split scaler.
10. Generate sliding windows with correct feature/metadata alignment.
11. Save processed datasets, labels, metadata arrays, and scaler.

## Final Sensor Channels

Each processed window contains 6 channels:

| Index | Channel |
| ---: | --- |
| 0 | `acc1_x` |
| 1 | `acc1_y` |
| 2 | `acc1_z` |
| 3 | `gyro_x` |
| 4 | `gyro_y` |
| 5 | `gyro_z` |

## Label Mapping

| Label | Class | Meaning |
| ---: | --- | --- |
| 0 | ADL | Activity of daily living |
| 1 | Fall | Fall activity |

## Window Configuration

| Parameter | Value |
| --- | ---: |
| Sampling rate after downsampling | 20 Hz |
| Window size | 64 samples |
| Window duration | 3.2 seconds |
| Stride | 16 samples |
| Stride duration | 0.8 seconds |
| Channels per window | 6 |
| Final input shape per sample | `(64, 6)` |

## Final Dataset Files

All final processed dataset files are stored in:

```text
data/processed/
```

| File | Shape / Type | Description |
| --- | --- | --- |
| `train.npy` | `(55423, 64, 6)` | Training feature windows |
| `train_labels.npy` | `(55423,)` | Training binary labels |
| `train_subject_ids.npy` | `(55423,)` | Subject ID for each training window |
| `train_recording_ids.npy` | `(55423,)` | Recording ID for each training window |
| `val.npy` | `(12320, 64, 6)` | Validation feature windows |
| `val_labels.npy` | `(12320,)` | Validation binary labels |
| `val_subject_ids.npy` | `(12320,)` | Subject ID for each validation window |
| `val_recording_ids.npy` | `(12320,)` | Recording ID for each validation window |
| `test.npy` | `(10874, 64, 6)` | Test feature windows |
| `test_labels.npy` | `(10874,)` | Test binary labels |
| `test_subject_ids.npy` | `(10874,)` | Subject ID for each test window |
| `test_recording_ids.npy` | `(10874,)` | Recording ID for each test window |
| `scaler.pkl` | Pickle object | Fitted training scaler |

## Final Split Summary

| Split | Windows | Shape | Subjects | Recordings |
| --- | ---: | --- | ---: | ---: |
| Train | 55,423 | `(55423, 64, 6)` | 26 | 2,942 |
| Validation | 12,320 | `(12320, 64, 6)` | 5 | 675 |
| Test | 10,874 | `(10874, 64, 6)` | 5 | 580 |
| Total | 78,617 | `(windows, 64, 6)` | 36 | 4,197 |

## Final Class Distribution

| Split | ADL Windows | ADL % | Fall Windows | Fall % | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 38,578 | 69.61% | 16,845 | 30.39% | 55,423 |
| Validation | 7,820 | 63.47% | 4,500 | 36.53% | 12,320 |
| Test | 7,499 | 68.96% | 3,375 | 31.04% | 10,874 |
| Total | 53,897 | 68.56% | 24,720 | 31.44% | 78,617 |

## Final EDA Status

The EDA was regenerated after the preprocessing fix.

Main report:

```text
data/processed/reports/phase2_eda_report.md
```

Key result:

```text
Fall median acceleration peaks are higher than ADL median peaks in train, validation, and test.
```

Final EDA conclusion:

```text
The corrected processed data is ready for CNN/CNN-LSTM training.
```

## Important Notes Before Training

- Use input shape `(64, 6)` for CNN/CNN-LSTM models.
- The dataset is moderately imbalanced toward ADL.
- Consider class weights, balanced batches, or threshold tuning.
- Keep validation and test subject-wise separation unchanged.
- Do not use peak acceleration as the only decision rule; use the full multichannel temporal window.
- The regenerated arrays should be used instead of any arrays saved before the sliding-window fix.

## Files Changed

Code files changed:

```text
src/data/sisfall_preprocessing.py
scripts/run_full_preprocessing.py
tests/test_sisfall_pipeline.py
```

Regenerated data/report outputs:

```text
data/processed/train.npy
data/processed/train_labels.npy
data/processed/train_subject_ids.npy
data/processed/train_recording_ids.npy
data/processed/val.npy
data/processed/val_labels.npy
data/processed/val_subject_ids.npy
data/processed/val_recording_ids.npy
data/processed/test.npy
data/processed/test_labels.npy
data/processed/test_subject_ids.npy
data/processed/test_recording_ids.npy
data/processed/scaler.pkl
data/processed/reports/phase2_eda_report.md
data/processed/reports/*.csv
data/processed/plots/eda_*.png
```

