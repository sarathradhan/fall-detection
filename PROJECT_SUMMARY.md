# SisFall Fall Detection Project Summary

## Project Overview
This project focuses on preparing the SisFall dataset for fall-detection modeling. The work completed so far includes dataset discovery, structured loading, preprocessing, validation, and generation of model-ready windowed arrays.

## Objectives Achieved
The main goals completed so far were:
- Set up a Python environment for data science and machine learning work.
- Load the SisFall dataset from raw text files.
- Build a preprocessing pipeline that transforms raw sensor recordings into a structured format for downstream modeling.
- Generate train, validation, and test datasets with sliding windows.
- Save processed artifacts for later model training.

## Environment Setup
The following environment steps were completed:
- Created and used a local virtual environment in the project folder.
- Installed necessary Python packages for data processing and visualization:
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - matplotlib
  - pytest

## Dataset Structure
The dataset used in this project is stored under the SisFall dataset folder.
Each activity file contains sensor readings for one recording and is organized by subject directory.
The pipeline was designed to:
- discover subject directories,
- read each activity file,
- parse the sensor values,
- preserve metadata such as subject ID, activity code, and source file name.

## Core Implementation Files
### Data Loader
The dataset loading logic was implemented in:
- [src/data/sisfall_loader.py](src/data/sisfall_loader.py)

This module is responsible for:
- resolving the dataset location,
- scanning subject folders,
- loading activity text files,
- converting them into pandas DataFrames,
- collecting metadata for each recording.

### Preprocessing Pipeline
The main preprocessing logic was implemented in:
- [src/data/sisfall_preprocessing.py](src/data/sisfall_preprocessing.py)

The pipeline includes the following stages:
1. Merge all activity files into a master dataframe.
2. Map activity codes to readable activity names.
3. Create binary labels for ADL vs fall data.
4. Generate timestamps per recording.
5. Validate the merged data.
6. Select the wearable IMU sensor channels.
7. Apply low-pass filtering to reduce noise.
8. Downsample recordings to a lower sampling rate.
9. Split data by subject into train, validation, and test sets.
10. Normalize features using the training split.
11. Generate sliding windows for temporal modeling.
12. Save processed datasets and scaler artifacts.

## Data Processing Results
The full preprocessing pipeline was run successfully on the complete SisFall dataset.

### Dataset scale
- Total rows in the merged master dataframe: 15,858,929
- Number of valid activity files loaded: 4,505
- Number of subjects discovered: 38

### Windowed dataset shapes
The final saved arrays have the following shapes:
- Train windows: $(55423, 64, 6)$
- Validation windows: $(12320, 64, 6)$
- Test windows: $(10874, 64, 6)$

Each window contains 6 sensor channels:
- acc1_x
- acc1_y
- acc1_z
- gyro_x
- gyro_y
- gyro_z

## Class Distribution
The processed windows were generated with label distribution as follows:
- Train: 38,578 ADL and 16,845 falls
- Validation: 7,820 ADL and 4,500 falls
- Test: 7,499 ADL and 3,375 falls

## Validation and Testing
Regression tests were created and executed to verify the preprocessing logic.
The relevant test files are:
- [tests/test_sisfall_preprocessing.py](tests/test_sisfall_preprocessing.py)
- [tests/test_sisfall_pipeline.py](tests/test_sisfall_pipeline.py)

Test result:
- 2 tests passed

## Generated Artifacts
The pipeline produced the following artifacts:
- Processed training windows: [data/processed/train.npy](data/processed/train.npy)
- Training labels: [data/processed/train_labels.npy](data/processed/train_labels.npy)
- Validation windows: [data/processed/val.npy](data/processed/val.npy)
- Validation labels: [data/processed/val_labels.npy](data/processed/val_labels.npy)
- Test windows: [data/processed/test.npy](data/processed/test.npy)
- Test labels: [data/processed/test_labels.npy](data/processed/test_labels.npy)
- Scaler artifact: [data/processed/scaler.pkl](data/processed/scaler.pkl)
- Filtering plot: [data/processed/filter_plot.png](data/processed/filter_plot.png)

Additional summary plots were also generated in:
- [data/processed/plots/class_balance.png](data/processed/plots/class_balance.png)
- [data/processed/plots/channel_boxplots.png](data/processed/plots/channel_boxplots.png)
- [data/processed/plots/train_first_window.png](data/processed/plots/train_first_window.png)
- [data/processed/plots/val_first_window.png](data/processed/plots/val_first_window.png)
- [data/processed/plots/test_first_window.png](data/processed/plots/test_first_window.png)

## Current Status
The project is now at the stage where the SisFall dataset has been:
- loaded,
- cleaned and enriched with metadata,
- preprocessed,
- split into train/validation/test sets,
- converted into sliding windows,
- and saved for future model training.

No model training was started in this phase. The focus was strictly on data preparation and pipeline validation.

## Next Possible Steps
Future work could include:
- training a baseline classifier,
- evaluating classical machine learning models,
- testing deep learning architectures such as CNNs or LSTMs,
- comparing performance across preprocessing settings,
- visualizing feature importance and model results.
