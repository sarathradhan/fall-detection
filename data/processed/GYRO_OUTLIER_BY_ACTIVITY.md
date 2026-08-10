# Gyro Peak by Activity Code

This report checks whether high gyro peaks in ADL/background windows are concentrated in specific activities.
A high gyro peak is not automatically a labeling error; it can be a legitimate fast-motion ADL.

## Reference Range

The fall-class gyro peaks reported in the processed dataset are approximately: mean 11.7, p95 about 17, and max roughly 37 to 49 depending on split.
An ADL activity is flagged here if its p95 or max reaches that rough fall range.

## Summary Table

| recording_type | activity_code | activity_name | windows | mean_gyro_peak | p95_gyro_peak | max_gyro_peak |
| --- | --- | --- | --- | --- | --- | --- |
| ADL recording | D04 | Jogging quickly | 3040 | 5.2222 | 7.4445 | 10.8291 |
| ADL recording | D13 | Sitting a moment, lying quickly, wait a moment, and sit again | 900 | 4.6955 | 7.8789 | 8.8842 |
| ADL recording | D14 | Being on one’s back change to lateral position, wait a moment, and change to one’s back | 1560 | 4.2592 | 8.4283 | 11.5786 |
| ADL recording | D10 | Quickly sit in a low height chair, wait a moment, and up quickly | 1500 | 4.1466 | 8.3048 | 12.0847 |
| ADL recording | D03 | Jogging slowly | 3272 | 3.9633 | 6.1436 | 8.5225 |
| ADL recording | D08 | Quickly sit in a half height chair, wait a moment, and up quickly | 1560 | 3.8897 | 7.7871 | 13.4258 |
| ADL recording | D18 | Stumble while walking | 900 | 3.2868 | 6.9258 | 10.0497 |
| ADL recording | D09 | Slowly sit in a low height chair, wait a moment, and up slowly | 1500 | 3.0619 | 5.9972 | 9.0073 |
| ADL recording | D11 | Sitting a moment, trying to get up, and collapse into a chair | 1548 | 2.9269 | 8.3990 | 11.3211 |
| ADL recording | D05 | Walking upstairs and downstairs slowly | 3640 | 2.8974 | 5.8873 | 45.9282 |
| ADL recording | D06 | Walking upstairs and downstairs quickly | 2044 | 2.8570 | 8.5915 | 12.5413 |
| ADL recording | D02 | Walking quickly | 3272 | 2.8100 | 4.5311 | 9.4181 |
| ADL recording | D17 | Standing, get into a car, remain seated and get out of the car | 3461 | 2.8003 | 5.5268 | 49.1466 |
| ADL recording | D12 | Sitting a moment, lying slowly, wait a moment, and sit again | 1560 | 2.7835 | 4.3699 | 5.9848 |
| ADL recording | D07 | Slowly sit in a half height chair, wait a moment, and up slowly | 1560 | 2.5825 | 4.6441 | 6.0460 |
| ADL recording | D19 | Gently jump without falling (trying to reach a high object) | 900 | 2.4873 | 5.6550 | 8.4455 |
| ADL recording | D16 | Standing, slowly bending without bending knees, and getting up | 1545 | 2.4183 | 5.3347 | 6.9607 |
| Fall recording background | F07 | Fall forward while jogging caused by a slip | 825 | 2.2791 | 6.4116 | 16.2842 |
| Fall recording background | F06 | Lateral fall while walking caused by a trip | 826 | 2.2478 | 6.7584 | 15.3638 |
| Fall recording background | F05 | Fall backward while walking caused by a trip | 826 | 2.1413 | 7.5037 | 37.1098 |
| ADL recording | D01 | Walking slowly | 3271 | 1.9859 | 3.0595 | 5.9170 |
| ADL recording | D15 | Standing, slowly bending at knees, and getting up | 1545 | 1.9384 | 4.0674 | 5.5496 |
| Fall recording background | F04 | Fall forward while walking caused by a trip | 826 | 1.9346 | 6.0575 | 16.2746 |
| Fall recording background | F03 | Lateral fall while walking caused by a slip | 825 | 1.9197 | 6.0617 | 15.4088 |
| Fall recording background | F01 | Fall forward while walking caused by a slip | 815 | 1.9081 | 6.9734 | 30.0718 |
| Fall recording background | F02 | Fall backward while walking caused by a slip | 825 | 1.8663 | 5.3020 | 16.0049 |
| Fall recording background | F09 | Lateral fall while jogging caused by a slip | 825 | 1.5215 | 5.9635 | 16.3478 |
| Fall recording background | F08 | Fall backward while jogging caused by a slip | 825 | 1.4522 | 5.9246 | 14.6801 |
| Fall recording background | F13 | Fall forward caused by fainting | 825 | 1.2845 | 6.2602 | 13.9941 |
| Fall recording background | F11 | Fall backward while jogging caused by a trip | 825 | 1.0944 | 5.5171 | 12.5359 |
| Fall recording background | F12 | Lateral fall while jogging caused by a trip | 826 | 1.0890 | 4.5813 | 12.9620 |
| Fall recording background | F15 | Fall lateral caused by fainting | 826 | 1.0709 | 5.7010 | 12.4280 |
| Fall recording background | F10 | Fall forward while jogging caused by a trip | 814 | 1.0204 | 4.2909 | 11.7003 |
| Fall recording background | F14 | Fall backward caused by fainting | 827 | 0.9427 | 4.2011 | 12.9691 |

## Flagged ADL Tail Activities

| recording_type | activity_code | activity_name | windows | mean_gyro_peak | p95_gyro_peak | max_gyro_peak | is_adl_recording | p95_near_fall_range | max_near_fall_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADL recording | D05 | Walking upstairs and downstairs slowly | 3640 | 2.8974 | 5.8873 | 45.9282 | True | False | True |
| ADL recording | D17 | Standing, get into a car, remain seated and get out of the car | 3461 | 2.8003 | 5.5268 | 49.1466 | True | False | True |

## Background Windows From Fall Recordings

| recording_type | activity_code | activity_name | windows | mean_gyro_peak | p95_gyro_peak | max_gyro_peak | is_adl_recording | p95_near_fall_range | max_near_fall_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fall recording background | F05 | Fall backward while walking caused by a trip | 826 | 2.1413 | 7.5037 | 37.1098 | False | False | True |

Boxplot saved to: D:/Fall Detection/fall-detection/data/processed/plots/eda_gyro_peak_by_activity.png
