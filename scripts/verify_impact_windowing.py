"""Verify impact-centered windowing statistics on processed arrays."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
CHANNELS = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"]
SPLITS = ["train", "val", "test"]


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(PROCESSED / f"{name}.npy")
    y = np.load(PROCESSED / f"{name}_labels.npy")
    return x, y


def peak_magnitude(windows: np.ndarray, channel_indices: tuple[int, ...]) -> np.ndarray:
    values = windows[:, :, channel_indices].astype(np.float64)
    magnitudes = np.sqrt(np.sum(values**2, axis=-1))
    return np.max(magnitudes, axis=1)


def class_peak_stats(windows: np.ndarray, labels: np.ndarray, channel_indices: tuple[int, ...]) -> dict[str, float]:
    peaks = peak_magnitude(windows, channel_indices)
    for label, name in [(0, "adl"), (1, "fall")]:
        values = peaks[labels == label]
        if values.size == 0:
            continue
        yield name, {
            "windows": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }


def main() -> None:
    print("=== Impact-centered windowing verification ===\n")

    total_windows = 0
    total_fall = 0
    print("Class balance per split:")
    for split in SPLITS:
        _, labels = load_split(split)
        fall_count = int((labels == 1).sum())
        total = len(labels)
        total_windows += total
        total_fall += fall_count
        print(f"  {split}: ADL={( labels == 0).sum():6d}  Fall={fall_count:6d}  Fall%={100*fall_count/total:6.2f}%  total={total}")

    print(f"\nOverall: {total_windows} windows, {total_fall} fall ({100*total_fall/total_windows:.2f}%)")

    print("\nPeak acceleration magnitude (normalized acc channels):")
    for split in SPLITS:
        windows, labels = load_split(split)
        print(f"  [{split}]")
        for name, stats in class_peak_stats(windows, labels, (0, 1, 2)):
            print(
                f"    {name:4s}: n={stats['windows']:6d}  "
                f"mean={stats['mean']:.3f}  median={stats['median']:.3f}  "
                f"p95={stats['p95']:.3f}  max={stats['max']:.3f}"
            )

    print("\nPeak gyroscope magnitude (normalized gyro channels):")
    for split in SPLITS:
        windows, labels = load_split(split)
        print(f"  [{split}]")
        for name, stats in class_peak_stats(windows, labels, (3, 4, 5)):
            print(
                f"    {name:4s}: n={stats['windows']:6d}  "
                f"mean={stats['mean']:.3f}  median={stats['median']:.3f}  "
                f"p95={stats['p95']:.3f}  max={stats['max']:.3f}"
            )


if __name__ == "__main__":
    main()
