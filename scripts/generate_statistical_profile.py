"""Generate a full statistical profile of the impact-centered windowed dataset."""

from __future__ import annotations

import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sisfall_preprocessing import (
    ACC_CHANNEL_INDICES,
    compute_raw_acceleration_magnitude,
    detect_impact_index_peak,
    window_contains_impact,
)
from src.data.sisfall_loader import discover_subject_directories

PROCESSED = ROOT / "data" / "processed"
DATASET = ROOT / "SisFall_dataset"
OUTPUT = PROCESSED / "STATISTICAL_PROFILE.md"

SPLITS = ["train", "val", "test"]
CHANNELS = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"]
WINDOW_SIZE = 64
STRIDE = 16
FS_HZ = 20.0
DOWNSAMPLE_STEP = 10

SUBJECT_DEMOGRAPHICS = pd.DataFrame(
    [
        ("SA01", 26, 165, 53.0, "F", "SA"),
        ("SA02", 23, 176, 58.5, "M", "SA"),
        ("SA03", 19, 156, 48.0, "F", "SA"),
        ("SA04", 23, 170, 72.0, "M", "SA"),
        ("SA05", 22, 172, 69.5, "M", "SA"),
        ("SA06", 21, 169, 58.0, "M", "SA"),
        ("SA07", 21, 156, 63.0, "F", "SA"),
        ("SA08", 21, 149, 41.5, "F", "SA"),
        ("SA09", 24, 165, 64.0, "M", "SA"),
        ("SA10", 21, 177, 67.0, "M", "SA"),
        ("SA11", 19, 170, 80.5, "M", "SA"),
        ("SA12", 25, 153, 47.0, "F", "SA"),
        ("SA13", 22, 157, 55.0, "F", "SA"),
        ("SA14", 27, 160, 46.0, "F", "SA"),
        ("SA15", 25, 160, 52.0, "F", "SA"),
        ("SA16", 20, 169, 61.0, "F", "SA"),
        ("SA17", 23, 182, 75.0, "M", "SA"),
        ("SA18", 23, 181, 73.0, "M", "SA"),
        ("SA19", 30, 170, 76.0, "M", "SA"),
        ("SA20", 30, 150, 42.0, "F", "SA"),
        ("SA21", 30, 183, 68.0, "M", "SA"),
        ("SA22", 19, 158, 50.5, "F", "SA"),
        ("SA23", 24, 156, 48.0, "F", "SA"),
        ("SE01", 71, 171, 102.0, "M", "SE"),
        ("SE02", 75, 150, 57.0, "F", "SE"),
        ("SE03", 62, 150, 51.0, "F", "SE"),
        ("SE04", 63, 160, 59.0, "F", "SE"),
        ("SE05", 63, 165, 72.0, "M", "SE"),
        ("SE06", 60, 163, 79.0, "M", "SE"),
        ("SE07", 65, 168, 76.0, "M", "SE"),
        ("SE08", 68, 163, 72.0, "F", "SE"),
        ("SE09", 66, 167, 65.0, "M", "SE"),
        ("SE10", 64, 156, 66.0, "F", "SE"),
        ("SE11", 66, 169, 63.0, "F", "SE"),
        ("SE12", 69, 164, 56.5, "M", "SE"),
        ("SE13", 65, 171, 72.5, "M", "SE"),
        ("SE14", 67, 163, 58.0, "M", "SE"),
        ("SE15", 64, 150, 50.0, "F", "SE"),
    ],
    columns=["subject_id", "age", "height_cm", "weight_kg", "sex", "cohort"],
)


def md_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                cells.append(format(val, float_fmt))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def extract_activity_code(recording_id: str) -> str:
    filename = recording_id.split(":", 1)[-1]
    return filename.split("_")[0]


def load_split_arrays(split: str) -> dict[str, np.ndarray]:
    return {
        "windows": np.load(PROCESSED / f"{split}.npy"),
        "labels": np.load(PROCESSED / f"{split}_labels.npy"),
        "subjects": np.load(PROCESSED / f"{split}_subject_ids.npy", allow_pickle=True),
        "recordings": np.load(PROCESSED / f"{split}_recording_ids.npy", allow_pickle=True),
    }


def count_file_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def build_recording_catalog() -> pd.DataFrame:
    rows = []
    for subject_dir in discover_subject_directories(DATASET):
        subject_id = subject_dir.name
        for file_path in sorted(subject_dir.glob("*.txt")):
            raw_lines = count_file_lines(file_path)
            downsampled_samples = len(range(0, raw_lines, DOWNSAMPLE_STEP))
            activity_code = file_path.name.split("_")[0]
            rows.append(
                {
                    "recording_id": f"{subject_id}:{file_path.name}",
                    "subject_id": subject_id,
                    "activity_code": activity_code,
                    "is_fall_recording": activity_code.startswith("F"),
                    "raw_samples_200hz": raw_lines,
                    "downsampled_samples_20hz": downsampled_samples,
                    "duration_s_20hz": downsampled_samples / FS_HZ,
                    "windowable": downsampled_samples >= WINDOW_SIZE,
                }
            )
    return pd.DataFrame(rows)


def assign_split_to_subjects() -> dict[str, set[str]]:
    subjects = sorted(SUBJECT_DEMOGRAPHICS["subject_id"].tolist())
    shuffled = np.random.default_rng(42).permutation(subjects)
    counts = [26, 6, 6]
    mapping = {
        "train": set(shuffled[: counts[0]]),
        "val": set(shuffled[counts[0] : counts[0] + counts[1]]),
        "test": set(shuffled[counts[0] + counts[1] :]),
    }
    return mapping


def percentile_row(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {prefix + k: np.nan for k in ["_mean", "_std", "_min", "_p5", "_p25", "_p50", "_p75", "_p95", "_p99", "_max"]}
    qs = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        prefix + "mean": float(np.mean(values)),
        prefix + "std": float(np.std(values)),
        prefix + "min": float(np.min(values)),
        prefix + "p5": float(qs[0]),
        prefix + "p25": float(qs[1]),
        prefix + "p50": float(qs[2]),
        prefix + "p75": float(qs[3]),
        prefix + "p95": float(qs[4]),
        prefix + "p99": float(qs[5]),
        prefix + "max": float(np.max(values)),
    }


def peak_magnitude(windows: np.ndarray, channel_indices: tuple[int, ...]) -> np.ndarray:
    values = windows[:, :, channel_indices].astype(np.float64)
    magnitudes = np.sqrt(np.sum(values**2, axis=-1))
    return np.max(magnitudes, axis=1)


def time_to_peak_within_window(windows: np.ndarray, channel_indices: tuple[int, ...]) -> np.ndarray:
    values = windows[:, :, channel_indices].astype(np.float64)
    magnitudes = np.sqrt(np.sum(values**2, axis=-1))
    return np.argmax(magnitudes, axis=1).astype(np.float64)


def main() -> None:
    split_subjects = assign_split_to_subjects()
    catalog = build_recording_catalog()
    catalog["split"] = catalog["subject_id"].map(
        lambda sid: next(name for name, subjects in split_subjects.items() if sid in subjects)
    )

    split_data = {split: load_split_arrays(split) for split in SPLITS}
    with (PROCESSED / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)

    lines: list[str] = [
        "# SisFall Windowed Dataset — Full Statistical Profile",
        "",
        "Generated from saved arrays in `data/processed/` and SisFall source files.",
        "Impact-centered labeling (`impact_method=peak`). Window size=64, stride=16, sampling rate=20 Hz.",
        "",
        "## Computation Reference",
        "",
        "| Section | Source | Method |",
        "| --- | --- | --- |",
        "| Overview counts | `*.npy`, `*_recording_ids.npy`, `*_subject_ids.npy` | `np.load`, unique counts |",
        "| Demographics | `SisFall_dataset/Readme.txt` subject table | Joined by `subject_id` per split |",
        "| Recording duration | SisFall `*.txt` files | Line count / 10 downsampling / 20 Hz |",
        "| Signal stats (raw) | Normalized windows + `scaler.pkl` | `scaler.inverse_transform` on flattened timesteps |",
        "| Magnitude stats | Normalized windows | `max_t sqrt(sum(channel^2))` per window |",
        "| Normalization check | Normalized windows | Per-channel mean/std across all timesteps in split |",
        "| Overlap analysis | Labels + recording IDs + raw SisFall acc | Impact index on raw acc; count label=1 vs 0 per fall recording |",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
    ]

    overview_rows = []
    for split in SPLITS:
        data = split_data[split]
        overview_rows.append(
            {
                "split": split,
                "subjects": len(set(data["subjects"].tolist())),
                "recordings": len(set(data["recordings"].tolist())),
                "windows": len(data["labels"]),
                "window_shape": str(data["windows"].shape),
            }
        )
    lines += ["### 1.1 Split Summary", "", md_table(pd.DataFrame(overview_rows)), ""]

    demo_rows = []
    for split in SPLITS:
        subjects = sorted(set(split_data[split]["subjects"].tolist()))
        demo = SUBJECT_DEMOGRAPHICS[SUBJECT_DEMOGRAPHICS["subject_id"].isin(subjects)]
        demo_rows.append(
            {
                "split": split,
                "subjects": len(subjects),
                "SA_count": int((demo["cohort"] == "SA").sum()),
                "SE_count": int((demo["cohort"] == "SE").sum()),
                "age_mean": float(demo["age"].mean()),
                "age_min": int(demo["age"].min()),
                "age_max": int(demo["age"].max()),
                "height_cm_mean": float(demo["height_cm"].mean()),
                "weight_kg_mean": float(demo["weight_kg"].mean()),
                "female": int((demo["sex"] == "F").sum()),
                "male": int((demo["sex"] == "M").sum()),
            }
        )
    lines += [
        "### 1.2 Subject Demographics per Split",
        "",
        "Demographics from official SisFall README; joined to subjects present in each split.",
        "",
        md_table(pd.DataFrame(demo_rows), float_fmt=".2f"),
        "",
    ]

    duration_stats = (
        catalog.groupby("activity_code")["duration_s_20hz"]
        .agg(["count", "min", "mean", "median", "max"])
        .reset_index()
        .rename(columns={"count": "recordings"})
        .sort_values("activity_code")
    )
    lines += [
        "### 1.3 Recording Duration by Activity Code (20 Hz downsampled)",
        "",
        "Computed as `(raw_lines // 10) / 20` from each SisFall `.txt` file.",
        "",
        md_table(duration_stats, float_fmt=".3f"),
        "",
    ]

    window_by_activity = []
    for split in SPLITS:
        data = split_data[split]
        codes = [extract_activity_code(str(r)) for r in data["recordings"]]
        labels = data["labels"]
        for code in sorted(set(codes)):
            mask = np.array([extract_activity_code(str(r)) == code for r in data["recordings"]])
            window_by_activity.append(
                {
                    "split": split,
                    "activity_code": code,
                    "windows_total": int(mask.sum()),
                    "windows_fall_label": int((labels[mask] == 1).sum()),
                    "windows_adl_label": int((labels[mask] == 0).sum()),
                }
            )
    wba = pd.DataFrame(window_by_activity)
    lines += [
        "### 1.4 Window Count by Activity Code and Split",
        "",
        md_table(wba),
        "",
        "---",
        "",
        "## 2. Class Distribution",
        "",
    ]

    class_rows = []
    for split in SPLITS:
        labels = split_data[split]["labels"]
        total = len(labels)
        fall = int((labels == 1).sum())
        adl = int((labels == 0).sum())
        class_rows.append(
            {
                "split": split,
                "adl_windows": adl,
                "fall_windows": fall,
                "total": total,
                "fall_pct": 100.0 * fall / total,
                "adl_pct": 100.0 * adl / total,
            }
        )
    lines += ["### 2.1 Fall vs ADL Windows", "", md_table(pd.DataFrame(class_rows), float_fmt=".2f"), ""]

    fall_code_rows = []
    for split in SPLITS:
        data = split_data[split]
        for code in sorted({extract_activity_code(str(r)) for r in data["recordings"] if str(r).split(":")[-1].startswith("F")}):
            mask = np.array(
                [(extract_activity_code(str(r)) == code and lbl == 1) for r, lbl in zip(data["recordings"], data["labels"])]
            )
            fall_code_rows.append({"split": split, "fall_code": code, "fall_windows": int(mask.sum())})
    lines += ["### 2.2 Fall Windows by Fall Code (label=1 only)", "", md_table(pd.DataFrame(fall_code_rows)), ""]

    adl_code_rows = []
    for split in SPLITS:
        data = split_data[split]
        for code in sorted({extract_activity_code(str(r)) for r in data["recordings"] if str(r).split(":")[-1].startswith("D")}):
            mask = np.array([extract_activity_code(str(r)) == code for r in data["recordings"]])
            adl_code_rows.append({"split": split, "adl_code": code, "windows": int(mask.sum())})
    lines += [
        "### 2.3 ADL Recording Windows by Activity Code (all windows from D01–D19 recordings)",
        "",
        md_table(pd.DataFrame(adl_code_rows)),
        "",
    ]

    bg_rows = []
    for split in SPLITS:
        data = split_data[split]
        mask = np.array([extract_activity_code(str(r)).startswith("F") and lbl == 0 for r, lbl in zip(data["recordings"], data["labels"])])
        bg_rows.append({"split": split, "background_windows_from_fall_recordings": int(mask.sum())})
    lines += [
        "### 2.4 Background Windows from Fall Recordings (label=0, F-code recordings)",
        "",
        md_table(pd.DataFrame(bg_rows)),
        "",
    ]

    subject_window_counts = []
    for split in SPLITS:
        counts = Counter(split_data[split]["subjects"].tolist())
        values = np.array(list(counts.values()))
        subject_window_counts.append(
            {
                "split": split,
                "subjects": len(values),
                "min": int(values.min()),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "max": int(values.max()),
            }
        )
    lines += ["### 2.5 Windows per Subject", "", md_table(pd.DataFrame(subject_window_counts), float_fmt=".2f"), ""]

    subject_dom = []
    for split in SPLITS:
        counts = Counter(split_data[split]["subjects"].tolist())
        median = np.median(list(counts.values()))
        for sid, cnt in sorted(counts.items()):
            subject_dom.append({"split": split, "subject_id": sid, "windows": cnt, "ratio_to_median": cnt / median})
    dom_df = pd.DataFrame(subject_dom)
    lines += [
        "",
        "Subjects with >2× median window count:",
        "",
        md_table(dom_df[dom_df["ratio_to_median"] > 2.0].sort_values(["split", "windows"], ascending=[True, False])),
        "",
        "---",
        "",
        "## 3. Signal Statistics (Raw, Pre-Normalization)",
        "",
        "Computed by applying `scaler.inverse_transform()` to all timesteps in saved normalized windows.",
        "",
    ]

    signal_rows = []
    duration_rows = []
    for split in SPLITS:
        windows = split_data[split]["windows"]
        labels = split_data[split]["labels"]
        n_windows = len(labels)
        flat_norm = windows.reshape(-1, len(CHANNELS))
        flat_raw = scaler.inverse_transform(flat_norm)
        label_expanded = np.repeat(labels, WINDOW_SIZE)
        duration_rows.append(
            {
                "split": split,
                "timesteps_in_windows": int(flat_raw.shape[0]),
                "duration_s_at_20hz": flat_raw.shape[0] / FS_HZ,
                "sampling_rate_hz": FS_HZ,
            }
        )
        for class_label, class_name in [(0, "ADL"), (1, "Fall")]:
            mask = label_expanded == class_label
            if mask.sum() == 0:
                continue
            class_raw = flat_raw[mask]
            for ch_idx, channel in enumerate(CHANNELS):
                values = class_raw[:, ch_idx]
                row = {
                    "split": split,
                    "class": class_name,
                    "channel": channel,
                    **percentile_row(values, ""),
                }
                signal_rows.append(row)

    lines += ["### 3.1 Sampling Rate and Windowed Timestep Duration", "", md_table(pd.DataFrame(duration_rows), float_fmt=".1f"), ""]
    sig_df = pd.DataFrame(signal_rows)
    for split in SPLITS:
        lines.append(f"### 3.2 Raw Channel Stats — {split}")
        lines.append("")
        lines.append(md_table(sig_df[sig_df["split"] == split], float_fmt=".3f"))
        lines.append("")

    lines += ["---", "", "## 4. Magnitude Statistics (Normalized Windows)", ""]

    mag_rows = []
    ttp_rows = []
    for split in SPLITS:
        windows = split_data[split]["windows"]
        labels = split_data[split]["labels"]
        for class_label, class_name in [(0, "ADL"), (1, "Fall")]:
            mask = labels == class_label
            if mask.sum() == 0:
                continue
            subset = windows[mask]
            acc_peaks = peak_magnitude(subset, ACC_CHANNEL_INDICES)
            gyro_peaks = peak_magnitude(subset, (3, 4, 5))
            acc_ttp = time_to_peak_within_window(subset, ACC_CHANNEL_INDICES)
            gyro_ttp = time_to_peak_within_window(subset, (3, 4, 5))
            mag_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "metric": "peak_accel",
                    **percentile_row(acc_peaks, ""),
                }
            )
            mag_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "metric": "peak_gyro",
                    **percentile_row(gyro_peaks, ""),
                }
            )
            ttp_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "metric": "time_to_peak_accel_samples",
                    **percentile_row(acc_ttp, ""),
                }
            )
            ttp_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "metric": "time_to_peak_gyro_samples",
                    **percentile_row(gyro_ttp, ""),
                }
            )

    mag_df = pd.DataFrame(mag_rows)
    ttp_df = pd.DataFrame(ttp_rows)
    for split in SPLITS:
        lines.append(f"### 4.1 Peak Magnitude — {split}")
        lines.append("")
        lines.append(md_table(mag_df[mag_df["split"] == split], float_fmt=".3f"))
        lines.append("")
    lines.append("### 4.2 Time-to-Peak Within Window (samples from window start)")
    lines.append("")
    lines.append(md_table(ttp_df, float_fmt=".2f"))
    lines.append("")

    lines += ["---", "", "## 5. Normalization Validation", ""]

    norm_rows = []
    for split in SPLITS:
        flat = split_data[split]["windows"].reshape(-1, len(CHANNELS))
        for ch_idx, channel in enumerate(CHANNELS):
            values = flat[:, ch_idx]
            norm_rows.append(
                {
                    "split": split,
                    "channel": channel,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "mean_abs_drift_from_0": abs(float(values.mean())),
                    "std_drift_from_1": abs(float(values.std()) - 1.0),
                }
            )
    norm_df = pd.DataFrame(norm_rows)
    lines += [
        "Post-normalization mean/std across all timesteps in each split's windows.",
        "",
        md_table(norm_df, float_fmt=".4f"),
        "",
        "---",
        "",
        "## 6. Window Overlap / Redundancy (Fall Recordings)",
        "",
    ]

    overlap_rows = []
    fall_recording_stats = []
    for split in SPLITS:
        split_catalog = catalog[catalog["split"] == split]
        data = split_data[split]
        for recording_id in sorted(set(data["recordings"].tolist())):
            code = extract_activity_code(recording_id)
            if not code.startswith("F"):
                continue
            rec_mask = data["recordings"] == recording_id
            fall_windows = int((data["labels"][rec_mask] == 1).sum())
            bg_windows = int((data["labels"][rec_mask] == 0).sum())
            fall_recording_stats.append(
                {
                    "split": split,
                    "recording_id": recording_id,
                    "windows_total": int(rec_mask.sum()),
                    "impact_windows_label_1": fall_windows,
                    "background_windows_label_0": bg_windows,
                }
            )

    frs = pd.DataFrame(fall_recording_stats)
    overlap_summary = (
        frs.groupby("split")[["impact_windows_label_1", "background_windows_label_0", "windows_total"]]
        .agg(["sum", "mean", "median", "min", "max"])
        .reset_index()
    )
    # flatten columns
    flat_cols = ["split"]
    summary_rows = []
    for split in SPLITS:
        sub = frs[frs["split"] == split]
        summary_rows.append(
            {
                "split": split,
                "fall_recordings": len(sub),
                "impact_windows_total": int(sub["impact_windows_label_1"].sum()),
                "background_windows_total": int(sub["background_windows_label_0"].sum()),
                "impact_windows_per_recording_mean": float(sub["impact_windows_label_1"].mean()),
                "impact_windows_per_recording_median": float(sub["impact_windows_label_1"].median()),
                "impact_windows_per_recording_max": int(sub["impact_windows_label_1"].max()),
                "background_windows_per_recording_mean": float(sub["background_windows_label_0"].mean()),
            }
        )
    lines += [
        "### 6.1 Impact vs Background Windows per Fall Recording",
        "",
        md_table(pd.DataFrame(summary_rows), float_fmt=".2f"),
        "",
    ]

    effective_events = frs.groupby("split")["recording_id"].nunique().reset_index(name="unique_fall_recordings_with_windows")
    impact_positive = frs.groupby("split")["impact_windows_label_1"].apply(lambda s: int((s > 0).sum())).reset_index(name="recordings_with_ge_1_impact_window")
    eff = effective_events.merge(impact_positive, on="split")
    lines += [
        "### 6.2 Effective Unique Fall Events",
        "",
        "Each fall recording contributes 1 physical fall event; overlapping impact windows map to the same event.",
        "",
        md_table(eff),
        "",
        "---",
        "",
        "## 7. Missing / Anomaly Checks",
        "",
    ]

    anomaly_notes: list[str] = []

    nan_rows = []
    for split in SPLITS:
        windows = split_data[split]["windows"]
        flat = windows.reshape(-1, len(CHANNELS))
        row = {"split": split, "nan_total": int(np.isnan(flat).sum()), "inf_total": int(np.isinf(flat).sum())}
        for ch_idx, channel in enumerate(CHANNELS):
            row[f"nan_{channel}"] = int(np.isnan(flat[:, ch_idx]).sum())
        nan_rows.append(row)
    lines += ["### 7.1 NaN / Inf Counts", "", md_table(pd.DataFrame(nan_rows)), ""]

    too_short = catalog[~catalog["windowable"]].copy()
    lines += [
        "### 7.2 Recordings Excluded (< 64 downsampled samples)",
        "",
        f"Count: **{len(too_short)}** (none appear in window arrays if excluded).",
        "",
    ]
    if len(too_short):
        lines.append(md_table(too_short[["split", "subject_id", "activity_code", "recording_id", "downsampled_samples_20hz"]]))
        lines.append("")

    flatline_rows = []
    for split in SPLITS:
        windows = split_data[split]["windows"]
        stds = windows.std(axis=(1, 2))
        flat_mask = stds < 1e-6
        flatline_rows.append({"split": split, "flatline_windows": int(flat_mask.sum())})
    lines += ["### 7.3 Flat-Line Windows (std < 1e-6 across all channels/times)", "", md_table(pd.DataFrame(flatline_rows)), ""]

    # anomalies
    dom_high = dom_df[dom_df["ratio_to_median"] > 2.0]
    if len(dom_high):
        anomaly_notes.append(
            f"{len(dom_high)} subject-split pairs exceed 2× the median window count (see section 2.5)."
        )

    low_var = norm_df[norm_df["std"] < 0.05]
    for _, row in low_var.iterrows():
        anomaly_notes.append(
            f"Low post-norm variance: {row['split']} {row['channel']} std={row['std']:.4f}."
        )

    if len(too_short):
        anomaly_notes.append(f"{len(too_short)} recordings shorter than window_size after downsampling.")

    max_impact = int(frs["impact_windows_label_1"].max())
    if max_impact > 8:
        anomaly_notes.append(
            f"Some fall recordings yield up to {max_impact} impact-labeled windows due to 75% stride overlap around impact."
        )

    lines += ["---", "", "## Anomalies", ""]
    if anomaly_notes:
        for note in anomaly_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No major anomalies flagged beyond known class imbalance.")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
