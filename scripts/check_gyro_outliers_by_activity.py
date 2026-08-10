from __future__ import annotations

"""Check whether ADL windows with high gyro peaks are concentrated in specific activities.

This script is additive validation only. It reads saved processed windows and
their metadata, computes the same per-window gyro peak proxy used elsewhere in
the repository, and summarizes ADL/background windows by activity code.

Gyro peak is defined as the maximum over time of the gyroscope magnitude
sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2) for each window.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PLOTS_DIR = PROCESSED_DIR / "plots"
REPORT_PATH = PROCESSED_DIR / "GYRO_OUTLIER_BY_ACTIVITY.md"


ACTIVITY_MAPPING = {
    "D01": "Walking slowly",
    "D02": "Walking quickly",
    "D03": "Jogging slowly",
    "D04": "Jogging quickly",
    "D05": "Walking upstairs and downstairs slowly",
    "D06": "Walking upstairs and downstairs quickly",
    "D07": "Slowly sit in a half height chair, wait a moment, and up slowly",
    "D08": "Quickly sit in a half height chair, wait a moment, and up quickly",
    "D09": "Slowly sit in a low height chair, wait a moment, and up slowly",
    "D10": "Quickly sit in a low height chair, wait a moment, and up quickly",
    "D11": "Sitting a moment, trying to get up, and collapse into a chair",
    "D12": "Sitting a moment, lying slowly, wait a moment, and sit again",
    "D13": "Sitting a moment, lying quickly, wait a moment, and sit again",
    "D14": "Being on one’s back change to lateral position, wait a moment, and change to one’s back",
    "D15": "Standing, slowly bending at knees, and getting up",
    "D16": "Standing, slowly bending without bending knees, and getting up",
    "D17": "Standing, get into a car, remain seated and get out of the car",
    "D18": "Stumble while walking",
    "D19": "Gently jump without falling (trying to reach a high object)",
    "F01": "Fall forward while walking caused by a slip",
    "F02": "Fall backward while walking caused by a slip",
    "F03": "Lateral fall while walking caused by a slip",
    "F04": "Fall forward while walking caused by a trip",
    "F05": "Fall backward while walking caused by a trip",
    "F06": "Lateral fall while walking caused by a trip",
    "F07": "Fall forward while jogging caused by a slip",
    "F08": "Fall backward while jogging caused by a slip",
    "F09": "Lateral fall while jogging caused by a slip",
    "F10": "Fall forward while jogging caused by a trip",
    "F11": "Fall backward while jogging caused by a trip",
    "F12": "Lateral fall while jogging caused by a trip",
    "F13": "Fall forward caused by fainting",
    "F14": "Fall backward caused by fainting",
    "F15": "Fall lateral caused by fainting",
}


FALL_GYRO_PEAK_REFERENCE = {
    "mean": 11.7,
    "p95": 17.0,
    "max_low": 37.0,
    "max_high": 49.0,
}


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = np.load(PROCESSED_DIR / f"{split}.npy")
    labels = np.load(PROCESSED_DIR / f"{split}_labels.npy")
    recording_ids = np.load(PROCESSED_DIR / f"{split}_recording_ids.npy", allow_pickle=True).astype(str)
    return windows, labels, recording_ids


def _extract_activity_code(recording_id: str) -> str:
    """Recover the SisFall activity code from the saved recording identifier."""

    try:
        _, filename = recording_id.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Unexpected recording_id format: {recording_id}") from exc
    return filename.split("_", 1)[0]


def _compute_gyro_peak(windows: np.ndarray) -> np.ndarray:
    gyro_mag = np.linalg.norm(windows[:, :, 3:], axis=2)
    return gyro_mag.max(axis=1)


def _summarize_by_activity(activity_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for activity_code, group in activity_df.groupby(["recording_type", "activity_code"], sort=True):
        recording_type, activity_code = activity_code
        peaks = group["gyro_peak"].to_numpy()
        rows.append(
            {
                "recording_type": recording_type,
                "activity_code": activity_code,
                "activity_name": ACTIVITY_MAPPING.get(activity_code, activity_code),
                "windows": int(len(group)),
                "mean_gyro_peak": float(peaks.mean()),
                "p95_gyro_peak": float(np.quantile(peaks, 0.95)),
                "max_gyro_peak": float(peaks.max()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_gyro_peak", ascending=False).reset_index(drop=True)
    return summary


def _flag_tail_activities(summary: pd.DataFrame) -> pd.DataFrame:
    """Flag ADL activities whose tails overlap the fall-range reference values."""

    flagged = summary.copy()
    flagged["is_adl_recording"] = flagged["recording_type"] == "ADL recording"
    flagged["p95_near_fall_range"] = flagged["p95_gyro_peak"] >= FALL_GYRO_PEAK_REFERENCE["p95"]
    flagged["max_near_fall_range"] = flagged["max_gyro_peak"] >= FALL_GYRO_PEAK_REFERENCE["max_low"]
    return flagged


def _plot_activity_boxplot(activity_df: pd.DataFrame) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = activity_df.groupby("activity_code")["gyro_peak"].median().sort_values(ascending=False).index.tolist()
    data = [activity_df.loc[activity_df["activity_code"] == code, "gyro_peak"].to_numpy() for code in ordered]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels(ordered)
    ax.set_title("Gyro peak magnitude by ADL activity code (train/background windows only)")
    ax.set_xlabel("Activity code")
    ax.set_ylabel("Per-window gyro peak")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out_path = PLOTS_DIR / "eda_gyro_peak_by_activity.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _write_report(summary: pd.DataFrame, flagged: pd.DataFrame, plot_path: Path) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "| | |\n| --- | --- |\n| (no rows) | (no rows) |"

        columns = list(df.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"

        def _format_value(value: object) -> str:
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        rows = [
            "| " + " | ".join(_format_value(value) for value in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *rows])

    report = [
        "# Gyro Peak by Activity Code",
        "",
        "This report checks whether high gyro peaks in ADL/background windows are concentrated in specific activities.",
        "A high gyro peak is not automatically a labeling error; it can be a legitimate fast-motion ADL.",
        "",
        "## Reference Range",
        "",
        "The fall-class gyro peaks reported in the processed dataset are approximately: mean 11.7, p95 about 17, and max roughly 37 to 49 depending on split.",
        "An ADL activity is flagged here if its p95 or max reaches that rough fall range.",
        "",
        "## Summary Table",
        "",
        _markdown_table(summary),
        "",
        "## Flagged ADL Tail Activities",
        "",
        _markdown_table(
            flagged.loc[
                flagged["is_adl_recording"]
                & (flagged["p95_near_fall_range"] | flagged["max_near_fall_range"])
            ]
        ),
        "",
        "## Background Windows From Fall Recordings",
        "",
        _markdown_table(
            flagged.loc[
                (~flagged["is_adl_recording"])
                & (flagged["p95_near_fall_range"] | flagged["max_near_fall_range"])
            ]
        ),
        "",
        f"Boxplot saved to: {plot_path.as_posix()}",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    windows, labels, recording_ids = _load_split("train")
    adl_mask = labels == 0
    adl_windows = windows[adl_mask]
    adl_recordings = recording_ids[adl_mask]

    gyro_peak = _compute_gyro_peak(adl_windows)
    activity_codes = np.asarray([_extract_activity_code(rec_id) for rec_id in adl_recordings], dtype=object)

    activity_df = pd.DataFrame(
        {
            "activity_code": activity_codes,
            "recording_type": np.where(np.char.startswith(activity_codes.astype(str), "D"), "ADL recording", "Fall recording background"),
            "gyro_peak": gyro_peak,
        }
    )

    summary = _summarize_by_activity(activity_df)
    flagged = _flag_tail_activities(summary)
    plot_path = _plot_activity_boxplot(activity_df)
    _write_report(summary, flagged, plot_path)

    adl_tail_rows = flagged.loc[
        flagged["is_adl_recording"] & (flagged["p95_near_fall_range"] | flagged["max_near_fall_range"]),
        ["activity_code", "activity_name", "windows", "mean_gyro_peak", "p95_gyro_peak", "max_gyro_peak"],
    ]
    background_tail_rows = flagged.loc[
        (~flagged["is_adl_recording"]) & (flagged["p95_near_fall_range"] | flagged["max_near_fall_range"]),
        ["activity_code", "activity_name", "windows", "mean_gyro_peak", "p95_gyro_peak", "max_gyro_peak"],
    ]

    if adl_tail_rows.empty:
        print("No ADL activities reached the fall-range gyro tail thresholds.")
    else:
        print("ADL activities driving the high-gyro tail:")
        for _, row in adl_tail_rows.iterrows():
            print(
                f"- {row['activity_code']} ({row['activity_name']}): windows={int(row['windows'])}, "
                f"mean={row['mean_gyro_peak']:.3f}, p95={row['p95_gyro_peak']:.3f}, max={row['max_gyro_peak']:.3f}"
            )

    if not background_tail_rows.empty:
        print("Background windows from fall recordings that also reach the fall-range tail:")
        for _, row in background_tail_rows.iterrows():
            print(
                f"- {row['activity_code']} ({row['activity_name']}): windows={int(row['windows'])}, "
                f"mean={row['mean_gyro_peak']:.3f}, p95={row['p95_gyro_peak']:.3f}, max={row['max_gyro_peak']:.3f}"
            )
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()