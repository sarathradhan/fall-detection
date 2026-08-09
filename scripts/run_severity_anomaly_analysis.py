from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEVERITY_DIR = ROOT / "data" / "processed" / "severity"
PLOTS_DIR = SEVERITY_DIR / "plots"
REPORTS_DIR = SEVERITY_DIR / "reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]


def _load_npy(name: str) -> np.ndarray:
    path = SEVERITY_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing severity artifact: {path}")
    return np.load(path)


def _load_csv(name: str) -> pd.DataFrame:
    path = SEVERITY_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing severity artifact: {path}")
    return pd.read_csv(path)


def _load_json(name: str) -> dict:
    path = SEVERITY_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing severity artifact: {path}")
    return pd.read_json(path)


def load_feature_names() -> list[str]:
    path = SEVERITY_DIR / "feature_names.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature_names.json at {path}")
    return list(pd.read_json(path))


def plot_anomaly_score_distribution() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for split in SPLITS:
        scores = _load_npy(f"{split}_anomaly_scores.npy")
        if scores.size == 0:
            continue
        ax.hist(scores, bins=50, alpha=0.5, density=True, label=split)
    ax.set_title("Isolation Forest Anomaly Score Distribution")
    ax.set_xlabel("Anomaly score (higher = more normal)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "anomaly_score_distribution.png", dpi=200)
    plt.close(fig)


def plot_anomaly_rate_by_severity() -> None:
    df = _load_csv("anomaly_by_severity.csv")
    severity_order = ["Mild", "Moderate", "Severe"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = np.arange(len(severity_order))

    for index, split in enumerate(SPLITS):
        split_df = df[df["split"] == split].copy()
        split_df["severity_name"] = pd.Categorical(split_df["severity_name"], categories=severity_order, ordered=True)
        split_df = split_df.sort_values("severity_name")
        ax.bar(x + index * width, split_df["anomaly_rate"].values, width, label=split)

    ax.set_xticks(x + width)
    ax.set_xticklabels(severity_order)
    ax.set_ylabel("Anomaly rate")
    ax.set_title("Anomaly rate by severity label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "anomaly_rate_by_severity.png", dpi=200)
    plt.close(fig)


def plot_anomaly_rate_by_subject() -> None:
    df = _load_csv("anomaly_by_subject.csv")
    for split in SPLITS:
        split_df = df[df["split"] == split].copy()
        split_df = split_df.sort_values("anomaly_rate", ascending=False).head(20)
        if split_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(split_df["subject_id"].astype(str), split_df["anomaly_rate"], color="tab:orange")
        ax.set_title(f"Top 20 anomaly rates by subject ({split})")
        ax.set_xlabel("Subject ID")
        ax.set_ylabel("Anomaly rate")
        ax.set_ylim(0, min(0.5, split_df["anomaly_rate"].max() * 1.1))
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"anomaly_rate_by_subject_{split}.png", dpi=200)
        plt.close(fig)


def summarize_feature_statistics() -> pd.DataFrame:
    feature_names = load_feature_names()
    selected_features = [
        "acc_mag_mean",
        "gyro_mag_mean",
        "acc_mag_peak",
        "gyro_mag_peak",
        "acc_mag_rms",
        "gyro_mag_rms",
    ]
    selected_indices = [feature_names.index(name) for name in selected_features if name in feature_names]
    rows: list[dict[str, object]] = []

    for split in SPLITS:
        features_path = SEVERITY_DIR / "clustering_features" / f"{split}_fall_features.npy"
        flags_path = SEVERITY_DIR / f"{split}_anomaly_flags.npy"
        if not features_path.exists() or not flags_path.exists():
            continue
        features = np.load(features_path)
        flags = np.load(flags_path)
        if features.shape[0] == 0:
            continue
        for status, status_name in [(0, "normal"), (1, "anomalous")]:
            mask = flags == status
            if not np.any(mask):
                continue
            selected = features[mask][:, selected_indices]
            stats = {
                "split": split,
                "status": status_name,
                "count": int(mask.sum()),
            }
            for idx, feature_name in zip(selected_indices, selected_features):
                stats[f"{feature_name}_mean"] = float(np.mean(features[mask, idx]))
                stats[f"{feature_name}_std"] = float(np.std(features[mask, idx]))
            rows.append(stats)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(REPORTS_DIR / "anomalous_feature_statistics.csv", index=False)
    return summary_df


def report_csv_summary() -> None:
    anomaly_summary = _load_csv("anomaly_summary.csv")
    anomaly_by_severity = _load_csv("anomaly_by_severity.csv")
    anomaly_by_subject = _load_csv("anomaly_by_subject.csv")
    anomaly_summary.to_csv(REPORTS_DIR / "anomaly_summary_report.csv", index=False)
    anomaly_by_severity.to_csv(REPORTS_DIR / "anomaly_by_severity_report.csv", index=False)
    anomaly_by_subject.to_csv(REPORTS_DIR / "anomaly_by_subject_report.csv", index=False)


def main() -> None:
    print(f"Loading severity artifacts from {SEVERITY_DIR}")
    plot_anomaly_score_distribution()
    plot_anomaly_rate_by_severity()
    plot_anomaly_rate_by_subject()
    summary_df = summarize_feature_statistics()
    report_csv_summary()
    print("Severity anomaly analysis complete.")
    print("Generated plots:")
    for plot in sorted(PLOTS_DIR.glob("*.png")):
        print(f" - {plot.name}")
    print("Generated report CSVs:")
    for report in sorted(REPORTS_DIR.glob("*.csv")):
        print(f" - {report.name}")
    print("Feature statistics summary saved to:", REPORTS_DIR / "anomalous_feature_statistics.csv")


if __name__ == "__main__":
    main()
