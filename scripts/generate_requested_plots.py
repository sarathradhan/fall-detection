from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEVERITY_DIR = ROOT / "data" / "processed" / "severity" / "runs" / "k3_20260913T135813Z"
DEFAULT_EVAL_DIR = ROOT / "results" / "cnn_lstm_baseline" / "severity_eval" / "k3_20260913T135813Z"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "requested_plots"
SEVERITY_NAMES = {-1: "Uncertain", 0: "Mild", 1: "Moderate", 2: "Severe"}
SEVERITY_COLORS = {-1: "#7A7A7A", 0: "#4C78A8", 1: "#59A14F", 2: "#E15759"}


def _load_mapping(severity_dir: Path) -> dict[int, int]:
    mapping_path = severity_dir / "severity_mapping.json"
    with mapping_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(cluster_id): int(label) for cluster_id, label in payload["cluster_to_severity"].items()}


def _cluster_display(cluster_id: int, cluster_to_severity: dict[int, int]) -> str:
    severity = cluster_to_severity[cluster_id]
    return f"C{cluster_id} {SEVERITY_NAMES[severity]}"


def load_test_clusters(severity_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[int, int]]:
    features_path = severity_dir / "clustering_features" / "test_fall_features.npy"
    scaler_path = severity_dir / "feature_scaler.pkl"
    model_path = severity_dir / "kmeans_model.pkl"
    feature_names_path = severity_dir / "feature_names.json"

    features = np.load(features_path)
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    with model_path.open("rb") as handle:
        kmeans = pickle.load(handle)
    with feature_names_path.open(encoding="utf-8") as handle:
        feature_names = json.load(handle)

    clusters = kmeans.predict(scaler.transform(features))
    return features, clusters, feature_names, _load_mapping(severity_dir)


def save_test_cluster_counts(clusters: np.ndarray, cluster_to_severity: dict[int, int], output_dir: Path) -> Path:
    cluster_order = sorted(cluster_to_severity, key=lambda cid: cluster_to_severity[cid])
    counts = np.array([int(np.sum(clusters == cid)) for cid in cluster_order])
    labels = [_cluster_display(cid, cluster_to_severity) for cid in cluster_order]
    colors = [SEVERITY_COLORS[cluster_to_severity[cid]] for cid in cluster_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors)
    ax.set_title("KMeans Severity Clusters on Test Fall Windows")
    ax.set_xlabel("Cluster and severity mapping")
    ax.set_ylabel("Test fall windows")
    ax.bar_label(bars, padding=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / "kmeans_severity_test_cluster_counts.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_test_feature_scatter(
    features: np.ndarray,
    clusters: np.ndarray,
    feature_names: list[str],
    cluster_to_severity: dict[int, int],
    output_dir: Path,
) -> Path:
    acc_idx = feature_names.index("acc_mag_peak")
    gyro_idx = feature_names.index("gyro_mag_peak")
    cluster_order = sorted(cluster_to_severity, key=lambda cid: cluster_to_severity[cid])

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in cluster_order:
        mask = clusters == cid
        severity = cluster_to_severity[cid]
        ax.scatter(
            features[mask, acc_idx],
            features[mask, gyro_idx],
            s=18,
            alpha=0.55,
            color=SEVERITY_COLORS[severity],
            label=f"{_cluster_display(cid, cluster_to_severity)} (n={int(mask.sum())})",
        )
    ax.set_title("Test Severity Clusters: Acceleration vs Gyroscope Peak")
    ax.set_xlabel("acc_mag_peak")
    ax.set_ylabel("gyro_mag_peak")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / "kmeans_severity_test_feature_scatter.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_test_pca_plot(
    severity_dir: Path,
    test_features: np.ndarray,
    test_clusters: np.ndarray,
    cluster_to_severity: dict[int, int],
    output_dir: Path,
) -> Path:
    train_features = np.load(severity_dir / "clustering_features" / "train_fall_features.npy")
    with (severity_dir / "feature_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaler.transform(train_features))
    projected = pca.transform(scaler.transform(test_features))
    explained = pca.explained_variance_ratio_
    cluster_order = sorted(cluster_to_severity, key=lambda cid: cluster_to_severity[cid])

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in cluster_order:
        mask = test_clusters == cid
        severity = cluster_to_severity[cid]
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=18,
            alpha=0.55,
            color=SEVERITY_COLORS[severity],
            label=f"{_cluster_display(cid, cluster_to_severity)} (n={int(mask.sum())})",
        )
    ax.set_title("KMeans Severity Test Clusters in PCA Space")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / "kmeans_severity_test_pca.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_cnn_lstm_severity_metrics(eval_dir: Path, output_dir: Path) -> Path:
    window = pd.read_csv(eval_dir / "window_metrics_by_severity.csv")
    recording = pd.read_csv(eval_dir / "recording_summary_by_severity.csv")
    subject = pd.read_csv(eval_dir / "subject_summary_by_severity.csv")

    order = [-1, 0, 1, 2]
    window = window.set_index("severity_label").loc[order].reset_index()
    recording = recording.set_index("severity_label").loc[order].reset_index()
    subject = subject.set_index("severity_label").loc[order].reset_index()
    labels = [SEVERITY_NAMES[int(label)] for label in window["severity_label"]]
    colors = [SEVERITY_COLORS[int(label)] for label in window["severity_label"]]

    recording["recording_recall"] = recording["detected_recordings"] / recording["recordings"]
    recording["impact_verified_recall"] = recording["impact_verified_recordings"] / recording["recordings"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    axes[0].bar(labels, window["window_recall"], color=colors)
    axes[0].set_title("Window Recall by Severity")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Recall")
    for idx, value in enumerate(window["window_recall"]):
        axes[0].text(idx, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)

    x = np.arange(len(labels))
    width = 0.38
    axes[1].bar(x - width / 2, window["window_precision"], width, label="Precision", color="#4C78A8")
    axes[1].bar(x + width / 2, window["window_f1"], width, label="F1", color="#F58518")
    axes[1].set_title("Window Precision and F1 by Severity")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False)

    axes[2].bar(x - width / 2, recording["recording_recall"], width, label="Detected", color="#59A14F")
    axes[2].bar(x + width / 2, recording["impact_verified_recall"], width, label="Impact verified", color="#E15759")
    axes[2].set_title("Recording-Level Detection by Severity")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Recall")
    axes[2].legend(frameon=False)

    axes[3].bar(labels, subject["mean_subject_recall"], color=colors)
    axes[3].scatter(labels, subject["min_subject_recall"], color="black", label="Minimum subject recall", zorder=3)
    axes[3].set_title("Subject-Level Recall by Severity")
    axes[3].set_ylim(0, 1.05)
    axes[3].set_ylabel("Recall")
    axes[3].legend(frameon=False)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("CNN-LSTM Severity Evaluation Metrics at Threshold 0.50", fontsize=14)
    fig.tight_layout()

    output_path = output_dir / "cnn_lstm_severity_evaluation_metrics.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--severity-dir", type=Path, default=DEFAULT_SEVERITY_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features, clusters, feature_names, cluster_to_severity = load_test_clusters(args.severity_dir.resolve())
    outputs = [
        save_test_cluster_counts(clusters, cluster_to_severity, output_dir),
        save_test_feature_scatter(features, clusters, feature_names, cluster_to_severity, output_dir),
        save_test_pca_plot(args.severity_dir.resolve(), features, clusters, cluster_to_severity, output_dir),
        save_cnn_lstm_severity_metrics(args.eval_dir.resolve(), output_dir),
    ]

    print("Generated requested plots:")
    for output in outputs:
        print(f"  - {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
