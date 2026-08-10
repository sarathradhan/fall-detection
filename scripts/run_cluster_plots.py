from __future__ import annotations

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
SEVERITY_DIR = ROOT / "data" / "processed" / "severity"
CLUSTER_FEATURES_DIR = SEVERITY_DIR / "clustering_features"
PLOTS_DIR = SEVERITY_DIR / "plots" / "clusters"
REPORTS_DIR = SEVERITY_DIR / "reports"
SPLITS = ["train", "val", "test"]

CLUSTER_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
SEVERITY_NAMES = {0: "Mild", 1: "Moderate", 2: "Severe"}


def _load_cluster_assignments() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], dict[int, int]]:
    with (SEVERITY_DIR / "feature_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (SEVERITY_DIR / "kmeans_model.pkl").open("rb") as handle:
        kmeans = pickle.load(handle)

    with (SEVERITY_DIR / "severity_mapping.json").open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    cluster_to_severity = {int(k): int(v) for k, v in mapping["cluster_to_severity"].items()}

    with (SEVERITY_DIR / "feature_names.json").open(encoding="utf-8") as handle:
        feature_names = json.load(handle)

    cluster_ids: dict[str, np.ndarray] = {}
    features_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        features = np.load(CLUSTER_FEATURES_DIR / f"{split}_fall_features.npy")
        features_by_split[split] = features
        if features.shape[0] == 0:
            cluster_ids[split] = np.empty((0,), dtype=np.int64)
        else:
            scaled = scaler.transform(features)
            cluster_ids[split] = kmeans.predict(scaled)

    return cluster_ids, features_by_split, feature_names, cluster_to_severity


def _cluster_label(cluster_id: int, cluster_to_severity: dict[int, int]) -> str:
    severity = cluster_to_severity[cluster_id]
    return f"Cluster {cluster_id} ({SEVERITY_NAMES[severity]})"


def plot_cluster_sizes(cluster_ids: dict[str, np.ndarray], cluster_to_severity: dict[int, int]) -> None:
    cluster_order = sorted(cluster_to_severity.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = []
    for split in SPLITS:
        ids = cluster_ids[split]
        row = {cid: int((ids == cid).sum()) for cid in cluster_order}
        row["split"] = split
        counts.append(row)
    df = pd.DataFrame(counts).set_index("split")

    x = np.arange(len(cluster_order))
    width = 0.25
    for index, split in enumerate(SPLITS):
        axes[0].bar(
            x + index * width,
            df.loc[split, cluster_order].values,
            width,
            label=split,
            color=plt.cm.tab10(index),
        )
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([_cluster_label(cid, cluster_to_severity) for cid in cluster_order])
    axes[0].set_ylabel("Fall windows")
    axes[0].set_title("Cluster sizes by split")
    axes[0].legend()

    total_counts = df[cluster_order].sum()
    colors = [CLUSTER_COLORS[cid] for cid in cluster_order]
    axes[1].pie(
        total_counts.values,
        labels=[_cluster_label(cid, cluster_to_severity) for cid in cluster_order],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    axes[1].set_title("Overall cluster distribution (all splits)")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cluster_sizes.png", dpi=200)
    plt.close(fig)


def plot_cluster_intensity(cluster_stats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("acc_mag_peak_mean", "Acc magnitude peak"),
        ("gyro_mag_peak_mean", "Gyro magnitude peak"),
        ("cluster_intensity_score", "Combined intensity score"),
    ]

    sorted_stats = cluster_stats.sort_values("severity_label")
    cluster_order = sorted_stats["cluster_id"].tolist()
    labels = [
        f"C{int(row.cluster_id)}\n({row.severity_name})"
        for row in sorted_stats.itertuples()
    ]
    colors = [CLUSTER_COLORS[cid] for cid in cluster_order]

    for ax, (col, title) in zip(axes, metrics):
        values = [cluster_stats.loc[cluster_stats["cluster_id"] == cid, col].item() for cid in cluster_order]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Mean feature value (normalized space)")

    fig.suptitle("Cluster intensity profiles (train-set centroids)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cluster_intensity_profiles.png", dpi=200)
    plt.close(fig)


def plot_cluster_feature_boxplots(
    features_by_split: dict[str, np.ndarray],
    cluster_ids: dict[str, np.ndarray],
    feature_names: list[str],
    cluster_to_severity: dict[int, int],
) -> None:
    key_features = [
        "acc_mag_peak",
        "gyro_mag_peak",
        "acc_mag_rms",
        "gyro_mag_rms",
        "acc_mag_mean",
        "gyro_mag_mean",
    ]
    feature_indices = [feature_names.index(name) for name in key_features]

    all_features = np.vstack([features_by_split[split] for split in SPLITS if features_by_split[split].shape[0] > 0])
    all_clusters = np.concatenate([cluster_ids[split] for split in SPLITS if cluster_ids[split].size > 0])
    cluster_order = sorted(cluster_to_severity.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, feat_idx, feat_name in zip(axes, feature_indices, key_features):
        data = [all_features[all_clusters == cid, feat_idx] for cid in cluster_order]
        bp = ax.boxplot(data, tick_labels=[f"C{cid}" for cid in cluster_order], patch_artist=True)
        for patch, cid in zip(bp["boxes"], cluster_order):
            patch.set_facecolor(CLUSTER_COLORS[cid])
            patch.set_alpha(0.7)
        ax.set_title(feat_name)
        ax.set_ylabel("Feature value")

    fig.suptitle("Feature distributions by cluster (all splits combined)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cluster_feature_boxplots.png", dpi=200)
    plt.close(fig)


def plot_cluster_pairwise_scatter(
    features_by_split: dict[str, np.ndarray],
    cluster_ids: dict[str, np.ndarray],
    feature_names: list[str],
    cluster_to_severity: dict[int, int],
) -> None:
    acc_idx = feature_names.index("acc_mag_peak")
    gyro_idx = feature_names.index("gyro_mag_peak")
    cluster_order = sorted(cluster_to_severity.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, split in zip(axes, SPLITS):
        features = features_by_split[split]
        ids = cluster_ids[split]
        if features.shape[0] == 0:
            ax.set_title(f"{split} (no data)")
            continue
        for cid in cluster_order:
            mask = ids == cid
            ax.scatter(
                features[mask, acc_idx],
                features[mask, gyro_idx],
                c=CLUSTER_COLORS[cid],
                alpha=0.35,
                s=12,
                label=_cluster_label(cid, cluster_to_severity),
            )
        ax.set_xlabel("acc_mag_peak")
        ax.set_ylabel("gyro_mag_peak")
        ax.set_title(f"{split} ({mask.sum() if features.shape[0] else 0} fall windows)")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Cluster separation: acc vs gyro peak magnitude", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cluster_pairwise_scatter.png", dpi=200)
    plt.close(fig)


def plot_cluster_pca(
    features_by_split: dict[str, np.ndarray],
    cluster_ids: dict[str, np.ndarray],
    cluster_to_severity: dict[int, int],
) -> None:
    with (SEVERITY_DIR / "feature_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)

    train_features = features_by_split["train"]
    if train_features.shape[0] == 0:
        return

    train_scaled = scaler.transform(train_features)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(train_scaled)

    cluster_order = sorted(cluster_to_severity.keys())
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, split in zip(axes, SPLITS):
        features = features_by_split[split]
        ids = cluster_ids[split]
        if features.shape[0] == 0:
            ax.set_title(f"{split} (no data)")
            continue
        projected = pca.transform(scaler.transform(features))
        for cid in cluster_order:
            mask = ids == cid
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                c=CLUSTER_COLORS[cid],
                alpha=0.35,
                s=12,
                label=_cluster_label(cid, cluster_to_severity),
            )
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
        ax.set_title(split)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("PCA projection of fall-window features by cluster (PCA fit on train)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cluster_pca_scatter.png", dpi=200)
    plt.close(fig)


def plot_cluster_centroid_heatmap(
    feature_names: list[str],
    cluster_to_severity: dict[int, int],
) -> None:
    with (SEVERITY_DIR / "feature_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (SEVERITY_DIR / "kmeans_model.pkl").open("rb") as handle:
        kmeans = pickle.load(handle)

    key_features = [
        "acc_mag_peak",
        "gyro_mag_peak",
        "acc_mag_rms",
        "gyro_mag_rms",
        "acc_mag_mean",
        "gyro_mag_mean",
        "acc_mag_energy",
        "gyro_mag_energy",
    ]
    feat_indices = [feature_names.index(name) for name in key_features]

    cluster_order = sorted(cluster_to_severity.keys())
    centroids = kmeans.cluster_centers_[:, feat_indices]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(centroids, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(key_features)))
    ax.set_xticklabels(key_features, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cluster_order)))
    ax.set_yticklabels(
        [_cluster_label(cid, cluster_to_severity) for cid in cluster_order]
    )
    ax.set_title("KMeans centroids (scaled feature space) — key intensity features")
    fig.colorbar(im, ax=ax, label="Scaled centroid value")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cluster_centroid_heatmap.png", dpi=200)
    plt.close(fig)


def plot_per_cluster_histograms(
    features_by_split: dict[str, np.ndarray],
    cluster_ids: dict[str, np.ndarray],
    feature_names: list[str],
    cluster_to_severity: dict[int, int],
) -> None:
    acc_idx = feature_names.index("acc_mag_peak")
    all_features = np.vstack([features_by_split[s] for s in SPLITS if features_by_split[s].shape[0] > 0])
    all_clusters = np.concatenate([cluster_ids[s] for s in SPLITS if cluster_ids[s].size > 0])

    cluster_order = sorted(cluster_to_severity.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, cid in zip(axes, cluster_order):
        mask = all_clusters == cid
        ax.hist(all_features[mask, acc_idx], bins=40, color=CLUSTER_COLORS[cid], alpha=0.85, edgecolor="white")
        ax.set_title(_cluster_label(cid, cluster_to_severity))
        ax.set_xlabel("acc_mag_peak")
        ax.set_ylabel("Count")
        ax.axvline(all_features[mask, acc_idx].mean(), color="black", linestyle="--", linewidth=1.2, label="mean")
        ax.legend()

    fig.suptitle("acc_mag_peak distribution per cluster (all splits)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cluster_acc_peak_histograms.png", dpi=200)
    plt.close(fig)


def export_cluster_report(
    cluster_ids: dict[str, np.ndarray],
    features_by_split: dict[str, np.ndarray],
    feature_names: list[str],
    cluster_to_severity: dict[int, int],
) -> None:
    key_features = ["acc_mag_peak", "gyro_mag_peak", "acc_mag_rms", "gyro_mag_rms"]
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        features = features_by_split[split]
        ids = cluster_ids[split]
        for cid in sorted(cluster_to_severity.keys()):
            mask = ids == cid
            row: dict[str, object] = {
                "split": split,
                "cluster_id": cid,
                "severity_label": cluster_to_severity[cid],
                "severity_name": SEVERITY_NAMES[cluster_to_severity[cid]],
                "count": int(mask.sum()),
            }
            if mask.any():
                for name in key_features:
                    idx = feature_names.index(name)
                    row[f"{name}_mean"] = float(features[mask, idx].mean())
                    row[f"{name}_std"] = float(features[mask, idx].std())
            rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(REPORTS_DIR / "cluster_feature_report.csv", index=False)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading severity artifacts from {SEVERITY_DIR}")
    cluster_ids, features_by_split, feature_names, cluster_to_severity = _load_cluster_assignments()
    cluster_stats = pd.read_csv(SEVERITY_DIR / "cluster_statistics.csv")

    plot_cluster_sizes(cluster_ids, cluster_to_severity)
    plot_cluster_intensity(cluster_stats)
    plot_cluster_feature_boxplots(features_by_split, cluster_ids, feature_names, cluster_to_severity)
    plot_cluster_pairwise_scatter(features_by_split, cluster_ids, feature_names, cluster_to_severity)
    plot_cluster_pca(features_by_split, cluster_ids, cluster_to_severity)
    plot_cluster_centroid_heatmap(feature_names, cluster_to_severity)
    plot_per_cluster_histograms(features_by_split, cluster_ids, feature_names, cluster_to_severity)
    export_cluster_report(cluster_ids, features_by_split, feature_names, cluster_to_severity)

    print("Cluster plots generated:")
    for plot in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  - {plot.relative_to(ROOT)}")
    print(f"Report saved to: {REPORTS_DIR / 'cluster_feature_report.csv'}")


if __name__ == "__main__":
    main()
