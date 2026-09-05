from __future__ import annotations

"""Validate the severity clustering stage without modifying production outputs.

This script evaluates the saved train-fall-window KMeans model using standard
cluster-quality metrics:
- Silhouette score: higher is better; measures cohesion and separation.
- Davies-Bouldin index: lower is better; measures within-cluster scatter versus
  between-cluster separation.
- Calinski-Harabasz score: higher is better; rewards compact, well-separated
  clusters.

It also fits exploratory KMeans models for k=2..6 on the same train fall-window
feature matrix so the chosen k=3 solution can be compared against nearby values.
"""

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SEVERITY_DIR = PROCESSED_DIR / "severity"
FEATURE_DIR = SEVERITY_DIR / "clustering_features"
PLOTS_DIR = SEVERITY_DIR / "plots" / "clusters"
REPORT_PATH = SEVERITY_DIR / "CLUSTER_VALIDATION.md"


def _load_artifact(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_train_features() -> np.ndarray:
    path = FEATURE_DIR / "train_fall_features.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing train fall feature matrix: {path}")
    return np.load(path)


def _load_scaler():
    path = SEVERITY_DIR / "feature_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature scaler: {path}")
    return _load_artifact(path)


def _load_kmeans_model():
    path = SEVERITY_DIR / "kmeans_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing KMeans model: {path}")
    return _load_artifact(path)


def _compute_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return standard clustering quality metrics for one fitted labeling.

    Silhouette scores are only defined when at least two clusters are present and
    no cluster is empty. This helper assumes the caller already supplied valid
    labels.
    """

    metrics = {
        "silhouette": float(silhouette_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
    }
    return metrics


def _per_cluster_silhouette(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    sample_scores = silhouette_samples(features, labels)
    rows = []
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_scores = sample_scores[mask]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(mask.sum()),
                "mean_silhouette": float(cluster_scores.mean()),
                "median_silhouette": float(np.median(cluster_scores)),
                "min_silhouette": float(cluster_scores.min()),
                "max_silhouette": float(cluster_scores.max()),
            }
        )
    return pd.DataFrame(rows)


def _sweep_k_values(features: np.ndarray, k_values: range) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(features)
        metrics = _compute_metrics(features, labels)
        rows.append(
            {
                "k": int(k),
                **metrics,
            }
        )
    df = pd.DataFrame(rows)
    df["silhouette_rank"] = df["silhouette"].rank(ascending=False, method="min").astype(int)
    df["davies_bouldin_rank"] = df["davies_bouldin"].rank(ascending=True, method="min").astype(int)
    return df


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


def _plot_silhouette_vs_k(sweep_df: pd.DataFrame, current_k: int) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_df["k"], sweep_df["silhouette"], marker="o", linewidth=2, color="tab:blue")
    ax.axvline(current_k, linestyle="--", color="tab:orange", alpha=0.8, label="current k=3")

    best_row = sweep_df.sort_values("silhouette", ascending=False).iloc[0]
    ax.scatter([best_row["k"]], [best_row["silhouette"]], color="tab:green", s=80, zorder=3, label=f"best k={int(best_row['k'])}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Severity clustering silhouette score vs k")
    ax.set_xticks(list(sweep_df["k"]))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = PLOTS_DIR / "silhouette_vs_k.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _write_report(
    train_shape: tuple[int, int],
    current_k: int,
    current_metrics: dict[str, float],
    per_cluster_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    plot_path: Path,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_sil_row = sweep_df.sort_values("silhouette", ascending=False).iloc[0]
    best_db_row = sweep_df.sort_values("davies_bouldin", ascending=True).iloc[0]

    report = [
        "# Severity Cluster Validation",
        "",
        "This report evaluates the saved severity clustering model on the train fall-window feature matrix only.",
        "Lower Davies-Bouldin is better; higher silhouette and Calinski-Harabasz are better.",
        "",
        "## Validation Setup",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Train fall feature matrix | {train_shape[0]:,} x {train_shape[1]} |",
        f"| Saved production model k | {current_k} |",
        f"| k sweep | 2..6 |",
        "",
        "## Saved Model Metrics",
        "",
        "| k | silhouette | davies_bouldin | calinski_harabasz |",
        "| --- | ---: | ---: | ---: |",
        f"| {current_k} | {current_metrics['silhouette']:.4f} | {current_metrics['davies_bouldin']:.4f} | {current_metrics['calinski_harabasz']:.4f} |",
        "",
        "## Per-Cluster Silhouette for k=3",
        "",
        _markdown_table(per_cluster_df),
        "",
        "## k Sweep Summary",
        "",
        _markdown_table(sweep_df),
        "",
        "## Interpretation",
        "",
        f"Best silhouette: k={int(best_sil_row['k'])} ({best_sil_row['silhouette']:.4f}).",
        f"Best Davies-Bouldin: k={int(best_db_row['k'])} ({best_db_row['davies_bouldin']:.4f}).",
        f"Silhouette plot saved to: {plot_path.as_posix()}",
        "",
        "A k=3 model is considered acceptable for exploratory severity labeling if its silhouette score is close to the best k and its Davies-Bouldin score is not materially worse than the best alternative.",
    ]

    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _pass_fail_summary(sweep_df: pd.DataFrame, current_k: int, current_metrics: dict[str, float]) -> str:
    best_sil_row = sweep_df.sort_values("silhouette", ascending=False).iloc[0]
    best_db_row = sweep_df.sort_values("davies_bouldin", ascending=True).iloc[0]
    current_row = sweep_df.loc[sweep_df["k"] == current_k].iloc[0]

    silhouette_gap = float(best_sil_row["silhouette"] - current_row["silhouette"])
    silhouette_gap_ratio = silhouette_gap / max(abs(float(best_sil_row["silhouette"])), 1e-12)
    db_gap_ratio = float(current_row["davies_bouldin"] - best_db_row["davies_bouldin"]) / max(float(best_db_row["davies_bouldin"]), 1e-12)

    if current_k == int(best_sil_row["k"]) or (silhouette_gap_ratio <= 0.05 and db_gap_ratio <= 0.10):
        verdict = "PASS"
        message = (
            f"{verdict}: k={current_k} is within a reasonable margin of the best-scoring k. "
            f"Silhouette gap={silhouette_gap_ratio:.2%}, DB gap={db_gap_ratio:.2%}."
        )
    else:
        verdict = "REVIEW"
        message = (
            f"{verdict}: k={current_k} is not the strongest option by validation metrics. "
            f"Consider k={int(best_sil_row['k'])} by silhouette or k={int(best_db_row['k'])} by Davies-Bouldin. "
            f"Silhouette gap={silhouette_gap_ratio:.2%}, DB gap={db_gap_ratio:.2%}."
        )

    return (
        f"{message} Current metrics: silhouette={current_metrics['silhouette']:.4f}, "
        f"davies_bouldin={current_metrics['davies_bouldin']:.4f}, calinski_harabasz={current_metrics['calinski_harabasz']:.4f}."
    )


def main() -> None:
    train_features = _load_train_features()
    scaler = _load_scaler()
    model = _load_kmeans_model()

    scaled_train = scaler.transform(train_features)
    current_labels = model.predict(scaled_train)
    current_k = int(getattr(model, "n_clusters", len(np.unique(current_labels))))

    current_metrics = _compute_metrics(scaled_train, current_labels)
    per_cluster_df = _per_cluster_silhouette(scaled_train, current_labels)
    sweep_df = _sweep_k_values(scaled_train, range(2, 7))
    plot_path = _plot_silhouette_vs_k(sweep_df, current_k=current_k)

    _write_report(train_features.shape, current_k, current_metrics, per_cluster_df, sweep_df, plot_path)

    print(_pass_fail_summary(sweep_df, current_k, current_metrics))
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()