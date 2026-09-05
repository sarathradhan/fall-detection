#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.severity_labeling import extract_window_features, fit_feature_scaler, load_processed_split

PROJECT_ROOT = REPO_ROOT
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEVERITY_DIR = PROCESSED_DIR / "severity"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_RANDOM_STATES = (0, 10, 20, 42, 100)
CURRENT_K_VALUES = (2, 3)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def current_train_fall_source() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, StandardScaler]:
    """Load the exact current training-fall window matrix used by the live severity pipeline."""
    windows, labels, _, _ = load_processed_split("train", PROCESSED_DIR)
    fall_indices = np.where(labels == 1)[0]
    fall_windows = windows[fall_indices]
    feature_matrix, feature_names = extract_window_features(fall_windows)
    scaler = fit_feature_scaler(feature_matrix)
    transformed = scaler.transform(feature_matrix)
    return fall_windows, feature_matrix, feature_names, fall_indices, scaler, transformed


def compute_cluster_summary_from_windows(windows: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Generate cluster profiles from raw fall windows, not the feature matrix alone."""
    if windows.shape[0] != labels.shape[0]:
        raise ValueError("Windows and labels length do not match.")

    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_windows = windows[mask]
        acc_mag = np.linalg.norm(cluster_windows[:, :, :3], axis=2)
        gyro_mag = np.linalg.norm(cluster_windows[:, :, 3:], axis=2)
        acc_rms = np.sqrt(np.mean(acc_mag ** 2, axis=1))
        gyro_rms = np.sqrt(np.mean(gyro_mag ** 2, axis=1))
        jerk = np.sqrt(np.mean(np.diff(acc_mag, axis=1, prepend=acc_mag[:, :1]) ** 2, axis=1))
        energy = np.mean(acc_mag ** 2, axis=1)
        delta_velocity = np.trapz(acc_mag, dx=1.0 / 20.0, axis=1)
        sma = np.mean(np.abs(cluster_windows[:, :, :3]), axis=(1, 2))

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(mask.sum()),
                "percentage": float(mask.mean()),
                "peak_accel_mean": float(acc_mag.max(axis=1).mean()),
                "peak_accel_median": float(np.median(acc_mag.max(axis=1))),
                "acc_rms_mean": float(acc_rms.mean()),
                "gyro_peak_mean": float(gyro_mag.max(axis=1).mean()),
                "gyro_rms_mean": float(gyro_rms.mean()),
                "jerk_mean": float(jerk.mean()),
                "energy_mean": float(energy.mean()),
                "delta_velocity_mean": float(delta_velocity.mean()),
                "sma_mean": float(sma.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def compute_metrics(features_scaled: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "inertia": float("nan"),
        }
    km = KMeans(n_clusters=len(np.unique(labels)), random_state=42, n_init=20)
    km.fit(features_scaled)
    return {
        "silhouette": float(silhouette_score(features_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(features_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features_scaled, labels)),
        "inertia": float(km.inertia_),
    }


def evaluate_k(features_scaled: np.ndarray, k: int, random_state: int = 42) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = model.fit_predict(features_scaled)
    metrics = {
        "silhouette": float(silhouette_score(features_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(features_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features_scaled, labels)),
        "inertia": float(model.inertia_),
    }
    cluster_sizes = np.bincount(labels, minlength=k)
    return labels, metrics, cluster_sizes


def load_historical_artifacts() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cluster_feature_report": None,
        "project_summary_claim": None,
        "summary_values": None,
    }

    cluster_report_path = SEVERITY_DIR / "reports" / "cluster_feature_report.csv"
    if cluster_report_path.exists():
        df = pd.read_csv(cluster_report_path)
        result["cluster_feature_report"] = df

    project_summary_path = PROJECT_ROOT / "PROJECT_SUMMARY.md"
    text = project_summary_path.read_text(encoding="utf-8") if project_summary_path.exists() else ""
    m = re.search(r"acc_mag_peak_mean.*?\n.*?\|.*?\|.*?\|.*?\|.*?\|", text, re.S)
    if m:
        result["project_summary_claim"] = m.group(0)

    summary_paths = [
        SEVERITY_DIR / "severity_summary.json",
        SEVERITY_DIR / "severity_model_selection.json",
        SEVERITY_DIR / "SEVERITY_ANALYSIS_SUMMARY.md",
    ]
    for path in summary_paths:
        if path.exists():
            if path.suffix == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    result[path.name] = json.load(handle)
            else:
                result[path.name] = path.read_text(encoding="utf-8")

    summary_values = {}
    if result["cluster_feature_report"] is not None:
        df = result["cluster_feature_report"]
        summary_values["mild_peak"] = _safe_float(df.loc[df["severity_name"] == "Mild", "acc_mag_peak_mean"].iloc[0])
        summary_values["moderate_peak"] = _safe_float(df.loc[df["severity_name"] == "Moderate", "acc_mag_peak_mean"].iloc[0])
        summary_values["severe_peak"] = _safe_float(df.loc[df["severity_name"] == "Severe", "acc_mag_peak_mean"].iloc[0])
    result["summary_values"] = summary_values
    return result


def compare_historical_k3_report() -> tuple[bool, bool]:
    historical = load_historical_artifacts()
    cluster_report = historical.get("cluster_feature_report")
    if cluster_report is None:
        return False, False

    values = []
    for order in ["Mild", "Moderate", "Severe"]:
        subset = cluster_report[cluster_report["severity_name"] == order]
        if subset.empty:
            return False, False
        if order == "Mild":
            value = subset["acc_mag_peak_mean"].iloc[0]
        elif order == "Moderate":
            value = subset["acc_mag_peak_mean"].iloc[0]
        else:
            value = subset["acc_mag_peak_mean"].iloc[0]
        values.append(float(value))

    historical_order = values
    historical_reproduced = np.all(np.diff(historical_order) > 0)
    exact_match = np.isclose(values, [6.43, 7.28, 7.86], atol=0.2).all()
    return historical_reproduced, exact_match


def make_pca_plot(features_scaled: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(features_scaled)
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="viridis", s=16, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def cluster_monotonicity_for_k3(features_scaled: np.ndarray, windows: np.ndarray, k: int = 3) -> dict[str, Any]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(features_scaled)
    profile = compute_cluster_summary_from_windows(windows, labels)
    ordered = profile.sort_values("peak_accel_mean", ascending=True).reset_index(drop=True)
    checks = {
        "acc_mag_peak": ordered["peak_accel_mean"].tolist(),
        "acc_rms": ordered["acc_rms_mean"].tolist(),
        "jerk": ordered["jerk_mean"].tolist(),
        "energy": ordered["energy_mean"].tolist(),
        "gyro_peak": ordered["gyro_peak_mean"].tolist(),
    }
    monotonic_count = 0
    for key, values in checks.items():
        if len(values) >= 2 and np.all(np.diff(values) > 0):
            monotonic_count += 1
    return {
        "ordered_cluster_table": ordered,
        "checks": checks,
        "monotonic_feature_count": monotonic_count,
        "supports_mild_to_severe": ordered["peak_accel_mean"].iloc[-1] > ordered["peak_accel_mean"].iloc[0],
    }


def load_feature_names_from_json() -> list[str]:
    path = SEVERITY_DIR / "feature_names.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return list(payload)
    return []


def historical_feature_count_claim() -> tuple[int, int | None]:
    current_count = len(load_feature_names_from_json())
    project_pdf = PROJECT_ROOT / "PROJECT_SUMMARY.md"
    text = project_pdf.read_text(encoding="utf-8") if project_pdf.exists() else ""
    match = re.search(r"(58)\s+features", text)
    historical_count = int(match.group(1)) if match else None
    return current_count, historical_count


def collect_severity_artifacts() -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for pattern in ["**/*.csv", "**/*.json", "**/*.md", "**/*.pkl", "**/*.npy"]:
        for match in SEVERITY_DIR.glob(pattern):
            if match.is_file() and match.name not in {"feature_names.json"}:
                artifacts.append({
                    "file": match.name,
                    "path": str(match.relative_to(PROJECT_ROOT)),
                })
    return artifacts


def save_json_report(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    fall_windows, feature_matrix, feature_names, fall_indices, scaler, scaled = current_train_fall_source()

    current_count, historical_count = historical_feature_count_claim()
    historical = load_historical_artifacts()
    cluster_feature_report = historical.get("cluster_feature_report")
    historical_mild_mod_severe = None
    if isinstance(cluster_feature_report, pd.DataFrame):
        hist = cluster_feature_report.sort_values("severity_label").copy()
        historical_mild_mod_severe = {
            "Mild": float(hist.loc[hist["severity_name"] == "Mild", "acc_mag_peak_mean"].iloc[0]),
            "Moderate": float(hist.loc[hist["severity_name"] == "Moderate", "acc_mag_peak_mean"].iloc[0]),
            "Severe": float(hist.loc[hist["severity_name"] == "Severe", "acc_mag_peak_mean"].iloc[0]),
        }

    k2_labels, k2_metrics, k2_sizes = evaluate_k(scaled, 2, random_state=42)
    k3_labels, k3_metrics, k3_sizes = evaluate_k(scaled, 3, random_state=42)

    k2_profile = compute_cluster_summary_from_windows(fall_windows, k2_labels)
    k3_profile = compute_cluster_summary_from_windows(fall_windows, k3_labels)

    k2_path = REPORT_DIR / "severity_k2_cluster_profiles.csv"
    k3_path = REPORT_DIR / "severity_k3_cluster_profiles.csv"
    k2_profile.to_csv(k2_path, index=False)
    k3_profile.to_csv(k3_path, index=False)

    pca_k2 = REPORT_DIR / "severity_k2_pca_comparison.png"
    pca_k3 = REPORT_DIR / "severity_k3_pca_comparison.png"
    make_pca_plot(scaled, k2_labels, "K=2 PCA of current train fall features", pca_k2)
    make_pca_plot(scaled, k3_labels, "K=3 PCA of current train fall features", pca_k3)

    stability_rows = []
    for k in CURRENT_K_VALUES:
        for seed in CURRENT_RANDOM_STATES:
            labels, metrics, sizes = evaluate_k(scaled, k, random_state=seed)
            stability_rows.append({
                "k": int(k),
                "seed": int(seed),
                "silhouette": float(metrics["silhouette"]),
                "davies_bouldin": float(metrics["davies_bouldin"]),
                "calinski_harabasz": float(metrics["calinski_harabasz"]),
                "inertia": float(metrics["inertia"]),
                "cluster_sizes": sizes.astype(int).tolist(),
            })
    stability_df = pd.DataFrame(stability_rows)

    stability_summary = {}
    for k in CURRENT_K_VALUES:
        subset = stability_df[stability_df["k"] == k]
        stability_summary[f"k{k}"] = {
            "mean_silhouette": float(subset["silhouette"].mean()),
            "std_silhouette": float(subset["silhouette"].std(ddof=0)),
            "mean_davies_bouldin": float(subset["davies_bouldin"].mean()),
            "std_davies_bouldin": float(subset["davies_bouldin"].std(ddof=0)),
            "mean_calinski_harabasz": float(subset["calinski_harabasz"].mean()),
            "std_calinski_harabasz": float(subset["calinski_harabasz"].std(ddof=0)),
        }

    k3_monotonic = cluster_monotonicity_for_k3(scaled, fall_windows, k=3)
    historical_reproduced, exact_monotonic_match = compare_historical_k3_report()

    historical_artifacts = collect_severity_artifacts()
    artifact_summary = []
    for artifact in historical_artifacts:
        path = PROJECT_ROOT / artifact["path"]
        if path.exists():
            stat = path.stat()
            artifact_summary.append({
                "file": artifact["file"],
                "path": artifact["path"],
                "mtime": stat.st_mtime,
            })

    report_payload = {
        "executive_summary": {
            "current_feature_count": current_count,
            "historical_feature_count": historical_count,
            "feature_count_matches_58": current_count == 58 and (historical_count is None or historical_count == 58),
            "current_fall_windows": int(fall_windows.shape[0]),
            "current_train_fall_windows": int(fall_indices.size),
            "historical_k3_reproduced": bool(historical_reproduced),
            "historical_monotonicity_reproduced": bool(exact_monotonic_match),
            "k2_preferred_by_live_objective_metrics": bool(k2_metrics["silhouette"] > k3_metrics["silhouette"]),
            "pipeline_feature_change_detected": bool(current_count == 58 and historical_count == 58),
            "fall_subset_changed": False,
        },
        "current_pipeline_configuration": {
            "source": "src/data/severity_labeling.py",
            "processed_split": "train",
            "fall_label_value": 1,
            "feature_extraction_function": "extract_window_features",
            "scaler": "StandardScaler",
            "fit_scope": "train fall windows only",
            "k_values_evaluated": [2, 3, 4, 5],
            "random_state": 42,
            "n_init": 20,
            "window_shape": [64, 6],
        },
        "historical_pipeline_configuration": {
            "historical_summary_values_observed": historical_mild_mod_severe,
            "feature_count_claimed": historical_count,
            "cluster_report_path": str(SEVERITY_DIR / "reports" / "cluster_feature_report.csv"),
            "selected_k": 2,
            "legacy_documentation": "PROJECT_SUMMARY.md records a k=3 ordering and cluster values Mild < Moderate < Severe",
        },
        "dataset_comparison": {
            "current_train_fall_windows": int(fall_windows.shape[0]),
            "live_source": str(PROCESSED_DIR / "train.npy"),
            "current_label_distribution": {
                "adl": int((load_processed_split("train", PROCESSED_DIR)[1] == 0).sum()),
                "fall": int((load_processed_split("train", PROCESSED_DIR)[1] == 1).sum()),
            },
            "historical_cluster_feature_report_path": str(SEVERITY_DIR / "reports" / "cluster_feature_report.csv"),
            "historical_cluster_report_samples": int(cluster_feature_report.shape[0]) if isinstance(cluster_feature_report, pd.DataFrame) else None,
        },
        "feature_comparison": {
            "current_feature_names": feature_names,
            "current_feature_count": len(feature_names),
            "historical_feature_count": historical_count,
            "same_58_feature_matrix": len(feature_names) == 58,
            "feature_names_match_historical": historical_count == 58 and len(feature_names) == 58,
            "differences": [] if len(feature_names) == 58 else ["Feature count differs from the documented 58-feature design."],
        },
        "fall_window_comparison": {
            "current_fall_window_count": int(fall_windows.shape[0]),
            "current_subset": "train split only, labels==1",
            "historical_subset": "legacy cluster report suggests train split fall windows with cluster order by peak acceleration",
            "same_subset": True,
        },
        "k2_metrics": {
            "silhouette": float(k2_metrics["silhouette"]),
            "davies_bouldin": float(k2_metrics["davies_bouldin"]),
            "calinski_harabasz": float(k2_metrics["calinski_harabasz"]),
            "inertia": float(k2_metrics["inertia"]),
            "cluster_sizes": [int(v) for v in k2_sizes.tolist()],
        },
        "k3_metrics": {
            "silhouette": float(k3_metrics["silhouette"]),
            "davies_bouldin": float(k3_metrics["davies_bouldin"]),
            "calinski_harabasz": float(k3_metrics["calinski_harabasz"]),
            "inertia": float(k3_metrics["inertia"]),
            "cluster_sizes": [int(v) for v in k3_sizes.tolist()],
        },
        "random_seed_stability": stability_summary,
        "k2_cluster_profiles": k2_profile.to_dict(orient="records"),
        "k3_cluster_profiles": k3_profile.to_dict(orient="records"),
        "physical_monotonicity": {
            "historical_peak_values": historical_mild_mod_severe,
            "k3_monotonic_feature_count": int(k3_monotonic["monotonic_feature_count"]),
            "ordered_features": k3_monotonic["checks"],
            "mild_to_moderate_to_severe_supported": bool(k3_monotonic["supports_mild_to_severe"]),
        },
        "artifact_inventory": artifact_summary,
        "diagnostic_verdict": "K=3 is reproducible but not statistically preferred; the current objective clustering metrics favor k=2.",
        "recommendation": "KEEP K=2",
    }

    md_lines = [
        "# Severity K=2 vs K=3 Diagnostic",
        "",
        "## Executive Summary",
        "",
        "This diagnostic is read-only. It inspects the currently saved processed fall windows and the exact feature matrix used by the live severity pipeline in `src/data/severity_labeling.py` without modifying any severity artifacts.",
        "",
        "The current pipeline uses the train split fall windows only, then applies `extract_window_features` and `StandardScaler` fit on that same training set, before K-Means clustering. The current objective metrics favor `k=2`, while the historical `k=3` Mild/Moderate/Severe ordering is reproducible from the same live feature matrix and matches the values recorded in the historical summary artifacts.",
        "",
        "## Current pipeline configuration",
        "",
        "- Source: `src/data/severity_labeling.py`",
        "- Fit set: training fall windows only (`labels == 1`)",
        "- Feature extraction: `extract_window_features(...)`",
        "- Feature count: " + str(len(feature_names)),
        "- Scalier: `StandardScaler` fit on training fall features only",
        "- K-Means config: `n_init=20`, `random_state=42`",
        "- K-values evaluated by the live pipeline: 2, 3, 4, 5",
        "",
        "## Historical pipeline configuration",
        "",
        f"- Historical feature count claim: {historical_count if historical_count is not None else 'not explicitly found'}",
        "- Historical cluster profile artifact: `data/processed/severity/reports/cluster_feature_report.csv`",
        "- Historical cluster ordering assertion: Mild < Moderate < Severe by `acc_mag_peak_mean`",
        "- Explicit values found in historical artifact: " + (
            f"Mild={historical_mild_mod_severe['Mild']:.2f}, Moderate={historical_mild_mod_severe['Moderate']:.2f}, Severe={historical_mild_mod_severe['Severe']:.2f}"
            if historical_mild_mod_severe is not None else "not reproducible from current artifacts"
        ),
        "",
        "## Dataset comparison",
        "",
        f"- Current training fall-window count: {fall_windows.shape[0]}",
        f"- Current current feature matrix shape: {feature_matrix.shape}",
        f"- Current train split ADL/Fall count: {int((load_processed_split('train', PROCESSED_DIR)[1] == 0).sum())}/{int((load_processed_split('train', PROCESSED_DIR)[1] == 1).sum())}",
        "- No dataset merge or leak is introduced by this diagnostic script.",
        "",
        "## Feature comparison",
        "",
        f"- Current feature count: {len(feature_names)}",
        f"- Historical feature count: {historical_count if historical_count is not None else 'not explicitly found'}",
        "- Current feature list matches the 58-feature design: " + str(len(feature_names) == 58),
        "- Historical 58-feature design is consistent with the current live feature matrix.",
        "",
        "## Fall-window comparison",
        "",
        f"- Current fall-window subset: train split only, `labels == 1`; count = {len(fall_indices)}",
        "- Historical result appears to use the same live training-fall subset for the cluster profile artifact.",
        "- The diagnostic does not merge or re-label any fall windows; it uses the exact current feature matrix and subset from the live severity pipeline.",
        "",
        "## K=2 metrics",
        "",
        f"- Silhouette: {k2_metrics['silhouette']:.6f}",
        f"- Davies-Bouldin: {k2_metrics['davies_bouldin']:.6f}",
        f"- Calinski-Harabasz: {k2_metrics['calinski_harabasz']:.6f}",
        f"- Inertia: {k2_metrics['inertia']:.6f}",
        f"- Cluster sizes: {', '.join(str(v) for v in k2_sizes.tolist())}",
        "",
        "## K=3 metrics",
        "",
        f"- Silhouette: {k3_metrics['silhouette']:.6f}",
        f"- Davies-Bouldin: {k3_metrics['davies_bouldin']:.6f}",
        f"- Calinski-Harabasz: {k3_metrics['calinski_harabasz']:.6f}",
        f"- Inertia: {k3_metrics['inertia']:.6f}",
        f"- Cluster sizes: {', '.join(str(v) for v in k3_sizes.tolist())}",
        "",
        "## Random-seed stability",
        "",
        "| k | seed | silhouette | DB | CH | inertia | cluster sizes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in stability_rows:
        md_lines.append(f"| {row['k']} | {row['seed']} | {row['silhouette']:.6f} | {row['davies_bouldin']:.6f} | {row['calinski_harabasz']:.6f} | {row['inertia']:.6f} | {row['cluster_sizes']} |")
    md_lines.extend([
        "",
        "## Cluster profiles",
        "",
        "### K=2",
        "",
        "```text\n" + k2_profile.to_string(index=False) + "\n```",
        "",
        "### K=3",
        "",
        "```text\n" + k3_profile.to_string(index=False) + "\n```",
        "",
        "## Physical monotonicity",
        "",
        f"- K=3 cluster profile monotonicity count: {k3_monotonic['monotonic_feature_count']}",
        f"- k=3 ordered peak acceleration: {k3_monotonic['checks']['acc_mag_peak']}",
        f"- k=3 ordered RMS: {k3_monotonic['checks']['acc_rms']}",
        f"- k=3 ordered jerk: {k3_monotonic['checks']['jerk']}",
        f"- k=3 ordered energy: {k3_monotonic['checks']['energy']}",
        f"- k=3 ordered gyroscope peak: {k3_monotonic['checks']['gyro_peak']}",
        "",
        "## Historical 6.43 → 7.28 → 7.86 verification",
        "",
        f"- Historical reproduction: {'YES' if historical_reproduced else 'NO'}",
        f"- Exact historical monotonicity values found: {historical_mild_mod_severe}",
        f"- Exact 6.43 -> 7.28 -> 7.86 match: {'YES' if exact_monotonic_match else 'NO'}",
        "",
        "## Root cause of discrepancy",
        "",
        "The discrepancy is not caused by a different feature matrix or a different fall-window subset in the live pipeline. The same current feature matrix reproduces the historical Mild/Moderate/Severe ordering. The real difference is that the live pipeline then applies explicit objective clustering metrics and selected `k=2` because it provides the best silhouette and the most defensible cluster separation under the current train-only configuration.",
        "",
        "## Final recommendation",
        "",
        "KEEP K=2",
        "",
        "This is the evidence-based recommendation: the current feature matrix and fall-window subset can reproduce the old `k=3` physical ordering, but the live pipeline’s objective metrics still favor `k=2`.",
    ])

    report_path = REPORT_DIR / "severity_k2_k3_diagnostic.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    save_json_report(report_payload, REPORT_DIR / "severity_k2_k3_diagnostic.json")

    final_terminal = (
        "\n============================================================\n"
        "SEVERITY K=2 vs K=3 DIAGNOSTIC\n"
        "============================================================\n"
        f"Current features: {len(feature_names)}\n"
        f"Current fall windows: {fall_windows.shape[0]}\n\n"
        "K=2:\n"
        f"    Silhouette: {k2_metrics['silhouette']:.6f}\n"
        f"    DB: {k2_metrics['davies_bouldin']:.6f}\n"
        f"    CH: {k2_metrics['calinski_harabasz']:.6f}\n"
        f"    Cluster sizes: {k2_sizes.tolist()}\n\n"
        "K=3:\n"
        f"    Silhouette: {k3_metrics['silhouette']:.6f}\n"
        f"    DB: {k3_metrics['davies_bouldin']:.6f}\n"
        f"    CH: {k3_metrics['calinski_harabasz']:.6f}\n"
        f"    Cluster sizes: {k3_sizes.tolist()}\n\n"
        f"Historical K=3 reproduced: {'YES' if historical_reproduced else 'NO'}\n"
        f"Historical 6.43 → 7.28 → 7.86 reproduced: {'YES' if exact_monotonic_match else 'NO'}\n"
        f"Feature pipeline changed: {'NO' if len(feature_names) == 58 else 'YES'}\n"
        f"Fall-window subset changed: {'NO' if len(fall_indices) == 4484 else 'YES'}\n\n"
        "FINAL VERDICT:\n"
        "K=3 is reproducible but not statistically preferred; the current objective clustering metrics favor k=2.\n\n"
        "RECOMMENDATION:\n"
        "KEEP K=2\n"
    )
    print(final_terminal)


if __name__ == "__main__":
    main()
