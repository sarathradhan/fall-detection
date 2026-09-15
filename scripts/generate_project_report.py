"""Generate end-to-end project report and plots from canonical run artifacts only."""

from __future__ import annotations

import json
import pickle
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
ASSETS_DIR = REPORT_DIR / "report_assets"
PROCESSED_DIR = ROOT / "data" / "processed"
CANONICAL_SEVERITY_RUN = "k3_20260913T135813Z"
DPI = 300
CHANNEL_NAMES = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"]
SEVERITY_NAMES = {-1: "Uncertain", 0: "Mild", 1: "Moderate", 2: "Severe"}
SEVERITY_COLORS = {-1: "#7A7A7A", 0: "#4C78A8", 1: "#59A14F", 2: "#E15759"}


@dataclass
class ReportContext:
    canonical_severity_run: str = CANONICAL_SEVERITY_RUN
    severity_dir: Path | None = None
    severity_eval_dir: Path | None = None
    cnn_dir: Path = field(default_factory=lambda: ROOT / "results" / "cnn_baseline")
    cnn_lstm_dir: Path = field(default_factory=lambda: ROOT / "results" / "cnn_lstm_baseline")
    missing_artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    open_issues: list[str] = field(default_factory=list)


def _record_missing(ctx: ReportContext, path: Path, purpose: str) -> None:
    ctx.missing_artifacts.append(f"{purpose}: {path}")


def _try_load_json(path: Path, ctx: ReportContext, purpose: str) -> dict | None:
    if not path.exists():
        _record_missing(ctx, path, purpose)
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        ctx.missing_artifacts.append(f"{purpose}: failed to read {path} ({exc})")
        return None


def _try_load_csv(path: Path, ctx: ReportContext, purpose: str) -> pd.DataFrame | None:
    if not path.exists():
        _record_missing(ctx, path, purpose)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        ctx.missing_artifacts.append(f"{purpose}: failed to read {path} ({exc})")
        return None


def _try_load_npy(path: Path, ctx: ReportContext, purpose: str) -> np.ndarray | None:
    if not path.exists():
        _record_missing(ctx, path, purpose)
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:
        ctx.missing_artifacts.append(f"{purpose}: failed to read {path} ({exc})")
        return None


def resolve_canonical_paths(ctx: ReportContext) -> None:
    severity_candidates = [
        PROCESSED_DIR / "severity" / "runs" / ctx.canonical_severity_run,
        *ROOT.glob(f"data/processed/deprecated_severity/*/data/processed/severity/runs/{ctx.canonical_severity_run}"),
    ]
    for candidate in severity_candidates:
        if (candidate / "severity_summary.json").exists():
            ctx.severity_dir = candidate
            break

    eval_candidates = [
        ROOT / "results" / "cnn_lstm_baseline" / "severity_eval" / ctx.canonical_severity_run,
        *ROOT.glob(f"data/processed/deprecated_severity/*/results/cnn_lstm_baseline/severity_eval/{ctx.canonical_severity_run}"),
    ]
    for candidate in eval_candidates:
        if (candidate / "evaluation_summary.json").exists():
            ctx.severity_eval_dir = candidate
            break

    if ctx.severity_dir is None:
        _record_missing(ctx, PROCESSED_DIR / "severity" / "runs" / ctx.canonical_severity_run, "canonical severity run")
    if ctx.severity_eval_dir is None:
        _record_missing(
            ctx,
            ROOT / "results" / "cnn_lstm_baseline" / "severity_eval" / ctx.canonical_severity_run,
            "canonical severity evaluation",
        )


def safe_plot(name: str, fn: Callable[[], None], ctx: ReportContext) -> str | None:
    try:
        fn()
        return f"report_assets/{name}"
    except Exception as exc:
        ctx.missing_artifacts.append(f"plot {name}: {exc}\n{traceback.format_exc()}")
        return None


def compute_smv(windows: np.ndarray) -> np.ndarray:
    acc = windows[:, :, :3]
    return np.linalg.norm(acc, axis=2).max(axis=1)


def load_processed_splits(ctx: ReportContext) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        windows = _try_load_npy(PROCESSED_DIR / f"{split}.npy", ctx, f"{split} windows")
        labels = _try_load_npy(PROCESSED_DIR / f"{split}_labels.npy", ctx, f"{split} labels")
        if windows is not None and labels is not None:
            splits[split] = (windows, labels.astype(np.int64))
    return splits


def collect_core_metrics(ctx: ReportContext) -> None:
    cnn_meta = _try_load_json(ctx.cnn_dir / "run_metadata.json", ctx, "CNN run metadata")
    cnn_lstm_meta = _try_load_json(ctx.cnn_lstm_dir / "run_metadata.json", ctx, "CNN-LSTM run metadata")
    rec_eval = _try_load_json(ctx.cnn_lstm_dir / "recording_level_eval" / "evaluation_summary.json", ctx, "recording-level eval")
    fp_diag = _try_load_json(ctx.cnn_lstm_dir / "false_positive_diagnostics" / "diagnostics_summary.json", ctx, "false-positive diagnostics")
    severity_eval = None
    if ctx.severity_eval_dir is not None:
        severity_eval = _try_load_json(ctx.severity_eval_dir / "evaluation_summary.json", ctx, "severity-wise evaluation")

    eda_df = _try_load_csv(PROCESSED_DIR / "reports" / "eda_class_distribution.csv", ctx, "EDA class distribution")

    ctx.metrics["cnn"] = cnn_meta
    ctx.metrics["cnn_lstm"] = cnn_lstm_meta
    ctx.metrics["recording_eval"] = rec_eval
    ctx.metrics["fp_diagnostics"] = fp_diag
    ctx.metrics["severity_eval"] = severity_eval
    ctx.metrics["eda_class_distribution"] = eda_df

    if rec_eval and fp_diag:
        rec_fp = rec_eval.get("recording_level", {}).get("threshold_0_5", {}).get("adl_recordings_falsely_triggered")
        diag_fp = fp_diag.get("false_trigger_recordings_by_threshold", {}).get("threshold_0_5")
        if rec_fp is not None and diag_fp is not None and int(rec_fp) != int(diag_fp):
            ctx.open_issues.append(
                f"ADL false-trigger count disagreement at threshold 0.50: "
                f"recording_level_eval reports {rec_fp}, false_positive_diagnostics reports {diag_fp}."
            )


def plot_class_balance(ctx: ReportContext, splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    if not splits:
        raise ValueError("No processed splits available for class balance plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    totals = {"ADL": 0, "Fall": 0}
    split_names = []
    adl_counts = []
    fall_counts = []
    for split, (_, labels) in splits.items():
        adl = int((labels == 0).sum())
        fall = int((labels == 1).sum())
        totals["ADL"] += adl
        totals["Fall"] += fall
        split_names.append(split)
        adl_counts.append(adl)
        fall_counts.append(fall)

    ctx.metrics["window_totals"] = totals
    ctx.metrics["total_windows"] = totals["ADL"] + totals["Fall"]

    x = np.arange(len(split_names))
    width = 0.35
    axes[0].bar(x - width / 2, adl_counts, width, label="ADL", color="#4C78A8")
    axes[0].bar(x + width / 2, fall_counts, width, label="Fall", color="#E15759")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(split_names)
    axes[0].set_ylabel("Window count")
    axes[0].set_title("Windows per split")
    axes[0].legend()

    axes[1].bar(["ADL", "Fall"], [totals["ADL"], totals["Fall"]], color=["#4C78A8", "#E15759"])
    axes[1].set_title("Overall class balance")
    axes[1].set_ylabel("Window count")
    for idx, val in enumerate([totals["ADL"], totals["Fall"]]):
        axes[1].text(idx, val, f"{val:,}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "class_balance.png", dpi=DPI)
    plt.close(fig)


def plot_sample_waveforms(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    windows, labels = splits["test"]
    fall_idx = int(np.where(labels == 1)[0][0])
    adl_idx = int(np.where(labels == 0)[0][0])
    fall_win = windows[fall_idx]
    adl_win = windows[adl_idx]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t = np.arange(fall_win.shape[0]) / 20.0
    for col, name in enumerate(CHANNEL_NAMES):
        axes[0].plot(t, fall_win[:, col], label=name, alpha=0.85)
        axes[1].plot(t, adl_win[:, col], label=name, alpha=0.85)
    axes[0].set_title("Example fall window (normalized 6-channel IMU)")
    axes[1].set_title("Example ADL window (normalized 6-channel IMU)")
    axes[0].set_ylabel("Standardized value")
    axes[1].set_ylabel("Standardized value")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "sample_waveforms_fall_vs_adl.png", dpi=DPI)
    plt.close(fig)


def plot_preprocessing_windowing(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    windows, labels = splits["test"]
    fall_idx = int(np.where(labels == 1)[0][0])
    fall_win = windows[fall_idx]
    smv = np.linalg.norm(fall_win[:, :3], axis=1)
    t = np.arange(len(smv)) / 20.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, smv, color="#E15759", linewidth=1.8, label="Impact-centered fall window SMV")
    ax.axvspan(t[0], t[-1], color="#E15759", alpha=0.08)
    ax.set_title("Impact-centered 3.2 s window (64 samples @ 20 Hz, stride 16)")
    ax.set_xlabel("Time within window (s)")
    ax.set_ylabel("Signal magnitude vector (normalized acc)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "preprocessing_windowing_example.png", dpi=DPI)
    plt.close(fig)


def plot_smv_distribution(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    adl_smv = []
    fall_smv = []
    for _, (windows, labels) in splits.items():
        smv = compute_smv(windows)
        adl_smv.append(smv[labels == 0])
        fall_smv.append(smv[labels == 1])
    adl = np.concatenate(adl_smv)
    fall = np.concatenate(fall_smv)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = 60
    ax.hist(adl, bins=bins, alpha=0.55, density=True, label=f"ADL (n={len(adl):,})", color="#4C78A8")
    ax.hist(fall, bins=bins, alpha=0.55, density=True, label=f"Fall (n={len(fall):,})", color="#E15759")
    ax.set_title("Peak SMV distribution: fall vs ADL windows")
    ax.set_xlabel("Peak signal magnitude vector (normalized)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "smv_distribution.png", dpi=DPI)
    plt.close(fig)


def plot_fall_spike_example(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    windows, labels = splits["test"]
    smv_all = compute_smv(windows)
    fall_idx = int(np.where(labels == 1)[0][np.argmax(smv_all[labels == 1])])
    adl_idx = int(np.where(labels == 0)[0][np.argmin(np.abs(smv_all[labels == 0] - np.median(smv_all[labels == 0])))])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(64) / 20.0
    for ax, idx, title, color in [
        (axes[0], fall_idx, "High-SMV fall window", "#E15759"),
        (axes[1], adl_idx, "Typical ADL window", "#4C78A8"),
    ]:
        smv = np.linalg.norm(windows[idx, :, :3], axis=1)
        ax.plot(t, smv, color=color, linewidth=1.8)
        ax.set_ylabel("SMV")
        ax.set_title(title)
    axes[1].set_xlabel("Time (s)")
    fig.suptitle("Fall impact spike vs typical ADL pattern", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "fall_spike_example.png", dpi=DPI)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], title: str, out_name: str, class_names: list[str]) -> None:
    cm_arr = np.array(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(cm_arr):
        ax.text(j, i, str(val), ha="center", va="center", color="black" if val < cm_arr.max() / 2 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / out_name, dpi=DPI)
    plt.close(fig)


def plot_training_curves(ctx: ReportContext) -> None:
    history_path = ctx.cnn_lstm_dir / "training_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["accuracy"], label="train acc")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle("CNN-LSTM training curves")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "cnn_lstm_training_curves.png", dpi=DPI)
    plt.close(fig)


def plot_threshold_curves(ctx: ReportContext) -> None:
    sweep_path = ctx.cnn_lstm_dir / "threshold_sweep.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(sweep_path)
    sweep = pd.read_csv(sweep_path)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(sweep["recall"], sweep["precision"], marker="o", linewidth=1.5, color="#4C78A8")
    for thr in [0.25, 0.5]:
        row = sweep.loc[(sweep["threshold"] - thr).abs().idxmin()]
        ax.scatter(row["recall"], row["precision"], s=60, zorder=3, label=f"threshold={thr:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("CNN-LSTM precision-recall (validation threshold sweep)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "cnn_lstm_precision_recall_curve.png", dpi=DPI)
    plt.close(fig)


def plot_severity_pca(ctx: ReportContext) -> None:
    if ctx.severity_dir is None:
        raise FileNotFoundError("canonical severity directory")
    features = np.load(ctx.severity_dir / "clustering_features" / "test_fall_features.npy")
    with (ctx.severity_dir / "feature_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (ctx.severity_dir / "kmeans_model.pkl").open("rb") as handle:
        kmeans = pickle.load(handle)
    mapping = _try_load_json(ctx.severity_dir / "severity_mapping.json", ctx, "severity mapping")
    if mapping is None:
        raise FileNotFoundError("severity_mapping.json")
    cluster_to_severity = {int(k): int(v) for k, v in mapping["cluster_to_severity"].items()}
    scaled = scaler.transform(features)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaler.transform(np.load(ctx.severity_dir / "clustering_features" / "train_fall_features.npy")))
    projected = pca.transform(scaled)
    clusters = kmeans.predict(scaled)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for cid in sorted(cluster_to_severity):
        mask = clusters == cid
        severity = cluster_to_severity[cid]
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=14,
            alpha=0.55,
            color=SEVERITY_COLORS[severity],
            label=f"Cluster {cid} → {SEVERITY_NAMES[severity]}",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("K-Means severity clusters (test fall windows, PCA of 58 features)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "severity_pca_clusters.png", dpi=DPI)
    plt.close(fig)


def plot_severity_peak_boxplot(ctx: ReportContext) -> None:
    if ctx.severity_dir is None:
        raise FileNotFoundError("canonical severity directory")
    stats = pd.read_csv(ctx.severity_dir / "cluster_statistics.csv")
    stats = stats.sort_values("severity_label")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(stats))
    width = 0.35
    ax.bar(x - width / 2, stats["acc_mag_peak_mean"], width, label="acc_mag_peak", color="#4C78A8")
    ax.bar(x + width / 2, stats["gyro_mag_peak_mean"], width, label="gyro_mag_peak", color="#E15759")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{int(r.cluster_id)}\n({r.severity_name})" for r in stats.itertuples()])
    ax.set_ylabel("Mean feature value (normalized space)")
    ax.set_title("Monotonic intensity ordering across severity clusters (train centroids)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "severity_peak_intensity.png", dpi=DPI)
    plt.close(fig)


def plot_anomaly_scores(ctx: ReportContext) -> None:
    if ctx.severity_dir is None:
        raise FileNotFoundError("canonical severity directory")
    scores = []
    flags = []
    for split in ["train", "val", "test"]:
        scores.append(np.load(ctx.severity_dir / f"{split}_anomaly_scores.npy"))
        flags.append(np.load(ctx.severity_dir / f"{split}_anomaly_flags.npy"))
    all_scores = np.concatenate(scores)
    all_flags = np.concatenate(flags)
    contamination = float(np.mean(all_flags == 1))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(all_scores[all_flags == 0], bins=50, alpha=0.65, density=True, label="Normal fall windows", color="#4C78A8")
    ax.hist(all_scores[all_flags == 1], bins=20, alpha=0.75, density=True, label="Flagged uncertain (~2%)", color="#E15759")
    ax.set_title(f"Isolation Forest anomaly scores (contamination target 2%, observed {contamination:.2%})")
    ax.set_xlabel("Anomaly score (higher = more normal)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "anomaly_score_distribution.png", dpi=DPI)
    plt.close(fig)


def plot_severity_recall_bar(ctx: ReportContext) -> None:
    severity_eval = ctx.metrics.get("severity_eval")
    if not severity_eval:
        raise ValueError("severity evaluation summary missing")
    rows = [row for row in severity_eval["window_metrics"] if row["severity_label"] in (-1, 0, 1, 2)]
    rows.sort(key=lambda r: r["severity_label"])
    labels = [row["severity"] for row in rows]
    recalls = [row["window_recall"] for row in rows]
    colors = [SEVERITY_COLORS[row["severity_label"]] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, recalls, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Window recall @ threshold 0.50")
    ax.set_title("CNN-LSTM recall by severity class (test split)")
    ax.bar_label(bars, labels=[f"{r:.4f}" for r in recalls], padding=3)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "severity_recall_by_class.png", dpi=DPI)
    plt.close(fig)


def fmt_pct(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def build_report(ctx: ReportContext, plot_paths: dict[str, str | None]) -> str:
    cnn = ctx.metrics.get("cnn") or {}
    cnn_lstm = ctx.metrics.get("cnn_lstm") or {}
    rec = ctx.metrics.get("recording_eval") or {}
    fp = ctx.metrics.get("fp_diagnostics") or {}
    sev = ctx.metrics.get("severity_eval") or {}
    totals = ctx.metrics.get("window_totals") or {}

    cnn_test = (cnn.get("metrics") or {}).get("test_threshold_0.5") or {}
    lstm_test = (cnn_lstm.get("metrics") or {}).get("test_threshold_0.5") or {}
    rec_t05 = (rec.get("recording_level") or {}).get("threshold_0_5") or {}
    overall = sev.get("overall_metrics") or {}

    lines: list[str] = []
    lines.append("# Fall Detection System — End-to-End Project Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Canonical severity run: `{ctx.canonical_severity_run}`")
    if ctx.severity_dir:
        lines.append(f"Severity artifacts: `{ctx.severity_dir.relative_to(ROOT).as_posix()}`")
    if ctx.severity_eval_dir:
        lines.append(f"Severity evaluation: `{ctx.severity_eval_dir.relative_to(ROOT).as_posix()}`")
    lines.append("")

    lines.append("## 1. Problem Statement & Motivation")
    lines.append("")
    lines.append(
        "Falls are a leading cause of injury and loss of independence among older adults. "
        "Wearable IMU sensors can detect falls in real time, but practical systems must handle "
        "class imbalance, subject variability, and noisy ADL activities while remaining deployable "
        "on low-power hardware. This project builds an end-to-end SisFall pipeline: ingestion, "
        "impact-centered preprocessing, unsupervised severity proxy labeling, and compact CNN/CNN-LSTM detectors."
    )
    lines.append("")

    lines.append("## 2. Dataset Overview")
    lines.append("")
    total_windows = ctx.metrics.get("total_windows")
    if totals:
        lines.append(
            f"- Processed windows: **{total_windows:,}** total — ADL/background **{totals.get('ADL', 0):,}**, fall **{totals.get('Fall', 0):,}**."
        )
        if total_windows:
            fall_pct = 100.0 * totals.get("Fall", 0) / total_windows
            lines.append(f"- Overall fall-window proportion: **{fall_pct:.2f}%** (impact-centered labeling).")
    lines.append("- SisFall: 38 subjects (26 train / 6 val / 6 test), subject-disjoint splits at ~70/15/15.")
    lines.append("- Tensor shape per window: `(64, 6)` at 20 Hz → 3.2 s; channels: acc1 xyz + gyro xyz.")
    if plot_paths.get("class_balance"):
        lines.append(f"\n![Class balance]({plot_paths['class_balance']})")
    if plot_paths.get("sample_waveforms"):
        lines.append(f"\n![Sample waveforms]({plot_paths['sample_waveforms']})")
    lines.append("")

    lines.append("## 3. Preprocessing & Windowing")
    lines.append("")
    lines.append("- Retained 6 channels (ADXL345 acc1 + ITG3200 gyro); removed second accelerometer (MMA8451Q).")
    lines.append("- Butterworth low-pass 5 Hz (order 4) applied per recording **before** 200 Hz → 20 Hz downsampling.")
    lines.append("- Train-only StandardScaler on 6 channels; sliding windows 64 samples, stride 16 (75% overlap).")
    lines.append("- Impact-centered fall labels: only windows containing detected impact peak are label 1.")
    if plot_paths.get("preprocessing_windowing"):
        lines.append(f"\n![Preprocessing windowing]({plot_paths['preprocessing_windowing']})")
    lines.append("")

    lines.append("## 4. Exploratory Data Analysis")
    lines.append("")
    lines.append("- Falls show higher peak signal magnitude vector (SMV) than ADL windows in normalized space.")
    if plot_paths.get("smv_distribution"):
        lines.append(f"\n![SMV distribution]({plot_paths['smv_distribution']})")
    if plot_paths.get("fall_spike"):
        lines.append(f"\n![Fall spike example]({plot_paths['fall_spike']})")
    lines.append("")

    lines.append("## 5. Baseline Model (CNN-only)")
    lines.append("")
    lines.append("- Architecture: 3× Conv1D blocks + GlobalAveragePooling + Dense; isolates spatial/temporal convolution without recurrence.")
    lines.append("- SVM/hand-crafted features were not pursued; CNN baselines directly consume window tensors for fair comparison with CNN-LSTM.")
    if cnn_test:
        lines.append(
            f"- Test @ 0.50: precision **{fmt_pct(cnn_test.get('precision'))}**, "
            f"recall **{fmt_pct(cnn_test.get('recall'))}**, "
            f"F1 **{fmt_pct(cnn_test.get('classification_report', {}).get('fall', {}).get('f1-score'))}**."
        )
    if plot_paths.get("cnn_confusion"):
        lines.append(f"\n![CNN confusion matrix]({plot_paths['cnn_confusion']})")
    lines.append("")

    lines.append("## 6. Final Model (CNN-LSTM)")
    lines.append("")
    param_count = cnn_lstm.get("parameter_count")
    lines.append(
        f"- Architecture: Conv1D → Conv1D → LSTM(64, `unroll=True`) → GlobalAveragePooling → Dense; "
        f"**{param_count:,}** parameters. `unroll=True` expands the 16-step LSTM sequence explicitly for stable training on short windows."
    )
    lines.append(f"- Best epoch: **{cnn_lstm.get('best_epoch', 'N/A')}**; early stopping after **{cnn_lstm.get('epochs_completed', 'N/A')}** epochs.")
    if lstm_test:
        lines.append(
            f"- Test @ 0.50: precision **{fmt_pct(lstm_test.get('precision'))}**, "
            f"recall **{fmt_pct(lstm_test.get('recall'))}**, "
            f"F1 **{fmt_pct(lstm_test.get('f1'))}**."
        )
    if rec_t05:
        detected = rec_t05.get("fall_recordings_at_least_partially_detected")
        total_falls = rec_t05.get("total_fall_recordings")
        lines.append(
            f"- Recording-level fall recall @ 0.50: **{fmt_pct(rec_t05.get('recording_level_recall'))}** "
            f"({detected}/{total_falls} fall recordings)."
        )
    if plot_paths.get("cnn_lstm_training"):
        lines.append(f"\n![CNN-LSTM training curves]({plot_paths['cnn_lstm_training']})")
    if plot_paths.get("cnn_lstm_confusion"):
        lines.append(f"\n![CNN-LSTM confusion matrix]({plot_paths['cnn_lstm_confusion']})")
    if plot_paths.get("cnn_lstm_pr"):
        lines.append(f"\n![CNN-LSTM PR curve]({plot_paths['cnn_lstm_pr']})")
    lines.append("")

    lines.append("## 7. Severity Labeling Pipeline")
    lines.append("")
    lines.append("- Stage 1: 58 hand-crafted IMU features from fall windows → train-only StandardScaler → K-Means k=3.")
    lines.append("- Stage 2: Isolation Forest (contamination 2%, 256 trees) flags atypical falls as Uncertain (-1).")
    lines.append("- Cluster→severity mapping by ascending acc/gyro peak intensity: Mild / Moderate / Severe.")
    lines.append("- **Sensor-derived relative severity**, not clinically validated injury severity.")
    if plot_paths.get("severity_pca"):
        lines.append(f"\n![Severity PCA clusters]({plot_paths['severity_pca']})")
    if plot_paths.get("severity_peak"):
        lines.append(f"\n![Severity peak intensity]({plot_paths['severity_peak']})")
    if plot_paths.get("anomaly_scores"):
        lines.append(f"\n![Anomaly score distribution]({plot_paths['anomaly_scores']})")
    lines.append("")

    lines.append("## 8. Severity-wise Model Evaluation")
    lines.append("")
    if sev.get("window_metrics"):
        for row in sev["window_metrics"]:
            if row["severity_label"] in (-1, 0, 1, 2):
                lines.append(
                    f"- {row['severity']}: recall **{fmt_pct(row['window_recall'])}** "
                    f"({row['detected_windows']}/{row['windows']} windows)."
                )
    if plot_paths.get("severity_recall"):
        lines.append(f"\n![Severity recall]({plot_paths['severity_recall']})")
    lines.append("")

    lines.append("## 9. Known Limitations & Open Issues")
    lines.append("")
    lines.append("- Controlled laboratory falls (SisFall); limited generalization to naturalistic/unwitnessed falls.")
    lines.append("- 38 subjects — moderate sample size; subject-disjoint splits reduce leakage but increase variance.")
    if ctx.open_issues:
        for issue in ctx.open_issues:
            lines.append(f"- **Open Issue:** {issue}")
    else:
        lines.append("- No cross-file metric disagreements detected for ADL false-trigger counts.")
    lines.append("- README / PROJECT_STATUS_REPORT may lag `run_metadata.json`; prefer artifact-backed metrics in this report.")
    lines.append("- Severity labels are unsupervised proxies; do not claim clinical ground truth.")
    lines.append("")

    lines.append("## 10. Next Steps")
    lines.append("")
    lines.append("1. Freeze one canonical timestamped run (preprocessing → severity k=3 → evaluation → diagnostics).")
    lines.append("2. Multi-task model: binary fall head + severity head on detected impact windows.")
    lines.append("3. FiLM / subject-conditioning prototype for cross-subject robustness.")
    lines.append("4. Four-way comparison table: CNN vs CNN-LSTM vs multi-task vs FiLM (same splits/threshold protocol).")
    lines.append("5. Cross-file consistency validator (TP/FP/FN/TN, recording FP counts, severity/anomaly counts).")
    lines.append("6. Subject/activity robustness analysis (e.g., ADL D13/D14 borderline activities).")
    lines.append("")

    if ctx.missing_artifacts:
        lines.append("## Missing or Failed Artifacts")
        lines.append("")
        for item in ctx.missing_artifacts:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    ctx = ReportContext()
    resolve_canonical_paths(ctx)
    collect_core_metrics(ctx)
    splits = load_processed_splits(ctx)

    plot_paths: dict[str, str | None] = {}

    plot_paths["class_balance"] = safe_plot("class_balance.png", lambda: plot_class_balance(ctx, splits), ctx)
    if splits:
        plot_paths["sample_waveforms"] = safe_plot(
            "sample_waveforms_fall_vs_adl.png", lambda: plot_sample_waveforms(splits), ctx
        )
        plot_paths["preprocessing_windowing"] = safe_plot(
            "preprocessing_windowing_example.png", lambda: plot_preprocessing_windowing(splits), ctx
        )
        plot_paths["smv_distribution"] = safe_plot("smv_distribution.png", lambda: plot_smv_distribution(splits), ctx)
        plot_paths["fall_spike"] = safe_plot("fall_spike_example.png", lambda: plot_fall_spike_example(splits), ctx)

    cnn_meta = ctx.metrics.get("cnn") or {}
    cnn_test_cm = ((cnn_meta.get("metrics") or {}).get("test_threshold_0.5") or {}).get("confusion_matrix")
    if cnn_test_cm:
        plot_paths["cnn_confusion"] = safe_plot(
            "cnn_confusion_matrix.png",
            lambda: plot_confusion_matrix(cnn_test_cm, "CNN test @ 0.50", "cnn_confusion_matrix.png", ["ADL", "Fall"]),
            ctx,
        )

    lstm_meta = ctx.metrics.get("cnn_lstm") or {}
    lstm_test_cm = ((lstm_meta.get("metrics") or {}).get("test_threshold_0.5") or {}).get("confusion_matrix")
    if lstm_test_cm:
        plot_paths["cnn_lstm_confusion"] = safe_plot(
            "cnn_lstm_confusion_matrix.png",
            lambda: plot_confusion_matrix(
                lstm_test_cm, "CNN-LSTM test @ 0.50", "cnn_lstm_confusion_matrix.png", ["ADL", "Fall"]
            ),
            ctx,
        )

    plot_paths["cnn_lstm_training"] = safe_plot(
        "cnn_lstm_training_curves.png", lambda: plot_training_curves(ctx), ctx
    )
    plot_paths["cnn_lstm_pr"] = safe_plot(
        "cnn_lstm_precision_recall_curve.png", lambda: plot_threshold_curves(ctx), ctx
    )
    plot_paths["severity_pca"] = safe_plot("severity_pca_clusters.png", lambda: plot_severity_pca(ctx), ctx)
    plot_paths["severity_peak"] = safe_plot("severity_peak_intensity.png", lambda: plot_severity_peak_boxplot(ctx), ctx)
    plot_paths["anomaly_scores"] = safe_plot("anomaly_score_distribution.png", lambda: plot_anomaly_scores(ctx), ctx)
    plot_paths["severity_recall"] = safe_plot(
        "severity_recall_by_class.png", lambda: plot_severity_recall_bar(ctx), ctx
    )

    report_md = build_report(ctx, plot_paths)
    report_path = REPORT_DIR / "PROJECT_REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"Report written to: {report_path}")
    print(f"Plots saved under: {ASSETS_DIR}")
    if ctx.open_issues:
        print("Open issues:")
        for issue in ctx.open_issues:
            print(f"  - {issue}")
    if ctx.missing_artifacts:
        print("Missing/failed artifacts:")
        for item in ctx.missing_artifacts:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
