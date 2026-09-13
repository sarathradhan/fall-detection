from __future__ import annotations

import json
import pickle
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SEVERITY_DIR = PROCESSED_DIR / "severity"


def _validate_run(run_dir: Path) -> None:
    with (run_dir / "kmeans_model.pkl").open("rb") as handle:
        kmeans = pickle.load(handle)
    if int(kmeans.n_clusters) != 3:
        raise ValueError(f"Expected k=3 in {run_dir / 'kmeans_model.pkl'}, found k={kmeans.n_clusters}.")

    expected_rows = {"train": 4484, "val": 1500, "test": 1200}
    import numpy as np
    import pandas as pd
    for split, expected_count in expected_rows.items():
        labels = np.load(run_dir / f"{split}_severity_labels.npy")
        refined = np.load(run_dir / f"{split}_refined_severity_labels.npy")
        flags = np.load(run_dir / f"{split}_anomaly_flags.npy")
        features = np.load(run_dir / "clustering_features" / f"{split}_fall_features.npy")
        if len(labels) != len(refined):
            raise ValueError(f"{split}: severity arrays are not aligned.")
        if len(flags) != len(features) or len(flags) != expected_count:
            raise ValueError(f"{split}: fall-window artifacts have unexpected length.")

    report = run_dir / "reports" / "post_isolation_forest_fall_windows.csv"
    if not report.exists():
        raise FileNotFoundError(f"Missing regenerated fall report: {report}")
    report_rows = len(pd.read_csv(report))
    if report_rows != sum(expected_rows.values()):
        raise ValueError(f"Expected 7,184 fall-report rows, found {report_rows}.")


def _promote_run(run_dir: Path) -> None:
    """Make the validated run the canonical artifact set without copying old runs."""

    for source in run_dir.iterdir():
        if source.name in {"artifact_provenance.json", "reports"}:
            continue
        destination = SEVERITY_DIR / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    reports_source = run_dir / "reports"
    reports_destination = SEVERITY_DIR / "reports"
    shutil.copytree(reports_source, reports_destination, dirs_exist_ok=True)


def main() -> None:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("k3_%Y%m%dT%H%M%SZ")
    run_dir = PROCESSED_DIR / "severity" / "runs" / run_id
    command = [sys.executable, str(ROOT / "scripts" / "run_versioned_k3_pipeline.py")]
    run_dir.mkdir(parents=True, exist_ok=False)

    severity_command = [
        sys.executable,
        str(ROOT / "scripts" / "run_severity_labeling.py"),
        "--output-dir",
        str(run_dir),
        "--n-clusters",
        "3",
    ]
    export_command = [
        sys.executable,
        str(ROOT / "scripts" / "export_post_isolation_forest_windows.py"),
        "--severity-dir",
        str(run_dir),
        "--output-dir",
        str(run_dir / "reports"),
    ]
    subprocess.run(severity_command, cwd=ROOT, check=True)
    subprocess.run(export_command, cwd=ROOT, check=True)
    _validate_run(run_dir)
    _promote_run(run_dir)

    provenance = {
        "run_id": run_id,
        "generated_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in command),
        "severity_command": " ".join(shlex.quote(part) for part in severity_command),
        "report_export_command": " ".join(shlex.quote(part) for part in export_command),
        "selected_k": 3,
        "canonical_artifacts": str(SEVERITY_DIR),
        "gyro_audit": {
            "source": "data/processed/GYRO_OUTLIER_BY_ACTIVITY.md",
            "status": "reviewed_existing_report; not rerun",
            "finding": "ADL gyro tail is concentrated in D05 and D17 maximum values; no relabeling change was made.",
        },
        "uncertain_window_policy": {
            "isolation_forest_flag": -1,
            "policy": "abstain_from_severity_training_and_evaluation; retain original cluster label separately",
            "reason": "Isolation Forest marks unusual fall windows as uncertain, not as a clinical severity class.",
        },
        "fit_scope": {
            "preprocessing_scaler": "preprocessing train split only",
            "severity_feature_scaler": "train fall windows only",
            "kmeans": "train fall windows only",
            "isolation_forest": "train fall windows only",
        },
    }
    provenance_path = run_dir / "artifact_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    shutil.copy2(provenance_path, SEVERITY_DIR / provenance_path.name)
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "provenance": str(run_dir / 'artifact_provenance.json')}, indent=2))


if __name__ == "__main__":
    main()