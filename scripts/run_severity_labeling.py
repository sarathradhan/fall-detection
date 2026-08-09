from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
from src.data.severity_labeling import run_severity_pipeline

if __name__ == "__main__":
    result = run_severity_pipeline(
        processed_dir=Path("data") / "processed",
        output_dir=Path("data") / "processed" / "severity",
        n_clusters=3,
        random_state=42,
        n_init=20,
    )
    print("Severity labeling complete.")
    print("Summary:")
    print(result.severity_summary)
    print("Cluster statistics saved to:")
    print(result.cluster_stats)
