from __future__ import annotations

import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
from src.data.severity_labeling import run_severity_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "processed" / "severity")
    parser.add_argument("--n-clusters", type=int, default=None)
    args = parser.parse_args()
    result = run_severity_pipeline(
        processed_dir=Path("data") / "processed",
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        random_state=42,
        n_init=20,
        k_values=[2, 3, 4, 5],
    )
    selected_k = int(result.severity_summary.get("selected_k", 3))
    print("\nBEST K =", selected_k)
    print("Reason:")
    print(f"- Tested k values: {result.severity_summary.get('tested_k_values', [])}")
    print(f"- Selection rule: {result.severity_summary.get('selection_rule', 'not available')}")
    print(f"- k=3 statement: {result.severity_summary.get('k3_statement', 'not available')}")
    print("- Cluster sizes and profiles were saved under data/processed/severity/")
    print("Severity labeling complete.")
    print("Summary:")
    print(result.severity_summary)
