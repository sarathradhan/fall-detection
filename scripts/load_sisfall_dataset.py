from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sisfall_loader import load_sisfall_dataset, print_summary


def main() -> None:
    """Run the dataset loading pipeline and print a summary."""

    dataset_payload = load_sisfall_dataset()
    print_summary(dataset_payload)


if __name__ == "__main__":
    main()
