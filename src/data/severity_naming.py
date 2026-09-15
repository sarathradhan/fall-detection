"""Shared severity category naming for a versioned severity run.

Used by every script that displays or groups by severity category (the CNN-LSTM
severity evaluator, cluster plots, requested plots, the project report, and the
post-isolation-forest window export) so the mapping can't drift out of sync
between them or silently assume a fixed cluster count.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_severity_names(severity_dir: Path) -> dict[int, str]:
    # No hardcoded fallback: a fixed category count here previously let a K=2
    # run silently report stale K=3 category names. Raise instead of guessing.
    summary_path = severity_dir / "severity_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"severity_summary.json not found under {severity_dir}; cannot determine "
            "severity category names for this run without it."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    names = summary.get("severity_names")
    if not isinstance(names, dict) or not names:
        raise ValueError(
            f"{summary_path} has no usable 'severity_names' mapping; refusing to fall "
            "back to a hardcoded category set that may not match this run's selected_k."
        )
    parsed = {int(label): str(name) for label, name in names.items()}
    if -1 in parsed:
        parsed[-1] = "Uncertain (-1)"

    selected_k = summary.get("selected_k")
    non_uncertain_categories = sorted(label for label in parsed if label >= 0)
    if selected_k is not None and len(non_uncertain_categories) != selected_k:
        raise ValueError(
            f"{summary_path} declares selected_k={selected_k} but severity_names has "
            f"{len(non_uncertain_categories)} non-uncertain categories "
            f"({non_uncertain_categories}); the mapping is inconsistent with the "
            "clustering run and would produce a mismatched evaluation."
        )
    return parsed
