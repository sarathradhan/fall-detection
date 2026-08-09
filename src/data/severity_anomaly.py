from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


ANOMALY_CONTAMINATION = 0.02
ANOMALY_N_ESTIMATORS = 256


def fit_isolation_forest(
    features: np.ndarray,
    contamination: float = ANOMALY_CONTAMINATION,
    random_state: int = 42,
    n_estimators: int = ANOMALY_N_ESTIMATORS,
) -> IsolationForest:
    if features.ndim != 2:
        raise ValueError("Expected 2D feature matrix for Isolation Forest training.")
    if features.shape[0] == 0:
        raise ValueError("Training features must contain at least one sample.")

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(features)
    return model


def compute_anomaly_scores(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("Expected 2D feature matrix for Isolation Forest scoring.")
    return model.decision_function(features)


def compute_anomaly_flags(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("Expected 2D feature matrix for Isolation Forest prediction.")
    return (model.predict(features) == -1).astype(np.int64)


def _save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
