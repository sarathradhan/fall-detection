"""Shared, version-skew-tolerant loading for this project's saved .keras models.

Used by every script that loads a previously-trained CNN or CNN-LSTM model
instead of building and training one, so the workaround lives in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def load_keras_model(tf: Any, model_path: Path, build_model_fn: Callable[[Any], Any]) -> Any:
    # Some saved .keras files carry an initializer config (e.g. GlorotUniform's
    # input_axes/output_axes) newer than the installed Keras can deserialize,
    # which breaks load_model() even though the weights themselves are fine.
    # Fall back to rebuilding the documented architecture (via build_model_fn,
    # e.g. train_cnn_lstm_baseline.build_model) and loading weights only,
    # which doesn't require deserializing the initializer configs.
    try:
        return tf.keras.models.load_model(model_path)
    except TypeError as exc:
        if "could not be deserialized properly" not in str(exc):
            raise
        model = build_model_fn(tf)
        model.load_weights(model_path)
        return model
