"""Load serialized artifacts and run inference (no training)."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import LABEL_NAMES, METADATA_PATH, MODEL_PATH, PREPROCESSOR_PATH
from src.preprocessing import prepare_features


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the Streamlit app cannot find saved model artifacts."""


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[object, object, dict]:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise ModelNotTrainedError(
            "Trained artifacts were not found. Run `python train.py` first "
            f"(expected {MODEL_PATH.name} and {PREPROCESSOR_PATH.name} in models/)."
        )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metadata: dict = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, preprocessor, metadata


def predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    model, preprocessor, _metadata = load_artifacts()
    features = prepare_features(frame, require_label=False)
    transformed = preprocessor.transform(features)
    labels = model.predict(transformed).astype(int)
    probabilities = _positive_class_proba(model, transformed)
    output = features.copy()
    output["prediction"] = labels
    output["prediction_label"] = output["prediction"].map(LABEL_NAMES)
    output["attack_probability"] = probabilities
    return output


def _positive_class_proba(model: object, transformed) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        return np.full(transformed.shape[0], np.nan)
    proba = model.predict_proba(transformed)
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in list(classes):
        idx = int(np.where(classes == 1)[0][0])
        return proba[:, idx]
    return proba[:, -1]
