"""Train the NIDS classifier and persist model + preprocessor artifacts.

This script is the only place that fits estimators. The Streamlit app only loads
the files written to models/.

Place official UNSW-NB15 CSVs in data/ as:
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv

If those files are missing, a labeled synthetic set with the same feature names
is used so the application can still be demonstrated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    DATA_DIR,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    TEST_CSV,
    TRAIN_CSV,
)
from src.preprocessing import build_preprocessor, make_synthetic_dataset, split_xy


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_CSV.exists():
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV) if TEST_CSV.exists() else None
        source = "unsw-nb15"
        print(f"Loaded training CSV: {TRAIN_CSV} ({len(train_df):,} rows)")
        if test_df is not None:
            print(f"Loaded testing CSV: {TEST_CSV} ({len(test_df):,} rows)")
        return train_df, test_df, source

    print("UNSW-NB15 CSVs not found under data/. Training on a synthetic stand-in dataset.")
    full = make_synthetic_dataset(n_samples=5000)
    train_df, test_df = train_test_split(
        full, test_size=0.25, random_state=42, stratify=full["label"]
    )
    return train_df, test_df, "synthetic"


def train() -> dict:
    train_df, test_df, source = load_datasets()
    x_train, y_train = split_xy(train_df)

    preprocessor = build_preprocessor()
    x_train_t = preprocessor.fit_transform(x_train)

    # GradientBoostingClassifier has no class_weight argument; balanced sample
    # weights match the project's cost-sensitive learning setup.
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.85,
        random_state=42,
    )
    model.fit(x_train_t, y_train, sample_weight=sample_weight)

    if test_df is not None and len(test_df):
        x_eval, y_eval = split_xy(test_df)
    else:
        x_eval, y_eval = x_train, y_train

    x_eval_t = preprocessor.transform(x_eval)
    y_pred = model.predict(x_eval_t)

    metrics = {
        "accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
        "precision": round(float(precision_score(y_eval, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_eval, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_eval, y_pred, zero_division=0)), 4),
        "n_train": int(len(x_train)),
        "n_eval": int(len(x_eval)),
        "data_source": source,
        "model": "GradientBoostingClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "class_weight": "balanced (via sample_weight)",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(classification_report(y_eval, y_pred, target_names=["Normal", "Attack"], zero_division=0))
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {PREPROCESSOR_PATH}")
    print(f"Saved {METADATA_PATH}")
    return metrics


if __name__ == "__main__":
    train()
