"""Feature alignment and sklearn preprocessor used by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, DROP_COLS, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COL


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(NUMERIC_FEATURES)),
            ("cat", categorical_pipe, list(CATEGORICAL_FEATURES)),
        ]
    )


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = prepare_features(frame, require_label=True)
    y = cleaned[TARGET_COL].astype(int)
    x = cleaned[FEATURE_COLUMNS]
    return x, y


def prepare_features(frame: pd.DataFrame, require_label: bool = False) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(col).strip() for col in data.columns]

    for extra in DROP_COLS:
        if extra in data.columns:
            data = data.drop(columns=[extra])

    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(
            "CSV is missing required feature columns: " + ", ".join(missing)
        )

    if require_label and TARGET_COL not in data.columns:
        raise ValueError(f"Training data must include a '{TARGET_COL}' column (0 = Normal, 1 = Attack).")

    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype(str).str.strip().replace({"nan": "-", "None": "-"})
        data[col] = data[col].replace({"": "-"})

    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if TARGET_COL in data.columns:
        data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
        data = data.dropna(subset=[TARGET_COL])
        data[TARGET_COL] = data[TARGET_COL].astype(int)

    keep = list(FEATURE_COLUMNS)
    if TARGET_COL in data.columns:
        keep.append(TARGET_COL)
    return data[keep]


def default_row() -> dict[str, object]:
    """Sensible defaults for the single-flow form (benign-looking TCP/HTTP)."""
    values: dict[str, object] = {
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "dur": 0.25,
        "spkts": 10,
        "dpkts": 8,
        "sbytes": 580,
        "dbytes": 1400,
        "rate": 72.0,
        "sttl": 254,
        "dttl": 252,
        "sload": 18560.0,
        "dload": 44800.0,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 28.0,
        "dinpkt": 32.0,
        "sjit": 4.5,
        "djit": 5.2,
        "swin": 255,
        "stcpb": 1_000_000,
        "dtcpb": 1_200_000,
        "dwin": 255,
        "tcprtt": 0.012,
        "synack": 0.006,
        "ackdat": 0.006,
        "smean": 58,
        "dmean": 175,
        "trans_depth": 1,
        "response_body_len": 450,
        "ct_srv_src": 2,
        "ct_state_ttl": 1,
        "ct_dst_ltm": 2,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 1,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 1,
        "ct_src_ltm": 2,
        "ct_srv_dst": 2,
        "is_sm_ips_ports": 0,
    }
    return values


def row_to_frame(values: dict[str, object]) -> pd.DataFrame:
    ordered = {col: [values[col]] for col in FEATURE_COLUMNS}
    return prepare_features(pd.DataFrame(ordered), require_label=False)


def make_synthetic_dataset(n_samples: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate a labeled stand-in when UNSW-NB15 CSVs are not on disk."""
    rng = np.random.default_rng(seed)
    n_attack = n_samples // 2
    n_normal = n_samples - n_attack

    def _block(n: int, attack: bool) -> pd.DataFrame:
        proto = rng.choice(["tcp", "udp", "icmp", "arp"], size=n, p=[0.62, 0.28, 0.07, 0.03])
        service = rng.choice(["-", "http", "dns", "smtp", "ftp", "ssh", "ssl"], size=n)
        state = rng.choice(["FIN", "INT", "CON", "REQ", "RST"], size=n)
        rate_scale = 1800 if attack else 80
        ttl_src = 64 if attack else 254
        rows = {
            "proto": proto,
            "service": service,
            "state": state,
            "dur": np.clip(rng.lognormal(mean=-1.2 if attack else -0.4, sigma=0.8, size=n), 1e-6, 60),
            "spkts": rng.integers(1, 40 if attack else 25, size=n),
            "dpkts": rng.integers(0, 8 if attack else 22, size=n),
            "sbytes": rng.integers(40, 8000 if attack else 2500, size=n),
            "dbytes": rng.integers(0, 900 if attack else 4000, size=n),
            "rate": np.clip(rng.normal(rate_scale, rate_scale * 0.4, size=n), 0.1, None),
            "sttl": np.full(n, ttl_src) + rng.integers(-2, 3, size=n),
            "dttl": rng.choice([0, 29, 252], size=n),
            "sload": np.clip(rng.normal(5e5 if attack else 2e4, 8e4, size=n), 0, None),
            "dload": np.clip(rng.normal(1e3 if attack else 4e4, 1e4, size=n), 0, None),
            "sloss": rng.integers(0, 6 if attack else 2, size=n),
            "dloss": rng.integers(0, 3, size=n),
            "sinpkt": np.clip(rng.normal(2 if attack else 30, 8, size=n), 0, None),
            "dinpkt": np.clip(rng.normal(1 if attack else 28, 8, size=n), 0, None),
            "sjit": np.clip(rng.normal(15 if attack else 4, 6, size=n), 0, None),
            "djit": np.clip(rng.normal(12 if attack else 5, 6, size=n), 0, None),
            "swin": rng.choice([0, 255], size=n, p=[0.35 if attack else 0.1, 0.65 if attack else 0.9]),
            "stcpb": rng.integers(0, 2_000_000, size=n),
            "dtcpb": rng.integers(0, 2_000_000, size=n),
            "dwin": rng.choice([0, 255], size=n),
            "tcprtt": np.clip(rng.normal(0.03 if attack else 0.01, 0.01, size=n), 0, None),
            "synack": np.clip(rng.normal(0.02, 0.01, size=n), 0, None),
            "ackdat": np.clip(rng.normal(0.01, 0.008, size=n), 0, None),
            "smean": rng.integers(40, 400, size=n),
            "dmean": rng.integers(0, 500, size=n),
            "trans_depth": rng.integers(0, 4, size=n),
            "response_body_len": rng.integers(0, 2000, size=n),
            "ct_srv_src": rng.integers(1, 20 if attack else 6, size=n),
            "ct_state_ttl": rng.integers(0, 6, size=n),
            "ct_dst_ltm": rng.integers(1, 16 if attack else 5, size=n),
            "ct_src_dport_ltm": rng.integers(1, 16 if attack else 4, size=n),
            "ct_dst_sport_ltm": rng.integers(1, 16 if attack else 4, size=n),
            "ct_dst_src_ltm": rng.integers(1, 18 if attack else 5, size=n),
            "is_ftp_login": rng.integers(0, 2, size=n),
            "ct_ftp_cmd": rng.integers(0, 4, size=n),
            "ct_flw_http_mthd": rng.integers(0, 5, size=n),
            "ct_src_ltm": rng.integers(1, 12, size=n),
            "ct_srv_dst": rng.integers(1, 14 if attack else 5, size=n),
            "is_sm_ips_ports": rng.integers(0, 2, size=n),
            "label": np.full(n, 1 if attack else 0),
        }
        return pd.DataFrame(rows)

    data = pd.concat([_block(n_normal, False), _block(n_attack, True)], ignore_index=True)
    return data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
