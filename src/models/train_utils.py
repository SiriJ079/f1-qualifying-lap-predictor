import pandas as pd
import numpy as np
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTEFACTS_DIR = PROJECT_ROOT / "models" / "artefacts"
METRICS_DIR = PROJECT_ROOT / "models" / "metrics"
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "DeltaToFastest_s"
ID_COLS = ["Year", "RoundNumber", "EventName", "Driver", "Team"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except target and identifiers."""
    return [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]


def prepare_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y


def save_model(model, name: str):
    path = ARTEFACTS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def save_metrics(metrics: dict, name: str):
    path = METRICS_DIR / f"{name}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(path, index=False)
    print(f"Saved metrics to {path}")