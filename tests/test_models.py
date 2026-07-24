from pathlib import Path
import pandas as pd


def test_xgboost_model_saved():
    path = Path("models/artefacts/xgboost_v1.pkl")
    assert path.exists(), "XGBoost model artefact not found — run train_xgboost.py first"


def test_xgboost_beats_baseline():
    baseline_path = Path("models/metrics/baseline_best.csv")
    xgb_path = Path("models/metrics/xgboost_v1_metrics.csv")

    if not (baseline_path.exists() and xgb_path.exists()):
        return

    baseline_mae = pd.read_csv(baseline_path)["MAE"].iloc[0]
    xgb_mae = pd.read_csv(xgb_path)["MAE"].iloc[0]

    assert xgb_mae <= baseline_mae, (
        f"XGBoost MAE ({xgb_mae}) does not beat Baseline A ({baseline_mae}) — "
        "revisit features before proceeding to tuning"
    )