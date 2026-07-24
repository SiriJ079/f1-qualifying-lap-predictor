import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.data_split import load_features, time_based_split
from src.models.evaluate import evaluate_predictions


def predict_driver_rolling(df: pd.DataFrame) -> pd.Series:
    """Use the driver's 5-session rolling delta as the prediction."""
    return df["DriverRollingDelta_5"].fillna(df["DriverCareerMedianDelta"])


if __name__ == "__main__":
    df = load_features()
    train, val, test = time_based_split(df)

    val_preds = predict_driver_rolling(val)
    val = val.copy()
    val["Prediction"] = val_preds.fillna(val["DeltaToFastest_s"].median())

    metrics = evaluate_predictions(val["DeltaToFastest_s"], val["Prediction"], label="Baseline A — Driver Rolling")