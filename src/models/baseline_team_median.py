import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.data_split import load_features, time_based_split
from src.models.evaluate import evaluate_predictions


def predict_team_median(df: pd.DataFrame) -> pd.Series:
    """Use team's rolling pace, lightly adjusted by driver's own recent form."""
    team_component = df["TeamRollingDelta_5"].fillna(df["TeamCareerMedianDelta"])
    driver_adjustment = (df["DriverRollingDelta_5"] - df["DriverCareerMedianDelta"]).fillna(0) * 0.3
    return team_component + driver_adjustment


if __name__ == "__main__":
    df = load_features()
    train, val, test = time_based_split(df)

    val_preds = predict_team_median(val)
    val = val.copy()
    val["Prediction"] = val_preds.fillna(val["DeltaToFastest_s"].median())

    metrics = evaluate_predictions(val["DeltaToFastest_s"], val["Prediction"], label="Baseline C — Team Median")