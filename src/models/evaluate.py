import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, label: str = "Model") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))

    results = {
        "Model": label,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MedianAE": round(median_ae, 4),
    }
    print(f"{label}: MAE={mae:.3f}s | RMSE={rmse:.3f}s | MedianAE={median_ae:.3f}s")
    return results


def evaluate_by_group(df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str) -> pd.DataFrame:
    """Break down error by a grouping column — e.g. Team, EventName, Driver."""
    def group_mae(g):
        return mean_absolute_error(g[y_true_col], g[y_pred_col])

    result = df.groupby(group_col).apply(group_mae, include_groups=False).reset_index()
    result.columns = [group_col, "MAE"]
    return result.sort_values("MAE", ascending=False)