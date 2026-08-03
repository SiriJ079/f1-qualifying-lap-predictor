import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.data_split import load_features, time_based_split
from src.models.evaluate import evaluate_predictions
from src.models.train_utils import get_feature_columns, prepare_xy, save_model, save_metrics

BEST_PARAMS_PATH = Path("models/metrics/best_params.json")


def load_best_params() -> dict:
    """Load Optuna-tuned hyperparameters, falling back to manual defaults if missing."""
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            params = json.load(f)
        print(f"Loaded tuned params from {BEST_PARAMS_PATH}: {params}")
        return params
    print("No tuned params found — using manual defaults.")
    return {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_lambda": 8.0,
        "reg_alpha": 1.0,
        "min_child_weight": 10,
    }


def train_xgboost():
    df = load_features()
    train, val, test = time_based_split(df)

    feature_cols = get_feature_columns(df)
    X_train, y_train = prepare_xy(train, feature_cols)
    X_val, y_val = prepare_xy(val, feature_cols)
    X_test, y_test = prepare_xy(test, feature_cols)

    best_params = load_best_params()

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1
        )),
    ])

    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)

    val_metrics = evaluate_predictions(y_val, val_preds, label="XGBoost — VAL (tuned)")
    test_metrics = evaluate_predictions(y_test, test_preds, label="XGBoost — TEST (tuned)")

    save_model(pipeline, "xgboost_v2_tuned")
    save_metrics(val_metrics, "xgboost_v2_tuned_val")
    save_metrics(test_metrics, "xgboost_v2_tuned_test")

    return pipeline, val_metrics, test_metrics


if __name__ == "__main__":
    train_xgboost()