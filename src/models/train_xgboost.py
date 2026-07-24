import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from pathlib import Path

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.data_split import load_features, time_based_split
from src.models.evaluate import evaluate_predictions
from src.models.train_utils import get_feature_columns, prepare_xy, save_model, save_metrics


def train_xgboost():
    df = load_features()
    train, val, test = time_based_split(df)

    feature_cols = get_feature_columns(df)
    X_train, y_train = prepare_xy(train, feature_cols)
    X_val, y_val = prepare_xy(val, feature_cols)
    X_test, y_test = prepare_xy(test, feature_cols)  

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_lambda=8.0,
            reg_alpha=1.0,
            min_child_weight=10,
            random_state=42,
            n_jobs=-1
        )),
    ])

    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)  

    val_metrics = evaluate_predictions(y_val, val_preds, label="XGBoost — VAL")
    test_metrics = evaluate_predictions(y_test, test_preds, label="XGBoost — TEST")  

    residuals_df = test[["Year", "RoundNumber", "EventName", "Driver", "Team"]].copy()
    residuals_df["y_true"] = y_test.values
    residuals_df["y_pred"] = test_preds
    residuals_df["AbsResidual"] = (residuals_df["y_true"] - residuals_df["y_pred"]).abs()

    worst20 = residuals_df.sort_values("AbsResidual", ascending=False).head(20)
    worst20.to_csv(OUTPUT_DIR / "test_residuals_worst20.csv", index=False)

    print(worst20[["Year", "RoundNumber", "EventName", "Driver", "Team", "y_true", "y_pred", "AbsResidual"]].to_string(index=False))
    save_model(pipeline, "xgboost_v1")
    save_metrics(val_metrics, "xgboost_v1_val")
    save_metrics(test_metrics, "xgboost_v1_test") 

    return pipeline, val_metrics, test_metrics  


if __name__ == "__main__":
    train_xgboost()
