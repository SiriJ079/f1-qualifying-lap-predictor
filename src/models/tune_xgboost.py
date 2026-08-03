import optuna
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from src.models.data_split import load_features, time_based_split
from src.models.train_utils import get_feature_columns, prepare_xy

df = load_features()
train, val, test = time_based_split(df)
feature_cols = get_feature_columns(df)
X_train, y_train = prepare_xy(train, feature_cols)
X_val, y_val = prepare_xy(val, feature_cols)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 15.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "random_state": 42,
        "n_jobs": -1,
    }

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBRegressor(**params)),
    ])
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)
    return mean_absolute_error(y_val, val_preds)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Best MAE:", study.best_value)
print("Best params:", study.best_params)

import json
with open("models/metrics/best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)