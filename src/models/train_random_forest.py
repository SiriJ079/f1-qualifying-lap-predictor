import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.data_split import load_features, time_based_split
from src.models.evaluate import evaluate_predictions
from src.models.train_utils import get_feature_columns, prepare_xy, save_model, save_metrics


def train_random_forest():
    df = load_features()
    train, val, test = time_based_split(df)

    feature_cols = get_feature_columns(df)
    X_train, y_train = prepare_xy(train, feature_cols)
    X_val, y_val = prepare_xy(val, feature_cols)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )),
    ])

    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)

    metrics = evaluate_predictions(y_val, val_preds, label="Random Forest")
    save_model(pipeline, "random_forest")
    save_metrics(metrics, "random_forest")

    return pipeline, metrics


if __name__ == "__main__":
    train_random_forest()