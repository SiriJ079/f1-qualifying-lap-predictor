import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.data_split import load_features, time_based_split
from src.models.evaluate import evaluate_predictions
from src.models.train_utils import get_feature_columns, prepare_xy, save_model, save_metrics


def train_linear_model():
    df = load_features()
    train, val, test = time_based_split(df)

    feature_cols = get_feature_columns(df)
    X_train, y_train = prepare_xy(train, feature_cols)
    X_val, y_val = prepare_xy(val, feature_cols)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)

    metrics = evaluate_predictions(y_val, val_preds, label="Linear Regression (Ridge)")
    save_model(pipeline, "linear_ridge")
    save_metrics(metrics, "linear_ridge")

    return pipeline, metrics


if __name__ == "__main__":
    train_linear_model()