import shap
import pandas as pd
from pathlib import Path

from src.models.data_split import load_features, time_based_split
from src.models.train_utils import get_feature_columns, prepare_xy, load_model

pipeline = load_model("xgboost_v2_tuned")
df = load_features()
train, val, test = time_based_split(df)
feature_cols = get_feature_columns(df)

X_train, y_train = prepare_xy(train, feature_cols) 
X_test, y_test = prepare_xy(test, feature_cols)

X_test_imputed = pipeline.named_steps["imputer"].transform(X_test)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=feature_cols)

explainer = shap.TreeExplainer(pipeline.named_steps["model"])
shap_values = explainer.shap_values(X_test_imputed)

Path("output").mkdir(parents=True, exist_ok=True)
shap.summary_plot(shap_values, X_test_imputed, show=False)
import matplotlib.pyplot as plt
plt.savefig("output/shap_summary.png", bbox_inches="tight")

def explain_single_prediction(row_index: int):
    single_row = X_test_imputed.iloc[[row_index]]
    shap_val = explainer.shap_values(single_row)
    return pd.DataFrame({
        "feature": feature_cols,
        "shap_value": shap_val[0],
        "feature_value": single_row.values[0],
    }).sort_values("shap_value", key=abs, ascending=False)

import xgboost as xgb

X_train_imputed = pipeline.named_steps["imputer"].transform(X_train)
X_train_imputed = pd.DataFrame(X_train_imputed, columns=feature_cols)

def train_quantile_model(X_train, y_train, quantile: float):
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=150, max_depth=3, learning_rate=0.05,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

lower_model = train_quantile_model(X_train_imputed, y_train, 0.1)
upper_model = train_quantile_model(X_train_imputed, y_train, 0.9)

import joblib
joblib.dump(explainer, "models/artefacts/shap_explainer.pkl")
joblib.dump(lower_model, "models/artefacts/quantile_lower.pkl")
joblib.dump(upper_model, "models/artefacts/quantile_upper.pkl")