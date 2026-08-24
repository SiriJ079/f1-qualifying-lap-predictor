import pandas as pd
from pathlib import Path

from src.models.train_utils import load_model, get_feature_columns
from src.models.data_split import load_features
import joblib
import shap


class PredictorService:
    def __init__(self):
        self.pipeline = load_model("xgboost_v2_tuned")
        self.explainer = joblib.load("models/artefacts/shap_explainer.pkl")
        self.lower_model = joblib.load("models/artefacts/quantile_lower.pkl")
        self.upper_model = joblib.load("models/artefacts/quantile_upper.pkl")
        self.df = load_features()
        self.feature_cols = get_feature_columns(self.df)

    def get_metadata(self):
        current_season = self.df["Year"].max()
        current_season_df = self.df[self.df["Year"] == current_season]

        current_drivers = current_season_df["Driver"].unique().tolist()
        all_circuits = self.df["EventName"].unique().tolist()

        if "RoundNumber" in current_season_df.columns and not current_season_df.empty:
            latest_round = int(current_season_df["RoundNumber"].max())
            last_updated = f"{int(current_season)} season, through round {latest_round}"
        else:
            last_updated = f"{int(current_season)} season"

        return {
            "drivers": sorted(current_drivers),
            "circuits": sorted(all_circuits),
            "teams": sorted(current_season_df["Team"].unique().tolist()),
            "last_updated": last_updated,
        }

    def _build_feature_row(self, driver: str, circuit: str, year: int) -> pd.DataFrame:
        row = self.df[
            (self.df["Driver"] == driver) &
            (self.df["EventName"] == circuit)
        ].sort_values("Year").tail(1)

        if row.empty:
            raise ValueError(f"No historical data found for {driver} at {circuit}")

        return row[self.feature_cols]

    def predict(self, driver: str, circuit: str, year: int = 2026):
        X = self._build_feature_row(driver, circuit, year)
        X_imputed = self.pipeline.named_steps["imputer"].transform(X)
        pred = self.pipeline.named_steps["model"].predict(X_imputed)[0]
        lower = self.lower_model.predict(X_imputed)[0]
        upper = self.upper_model.predict(X_imputed)[0]

        shap_vals = self.explainer.shap_values(X_imputed)
        top_features = (
            pd.DataFrame({"feature": self.feature_cols, "impact": shap_vals[0]})
            .sort_values("impact", key=abs, ascending=False)
            .head(5)
            .to_dict(orient="records")
        )

        return {
            "driver": driver,
            "circuit": circuit,
            "predicted_delta_s": round(float(pred), 3),
            "lower_bound_s": round(float(lower), 3),
            "upper_bound_s": round(float(upper), 3),
            "top_features": top_features,
        }