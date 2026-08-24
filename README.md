# F1 Qualifying Lap Time Predictor

A machine learning model and interactive web application that predicts Formula 1
qualifying lap times for any driver at any circuit, with driver comparison and
model explainability.

## Project overview

**Question:** Given a driver, circuit, season context, and expected qualifying
conditions, what lap time will they set and how does it compare with the rest
of the grid?

**Data sources:** FastF1 (qualifying sessions, lap times, weather), OpenF1 (telemetry,
track conditions), Jolpica-F1 (historical driver and constructor records)

**Model approach:** Gradient boosting regression (XGBoost / LightGBM) with
time-aware cross-validation and SHAP-based explainability

**End product:** An interactive web app where you select a track and driver,
receive a predicted lap time with uncertainty range, and compare across multiple
drivers

## Key features

- Prediction for any driver–circuit combination with trained model
- Multi-driver comparison view with delta to predicted fastest
- Confidence interval display alongside point prediction
- SHAP feature contribution chart explaining each prediction
- Responsive UI with F1 visual identity

## Repository structure

- data/ Raw and processed datasets
- notebooks/ EDA, feature exploration, model experiments
- src/ Ingestion, feature engineering, model, and API modules
- app/ Front-end templates and static assets
- models/ Saved model artefacts and evaluation metrics
- tests/ Unit and integration tests

## Project Write-up

### Phase 4: Feature Engineering

Built `build_features.py` to construct the model-ready feature matrix, with every function carefully designed to avoid data leakage:

- **`load_clean_data()`** -- loads cleaned data and sorts strictly by Year/RoundNumber, since every rolling feature downstream depends on correct chronological order
- **`get_driver_session_best()`** -- collapses many lap rows per driver-session down to one row (their best qualifying lap), computing `DeltaToFastest_s` as the model target
- **`add_driver_rolling_features()`** -- rolling driver form over last 3/5/10 sessions and career median, using `.shift(1)` before `.rolling()` to guarantee only past sessions are used (critical leakage prevention)
- **`add_constructor_rolling_features()`** -- same rolling logic applied at team level
- **Circuit history, season context, and driver season-trend features** -- circuit-specific historical deltas, season progress, and within-season form trend, using `.expanding()` with shifting to avoid leakage
- **`add_compound_features()`** -- ordinal-encodes tyre compounds (SOFT=0 through WET=4) to preserve their natural performance ordering
- **`add_weather_features()`** -- fills missing weather values with circuit-specific medians and converts rainfall to a numeric flag

**Problems encountered and fixed:**

- **Missing rounds in the collected dataset.** Built a diagnostic cell comparing each season's official FastF1 event schedule against rounds actually present in the data, producing a clean "missing rounds" table by year and event name to identify exactly which sessions needed re-fetching.
- **Rebuilding features after a mid-season data update.** When updating the dataset with newly-completed 2026 races, the full pipeline (ingestion -> circuit metadata -> cleaning -> feature engineering -> tests) had to be re-run in the correct order, since rolling/historical features needed recalculating across the extended timeline rather than just appending new rows.

Added feature-specific tests: confirming the feature file exists, checking for target-leakage columns (anything containing "LapTime" or "SessionFastest" in the final matrix), and confirming rolling features are properly lagged (first session for any driver should have NaN rolling delta, since there's nothing to roll over yet).

**Result:** A clean, leakage-checked, model-ready feature matrix saved to `data/features/qualifying_features.parquet`.

---

### Phase 5: Baseline Models

Built three simple, honest baselines before touching XGBoost, on the principle that a real ML model should have to clearly beat naive heuristics before its added complexity is justified:

| Baseline | Logic |
|---|---|
| A -- Rolling driver average | Predict using the driver's own recent rolling delta |
| B -- Last result at circuit | Predict using the driver's last visit to this specific circuit |
| C -- Team median pace | Predict using the team's median delta, adjusted by driver form |

Implemented a chronological (not random) train/validation/test split via `time_based_split()` in `data_split.py`, since this is time-series data and a random split would leak future information into training.

Built each baseline as its own script (`baseline_circuit_history.py`, `baseline_team_median.py`, etc.), each loading the same feature matrix, generating predictions, filling any nulls with the training median, and evaluating with a shared `evaluate_predictions()` function for consistent MAE/RMSE reporting -- then compared all three side by side in a dedicated notebook. Baseline A's rolling-average approach set the benchmark MAE of 0.835s that the eventual XGBoost model needed to beat.

---

### Phase 6: Model Building & Comparison

Trained an XGBoost regression pipeline and evaluated it against the Phase 5 baseline using the same time-based split:

| Split | Period | Rows |
|---|---|---|
| Train | 2021-2023 | 1,231 |
| Validation | 2024 | 451 |
| Test | 2025 onward | 681 |

**Problems encountered and fixed:**

- **Extreme outlier laps distorting evaluation.** A review of the worst predictions showed deltas of 33.6s, 19.7s, and 14.0s -- far beyond any realistic qualifying gap -- concentrated in midfield/backmarker teams (Alpine, Williams, Kick Sauber, Haas), caused by crashes, red flags, and mechanical failures rather than genuine pace differences. Fixed by adding a `filter_extreme_deltas()` function removing any lap with `DeltaToSessionFastest_s > 5.0` seconds.
- **Held-out test set defined but never evaluated.** `time_based_split()` already returned a `test` set, but the training script only checked validation performance. Fixed by adding a parallel evaluation block using the same trained pipeline (no refitting) on the test split.

**Final results:**

| Split | Rows | MAE | RMSE | MedianAE |
|---|---|---|---|---|
| Train (2021-2023) | 1,231 | 0.162s | 0.220s | 0.122s |
| Validation (2024) | 451 | 0.412s | 0.538s | 0.324s |
| Test (2025 onward) | 681 | 0.429s | 0.645s | 0.292s |

The close alignment between validation MAE (0.412s) and test MAE (0.429s) confirmed the model generalises well to genuinely unseen future seasons rather than overfitting -- both comfortably beat the 0.835s baseline benchmark.

**Residual analysis** on the test set's 20 largest errors revealed two dominant, explainable patterns:

- **Cadillac (BOT, PER)** -- a brand-new 2026 team with zero history in the 2021-2025 training window, causing the model's imputer to fall back to generic midfield estimates and producing the single largest errors (up to 3.56s)
- **Aston Martin (ALO, STR)** -- 9 of the 20 largest residuals despite full training history, pointing to a genuine mid-season competitive decline that backward-looking rolling-average features can't anticipate

### Limitations

- **Cold-start teams have no historical signal.** New entrants (e.g. Cadillac in 2026) have no prior-season data for rolling/historical features, so predictions default to generic imputed values rather than true team pace.
- **Rolling features lag real competitive shifts.** Teams with sharp mid-season form changes (e.g. Aston Martin's 2026 decline) are systematically mispredicted since trailing averages are inherently backward-looking.
- **A small number of large residuals inflate RMSE more than MAE.** Test RMSE (0.645s) rose more than MAE relative to validation, though MedianAE actually improved -- most predictions are highly accurate, with a few hard cases skewing RMSE.
- **The outlier threshold is a fixed heuristic.** The 5.0-second cutoff for filtering crash/DNF-style laps was chosen by inspecting the data distribution, not derived from a principled statistical method or explicit incident flags in the source data.

### Phase 7: Hyperparameter Tuning & Robustness Checks

Used Optuna to systematically search XGBoost hyperparameters rather than relying on hand-picked values, optimising directly for validation MAE with a time-respecting train/val split (no shuffling, since this is chronological data).

Best parameters were saved to `models/metrics/best_params.json` and loaded into `train_xgboost.py` in place of the original manual values, with a safe fallback to the manual defaults if the tuned file is missing.

**Result:**

| Model | Val MAE | Test MAE |
|---|---|---|
| XGBoost (manual params) | 0.412s | 0.429s |
| XGBoost (Optuna-tuned) | 0.399s | 0.415s |

Tuning produced a small, genuine improvement (roughly 1-3% on both splits), confirming the manual parameters from Phase 6 were already close to optimal. Re-running the Optuna search a second time produced near-identical results (0.400s vs 0.399s val MAE) -- this run-to-run variation is well within normal Bayesian-search noise rather than a real further improvement, so tuning was stopped at that point rather than repeated indefinitely for diminishing returns. This is an important discipline point: hyperparameter tuning should be treated as converged once successive runs stop meaningfully changing the result, since continuing to search and check against the validation or test set repeatedly risks quietly overfitting evaluation decisions to those specific splits.

---

### Phase 8: Explainability & Prediction Intervals

Built `src/models/explain_model.py` to generate SHAP values for the tuned model, giving both a global feature-importance view and the ability to explain any single prediction -- essential for the app's planned "why this prediction" feature.

Also trained two supplementary XGBoost models using quantile regression (10th and 90th percentile) to produce an 80% prediction interval around each point estimate, rather than a falsely precise single number:

**Problems encountered and fixed:**

- **`NameError` for `X_train`/`y_train`** The explainability script initially only loaded the test split (for SHAP), but the quantile interval models needed a training split that was never pulled in. Fixed by extracting `X_train, y_train` via `prepare_xy(train, feature_cols)` alongside the existing test split, and imputing both sets using the already-fitted imputer from the saved pipeline rather than refitting it.

**Result:** The SHAP summary plot confirmed the model relies on sensible, expected signals -- `DriverRollingDelta_10` and `DriverRollingDelta_5` (recent driver form) were by far the strongest predictors, followed by `TeamCircuitHistoricalDelta` and team-level rolling/median deltas. Weather and circuit-context features (wind, humidity, altitude, street circuit flag) contributed at the margins, as expected. This gave confidence the model had learned genuine motorsport relationships rather than spurious patterns.

---

### Phase 9: Backend API

Built a FastAPI service wrapping the trained model, SHAP explainer, and quantile interval models behind a clean set of endpoints, structured as three files:

- **`src/api/schemas.py`** -- Pydantic request/response models for `/predict`, `/compare`, and `/metadata`
- **`src/api/predictor_service.py`** -- a `PredictorService` class loading all artefacts once at startup (not per-request), building a feature row from each driver/circuit's most recent historical data, and returning a point prediction, an 80% interval, and the top 5 SHAP-ranked features driving that specific prediction
- **`src/api/main.py`** -- the FastAPI app itself, exposing `/predict`, `/compare`, `/metadata`, and `/health` endpoints with CORS enabled for the future frontend

**Problem encountered and fixed:**

- **`404 Not Found` on initial local run.** Visiting the root URL (`/`) returned a 404, which looked alarming but was actually expected -- no route was ever defined for the bare root path, and the log confirmed `Application startup complete` with no real errors. Fixed by testing the actual defined endpoints (`/docs`, `/health`, `/metadata`) instead, and optionally adding a simple root route (`GET /`) returning a welcome message for convenience during development.

**Result:** All endpoints verified working locally via FastAPI's auto-generated Swagger UI (`/docs`), confirming the model, SHAP explainer, and quantile models all load correctly at startup and return valid predictions end-to-end. The backend is now ready to be connected to a frontend in Phase 10.

## Setup

```bash
git clone https://github.com/SiriJ079/f1-qualifying-lap-predictor.git
cd f1-qualifying-lap-predictor
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Status

- Completed — Phase 1 (Repo & Environment setup) 
- Completed — Phase 2 (Data Ingestion : Collect and Cache the raw data)
- Completed — Phase 3 (EDA : Exploratory Data Analysis)
- Completed — Phase 4 (Feature Engineering : Prepping (transforming) data to be ready for an ML Model) 
- Completed — Phase 5 (Baseline models (driver rolling, circuit history, team median) with evaluation framework) 
- Completed — Phase 6 (Model Training, Outlier Handling & Evaluation)
- Completed — Phase 7 (Tune & stress-testing model)
- Completed — Phase 8 (Adding interpretability and uncertainty)
- Completed — Phase 9 (Building back-end API)
<!-- - Completed — Phase 10 (Building front-end (App UI))
- Completed — Phase 11 (Test & Deploy App) -->