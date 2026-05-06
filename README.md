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

## Setup

```bash
git clone https://github.com/SiriJ079/f1-qualifying-lap-predictor.git
cd f1-qualifying-lap-predictor
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Status

🚧 Completed — Phase 1 (Repo & Environment setup) 
🚧 In development — Phase 2 (Data Ingestion : Collect and Cache the raw data)