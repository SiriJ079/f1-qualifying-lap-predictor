import pandas as pd
from pathlib import Path


def test_qualifying_file_exists():
    path = Path("data/raw/qualifying_laps_raw.parquet")
    assert path.exists(), "Qualifying parquet file not found — run fetch_qualifying.py first"


def test_qualifying_has_required_columns():
    path = Path("data/raw/qualifying_laps_raw.parquet")
    if not path.exists():
        return  # Skip if data not yet downloaded
    df = pd.read_parquet(path)
    required = ["Year", "EventName", "Driver", "LapTime", "Compound", "TrackTemp"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_circuit_metadata_exists():
    path = Path("data/raw/circuit_metadata.parquet")
    assert path.exists(), "Circuit metadata file not found — run fetch_circuit_metadata.py first"


def test_clean_file_exists():
    path = Path("data/processed/qualifying_clean.parquet")
    assert path.exists(), "Clean parquet not found — run clean_qualifying.py first"


def test_no_invalid_lap_times():
    path = Path("data/processed/qualifying_clean.parquet")
    if not path.exists():
        return
    df = pd.read_parquet(path)
    assert df["LapTime_s"].between(60, 200).all(), "Invalid lap times found outside 60–200s range"


def test_no_invalid_compounds():
    path = Path("data/processed/qualifying_clean.parquet")
    if not path.exists():
        return
    df = pd.read_parquet(path)
    valid = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
    assert set(df["Compound"].unique()).issubset(valid), "Invalid compound types found"

def test_feature_file_exists():
    path = Path("data/features/qualifying_features.parquet")
    assert path.exists(), "Feature parquet not found — run build_features.py first"


def test_no_target_leakage():
    path = Path("data/features/qualifying_features.parquet")
    if not path.exists():
        return
    df = pd.read_parquet(path)
    leakage_cols = [c for c in df.columns if "LapTime" in c or "SessionFastest" in c]
    assert len(leakage_cols) == 0, f"Potential leakage columns found: {leakage_cols}"


def test_rolling_features_are_lagged():
    path = Path("data/features/qualifying_features.parquet")
    if not path.exists():
        return
    df = pd.read_parquet(path)
    # First session for any driver should have NaN rolling delta (nothing to roll over)
    first_appearances = df.groupby("Driver").head(1)
    assert first_appearances["DriverRollingDelta_3"].isna().any(), \
        "Rolling features not properly lagged — first rows should be NaN"