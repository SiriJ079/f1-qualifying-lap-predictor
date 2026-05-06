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