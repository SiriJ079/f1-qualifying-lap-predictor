import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("./data/raw/qualifying_laps_raw.parquet")
CIRCUIT_PATH = Path("./data/raw/circuit_metadata.parquet")
OUTPUT_PATH = Path("./data/processed/qualifying_clean.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    laps = pd.read_parquet(RAW_PATH)
    circuits = pd.read_parquet(CIRCUIT_PATH)
    return laps, circuits


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return a lean, model-ready DataFrame."""

    # 1. Keep only laps with a valid recorded lap time
    df = df[df["LapTime"].notna()].copy()

    # 2. Convert LapTime (timedelta) to seconds as a float
    df["LapTime_s"] = df["LapTime"].dt.total_seconds()

    # 3. Remove physically impossible lap times
    # Below 60s is impossible; above 200s is likely an out/in lap or red flag lap
    df = df[(df["LapTime_s"] >= 60) & (df["LapTime_s"] <= 200)]

    # 4. Keep only standard compound types — drop TEST_UNKNOWN or missing
    valid_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    df = df[df["Compound"].isin(valid_compounds)]

    # 5. Keep only Q1, Q2, Q3 sessions (drop any stray practice laps)
    if "Session" in df.columns:
        df = df[df["Session"].isin(["Q1", "Q2", "Q3"])]

    # 6. Drop deleted laps (track limit violations etc.)
    if "Deleted" in df.columns:
        df = df[df["Deleted"] != True]

    # 7. Convert sector times to seconds
    for sector in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if sector in df.columns:
            df[f"{sector}_s"] = df[sector].dt.total_seconds()

    # 8. Select and rename only the columns we need going forward
    keep_cols = [
        "Year", "RoundNumber", "EventName", "Country", "CircuitShortName",
        "Driver", "Team", "LapTime_s", "Compound",
        "Sector1Time_s", "Sector2Time_s", "Sector3Time_s",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "TrackTemp", "AirTemp", "Humidity", "WindSpeed", "Rainfall",
        "TyreLife", "FreshTyre", "IsPersonalBest",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df.reset_index(drop=True)


def compute_session_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta-to-fastest columns per session for comparison."""

    # Fastest lap per driver per session round
    df["BestLap_s"] = df.groupby(["Year", "RoundNumber", "Driver"])["LapTime_s"].transform("min")

    # Fastest lap overall per session
    df["SessionFastest_s"] = df.groupby(["Year", "RoundNumber"])["LapTime_s"].transform("min")

    # Delta of each lap to that driver's own best
    df["DeltaToPersonalBest_s"] = df["LapTime_s"] - df["BestLap_s"]

    # Delta of driver's best to the session fastest (pole/fastest qualifier)
    df["DeltaToSessionFastest_s"] = df["BestLap_s"] - df["SessionFastest_s"]

    return df


def merge_circuit_metadata(laps: pd.DataFrame, circuits: pd.DataFrame) -> pd.DataFrame:
    """Join circuit metadata onto the laps table."""
    circuits_slim = circuits[["Location", "IsStreetCircuit", "Altitude_m"]].drop_duplicates("Location")
    merged = laps.merge(circuits_slim, left_on="CircuitShortName", right_on="Location", how="left")
    merged.drop(columns=["Location"], inplace=True)
    return merged


if __name__ == "__main__":
    print("Loading raw data...")
    laps, circuits = load_data()
    print(f"  Raw laps: {len(laps):,}")

    print("Cleaning laps...")
    clean = clean_laps(laps)
    print(f"  After cleaning: {len(clean):,}")

    print("Computing session deltas...")
    clean = compute_session_deltas(clean)

    print("Merging circuit metadata...")
    clean = merge_circuit_metadata(clean, circuits)

    print(f"Saving to {OUTPUT_PATH}...")
    clean.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n✅ Done — {len(clean):,} clean laps saved")
    print(f"   Columns: {list(clean.columns)}")
    print(clean[["Year", "EventName", "Driver", "LapTime_s", "DeltaToSessionFastest_s", "TrackTemp"]].head(10))