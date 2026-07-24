import pandas as pd
import numpy as np
from pathlib import Path

CLEAN_PATH = Path("data/processed/qualifying_clean.parquet")
OUTPUT_PATH = Path("data/features/qualifying_features.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_clean_data() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_PATH)
    # Sort chronologically — essential for all rolling/lagged features
    df = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    return df


# ── 1. Driver best lap per session ───────────────────────────────────────────

def get_driver_session_best(df: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per session — their best qualifying lap."""
    best = (
        df.groupby(["Year", "RoundNumber", "EventName", "Driver", "Team",
                    "IsStreetCircuit", "Altitude_m",
                    "TrackTemp", "AirTemp", "Humidity", "WindSpeed", "Rainfall"])
        .agg(
            BestLap_s=("LapTime_s", "min"),
            CompoundUsed=("Compound", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )

    # Session fastest for delta target
    session_fastest = (
        best.groupby(["Year", "RoundNumber"])["BestLap_s"]
        .min()
        .reset_index()
        .rename(columns={"BestLap_s": "SessionFastest_s"})
    )

    best = best.merge(session_fastest, on=["Year", "RoundNumber"])
    best["DeltaToFastest_s"] = best["BestLap_s"] - best["SessionFastest_s"]

    return best.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)


# ── 2. Rolling driver form features ──────────────────────────────────────────

def add_driver_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling averages of driver delta over last N sessions."""
    df = df.sort_values(["Driver", "Year", "RoundNumber"])

    for window in [3, 5, 10]:
        col = f"DriverRollingDelta_{window}"
        df[col] = (
            df.groupby("Driver")["DeltaToFastest_s"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Driver's career median delta — long-term quality signal
    df["DriverCareerMedianDelta"] = (
        df.groupby("Driver")["DeltaToFastest_s"]
        .transform(lambda x: x.shift(1).expanding().median())
    )

    return df


# ── 3. Rolling constructor form features ─────────────────────────────────────

def add_constructor_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling team pace averages — captures car development trajectory."""
    df = df.sort_values(["Team", "Year", "RoundNumber"])

    for window in [3, 5]:
        col = f"TeamRollingDelta_{window}"
        df[col] = (
            df.groupby("Team")["DeltaToFastest_s"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    df["TeamCareerMedianDelta"] = (
        df.groupby("Team")["DeltaToFastest_s"]
        .transform(lambda x: x.shift(1).expanding().median())
    )

    return df


# ── 4. Circuit historical features ───────────────────────────────────────────

def add_circuit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Driver's historical delta at this specific circuit."""
    df = df.sort_values(["Driver", "EventName", "Year", "RoundNumber"])

    df["DriverCircuitHistoricalDelta"] = (
        df.groupby(["Driver", "EventName"])["DeltaToFastest_s"]
        .transform(lambda x: x.shift(1).expanding().median())
    )

    df["DriverCircuitAppearances"] = (
        df.groupby(["Driver", "EventName"]).cumcount()
    )

    # Team historical pace at this circuit
    df["TeamCircuitHistoricalDelta"] = (
        df.groupby(["Team", "EventName"])["DeltaToFastest_s"]
        .transform(lambda x: x.shift(1).expanding().median())
    )

    return df


# ── 5. Season context features ────────────────────────────────────────────────

# def add_season_context_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Where in the season this round falls — early vs late development."""
#     season_length = df.groupby("Year")["RoundNumber"].transform("max")
#     df["SeasonProgress"] = df["RoundNumber"] / season_length

#     # Driver's cumulative delta improvement within the season
#     df["DriverSeasonDeltaTrend"] = (
#         df.groupby(["Driver", "Year"])["DeltaToFastest_s"]
#         .transform(lambda x: x.shift(1).expanding().mean())
#     )

#     return df


# ── 6. Compound encoding ──────────────────────────────────────────────────────

def add_compound_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode compound as ordinal — SOFT is fastest."""
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    df["CompoundOrdinal"] = df["CompoundUsed"].map(compound_map).fillna(2)
    return df


# ── 7. Weather normalisation ──────────────────────────────────────────────────

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing weather values with circuit historical medians."""
    for col in ["TrackTemp", "AirTemp", "Humidity", "WindSpeed"]:
        circuit_median = df.groupby("EventName")[col].transform("median")
        df[col] = df[col].fillna(circuit_median)

    df["Rainfall"] = df["Rainfall"].fillna(False).astype(int)
    df["TempDelta"] = df["TrackTemp"] - df["AirTemp"]

    return df


# ── 8. Assemble final feature matrix ─────────────────────────────────────────

FEATURE_COLS = [
    # Target
    "DeltaToFastest_s",
    # Identifiers (not model inputs — kept for validation/grouping)
    "Year", "RoundNumber", "EventName", "Driver", "Team",
    # Driver form
    "DriverRollingDelta_3", "DriverRollingDelta_5", "DriverRollingDelta_10",
    "DriverCareerMedianDelta",
    # Constructor form
    "TeamRollingDelta_3", "TeamRollingDelta_5", "TeamCareerMedianDelta",
    # Circuit history
    "DriverCircuitHistoricalDelta", "DriverCircuitAppearances",
    "TeamCircuitHistoricalDelta",
    # # Season context
    # "SeasonProgress", "DriverSeasonDeltaTrend",
    # Circuit profile
    "IsStreetCircuit", "Altitude_m",
    # Weather
    "TrackTemp", "AirTemp", "Humidity", "WindSpeed", "Rainfall", "TempDelta",
    # Tyre
    "CompoundOrdinal",
]


if __name__ == "__main__":
    print("Loading clean data...")
    df = load_clean_data()

    print("Building driver session bests...")
    df = get_driver_session_best(df)
    print(f"  {len(df):,} driver-session rows")

    print("Adding driver rolling features...")
    df = add_driver_rolling_features(df)

    print("Adding constructor rolling features...")
    df = add_constructor_rolling_features(df)

    print("Adding circuit history features...")
    df = add_circuit_history_features(df)

    # print("Adding season context features...")
    # df = add_season_context_features(df)

    print("Adding compound features...")
    df = add_compound_features(df)

    print("Adding weather features...")
    df = add_weather_features(df)

    # Select final columns
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_features = df[available_cols].copy()

    # Drop rows where target is null
    df_features = df_features[df_features["DeltaToFastest_s"].notna()]

    print(f"\nFinal feature matrix: {df_features.shape}")
    print(f"Null counts:\n{df_features.isnull().sum().sort_values(ascending=False).head(10)}")

    df_features.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved to {OUTPUT_PATH}")