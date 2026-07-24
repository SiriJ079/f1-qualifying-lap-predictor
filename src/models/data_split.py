import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "qualifying_features.parquet"


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    return df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)


def time_based_split(df: pd.DataFrame, test_season: int = 2025, val_season: int = 2024):
    """
    Train: everything before val_season
    Validation: val_season only
    Test: test_season only (held out, touched only at the very end)
    """
    train = df[df["Year"] < val_season].copy()
    val = df[df["Year"] == val_season].copy()
    test = df[df["Year"] >= test_season].copy()

    print(f"Train: {len(train):,} rows ({train['Year'].min()}–{train['Year'].max()})")
    print(f"Val:   {len(val):,} rows ({val_season})")
    print(f"Test:  {len(test):,} rows ({test_season} onward)")

    return train, val, test


if __name__ == "__main__":
    df = load_features()
    train, val, test = time_based_split(df)