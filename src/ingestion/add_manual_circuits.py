import pandas as pd
from pathlib import Path

RAW_PATH = Path("./data/raw/circuit_metadata.parquet")

MANUAL_CIRCUITS = [
    {
        "CircuitName": "Spanish Grand Prix",
        "Location": "Madrid",
        "Country": "Spain",
        "IsStreetCircuit": True,
        "Altitude_m": 667,
    },
]


def add_manual_circuits():
    df = pd.read_parquet(RAW_PATH)

    for circuit in MANUAL_CIRCUITS:
        exists = (
            (df["CircuitName"] == circuit["CircuitName"])
            & (df["Location"] == circuit["Location"])
        ).any()

        if exists:
            print(f"  – {circuit['CircuitName']} ({circuit['Location']}) already present, skipping")
            continue

        df = pd.concat([df, pd.DataFrame([circuit])], ignore_index=True)
        print(f"  ✓ Added {circuit['CircuitName']} ({circuit['Location']})")

    df.to_parquet(RAW_PATH, index=False)
    print(f"✅ Saved {len(df)} total circuits to {RAW_PATH}")


if __name__ == "__main__":
    add_manual_circuits()