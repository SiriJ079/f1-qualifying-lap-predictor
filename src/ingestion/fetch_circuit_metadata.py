import fastf1
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

CACHE_DIR = os.getenv("FASTF1_CACHE_DIR", "./data/raw/fastf1_cache")
OUTPUT_DIR = Path("./data/raw")
fastf1.Cache.enable_cache(CACHE_DIR)

# Street circuits — manually flagged (no reliable API field for this)
STREET_CIRCUITS = [
    "Melbourne", "Baku", "Monte Carlo", "Monaco", "Singapore",
    "Marina Bay", "Jeddah", "Miami", "Miami Gardens", "Las Vegas"
]

# Approximate circuit altitudes in metres
ALTITUDE_MAP = {
    "Melbourne": 20,
    "Sakhir": 5,
    "Shanghai": 4,
    "Baku": 28,
    "Barcelona": 109,
    "Monte Carlo": 7,
    "Monaco": 7,
    "Montréal": 18,
    "Le Castellet": 410,
    "Spielberg": 693,
    "Silverstone": 91,
    "Hockenheim": 110,
    "Budapest": 264,
    "Spa-Francorchamps": 401,
    "Monza": 162,
    "Singapore": 15,
    "Marina Bay": 15,
    "Sochi": 2,
    "Suzuka": 45,
    "Austin": 149,
    "Mexico City": 2285,
    "São Paulo": 799,
    "Yas Marina": 3,
    "Yas Island": 3,
    "Mugello": 330,
    "Nürburgring": 600,
    "Portimão": 108,
    "Imola": 38,
    "Istanbul": 99,
    "Zandvoort": 5,
    "Lusail": 14,
    "Jeddah": 15,
    "Miami": 3,
    "Miami Gardens": 3,
    "Las Vegas": 620,
}

def build_circuit_metadata(seasons: list[int]) -> pd.DataFrame:
    """Extract circuit metadata from FastF1 event schedules."""
    records = []
    seen = set()

    for year in seasons:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            name = event.get("EventName", "Unknown")
            location = event.get("Location", "Unknown")
            key = location

            if key in seen:
                continue
            seen.add(key)

            records.append({
                "CircuitName": name,
                "Location": location,
                "Country": event.get("Country", "Unknown"),
                "IsStreetCircuit": location in STREET_CIRCUITS,
                "Altitude_m": ALTITUDE_MAP.get(location, None),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Building circuit metadata...")
    metadata = build_circuit_metadata(list(range(2018, 2026)))
    output_path = OUTPUT_DIR / "circuit_metadata.parquet"
    metadata.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(metadata)} circuits to {output_path}")
    print(metadata)