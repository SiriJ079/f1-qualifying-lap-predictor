import fastf1
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

CACHE_DIR = os.getenv("FASTF1_CACHE_DIR", "./data/raw/fastf1_cache")
OUTPUT_DIR = Path("./data/raw")
fastf1.Cache.enable_cache(CACHE_DIR)

# Seasons to collect — adjust range as needed
SEASONS = list(range(2021, 2027))

def fetch_qualifying_session(year: int, round_number: int) -> pd.DataFrame | None:
    """Download one qualifying session and return it as a DataFrame."""
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(telemetry=False, weather=True, messages=True)

        laps = session.laps.copy()
        laps["Year"] = year
        laps["RoundNumber"] = round_number
        laps["EventName"] = session.event["EventName"]
        laps["Country"] = session.event["Country"]
        laps["CircuitShortName"] = session.event.get("Location", "Unknown")

        # Attach weather snapshot (session-level averages)
        if session.weather_data is not None and not session.weather_data.empty:
            weather = session.weather_data
            laps["TrackTemp"] = weather["TrackTemp"].mean()
            laps["AirTemp"] = weather["AirTemp"].mean()
            laps["Humidity"] = weather["Humidity"].mean()
            laps["WindSpeed"] = weather["WindSpeed"].mean()
            laps["Rainfall"] = weather["Rainfall"].any()
        else:
            for col in ["TrackTemp", "AirTemp", "Humidity", "WindSpeed", "Rainfall"]:
                laps[col] = None

        print(f"  ✓ {year} Round {round_number} — {session.event['EventName']} ({len(laps)} laps)")
        return laps

    except Exception as e:
        print(f"  ✗ {year} Round {round_number} failed: {e}")
        return None


def fetch_all_seasons(seasons: list[int]) -> pd.DataFrame:
    """Loop through all seasons and rounds, collect all qualifying sessions."""
    all_laps = []

    for year in seasons:
        print(f"\n--- Season {year} ---")
        schedule = fastf1.get_event_schedule(year, include_testing=False)

        for _, event in schedule.iterrows():
            round_num = event["RoundNumber"]
            df = fetch_qualifying_session(year, round_num)
            if df is not None:
                all_laps.append(df)

    return pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()


if __name__ == "__main__":
    print("Starting qualifying data ingestion...")
    qualifying_data = fetch_all_seasons(SEASONS)

    if not qualifying_data.empty:
        output_path = OUTPUT_DIR / "qualifying_laps_raw.parquet"
        qualifying_data.to_parquet(output_path, index=False)
        print(f"\n✅ Saved {len(qualifying_data)} rows to {output_path}")
        print(qualifying_data[["Year", "EventName", "Driver", "LapTime", "Compound", "TrackTemp"]].head(10))
    else:
        print("\n⚠️ No data collected — check your internet connection and FastF1 version.")
        