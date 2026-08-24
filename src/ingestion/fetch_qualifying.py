import fastf1
import pandas as pd
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

CACHE_DIR = Path(os.getenv("FASTF1_CACHE_DIR", "./data/raw/fastf1_cache"))
OUTPUT_DIR = Path("./data/raw")
OUTPUT_PATH = OUTPUT_DIR / "qualifying_laps_raw.parquet"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

SEASONS = list(range(2021, 2027))
REQUEST_DELAY_SECONDS = 3


def load_existing_data() -> pd.DataFrame:
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        print(f"Loaded existing dataset: {len(existing)} rows across {existing['Year'].nunique()} seasons")
        return existing
    print("No existing dataset found — starting fresh.")
    return pd.DataFrame()


def get_already_fetched_rounds(existing: pd.DataFrame) -> set[tuple[int, int]]:
    if existing.empty:
        return set()
    return set(zip(existing["Year"], existing["RoundNumber"]))


def fetch_qualifying_session(year: int, round_number: int) -> pd.DataFrame | None:
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(telemetry=False, weather=True, messages=True)

        laps = session.laps.copy()
        laps["Year"] = year
        laps["RoundNumber"] = round_number
        laps["EventName"] = session.event["EventName"]
        laps["Country"] = session.event["Country"]
        laps["CircuitShortName"] = session.event.get("Location", "Unknown")

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


def fetch_season_schedule(year: int):
    """Isolate schedule-fetch failures so one bad season doesn't kill the whole run."""
    try:
        return fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"  ✗ Could not load schedule for {year}: {e}")
        return None


def fetch_new_sessions(seasons: list[int], already_fetched: set[tuple[int, int]]) -> pd.DataFrame:
    all_laps = []
    now = datetime.now(timezone.utc)

    for year in seasons:
        print(f"\n--- Season {year} ---")
        schedule = fetch_season_schedule(year)
        if schedule is None:
            continue

        for _, event in schedule.iterrows():
            round_num = event["RoundNumber"]

            if (year, round_num) in already_fetched:
                continue

            event_date = event.get("EventDate")
            if pd.notna(event_date) and pd.Timestamp(event_date, tz="UTC") > now:
                print(f"  – {year} Round {round_num} — {event['EventName']} hasn't happened yet, skipping")
                continue

            df = fetch_qualifying_session(year, round_num)
            if df is not None:
                all_laps.append(df)

            time.sleep(REQUEST_DELAY_SECONDS)

    return pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()


def merge_and_save(existing: pd.DataFrame, new_data: pd.DataFrame):
    if new_data.empty:
        print("\nNo new sessions found — dataset already up to date.")
        return

    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Year", "RoundNumber", "Driver", "LapNumber"], keep="last")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(combined)} total rows ({len(new_data)} new) to {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Starting qualifying data ingestion...")
    existing_data = load_existing_data()
    already_fetched = get_already_fetched_rounds(existing_data)

    try:
        new_data = fetch_new_sessions(SEASONS, already_fetched)
    except Exception as e:
        print(f"Unexpected fatal error during fetch: {e}")
        new_data = pd.DataFrame()

    merge_and_save(existing_data, new_data)