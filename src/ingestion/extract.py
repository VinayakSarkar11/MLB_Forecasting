"""
Pull MLB Statcast pitch-by-pitch data for a given season and save
partitioned CSVs locally under data/raw/year=YYYY/month=MM/.
"""

import os
from pathlib import Path
import pandas as pd
from pybaseball import statcast
from pybaseball import cache

# Speed up repeat runs by caching pybaseball responses
cache.enable()

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# 2024 regular season: March 20 – September 29
SEASONS = {
    2024: ("2024-03-20", "2024-09-29"),
}


def extract_season(year: int, start: str, end: str) -> None:
    print(f"Pulling Statcast data for {year} ({start} → {end})...")
    df = statcast(start_dt=start, end_dt=end)

    if df.empty:
        print(f"No data returned for {year}.")
        return

    df["game_date"] = pd.to_datetime(df["game_date"])

    for month, group in df.groupby(df["game_date"].dt.month):
        out_dir = RAW_DIR / f"year={year}" / f"month={month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"statcast_{year}_{month:02d}.csv"
        group.to_csv(out_path, index=False)
        print(f"  Saved {len(group):,} rows → {out_path.relative_to(RAW_DIR.parent.parent)}")

    print(f"Done: {year} — {len(df):,} total pitches.\n")


if __name__ == "__main__":
    for year, (start, end) in SEASONS.items():
        extract_season(year, start, end)
