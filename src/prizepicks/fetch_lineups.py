"""Fetch today's MLB starting lineups from the MLB Stats API."""

import requests
import pandas as pd
from datetime import date

MLB_API = "https://statsapi.mlb.com/api/v1"


def fetch_today_games(game_date: date | None = None) -> list[dict]:
    """Return list of game dicts for the given date (default: today)."""
    d = game_date or date.today()
    url = f"{MLB_API}/schedule"
    resp = requests.get(url, params={
        "sportId": 1,
        "date": d.strftime("%Y-%m-%d"),
        "hydrate": "lineups,probablePitcher",
    }, timeout=15)
    resp.raise_for_status()
    games = []
    for date_entry in resp.json().get("dates", []):
        for g in date_entry.get("games", []):
            games.append(g)
    return games


def fetch_lineups(game_date: date | None = None) -> pd.DataFrame:
    """Return expected PA assignments for today.

    Columns: game_pk, home_team, batter_name, batter_mlbam_id,
             batting_order, team_side, starter_pitcher_id, starter_pitcher_name
    """
    games = fetch_today_games(game_date)
    rows  = []
    for g in games:
        game_pk    = g["gamePk"]
        home_team  = g["teams"]["home"]["team"].get("abbreviation", "")
        away_team  = g["teams"]["away"]["team"].get("abbreviation", "")

        for side in ("home", "away"):
            team_abbr  = home_team if side == "home" else away_team
            opp_side   = "away" if side == "home" else "home"
            opp_info   = g["teams"][opp_side]
            sp         = opp_info.get("probablePitcher", {})
            sp_id      = sp.get("id")
            sp_name    = sp.get("fullName", "Unknown")

            batters = (g.get("lineups", {})
                        .get(f"{side}Batters", []))
            for idx, batter in enumerate(batters):
                rows.append({
                    "game_pk":              game_pk,
                    "home_team":            home_team,
                    "batter_name":          batter.get("fullName", ""),
                    "batter_mlbam_id":      batter.get("id"),
                    "batting_order":        idx + 1,
                    "team_side":            side,
                    "starter_pitcher_id":   sp_id,
                    "starter_pitcher_name": sp_name,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = fetch_lineups()
    print(df.to_string(index=False))
