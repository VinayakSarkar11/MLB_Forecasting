"""Evening scraper: fetch MLB box scores and store actual batter stat counts.

Run after all games are complete — 11 PM PT or later is safe for West Coast games.
Default date uses Pacific time so the correct game date is used when run past midnight UTC.

    python -m src.prizepicks.scrape_results
    python -m src.prizepicks.scrape_results --date 2026-06-22

Only stores rows for completed (Final) games. Re-running is safe: duplicate
rows are replaced via the UNIQUE constraint on (game_date, game_pk, player_name).
"""

import argparse
import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import requests

from src.prizepicks.store import init_db, upsert_results

MLB_API = "https://statsapi.mlb.com/api/v1"


def _get_final_game_pks(game_date: date) -> list[int]:
    resp = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "date": game_date.strftime("%Y-%m-%d")},
        timeout=15,
    )
    resp.raise_for_status()

    pks = []
    for date_entry in resp.json().get("dates", []):
        for game in date_entry.get("games", []):
            state = game.get("status", {}).get("abstractGameState", "")
            if state == "Final":
                pks.append(game["gamePk"])
            else:
                print(f"  [skip] gamePk={game['gamePk']} status={state}")
    return pks


def _parse_boxscore(game_pk: int, game_date: date, fetched_at: str) -> list[dict]:
    resp = requests.get(f"{MLB_API}/game/{game_pk}/boxscore", timeout=15)
    if resp.status_code != 200:
        print(f"  [warn] boxscore fetch failed for game {game_pk}: {resp.status_code}")
        return []

    rows  = []
    teams = resp.json().get("teams", {})
    for side in ("home", "away"):
        team_data = teams.get(side, {})
        team_abbr = team_data.get("team", {}).get("abbreviation", "")
        for _, pdata in team_data.get("players", {}).items():
            batting = pdata.get("stats", {}).get("batting", {})
            if not batting or batting.get("atBats", 0) == 0:
                continue
            rows.append({
                "fetched_at":      fetched_at,
                "game_date":       game_date.isoformat(),
                "game_pk":         game_pk,
                "batter_mlbam_id": pdata.get("person", {}).get("id"),
                "player_name":     pdata.get("person", {}).get("fullName", ""),
                "team":            team_abbr,
                "at_bats":         batting.get("atBats",      0),
                "hits":            batting.get("hits",         0),
                "doubles":         batting.get("doubles",      0),
                "triples":         batting.get("triples",      0),
                "home_runs":       batting.get("homeRuns",     0),
                "rbi":             batting.get("rbi",          0),
                "walks":           batting.get("baseOnBalls",  0),
                "strikeouts":      batting.get("strikeOuts",   0),
                "runs":            batting.get("runs",         0),
            })
    return rows


def scrape_and_store(game_date: date | None = None) -> int:
    """Fetch box scores for all completed games on game_date, store to DB.

    Returns number of player-game rows stored.
    """
    init_db()
    game_date  = game_date or datetime.now(ZoneInfo("America/Los_Angeles")).date()
    fetched_at = datetime.now(timezone.utc).isoformat()

    print(f"[{fetched_at[:19]}] Fetching MLB box scores for {game_date}...")
    pks = _get_final_game_pks(game_date)
    if not pks:
        print("No completed games found.")
        return 0
    print(f"  {len(pks)} completed games: {pks}")

    all_rows = []
    for pk in pks:
        rows = _parse_boxscore(pk, game_date, fetched_at)
        print(f"  game {pk}: {len(rows)} batters")
        all_rows.extend(rows)
        time.sleep(0.1)   # stay polite to the MLB API

    if not all_rows:
        print("No batter rows parsed.")
        return 0

    upsert_results(all_rows)
    print(f"Stored {len(all_rows)} player-game rows.")
    return len(all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", type=date.fromisoformat, default=None,
        help="Game date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()
    scrape_and_store(args.date)
