"""Fetch today's MLB props from PrizePicks public API.

PrizePicks uses DataDome bot protection, so standard requests get a 403.
curl_cffi spoofs Chrome's TLS fingerprint which bypasses DataDome cleanly.
"""

from curl_cffi import requests
import pandas as pd

PP_URL = "https://api.prizepicks.com/projections"
MLB_LEAGUE_ID = 2

# PrizePicks stat_type strings we care about, mapped to our outcome names.
# "Hitter Strikeouts" = batter Ks (what we model).
# "Pitcher Strikeouts" = pitcher Ks — excluded; different model target.
STAT_MAP = {
    "Hitter Strikeouts": "strikeout",
    "Walks":             "walk",
    "Home Runs":         "home_run",
    "Hits":              "hit",
    "Hits+Runs+RBIs":    "hits_runs_rbis",
}


def fetch_props(stat_types: list[str] | None = None) -> pd.DataFrame:
    """Return today's PrizePicks MLB lines as a DataFrame.

    Columns: player_name, team, stat_type, outcome, line, pp_implied_prob
    """
    params = {
        "league_id": MLB_LEAGUE_ID,
        "per_page": 500,
        "single_stat": "true",
    }
    resp = requests.get(PP_URL, params=params, impersonate="chrome120", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    projections = data.get("data", [])
    included     = {item["id"]: item for item in data.get("included", [])}

    rows = []
    for proj in projections:
        attrs   = proj["attributes"]
        stat    = attrs.get("stat_type", "")
        outcome = STAT_MAP.get(stat)
        if outcome is None:
            continue
        if stat_types and stat not in stat_types:
            continue

        # Resolve player name from relationships
        player_rel = proj["relationships"].get("new_player", {}).get("data", {})
        player_id  = player_rel.get("id")
        player_obj = included.get(player_id, {})
        player_attrs = player_obj.get("attributes", {})

        line = float(attrs.get("line_score", 0))
        # PrizePicks more/less at -110 both sides → implied break-even ≈ 52.4%
        # Their "line" is roughly the median; we treat it as the threshold.
        rows.append({
            "player_name":     player_attrs.get("name", ""),
            "team":            player_attrs.get("team", ""),
            "stat_type":       stat,
            "outcome":         outcome,
            "line":            line,
            "pp_implied_prob": 0.524,   # standard -110 juice break-even
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = fetch_props()
    print(df.to_string(index=False))
