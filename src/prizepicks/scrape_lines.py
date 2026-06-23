"""Morning scraper: fetch today's PrizePicks lines + model predictions, store to DB.

Run 1–2 hours before first pitch, after lineups are posted (~11 AM ET on game days).

    python -m src.prizepicks.scrape_lines
    python -m src.prizepicks.scrape_lines --date 2026-06-22

Model predictions are optional: if Snowflake / model files are unavailable the
lines are still stored (model_prob = NULL) and predictions can be back-filled later.

Note: model_prob is always P(1+ outcome). For PP lines above 0.5 (e.g. 1.5 Ks)
the edge calculation is approximate — treat those rows accordingly in analysis.
"""

import argparse
from datetime import date, datetime, timezone

from src.prizepicks.fetch_props import fetch_props
from src.prizepicks.fetch_lineups import fetch_lineups
from src.prizepicks.store import init_db, upsert_lines

PP_IMPLIED = 0.524

# Maps our outcome name → game_preds column from pa_probs_to_game
_OUTCOME_TO_COL = {
    "strikeout": "p_1plus_k",
    "walk":      "p_1plus_walk",
    "home_run":  "p_1plus_hr",
    "hit":       "p_1plus_hit",
}


def _try_predict(lineups) -> "pd.DataFrame | None":
    """Run models and return game-level predictions, or None on any failure."""
    try:
        from src.prizepicks.predict_game import (
            load_models, fetch_player_features,
            build_pregame_pa_rows, pa_probs_to_game,
        )
        batter_ids  = lineups["batter_mlbam_id"].dropna().astype(int).tolist()
        pitcher_ids = lineups["starter_pitcher_id"].dropna().astype(int).tolist()
        if not batter_ids or not pitcher_ids:
            return None
        models    = load_models()
        features  = fetch_player_features(batter_ids, pitcher_ids)
        pa_rows   = build_pregame_pa_rows(lineups, features)
        return pa_probs_to_game(pa_rows, models)
    except Exception as e:
        print(f"  [warn] Model prediction skipped: {e}")
        return None


def scrape_and_store(game_date: date | None = None) -> int:
    """Fetch PP lines + model probs and write to DB. Returns number of rows stored."""
    init_db()
    game_date  = game_date or date.today()
    scraped_at = datetime.now(timezone.utc).isoformat()

    print(f"[{scraped_at[:19]}] Scraping PrizePicks lines for {game_date}...")
    props = fetch_props()
    if props.empty:
        print("No props available.")
        return 0
    print(f"  {len(props)} props found.")

    game_preds = None
    print("Fetching lineups...")
    lineups = fetch_lineups(game_date)
    if lineups.empty:
        print("  No lineups yet — storing lines without model predictions.")
    else:
        print(f"  {len(lineups)} batter slots. Running models...")
        game_preds = _try_predict(lineups)
        if game_preds is not None:
            print(f"  Predictions ready for {len(game_preds)} batters.")

    rows = []
    for _, prop in props.iterrows():
        model_prob = edge = bet_direction = None

        if game_preds is not None:
            pred_col = _OUTCOME_TO_COL.get(prop["outcome"])
            if pred_col and pred_col in game_preds.columns:
                match = game_preds[
                    game_preds["batter_name"].str.lower() == prop["player_name"].lower()
                ]
                if not match.empty:
                    model_prob    = round(float(match.iloc[0][pred_col]), 4)
                    edge          = round(model_prob - PP_IMPLIED, 4)
                    bet_direction = "MORE" if model_prob > PP_IMPLIED else "LESS"

        rows.append({
            "scraped_at":   scraped_at,
            "game_date":    game_date.isoformat(),
            "player_name":  prop["player_name"],
            "team":         prop["team"],
            "stat_type":    prop["stat_type"],
            "outcome":      prop["outcome"],
            "line":         float(prop["line"]),
            "pp_implied":   PP_IMPLIED,
            "model_prob":   model_prob,
            "edge":         edge,
            "bet_direction": bet_direction,
        })

    upsert_lines(rows)
    n_with_pred = sum(1 for r in rows if r["model_prob"] is not None)
    print(f"Stored {len(rows)} lines ({n_with_pred} with model predictions).")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", type=date.fromisoformat, default=None,
        help="Game date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()
    scrape_and_store(args.date)
