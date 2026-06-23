"""SQLite storage for PrizePicks scraped lines and actual MLB outcomes.

Schema:
  pp_lines      -- scraped props (player, line, model_prob) per game date
  game_results  -- actual batter stat counts from MLB box scores

Run as a script to print the current P&L summary:
    python -m src.prizepicks.store
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/prizepicks_history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pp_lines (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    scraped_at    TEXT    NOT NULL,
    game_date     TEXT    NOT NULL,
    player_name   TEXT    NOT NULL,
    team          TEXT,
    stat_type     TEXT    NOT NULL,
    outcome       TEXT    NOT NULL,
    line          REAL    NOT NULL,
    pp_implied    REAL    NOT NULL DEFAULT 0.524,
    model_prob    REAL,
    edge          REAL,
    bet_direction TEXT,
    UNIQUE(game_date, player_name, outcome) ON CONFLICT REPLACE
);

CREATE TABLE IF NOT EXISTS game_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at      TEXT    NOT NULL,
    game_date       TEXT    NOT NULL,
    game_pk         INTEGER NOT NULL,
    batter_mlbam_id INTEGER,
    player_name     TEXT    NOT NULL,
    team            TEXT,
    at_bats         INTEGER DEFAULT 0,
    hits            INTEGER DEFAULT 0,
    doubles         INTEGER DEFAULT 0,
    triples         INTEGER DEFAULT 0,
    home_runs       INTEGER DEFAULT 0,
    rbi             INTEGER DEFAULT 0,
    walks           INTEGER DEFAULT 0,
    strikeouts      INTEGER DEFAULT 0,
    runs            INTEGER DEFAULT 0,
    UNIQUE(game_date, game_pk, player_name) ON CONFLICT REPLACE
);
"""


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(_SCHEMA)


def upsert_lines(rows: list[dict]) -> None:
    with _conn() as con:
        con.executemany("""
            INSERT OR REPLACE INTO pp_lines
                (scraped_at, game_date, player_name, team, stat_type, outcome,
                 line, pp_implied, model_prob, edge, bet_direction)
            VALUES
                (:scraped_at, :game_date, :player_name, :team, :stat_type, :outcome,
                 :line, :pp_implied, :model_prob, :edge, :bet_direction)
        """, rows)


def upsert_results(rows: list[dict]) -> None:
    with _conn() as con:
        con.executemany("""
            INSERT OR REPLACE INTO game_results
                (fetched_at, game_date, game_pk, batter_mlbam_id, player_name,
                 team, at_bats, hits, doubles, triples, home_runs, rbi,
                 walks, strikeouts, runs)
            VALUES
                (:fetched_at, :game_date, :game_pk, :batter_mlbam_id, :player_name,
                 :team, :at_bats, :hits, :doubles, :triples, :home_runs, :rbi,
                 :walks, :strikeouts, :runs)
        """, rows)


def load_pnl_data(outcome: str | None = None) -> pd.DataFrame:
    """Return joined lines + results with P&L per bet.

    model_prob is always P(1+ outcome) regardless of line value.
    Filter to line = 0.5 for a clean apples-to-apples comparison, or
    use actual_count vs line directly for multi-level line analysis.
    """
    where = f"AND l.outcome = '{outcome}'" if outcome else ""
    query = f"""
        SELECT
            l.game_date,
            l.player_name,
            l.team,
            l.outcome,
            l.line,
            l.model_prob,
            l.edge,
            l.bet_direction,
            l.pp_implied,
            CASE l.outcome
                WHEN 'strikeout'      THEN r.strikeouts
                WHEN 'walk'           THEN r.walks
                WHEN 'home_run'       THEN r.home_runs
                WHEN 'hit'            THEN r.hits
                WHEN 'hits_runs_rbis' THEN r.hits + r.runs + r.rbi
            END AS actual_count,
            CASE
                WHEN CASE l.outcome
                    WHEN 'strikeout'      THEN r.strikeouts
                    WHEN 'walk'           THEN r.walks
                    WHEN 'home_run'       THEN r.home_runs
                    WHEN 'hit'            THEN r.hits
                    WHEN 'hits_runs_rbis' THEN r.hits + r.runs + r.rbi
                END > l.line THEN 1 ELSE 0
            END AS more_hit,
            CASE
                WHEN CASE l.outcome
                    WHEN 'strikeout'      THEN r.strikeouts
                    WHEN 'walk'           THEN r.walks
                    WHEN 'home_run'       THEN r.home_runs
                    WHEN 'hit'            THEN r.hits
                    WHEN 'hits_runs_rbis' THEN r.hits + r.runs + r.rbi
                END < l.line THEN 1 ELSE 0
            END AS less_hit
        FROM pp_lines l
        JOIN game_results r
            ON  l.game_date            = r.game_date
            AND LOWER(l.player_name)   = LOWER(r.player_name)
        WHERE l.model_prob IS NOT NULL
          {where}
        ORDER BY l.game_date, l.player_name
    """
    with _conn() as con:
        df = pd.read_sql_query(query, con)

    if df.empty:
        return df

    def _pnl(row):
        if row["bet_direction"] == "MORE":
            return 90.91 if row["more_hit"] == 1 else -100.0
        elif row["bet_direction"] == "LESS":
            return 90.91 if row["less_hit"] == 1 else -100.0
        return 0.0

    def _hit(row):
        return row["more_hit"] if row["bet_direction"] == "MORE" else row["less_hit"]

    df["pnl"]     = df.apply(_pnl, axis=1)
    df["bet_hit"] = df.apply(_hit, axis=1)
    return df


def summarize_pnl(df: pd.DataFrame) -> None:
    if df.empty:
        print("No matched data yet — keep scraping.")
        return

    print(f"\n{'='*68}")
    print(f"PrizePicks P&L  ({df['game_date'].min()} → {df['game_date'].max()})")
    print(f"{'='*68}")
    print(f"  {'Outcome':<14} {'Bets':>5} {'Acc':>6} {'Profit':>10} {'ROI':>8} {'Avg edge':>9}")
    print(f"  {'─'*14} {'─'*5} {'─'*6} {'─'*10} {'─'*8} {'─'*9}")

    for outcome, grp in df.groupby("outcome"):
        n      = len(grp)
        acc    = grp["bet_hit"].mean()
        profit = grp["pnl"].sum()
        roi    = profit / (n * 100)
        edge   = grp["edge"].mean()
        print(f"  {outcome:<14} {n:>5} {acc:>6.3f} ${profit:>9,.2f} {roi:>+8.2%} {edge:>+9.3f}")

    n_tot      = len(df)
    acc_tot    = df["bet_hit"].mean()
    profit_tot = df["pnl"].sum()
    roi_tot    = profit_tot / (n_tot * 100)
    print(f"  {'─'*14} {'─'*5} {'─'*6} {'─'*10} {'─'*8} {'─'*9}")
    print(f"  {'TOTAL':<14} {n_tot:>5} {acc_tot:>6.3f} ${profit_tot:>9,.2f} {roi_tot:>+8.2%}")
    print()

    # Edge bucket breakdown: how does accuracy vary with edge size?
    print(f"  Accuracy by edge bucket (all outcomes combined):")
    print(f"  {'Edge range':<18} {'Bets':>5} {'Acc':>6} {'ROI':>8}")
    print(f"  {'─'*18} {'─'*5} {'─'*6} {'─'*8}")
    buckets = [
        ("  0–5%",  0.00, 0.05),
        ("  5–10%", 0.05, 0.10),
        ("  10–15%",0.10, 0.15),
        ("  >15%",  0.15, 1.00),
    ]
    for label, lo, hi in buckets:
        sub = df[(df["edge"].abs() >= lo) & (df["edge"].abs() < hi)]
        if sub.empty:
            continue
        n   = len(sub)
        acc = sub["bet_hit"].mean()
        roi = sub["pnl"].sum() / (n * 100)
        print(f"  {label:<18} {n:>5} {acc:>6.3f} {roi:>+8.2%}")


if __name__ == "__main__":
    init_db()
    df = load_pnl_data()
    summarize_pnl(df)
