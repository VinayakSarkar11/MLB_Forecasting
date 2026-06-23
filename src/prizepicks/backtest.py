"""Historical backtesting of PrizePicks-style bets using the model test split.

For each (game_pk, batter) in the test set:
  1. Take the first pitch of each PA (0-0 count, pre-pitch state).
  2. Run all models → P(K per PA), P(walk per PA), P(HR per PA), P(hit per PA).
  3. Aggregate across PAs → P(1+ K), P(1+ walk), P(1+ HR), P(1+ hit).
  4. Compare to actual game-level outcomes.
  5. Report calibration by decile and P&L assuming -110 on every edge > threshold.

Usage:
    python -m src.prizepicks.backtest
    python -m src.prizepicks.backtest --model xgb --threshold 0.55
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.ml.load_data import load_model2_midab_features
from src.ml.train_model1 import time_based_split
from src.ml.train_model2 import (
    prepare_features, predict_cascade, apply_calibrators,
    NUMERIC_FEATURES, ALL_OUTCOMES, build_pa_labels,
)

MODEL_DIR     = Path("models/cascade_midab")
MODEL_TYPE    = "xgb"
PP_BREAK_EVEN = 0.524    # -110 juice break-even
BET_AMOUNT    = 100.0
WIN_RETURN    = 90.91    # profit on winning -110 bet


def load_models(model_type: str = MODEL_TYPE) -> dict:
    d = MODEL_DIR / model_type
    return {
        "stage1":           joblib.load(d / "stage1_inplay.pkl"),
        "stage2a_reg":      joblib.load(d / "stage2a_xwoba_reg.pkl"),
        "stage2a_cls":      joblib.load(d / "stage2a_cls.pkl"),
        "stage2a_le":       joblib.load(d / "stage2a_le.pkl"),
        "stage2a_cals":     joblib.load(d / "stage2a_calibrators.pkl"),
        "stage2b":          joblib.load(d / "stage2b_strikeout_walk.pkl"),
        "stage2b_cal":      joblib.load(d / "stage2b_calibrator.pkl"),
        "woba_cal":         joblib.load(d / "cascade_woba_calibrator.pkl"),
        "direct_ob":        joblib.load(d / "direct_onbase.pkl"),
        "direct_ob_cal":    joblib.load(d / "direct_onbase_calibrator.pkl"),
        "direct_3cls":      joblib.load(d / "direct_3class.pkl"),
        "direct_3cls_le":   joblib.load(d / "direct_3class_le.pkl"),
        "direct_3cls_cals": joblib.load(d / "direct_3class_calibrators.pkl"),
    }


def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate feature engineering from train_model2.main()."""
    df = df.copy()
    df["pa_outcome"] = df["pa_outcome"].replace(
        {"triple": "xbh", "double": "xbh", "hbp": "walk"}
    )
    df = build_pa_labels(df)
    df["same_hand"] = (df["stand"] == df["p_throws"]).astype(int)
    df["batter_platoon_woba_split"] = (
        df["batter_career_woba_vs_rhp"].astype(float)
        - df["batter_career_woba_vs_lhp"].astype(float)
    )
    df["pitcher_platoon_woba_split"] = (
        df["pitcher_career_woba_against_vs_rhb"].astype(float)
        - df["pitcher_career_woba_against_vs_lhb"].astype(float)
    )
    df["batter_career_babip_luck"] = (
        df["batter_career_babip"].astype(float)
        - (0.24 * df["batter_career_gb_rate"].astype(float)
           + 0.68 * df["batter_career_ld_rate"].astype(float))
    )
    return df


def _first_pitch_per_pa(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to one row per PA (minimum pitches_seen_in_ab = start of AB).

    This mirrors the pre-game prediction setting where we predict each PA at 0-0
    count before any pitches are thrown. Using start-of-AB rows avoids double-
    counting the same PA multiple times when aggregating game-level probabilities.
    """
    idx = df.groupby(["game_pk", "at_bat_number"])["pitches_seen_in_ab"].idxmin()
    return df.loc[idx].reset_index(drop=True)


def _score_pas(pa_df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Run all models on PA rows, return pa_df with prediction columns added."""
    X = prepare_features(pa_df)

    cascade = predict_cascade(
        X,
        models["stage1"], models["stage2a_reg"], models["stage2a_cls"],
        models["stage2a_le"], models["stage2a_cals"],
        models["stage2b"], models["stage2b_cal"],
    )
    cascade["expected_woba"] = models["woba_cal"].transform(
        cascade["expected_woba"].values
    )

    p_ob_raw = models["direct_ob"].predict_proba(X)[:, 1]
    p_ob     = models["direct_ob_cal"].transform(p_ob_raw)

    col_idx = [list(models["direct_3cls_le"].classes_).index(c)
               for c in ["out", "walk", "hit"]]
    p3_raw  = models["direct_3cls"].predict_proba(X)[:, col_idx]
    p3      = np.column_stack([
        models["direct_3cls_cals"][i].transform(p3_raw[:, i]) for i in range(3)
    ])
    p3 = p3 / p3.sum(axis=1, keepdims=True)

    out = pa_df.copy()
    for col in ALL_OUTCOMES:
        out[f"p_{col}"] = cascade[col].values
    out["p_onbase"]   = p_ob
    out["p_hit_dir"]  = p3[:, 2]
    out["p_walk_dir"] = p3[:, 1]
    out["pred_ewoba"] = cascade["expected_woba"].values
    return out


def _aggregate_to_game(pa_scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored PA rows → game-level P(1+) and actual outcomes."""

    def p_at_least_one(s):
        return 1.0 - float(np.prod(1.0 - s.values))

    def any_flag(s):
        return int(s.any())

    df = pa_scored.copy()
    df["act_k"]    = (df["pa_outcome"] == "strikeout").astype(int)
    df["act_walk"] = (df["pa_outcome"] == "walk").astype(int)
    df["act_hr"]   = (df["pa_outcome"] == "home_run").astype(int)
    df["act_hit"]  = df["pa_outcome"].isin({"single", "xbh", "home_run"}).astype(int)

    agg = (
        df
        .groupby(["game_pk", "batter"])
        .agg(
            game_date        =("game_date",    "first"),
            n_pas            =("p_strikeout",  "count"),
            p_1plus_k        =("p_strikeout",  p_at_least_one),
            p_1plus_walk     =("p_walk",       p_at_least_one),
            p_1plus_hr       =("p_home_run",   p_at_least_one),
            p_1plus_hit      =("p_hit_dir",    p_at_least_one),
            actual_k         =("act_k",        any_flag),
            actual_walk      =("act_walk",     any_flag),
            actual_hr        =("act_hr",       any_flag),
            actual_hit       =("act_hit",      any_flag),
        )
        .reset_index()
    )
    return agg


def calibration_table(
    game_df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bucket predictions into n_bins quantile bins; show actual rate per bucket."""
    bins = pd.qcut(game_df[pred_col], n_bins, duplicates="drop")
    tbl = (
        game_df
        .groupby(bins, observed=True)
        .agg(
            count       =(pred_col,   "count"),
            pred_mean   =(pred_col,   "mean"),
            actual_rate =(actual_col, "mean"),
        )
        .reset_index(drop=True)
    )
    tbl["residual"] = tbl["actual_rate"] - tbl["pred_mean"]
    return tbl


def simulate_pnl(
    game_df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    threshold: float = PP_BREAK_EVEN,
) -> dict:
    """Bet MORE on every row where model_prob > threshold at -110 odds."""
    bets = game_df[game_df[pred_col] > threshold].copy()
    if bets.empty:
        return {
            "n_bets": 0, "n_wins": 0, "n_losses": 0,
            "profit": 0.0, "roi": float("nan"), "accuracy": float("nan"),
        }
    wins   = int((bets[actual_col] == 1).sum())
    losses = len(bets) - wins
    profit = wins * WIN_RETURN - losses * BET_AMOUNT
    return {
        "n_bets":   len(bets),
        "n_wins":   wins,
        "n_losses": losses,
        "profit":   round(profit, 2),
        "roi":      round(profit / (len(bets) * BET_AMOUNT), 4),
        "accuracy": round(wins / len(bets), 4),
    }


def run_backtest(
    model_type: str = MODEL_TYPE,
    threshold: float = PP_BREAK_EVEN,
) -> pd.DataFrame:
    """End-to-end backtest. Returns the game-level prediction DataFrame."""
    print("Loading data from Snowflake...")
    raw = load_model2_midab_features()
    df  = _feature_engineer(raw)

    _, test_df, cutoff = time_based_split(df)
    print(f"Test split: {len(test_df):,} pitch rows starting {cutoff.date()}")

    pa_df = _first_pitch_per_pa(test_df)
    print(f"Unique PAs in test set: {len(pa_df):,}")

    print("Loading models...")
    models = load_models(model_type)

    print("Scoring PAs...")
    pa_scored = _score_pas(pa_df, models)

    print("Aggregating to game level...")
    game_df = _aggregate_to_game(pa_scored)
    print(f"Game-level rows: {len(game_df):,}  "
          f"({game_df['game_pk'].nunique():,} games, "
          f"{game_df['batter'].nunique():,} unique batters)")

    PROPS = [
        ("p_1plus_k",    "actual_k",    "K"),
        ("p_1plus_walk", "actual_walk", "BB"),
        ("p_1plus_hr",   "actual_hr",   "HR"),
        ("p_1plus_hit",  "actual_hit",  "H"),
    ]

    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS  ({model_type.upper()}, threshold={threshold:.3f})")
    print(f"{'='*70}")

    for pred_col, actual_col, name in PROPS:
        cal = calibration_table(game_df, pred_col, actual_col)
        pnl = simulate_pnl(game_df, pred_col, actual_col, threshold)

        actual_rate = game_df[actual_col].mean()
        print(f"\n── {name}  (base rate: {actual_rate:.3f}) {'─'*40}")
        print(f"  {'Decile':<8} {'Count':>6} {'Pred%':>8} {'Actual%':>9} {'Residual':>10}")
        print(f"  {'─'*8} {'─'*6} {'─'*8} {'─'*9} {'─'*10}")
        for i, row in cal.iterrows():
            print(f"  {i+1:<8} {int(row['count']):>6}"
                  f" {row['pred_mean']:>8.3f}"
                  f" {row['actual_rate']:>9.3f}"
                  f" {row['residual']:>+10.3f}")

        p = pnl
        print(f"\n  P&L (betting MORE when model > {threshold:.3f}):")
        print(f"  Bets {p['n_bets']:>5}  |  Wins {p['n_wins']:>5}  "
              f"Losses {p['n_losses']:>5}  |  Accuracy {p['accuracy']:.3f}")
        print(f"  Profit ${p['profit']:>10,.2f}  |  ROI {p['roi']:>+.2%}")

    return game_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=MODEL_TYPE, choices=["lgb", "xgb"])
    parser.add_argument("--threshold", default=PP_BREAK_EVEN, type=float)
    args = parser.parse_args()
    run_backtest(model_type=args.model, threshold=args.threshold)
