"""Aggregate PA-level model predictions to game-level PrizePicks props.

Pipeline:
  1. Load today's lineups (fetch_lineups)
  2. Pull career features for each batter + starting pitcher from Snowflake
  3. Construct feature rows for each expected PA (balls=0, strikes=0, outs=0,
     base state empty — pre-game best guess; updates live as game progresses)
  4. Run cascade + direct classifiers to get P(K), P(walk), P(HR), P(hit) per PA
  5. Aggregate across expected PAs → P(1+ Ks), P(1+ HRs), etc.
  6. Compare against PrizePicks implied probability
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.ml.snowflake_client import query_to_df
from src.ml.train_model2 import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS,
    prepare_features, predict_cascade, apply_calibrators, ALL_OUTCOMES,
)

MODEL_DIR = Path("models/cascade_midab")
MODEL_TYPE = "xgb"   # use XGB — consistently better across metrics

# Features to surface per outcome in explain_predictions
_EXPLAIN_FEATURES: dict[str, list[tuple[str, str]]] = {
    "strikeout": [
        ("batter_career_k_rate",        "Batter career K%"),
        ("batter_30d_k_rate",           "Batter 30d K%"),
        ("pitcher_career_k_rate",       "Pitcher career K%"),
        ("pitcher_30d_k_rate",          "Pitcher 30d K%"),
        ("park_k_rate",                 "Park K rate"),
        ("batter_career_contact_rate",  "Batter contact%"),
        ("pitcher_career_miss_rate",    "Pitcher miss%"),
        ("same_hand",                   "Same hand (0=platoon adv)"),
    ],
    "walk": [
        ("batter_career_bb_rate",       "Batter career BB%"),
        ("batter_30d_bb_rate",          "Batter 30d BB%"),
        ("pitcher_career_bb_rate",      "Pitcher career BB%"),
        ("pitcher_30d_bb_rate",         "Pitcher 30d BB%"),
        ("park_bb_rate",                "Park BB rate"),
        ("same_hand",                   "Same hand"),
    ],
    "home_run": [
        ("batter_career_hr_rate",       "Batter career HR%"),
        ("batter_career_hard_hit_rate", "Batter hard hit%"),
        ("batter_career_avg_exit_velo", "Batter exit velo"),
        ("pitcher_career_hr_rate",      "Pitcher career HR%"),
        ("park_hr_rate",                "Park HR rate"),
    ],
    "hit": [
        ("batter_career_woba",          "Batter career wOBA"),
        ("batter_30d_woba",             "Batter 30d wOBA"),
        ("batter_career_babip",         "Batter career BABIP"),
        ("batter_career_babip_luck",    "BABIP luck residual"),
        ("pitcher_career_woba_against", "Pitcher wOBA against"),
        ("batter_career_ld_rate",       "Batter LD rate"),
        ("batter_career_gb_rate",       "Batter GB rate"),
    ],
}

_OUTCOME_LABEL = {
    "strikeout": "K",
    "walk":      "BB",
    "home_run":  "HR",
    "hit":       "H",
}


def load_models(model_type: str = MODEL_TYPE) -> dict:
    d = MODEL_DIR / model_type
    return {
        "stage1":            joblib.load(d / "stage1_inplay.pkl"),
        "stage2a_reg":       joblib.load(d / "stage2a_xwoba_reg.pkl"),
        "stage2a_cls":       joblib.load(d / "stage2a_cls.pkl"),
        "stage2a_le":        joblib.load(d / "stage2a_le.pkl"),
        "stage2a_cals":      joblib.load(d / "stage2a_calibrators.pkl"),
        "stage2b":           joblib.load(d / "stage2b_strikeout_walk.pkl"),
        "stage2b_cal":       joblib.load(d / "stage2b_calibrator.pkl"),
        "woba_cal":          joblib.load(d / "cascade_woba_calibrator.pkl"),
        "direct_ob":         joblib.load(d / "direct_onbase.pkl"),
        "direct_ob_cal":     joblib.load(d / "direct_onbase_calibrator.pkl"),
        "direct_3cls":       joblib.load(d / "direct_3class.pkl"),
        "direct_3cls_le":    joblib.load(d / "direct_3class_le.pkl"),
        "direct_3cls_cals":  joblib.load(d / "direct_3class_calibrators.pkl"),
    }


def fetch_player_features(batter_ids: list[int], pitcher_ids: list[int]) -> pd.DataFrame:
    """Pull the most recent career feature row for each batter/pitcher pair."""
    b_ids = ",".join(str(i) for i in set(batter_ids))
    p_ids = ",".join(str(i) for i in set(pitcher_ids))

    query = f"""
        SELECT *
        FROM vw_model2_midab_features
        WHERE batter IN ({b_ids})
          AND pitcher IN ({p_ids})
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY batter, pitcher
            ORDER BY game_date DESC, at_bat_number DESC, pitch_number DESC
        ) = 1
    """
    df = query_to_df(query)
    df.columns = [c.lower() for c in df.columns]
    return df


def build_pregame_pa_rows(
    lineups: pd.DataFrame,
    features: pd.DataFrame,
    expected_pas: int = 4,
) -> pd.DataFrame:
    """Construct one feature row per expected PA (pre-game, count 0-0).

    We assume:
    - Count starts 0-0 (balls=0, strikes=0)
    - Bases empty, 0 outs (pre-game prior; update live for in-game use)
    - No within-AB sequence info yet (all sequence features = 0)
    """
    rows = []
    for _, batter_row in lineups.iterrows():
        batter_id  = batter_row["batter_mlbam_id"]
        pitcher_id = batter_row["starter_pitcher_id"]

        feat = features[
            (features["batter"]  == batter_id) &
            (features["pitcher"] == pitcher_id)
        ]
        if feat.empty:
            # Fallback: use any recent row for this batter
            feat = features[features["batter"] == batter_id]
        if feat.empty:
            continue

        base = feat.iloc[0].copy()
        # Override situational state to pre-game defaults
        base["balls"]   = 0;  base["strikes"] = 0
        base["outs_when_up"] = 0
        base["on_1b"]   = 0;  base["on_2b"]   = 0;  base["on_3b"] = 0
        base["num_runners_on"] = 0;  base["runner_in_scoring_position"] = 0
        base["inning"]  = 1;  base["bat_score_diff"] = 0
        # Within-AB sequence starts at zero
        for col in ["pitches_seen_in_ab", "fastballs_seen_in_ab",
                    "breaking_seen_in_ab", "offspeed_seen_in_ab",
                    "fouls_in_ab", "whiffs_in_ab"]:
            if col in base.index:
                base[col] = 0

        for _ in range(expected_pas):
            row = base.copy()
            row["batter_name"]  = batter_row["batter_name"]
            row["batting_order"] = batter_row["batting_order"]
            rows.append(row)

    return pd.DataFrame(rows)


def _add_engineered_features(pa_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the feature engineering from train_model2.main()."""
    pa_df = pa_df.copy()
    pa_df["same_hand"] = (pa_df["stand"] == pa_df["p_throws"]).astype(int)
    pa_df["batter_platoon_woba_split"] = (
        pa_df["batter_career_woba_vs_rhp"].astype(float)
        - pa_df["batter_career_woba_vs_lhp"].astype(float)
    )
    pa_df["pitcher_platoon_woba_split"] = (
        pa_df["pitcher_career_woba_against_vs_rhb"].astype(float)
        - pa_df["pitcher_career_woba_against_vs_lhb"].astype(float)
    )
    pa_df["batter_career_babip_luck"] = (
        pa_df["batter_career_babip"].astype(float)
        - (0.24 * pa_df["batter_career_gb_rate"].astype(float)
           + 0.68 * pa_df["batter_career_ld_rate"].astype(float))
    )
    return pa_df


def pa_probs_to_game(pa_df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Run models on PA rows and aggregate to game-level probabilities."""
    pa_df = _add_engineered_features(pa_df)
    X = prepare_features(pa_df)

    # Cascade
    cascade = predict_cascade(
        X,
        models["stage1"], models["stage2a_reg"], models["stage2a_cls"],
        models["stage2a_le"], models["stage2a_cals"],
        models["stage2b"], models["stage2b_cal"],
    )
    cascade["expected_woba"] = models["woba_cal"].transform(cascade["expected_woba"].values)

    # Direct classifiers
    p_ob_raw = models["direct_ob"].predict_proba(X)[:, 1]
    p_ob     = models["direct_ob_cal"].transform(p_ob_raw)

    col_idx  = [list(models["direct_3cls_le"].classes_).index(c)
                for c in ["out", "walk", "hit"]]
    p3_raw   = models["direct_3cls"].predict_proba(X)[:, col_idx]
    p3       = np.column_stack([
        models["direct_3cls_cals"][i].transform(p3_raw[:, i]) for i in range(3)
    ])
    p3       = p3 / p3.sum(axis=1, keepdims=True)

    pa_df = pa_df.copy()
    for col in ALL_OUTCOMES:
        pa_df[f"p_{col}"] = cascade[col].values
    pa_df["p_onbase"]   = p_ob
    pa_df["p_hit_dir"]  = p3[:, 2]   # "hit" column from direct 3-class
    pa_df["p_walk_dir"] = p3[:, 1]
    pa_df["pred_ewoba"] = cascade["expected_woba"].values

    # Aggregate: P(0 events) = product of (1 - p_i) across PAs
    def p_at_least_one(p_series):
        return 1.0 - np.prod(1.0 - p_series.values)

    agg = (
        pa_df
        .groupby("batter_name")
        .agg(
            expected_pas=("p_strikeout", "count"),
            p_1plus_k   =("p_strikeout", p_at_least_one),
            p_1plus_walk=("p_walk",       p_at_least_one),
            p_1plus_hr  =("p_home_run",   p_at_least_one),
            p_1plus_hit =("p_hit_dir",    p_at_least_one),
            p_1plus_ob  =("p_onbase",     p_at_least_one),
            mean_ewoba  =("pred_ewoba",   "mean"),
        )
        .reset_index()
        .round(3)
    )
    return agg


def compare_to_prizepicks(
    game_preds: pd.DataFrame,
    props: pd.DataFrame,
) -> pd.DataFrame:
    """Join model game-level predictions with PrizePicks lines.

    Returns rows where the model disagrees with the PrizePicks implied prob
    by more than a threshold — these are the potential edges.
    """
    OUTCOME_TO_PRED = {
        "strikeout": "p_1plus_k",
        "walk":      "p_1plus_walk",
        "home_run":  "p_1plus_hr",
        "hit":       "p_1plus_hit",
    }

    merged = props.merge(
        game_preds[["batter_name"] + list(OUTCOME_TO_PRED.values())],
        left_on="player_name", right_on="batter_name", how="inner"
    )
    merged["model_prob"] = merged.apply(
        lambda r: r[OUTCOME_TO_PRED.get(r["outcome"], "")], axis=1
    )
    # PrizePicks line is usually 0.5 for binary props (did player get 1+?)
    # pp_implied_prob is the break-even (~0.524 at -110)
    merged["edge"] = merged["model_prob"] - merged["pp_implied_prob"]
    merged["bet"]  = np.where(
        merged["model_prob"] > merged["pp_implied_prob"], "MORE", "LESS"
    )

    return (
        merged[["player_name", "team", "stat_type", "line",
                "model_prob", "pp_implied_prob", "edge", "bet"]]
        .sort_values("edge", ascending=False)
    )


def explain_predictions(
    player_name: str,
    pa_df: pd.DataFrame,
    models: dict,
    outcome: str = "strikeout",
    pp_line: float | None = None,
    pp_implied_prob: float = 0.524,
) -> None:
    """Print cascade breakdown and key feature drivers for one player-outcome.

    Args:
        player_name:    Must match `batter_name` in pa_df.
        pa_df:          All pre-game PA rows (all batters). Used for league-avg comparisons.
        models:         Loaded model dict from load_models().
        outcome:        One of "strikeout", "walk", "home_run", "hit".
        pp_line:        PrizePicks line value (e.g. 0.5 for 0.5+ Ks), optional.
        pp_implied_prob: Break-even probability at the juice (default -110 → 0.524).
    """
    player_pa = pa_df[pa_df["batter_name"] == player_name]
    if player_pa.empty:
        print(f"No PA rows found for {player_name}")
        return

    player_pa = _add_engineered_features(player_pa)
    X         = prepare_features(player_pa)
    n_pas     = len(player_pa)
    lbl       = _OUTCOME_LABEL.get(outcome, outcome)

    # Cascade intermediates
    p_ip  = models["stage1"].predict_proba(X)[:, 1]
    p_nip = 1.0 - p_ip

    raw_k      = models["stage2b"].predict_proba(X)[:, 1]
    p_k_nip    = models["stage2b_cal"].transform(raw_k)
    p_walk_nip = 1.0 - p_k_nip

    cascade = predict_cascade(
        X,
        models["stage1"], models["stage2a_reg"], models["stage2a_cls"],
        models["stage2a_le"], models["stage2a_cals"],
        models["stage2b"], models["stage2b_cal"],
    )
    cascade["expected_woba"] = models["woba_cal"].transform(
        cascade["expected_woba"].values
    )

    # Direct classifiers
    p_ob_raw = models["direct_ob"].predict_proba(X)[:, 1]
    p_ob     = models["direct_ob_cal"].transform(p_ob_raw)

    col_idx = [list(models["direct_3cls_le"].classes_).index(c)
               for c in ["out", "walk", "hit"]]
    p3_raw  = models["direct_3cls"].predict_proba(X)[:, col_idx]
    p3      = np.column_stack([
        models["direct_3cls_cals"][i].transform(p3_raw[:, i]) for i in range(3)
    ])
    p3 = p3 / p3.sum(axis=1, keepdims=True)

    # Per-PA probability array for the chosen outcome
    if outcome == "strikeout":
        p_per_pa = cascade["strikeout"].values
        detail = (f"P(NIP) × P(K|NIP) = {p_nip.mean():.3f} × {p_k_nip.mean():.3f}"
                  f" = {p_per_pa.mean():.3f} per PA")
    elif outcome == "walk":
        p_per_pa = cascade["walk"].values
        detail = (f"P(NIP) × P(BB|NIP) = {p_nip.mean():.3f} × {p_walk_nip.mean():.3f}"
                  f" = {p_per_pa.mean():.3f} per PA")
    elif outcome == "home_run":
        p_per_pa    = cascade["home_run"].values
        hr_given_ip = (p_per_pa / np.maximum(p_ip, 1e-9)).mean()
        detail = (f"P(IP) × P(HR|IP) = {p_ip.mean():.3f} × {hr_given_ip:.3f}"
                  f" = {p_per_pa.mean():.3f} per PA")
    elif outcome == "hit":
        p_per_pa = p3[:, 2]   # direct 3-class "hit"
        detail   = f"direct 3-class P(hit) = {p_per_pa.mean():.3f} per PA"
    else:
        print(f"Unknown outcome: {outcome!r}. Choose: strikeout, walk, home_run, hit")
        return

    p_0     = float(np.prod(1.0 - p_per_pa))
    p_1plus = 1.0 - p_0

    # Feature values: this player vs today's field average
    all_eng = _add_engineered_features(pa_df)
    league  = all_eng[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").mean()
    player_vals = player_pa[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").mean()

    sep = "─" * 62
    print(f"\n{player_name} — {lbl}")
    print(sep)
    print(f"\nCascade (avg over {n_pas} expected PAs at 0-0 count):")
    print(f"  {detail}")
    print(f"\nGame aggregation ({n_pas} PAs):")
    print(f"  P(0 {lbl}s)  = Π(1 - p_i)   = {p_0:.4f}")
    print(f"  P(1+ {lbl})  = 1 - {p_0:.4f}  = {p_1plus:.4f}  ← model")
    if pp_line is not None:
        print(f"  PP line     = {pp_line}")
    edge      = p_1plus - pp_implied_prob
    direction = "MORE" if edge > 0 else "LESS"
    print(f"  PP implied  = {pp_implied_prob:.3f}  edge = {edge:+.3f} → {direction}")

    feat_list = _EXPLAIN_FEATURES.get(outcome, [])
    if feat_list:
        print(f"\nKey drivers vs today's field average:")
        print(f"  {'Feature':<36} {'Player':>8} {'Avg':>8} {'Delta':>9}")
        print(f"  {'─'*36} {'─'*8} {'─'*8} {'─'*9}")
        for feat, feat_label in feat_list:
            pv = player_vals.get(feat, np.nan)
            lv = league.get(feat, np.nan)
            if pd.isna(pv) or pd.isna(lv):
                continue
            delta = pv - lv
            arrow = "↑" if delta > 0.001 else "↓" if delta < -0.001 else " "
            print(f"  {feat_label:<36} {pv:>8.3f} {lv:>8.3f} {delta:>+9.3f} {arrow}")
    print()


if __name__ == "__main__":
    from src.prizepicks.fetch_lineups import fetch_lineups
    from src.prizepicks.fetch_props   import fetch_props

    print("Loading models...")
    models  = load_models()

    print("Fetching lineups...")
    lineups = fetch_lineups()
    if lineups.empty:
        print("No lineups available yet — check back closer to game time.")
        raise SystemExit

    print(f"Found {len(lineups)} batter slots across {lineups['game_pk'].nunique()} games")

    batter_ids  = lineups["batter_mlbam_id"].dropna().astype(int).tolist()
    pitcher_ids = lineups["starter_pitcher_id"].dropna().astype(int).tolist()

    print("Pulling career features from Snowflake...")
    features = fetch_player_features(batter_ids, pitcher_ids)

    print("Building pre-game PA rows...")
    pa_rows = build_pregame_pa_rows(lineups, features)

    print("Running models...")
    game_preds = pa_probs_to_game(pa_rows, models)
    print("\nGame-level predictions:")
    print(game_preds.to_string(index=False))

    print("\nFetching PrizePicks props...")
    props = fetch_props()
    if props.empty:
        print("No props available right now.")
        raise SystemExit

    edges = compare_to_prizepicks(game_preds, props)
    print("\nModel vs PrizePicks (sorted by edge):")
    print(edges.to_string(index=False))

    # Explain top K edge and top HR edge
    for outcome_filter in ["strikeout", "home_run"]:
        subset = edges[edges["stat_type"].str.lower().str.contains(
            "strikeout" if outcome_filter == "strikeout" else "home run", na=False
        )]
        if subset.empty:
            continue
        top_row = subset.iloc[0]
        player  = top_row["player_name"]
        pp_line = top_row["line"]
        explain_predictions(
            player, pa_rows, models,
            outcome=outcome_filter,
            pp_line=pp_line,
        )
