# StatCast Surge: MLB Plate Appearance Outcome Model

End-to-end ML pipeline predicting MLB plate appearance outcomes from pitch-by-pitch Statcast data. Built to evaluate edge against PrizePicks prop lines (Ks, walks, HRs, hits).

**Tech Stack:** Python, SQL, XGBoost, LightGBM, scikit-learn, Snowflake, AWS S3, MLflow

---

## What We're Building

A daily betting evaluation loop:
1. Scrape today's PrizePicks lines (11 AM PT)
2. Run our model → game-level P(1+ K), P(1+ walk), P(1+ HR), P(1+ hit) per batter
3. Compare model probability vs PP implied probability (~52.4% at -110)
4. Store actual outcomes (11 PM PT) to measure P&L over time

---

## Architecture

### Data Pipeline

Raw pitch-by-pitch Statcast data → AWS S3 → Snowflake → `vw_model2_midab_features`

The Snowflake view constructs point-in-time features using window functions with `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` to prevent target leakage. One row per pitch; the final PA outcome is attached to every pitch in that AB.

### Failed Approach: Pitch-Level Cascade

The original design predicted immediate pitch outcomes (Ball / Called Strike / Swinging Strike / Foul / In-Play), then conditionally predicted the batted-ball result on In-Play pitches. This mirrors a natural reading of baseball as a sequence of pitch events.

**Why it was abandoned:**

- Prop bets are on PA-level outcomes (did the batter strike out?), not pitch-level outcomes. A pitch-level model requires chaining probabilities across pitches to produce a PA-level estimate — mathematically tractable but error compounds across 4–6 pitch chains.
- Pitch outcomes are extremely noisy targets. The difference between a called strike and a ball on a borderline pitch is near-random; the model learns umpire tendencies more than pitcher/batter skill.
- Count state at each pitch already encodes the cumulative pitch history. Predicting the PA outcome from the current state is a cleaner signal than trying to predict each individual pitch.

### Current Approach: Mid-AB PA Cascade

One row per pitch, but the **target is the final PA outcome** — not what happened on this specific pitch. The count, base state, and within-AB sequence at the time of each pitch serve as features describing "where this PA is headed."

This gives us within-AB dynamics (a 1-2 count is more likely to K than 2-0) while predicting a target that maps directly to prop bets.

```
Input: pitch-level features (count, base state, batter/pitcher career stats, park, AB sequence)
         │
         ▼
Stage 1 (binary): In-Play vs NIP
         │
    ─────┴─────────────────────────
    │                             │
 NIP (K or walk)              In-Play (BIP)
    │                             │
Stage 2b (binary):          Stage 2a (regression + multiclass):
K vs walk                   xwOBA regression
                            BIP class: single / xbh / HR / out_in_play
         │
         ▼
Cascade outputs: P(strikeout), P(walk), P(single), P(xbh), P(home_run), P(out_in_play), E[wOBA]
         │
         ▼
Direct classifiers (trained on simplified targets for cleaner gradient):
  Binary:    on-base vs out
  3-class:   hit / walk / out
         │
         ▼
Isotonic calibration on 15% holdout for all outputs
```

**Game-level aggregation:**

For a batter with N expected PAs: `P(1+ K) = 1 - Π(1 - P(K per PA))`

This is the number PrizePicks cares about (did the batter get at least one K today?).

---

## Model Performance

Evaluated on a time-based test split (last ~15% of dates). Primary metric is Brier Skill Score (BSS) vs a situation-aware naive baseline (count + outs + base state frequencies from training data). Log-loss is not the right metric here — prop betting only cares about per-outcome calibration, not joint 6-class distribution quality.

| Outcome | BSS | Notes |
|---|---|---|
| Strikeout | ~0.064 | Strongest signal. K rate is predictable from career and recent trends. |
| Walk | ~0.084 | Strongest signal. BB rate reflects pitcher control and batter discipline. |
| Home Run | ~0.004 | Weak but useful. Park factors and exit velo help; single-PA variance dominates. |
| Hit (single + xbh + HR) | ~0.009 | Near the pre-pitch ceiling. BABIP is the hardest outcome to forecast. |
| On-base (direct) | ~0.034 | Direct classifier outperforms cascade simplified split aggregate. |

### Strengths

- **K and walk prediction** — strong enough for real prop edge. Career K%, pitcher K%, 30-day trends, and within-AB state (count, whiffs, fouls) combine for meaningful lift.
- **Situation-aware** — count state, outs, base state, and within-AB pitch sequence are all features. A batter in a 3-2 count with 2 fouls is genuinely more likely to K than a batter in a 1-0 count.
- **Platoon effects** — batter and pitcher handedness splits (wOBA vs RHP/LHP, K rate vs RHB/LHB) are encoded as continuous features, not dummies. The model learns magnitudes.
- **Ballpark factors** — rolling park HR/K/BB rates (prior games only) capture Coors/Yankee Stadium effects.
- **Calibration** — isotonic regression on a shared holdout maps raw GBT outputs to empirical rates. BSS > 0 means we beat a well-specified baseline; the calibration curves are close to diagonal for K and walk.
- **Direct classifiers** — the on-base/out and hit/walk/out classifiers are trained on simplified targets with cleaner signal than the 6-class cascade.

### Weaknesses

- **Hit and HR prediction is near the pre-pitch ceiling.** BABIP is fundamentally noisy — the main driver is pitch location and trajectory (post-pitch), which we don't have. Career GB/LD rates and BABIP luck residual add marginal lift but BSS ~0.009 for hits is close to the theoretical maximum for a pre-pitch model.
- **No live pitch features.** Pitch velocity, spin rate, and location are the strongest predictors of contact quality. We only have pre-pitch (career, rolling, situational) features. Adding live pitch data would require a real-time inference pipeline.
- **PP lines above 0.5 are not modeled correctly.** `model_prob` is always P(1+ outcome). For a PP line of 1.5 Ks, we need P(2+ Ks), which requires a different aggregation formula. Current P&L analysis should be filtered to `line = 0.5` props for accuracy.
- **Player name matching is approximate.** PrizePicks names and MLB MLBAM names are matched with lowercase string equality. Edge cases (accents, suffixes, nicknames) will silently fail to join and be excluded from analysis.
- **Training data is ~2 seasons.** Rare events (HR in a specific park, a batter's platoon splits) have sparse denominators. More data would improve stability.

---

## Repository Structure

```
src/
  ml/
    load_data.py              Snowflake → DataFrame
    train_model2.py           Mid-AB cascade training (XGBoost + LightGBM)
  prizepicks/
    fetch_props.py            Scrape today's PP lines (curl_cffi / Chrome TLS spoof)
    fetch_lineups.py          MLB Stats API — today's starting lineups
    predict_game.py           Build pre-game PA rows → game-level predictions
    backtest.py               Model calibration on StatCast test split
    store.py                  SQLite DB for scraped lines + actual outcomes
    scrape_lines.py           Morning scraper (11 AM PT) — lines + model probs
    scrape_results.py         Evening scraper (11 PM PT) — MLB box scores
  snowflake/
    10_model2_midab_view.sql  Feature view (one row per pitch, PA outcome attached)

scripts/
  scrape_lines.sh             Shell wrapper for morning scraper
  scrape_results.sh           Shell wrapper for evening scraper
  launchd/
    install.sh                Load both launchd jobs (run once)
    com.mlbpred.scrape_lines.plist
    com.mlbpred.scrape_results.plist

data/
  prizepicks_history.db       SQLite — accumulates daily as scrapers run

models/
  cascade_midab/
    xgb/                      XGBoost model artifacts + calibrators
    lgb/                      LightGBM model artifacts + calibrators
```

---

## Next Steps

1. **Accumulate data** — scrapers run daily; meaningful P&L signal after ~4–6 weeks.
2. **Compare model vs PrizePicks** — `python -m src.prizepicks.store` once lines + results are joined.
3. **Tune model** — use P&L and BSS by edge bucket to identify where the model is and isn't adding value. Consider adding live pitch features for real-time inference.
4. **Quantify edge** — if K/walk BSS holds at game level, model should have measurable positive ROI on those props at -110.
