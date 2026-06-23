USE ROLE ACCOUNTADMIN;
USE DATABASE STATCAST_DB;
USE SCHEMA MLB_RAW;
USE WAREHOUSE STATCAST_WH;

-- Model 1 (Plate Discipline) — flat / non-rolling features.
-- No window functions yet: these are either raw columns or simple
-- per-row derivations that don't require looking at other rows.

SELECT
    -- identifiers (kept for joining/ordering in later steps, not features)
    game_pk,
    game_date,
    at_bat_number,
    pitch_number,
    batter,
    pitcher,

    -- pitch identity & physics
    pitch_type,
    release_speed,
    release_spin_rate,
    pfx_x,
    pfx_z,
    plate_x,
    plate_z,
    zone,
    arm_angle,

    -- matchup / count context
    stand,
    p_throws,
    balls,
    strikes,

    -- game state
    outs_when_up,
    inning,
    bat_score,
    fld_score,
    bat_score_diff,
    home_team,

    -- baserunners (derived from on_1b/on_2b/on_3b, which hold a runner's
    -- batter ID or NULL — not booleans)
    IFF(on_1b IS NOT NULL, 1, 0)
        + IFF(on_2b IS NOT NULL, 1, 0)
        + IFF(on_3b IS NOT NULL, 1, 0)                       AS num_runners_on,
    IFF(on_2b IS NOT NULL OR on_3b IS NOT NULL, 1, 0)         AS runner_in_scoring_position,

    -- fielding alignment (pre-pitch decision, safe to use)
    if_fielding_alignment,
    of_fielding_alignment,

    -- pitcher fatigue / familiarity
    pitcher_days_since_prev_game,
    n_thruorder_pitcher

FROM raw_statcast_pitches;
