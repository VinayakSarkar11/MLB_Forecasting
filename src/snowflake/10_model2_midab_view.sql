USE ROLE ACCOUNTADMIN;
USE DATABASE STATCAST_DB;
USE SCHEMA MLB_RAW;
USE WAREHOUSE STATCAST_WH;

-- One row per PITCH in every valid PA.
-- Career/30d stats are guarded with IFF(events IS NOT NULL, ...) so that
-- intermediate pitch rows in a PA don't double-count the PA's outcome.
-- Columns like woba_value, launch_speed, estimated_woba_using_speedangle are
-- already NULL on non-final pitches in Statcast, so their aggregations need
-- no extra guard. Only pa_outcome-based counts and COUNT(*) denominators do.
-- Within-AB sequence features partition by (game_pk, at_bat_number) and give
-- the state BEFORE the current pitch is thrown.

CREATE OR REPLACE VIEW vw_model2_midab_features AS

WITH

all_pitches AS (
    SELECT
        *,
        CASE
            WHEN pitch_type IN ('FF', 'FT', 'SI', 'FC') THEN 'Fastball'
            WHEN pitch_type IN ('SL', 'CU', 'KC', 'CS', 'SV', 'ST') THEN 'Breaking'
            WHEN pitch_type IN ('CH', 'FS', 'FO', 'SC', 'KN', 'EP') THEN 'Offspeed'
            ELSE 'Other'
        END AS pitch_category,
        IFF(description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt', 'missed_bunt', 'hit_into_play'
        ), 1, 0) AS is_swing,
        IFF(description IN ('foul', 'foul_tip', 'foul_bunt', 'hit_into_play'), 1, 0) AS is_contact
    FROM raw_statcast_pitches
    WHERE game_type = 'R'
      AND description NOT IN ('automatic_ball', 'automatic_strike')
),

pa_outcomes AS (
    SELECT
        game_pk,
        at_bat_number,
        CASE
            WHEN events IN ('strikeout', 'strikeout_double_play')   THEN 'strikeout'
            WHEN events IN ('walk', 'intent_walk')                  THEN 'walk'
            WHEN events = 'hit_by_pitch'                            THEN 'hbp'
            WHEN events = 'single'                                  THEN 'single'
            WHEN events = 'double'                                  THEN 'double'
            WHEN events = 'triple'                                  THEN 'triple'
            WHEN events = 'home_run'                                THEN 'home_run'
            WHEN events IN (
                'field_out', 'grounded_into_double_play', 'force_out',
                'double_play', 'sac_fly', 'sac_bunt', 'fielders_choice',
                'fielders_choice_out', 'field_error', 'other_out',
                'triple_play', 'sac_fly_double_play', 'sac_bunt_double_play'
            )                                                       THEN 'out_in_play'
            ELSE NULL
        END AS pa_outcome,
        estimated_woba_using_speedangle AS bip_xwoba,
        COALESCE(woba_value, 0)         AS actual_woba_value
    FROM all_pitches
    WHERE events IS NOT NULL
),

-- Every pitch joined to the outcome of its PA
base AS (
    SELECT
        ap.*,
        po.pa_outcome,
        po.bip_xwoba,
        po.actual_woba_value
    FROM all_pitches ap
    INNER JOIN pa_outcomes po
        ON  ap.game_pk       = po.game_pk
        AND ap.at_bat_number = po.at_bat_number
    WHERE po.pa_outcome IS NOT NULL
)

SELECT

    -- identifiers
    game_pk,
    game_date,
    at_bat_number,
    pitch_number,
    batter,
    pitcher,

    -- ── Ballpark ───────────────────────────────────────────────────────────────
    -- home_team is the park proxy (stable — teams rarely move).
    -- park_hr_rate: rolling HR/PA at this park prior to this game.
    -- park_k_rate / park_bb_rate: Coors altitude/humidity affect contact;
    --   most parks differ less here but the model will learn what matters.
    home_team,

    DIV0NULL(
        SUM(IFF(events = 'home_run', 1, 0)) OVER (
            PARTITION BY home_team
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (
            PARTITION BY home_team
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS park_hr_rate,

    DIV0NULL(
        SUM(IFF(events IN ('strikeout', 'strikeout_double_play'), 1, 0)) OVER (
            PARTITION BY home_team
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (
            PARTITION BY home_team
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS park_k_rate,

    DIV0NULL(
        SUM(IFF(events IN ('walk', 'intent_walk'), 1, 0)) OVER (
            PARTITION BY home_team
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (
            PARTITION BY home_team
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS park_bb_rate,

    -- ── Situational ────────────────────────────────────────────────────────────
    inning,
    outs_when_up,
    IFF(on_1b IS NOT NULL, 1, 0)                       AS on_1b,
    IFF(on_2b IS NOT NULL, 1, 0)                       AS on_2b,
    IFF(on_3b IS NOT NULL, 1, 0)                       AS on_3b,
    IFF(on_1b IS NOT NULL, 1, 0)
        + IFF(on_2b IS NOT NULL, 1, 0)
        + IFF(on_3b IS NOT NULL, 1, 0)                AS num_runners_on,
    IFF(on_2b IS NOT NULL OR on_3b IS NOT NULL, 1, 0)  AS runner_in_scoring_position,
    bat_score_diff,
    n_thruorder_pitcher,
    pitcher_days_since_prev_game,
    stand,
    p_throws,
    if_fielding_alignment,
    of_fielding_alignment,

    -- ── Batter career outcome stats ────────────────────────────────────────────
    -- Guard: IFF(events IS NOT NULL, ...) so only the final pitch of each PA
    -- contributes to the career count. woba_value/woba_denom are already NULL
    -- on intermediate pitches so no guard needed there.

    DIV0NULL(
        SUM(woba_value) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_woba,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_rate,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_bb_rate,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'hbp', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_hbp_rate,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'home_run', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_hr_rate,

    DIV0NULL(
        SUM(babip_value) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(babip_value) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_babip,

    AVG(launch_speed) OVER (PARTITION BY batter
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_exit_velo,

    AVG(launch_angle) OVER (PARTITION BY batter
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_launch_angle,

    DIV0NULL(
        SUM(IFF(launch_speed >= 95, 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(launch_speed) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_hard_hit_rate,

    -- Batted ball type rates — key predictors of BABIP:
    -- GB (~0.24 BABIP), LD (~0.68 BABIP). FB omitted: it's roughly 1 - GB - LD - popup
    -- and adds no independent information for a tree model.
    -- COUNT(bb_type) auto-filters to BIP-only since bb_type is NULL on non-contact pitches.
    DIV0NULL(
        SUM(IFF(bb_type = 'ground_ball', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(bb_type) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_gb_rate,

    DIV0NULL(
        SUM(IFF(bb_type = 'line_drive', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(bb_type) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_ld_rate,

    AVG(estimated_woba_using_speedangle) OVER (PARTITION BY batter
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_xwoba,

    -- ── Batter 30d outcome stats ───────────────────────────────────────────────
    -- RANGE windows exclude same-day rows but include all pitches from prior days.
    -- Guard needed on pa_outcome counts; woba/launch cols are already NULL on
    -- intermediate pitches.

    DIV0NULL(
        SUM(woba_value) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS batter_30d_woba,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS batter_30d_k_rate,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS batter_30d_bb_rate,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'home_run', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS batter_30d_hr_rate,

    DIV0NULL(
        SUM(babip_value) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        COUNT(babip_value) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS batter_30d_babip,

    AVG(launch_speed) OVER (PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
    ) AS batter_30d_avg_exit_velo,

    AVG(launch_angle) OVER (PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
    ) AS batter_30d_avg_launch_angle,

    AVG(estimated_woba_using_speedangle) OVER (PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
    ) AS batter_30d_avg_xwoba,

    -- ── Batter swing behavior (all pitch rows — correctly pitch-level here) ────
    DIV0NULL(
        SUM(is_contact) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(is_swing) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_contact_rate,

    DIV0NULL(
        SUM(is_swing) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(*) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_swing_rate,

    -- ── K:BB ratios ────────────────────────────────────────────────────────────
    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_bb_ratio,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_bb_ratio,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout' AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk' AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_bb_ratio_vs_rhp,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout' AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk' AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_bb_ratio_vs_lhp,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout' AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk' AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_bb_ratio_vs_rhb,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout' AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk' AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_bb_ratio_vs_lhb,

    -- ── Batter platoon splits ──────────────────────────────────────────────────
    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout' AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_rate_vs_rhp,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout' AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_rate_vs_lhp,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk' AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_bb_rate_vs_rhp,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk' AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_bb_rate_vs_lhp,

    DIV0NULL(
        SUM(IFF(p_throws = 'R', woba_value, NULL)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(p_throws = 'R', woba_denom, NULL)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_woba_vs_rhp,

    DIV0NULL(
        SUM(IFF(p_throws = 'L', woba_value, NULL)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(p_throws = 'L', woba_denom, NULL)) OVER (PARTITION BY batter
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_woba_vs_lhp,

    AVG(IFF(p_throws = 'R', launch_speed, NULL)) OVER (PARTITION BY batter
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_exit_velo_vs_rhp,

    AVG(IFF(p_throws = 'L', launch_speed, NULL)) OVER (PARTITION BY batter
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_exit_velo_vs_lhp,

    AVG(IFF(p_throws = 'R', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY batter
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_xwoba_vs_rhp,

    AVG(IFF(p_throws = 'L', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY batter
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_xwoba_vs_lhp,

    -- ── Pitcher career outcome stats ───────────────────────────────────────────
    DIV0NULL(
        SUM(woba_value) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_woba_against,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_rate,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_bb_rate,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'home_run', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_hr_rate,

    DIV0NULL(
        SUM(babip_value) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(babip_value) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_babip_against,

    DIV0NULL(
        SUM(IFF(bb_type = 'ground_ball', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(bb_type) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_gb_rate,

    -- ── Pitcher platoon splits ─────────────────────────────────────────────────
    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout' AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_rate_vs_rhb,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout' AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_rate_vs_lhb,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk' AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_bb_rate_vs_rhb,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk' AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(events IS NOT NULL AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_bb_rate_vs_lhb,

    DIV0NULL(
        SUM(IFF(stand = 'R', woba_value, NULL)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(stand = 'R', woba_denom, NULL)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_woba_against_vs_rhb,

    DIV0NULL(
        SUM(IFF(stand = 'L', woba_value, NULL)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(stand = 'L', woba_denom, NULL)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_woba_against_vs_lhb,

    -- ── Pitcher contact quality by pitch category ──────────────────────────────
    AVG(IFF(pitch_category = 'Fastball', launch_speed, NULL)) OVER (PARTITION BY pitcher
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_exit_velo_fastball,

    AVG(IFF(pitch_category = 'Breaking', launch_speed, NULL)) OVER (PARTITION BY pitcher
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_exit_velo_breaking,

    AVG(IFF(pitch_category = 'Offspeed', launch_speed, NULL)) OVER (PARTITION BY pitcher
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_exit_velo_offspeed,

    AVG(IFF(pitch_category = 'Fastball', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY pitcher
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_xwoba_fastball,

    AVG(IFF(pitch_category = 'Breaking', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY pitcher
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_xwoba_breaking,

    AVG(IFF(pitch_category = 'Offspeed', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY pitcher
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_xwoba_offspeed,

    -- ── Pitcher 30d outcome stats ──────────────────────────────────────────────
    DIV0NULL(
        SUM(woba_value) OVER (PARTITION BY pitcher ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY pitcher ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS pitcher_30d_woba_against,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS pitcher_30d_k_rate,

    DIV0NULL(
        SUM(IFF(events IS NOT NULL AND pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(IFF(events IS NOT NULL, 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS pitcher_30d_bb_rate,

    -- ── Pitcher pitch-level aggregate rates (all pitch rows — truly pitch-level) ─
    DIV0NULL(
        SUM(is_swing) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(*) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_swing_rate,

    DIV0NULL(
        SUM(IFF(is_swing = 1 AND is_contact = 0, 1, 0)) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(is_swing) OVER (PARTITION BY pitcher
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_miss_rate,

    AVG(release_speed) OVER (PARTITION BY pitcher
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_velo,

    -- ── Within-AB state (count and pitch sequence BEFORE this pitch) ───────────
    balls,
    strikes,

    pitch_number - 1                                                          AS pitches_seen_in_ab,

    SUM(IFF(pitch_category = 'Fastball', 1, 0)) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )                                                                         AS fastballs_seen_in_ab,

    SUM(IFF(pitch_category = 'Breaking', 1, 0)) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )                                                                         AS breaking_seen_in_ab,

    SUM(IFF(pitch_category = 'Offspeed', 1, 0)) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )                                                                         AS offspeed_seen_in_ab,

    SUM(IFF(description IN ('foul', 'foul_tip', 'foul_bunt'), 1, 0)) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )                                                                         AS fouls_in_ab,

    SUM(IFF(description IN ('swinging_strike', 'swinging_strike_blocked'), 1, 0)) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )                                                                         AS whiffs_in_ab,

    -- ── Targets ────────────────────────────────────────────────────────────────
    pa_outcome,
    bip_xwoba,
    actual_woba_value

FROM base
ORDER BY batter, game_date, at_bat_number, pitch_number;


-- Sanity checks
SELECT COUNT(*) AS total_pitch_rows FROM vw_model2_midab_features;

SELECT
    COUNT(DISTINCT game_pk || '-' || at_bat_number) AS unique_pas,
    COUNT(*)                                         AS total_rows,
    ROUND(COUNT(*) / COUNT(DISTINCT game_pk || '-' || at_bat_number), 2) AS avg_pitches_per_pa
FROM vw_model2_midab_features;

-- Spot-check: confirm within-AB features reset per PA and career stats are stable
SELECT pitch_number, balls, strikes, pitches_seen_in_ab,
       fastballs_seen_in_ab, whiffs_in_ab,
       batter_career_k_rate, pa_outcome
FROM vw_model2_midab_features
WHERE game_pk = (SELECT MIN(game_pk) FROM vw_model2_midab_features)
ORDER BY at_bat_number, pitch_number
LIMIT 30;
