USE ROLE ACCOUNTADMIN;
USE DATABASE STATCAST_DB;
USE SCHEMA MLB_RAW;
USE WAREHOUSE STATCAST_WH;

-- One row per plate appearance (final pitch of each PA where events IS NOT NULL).
-- Target: 8-class PA outcome — strikeout, walk, hbp, single, double, triple, home_run, out_in_play.
-- All batter/pitcher stats are computed over PRIOR PAs only (ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
-- or RANGE BETWEEN INTERVAL 'N days' PRECEDING AND INTERVAL '1 day' PRECEDING) to avoid leakage.
-- Swing behavior features are approximated over PA-ending pitch rows (proxy for full pitch-level rates).

CREATE OR REPLACE VIEW vw_model2_pa_features AS

WITH flags AS (
    SELECT
        *,
        CASE
            WHEN pitch_type IN ('FF', 'FT', 'SI', 'FC') THEN 'Fastball'
            WHEN pitch_type IN ('SL', 'CU', 'KC', 'CS', 'SV', 'ST') THEN 'Breaking'
            WHEN pitch_type IN ('CH', 'FS', 'FO', 'SC', 'KN', 'EP') THEN 'Offspeed'
            ELSE 'Other'
        END AS pitch_category,
        IFF(
            description IN (
                'swinging_strike', 'swinging_strike_blocked',
                'foul', 'foul_tip', 'foul_bunt', 'missed_bunt', 'hit_into_play'
            ), 1, 0
        ) AS is_swing,
        IFF(
            description IN ('foul', 'foul_tip', 'foul_bunt', 'hit_into_play'),
            1, 0
        ) AS is_contact
    FROM raw_statcast_pitches
    WHERE game_type = 'R'
      AND description NOT IN ('automatic_ball', 'automatic_strike')
),

pa_base AS (
    SELECT
        *,
        CASE
            WHEN events IN ('strikeout', 'strikeout_double_play')                  THEN 'strikeout'
            WHEN events IN ('walk', 'intent_walk')                                 THEN 'walk'
            WHEN events = 'hit_by_pitch'                                           THEN 'hbp'
            WHEN events = 'single'                                                 THEN 'single'
            WHEN events = 'double'                                                 THEN 'double'
            WHEN events = 'triple'                                                 THEN 'triple'
            WHEN events = 'home_run'                                               THEN 'home_run'
            WHEN events IN (
                'field_out', 'grounded_into_double_play', 'force_out',
                'double_play', 'sac_fly', 'sac_bunt', 'fielders_choice',
                'fielders_choice_out', 'field_error', 'other_out',
                'triple_play', 'sac_fly_double_play', 'sac_bunt_double_play'
            )                                                                      THEN 'out_in_play'
            ELSE NULL
        END AS pa_outcome
    FROM flags
    WHERE events IS NOT NULL
),

pa AS (
    SELECT * FROM pa_base WHERE pa_outcome IS NOT NULL
)

SELECT

    -- identifiers
    game_pk,
    game_date,
    at_bat_number,
    batter,
    pitcher,

    -- ── Situational (start-of-PA values; unchanged from first to last pitch) ──

    inning,
    outs_when_up,
    IFF(on_1b IS NOT NULL, 1, 0)
        + IFF(on_2b IS NOT NULL, 1, 0)
        + IFF(on_3b IS NOT NULL, 1, 0)               AS num_runners_on,
    IFF(on_2b IS NOT NULL OR on_3b IS NOT NULL, 1, 0) AS runner_in_scoring_position,
    bat_score_diff,
    n_thruorder_pitcher,
    pitcher_days_since_prev_game,

    -- handedness / alignment
    stand,
    p_throws,
    if_fielding_alignment,
    of_fielding_alignment,

    -- ── Batter career PA outcome stats ───────────────────────────────────────

    DIV0NULL(
        SUM(woba_value)  OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(woba_denom)  OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_woba,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(*)                                     OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_rate,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(*)                                     OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_bb_rate,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'hbp', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(*)                                     OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_hbp_rate,

    -- HR rate per PA in wOBA denominator (excludes sac bunts)
    DIV0NULL(
        SUM(IFF(pa_outcome = 'home_run', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(woba_denom)                              OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_hr_rate,

    -- BABIP: hits on balls in play / balls in play (babip_value is 1 for hit BIP, 0 for out BIP, NULL otherwise)
    DIV0NULL(
        SUM(babip_value)   OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(babip_value) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_babip,

    -- Exit velocity and batted ball quality (NULLs on K/BB/HBP; AVG/COUNT naturally exclude them)
    AVG(launch_speed) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_exit_velo,

    AVG(launch_angle) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_launch_angle,

    DIV0NULL(
        SUM(IFF(launch_speed >= 95, 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(launch_speed)                 OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_hard_hit_rate,

    -- xwOBA on contact: NULL for K/BB/HBP, so AVG computes over BIP only
    -- Combines exit velo + launch angle into a single run-value-weighted quality metric
    AVG(estimated_woba_using_speedangle) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_xwoba,

    -- ── Batter recent (30-day) PA outcome stats ───────────────────────────────

    DIV0NULL(
        SUM(woba_value) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS batter_30d_woba,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        COUNT(*) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS batter_30d_k_rate,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        COUNT(*) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS batter_30d_bb_rate,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'home_run', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS batter_30d_hr_rate,

    DIV0NULL(
        SUM(babip_value)   OVER (PARTITION BY batter ORDER BY game_date
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

    -- ── Batter swing behavior (PA-ending pitch rows as proxy for full pitch-level rates) ──

    DIV0NULL(
        SUM(is_contact) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(is_swing)   OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_contact_rate,

    DIV0NULL(
        SUM(is_swing) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(*)      OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_swing_rate,

    AVG(bat_speed) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_bat_speed,

    AVG(swing_length) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_swing_length,

    AVG(miss_distance) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_miss_distance,

    -- ── Pitcher career PA outcome stats ──────────────────────────────────────

    DIV0NULL(
        SUM(woba_value) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_woba_against,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(*)                                  OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_rate,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(*)                             OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_bb_rate,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'home_run', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(woba_denom)                          OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_hr_rate,

    DIV0NULL(
        SUM(babip_value)   OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(babip_value) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_babip_against,

    -- GB rate: ground balls / all balls in play (bb_type is non-NULL only on BIP)
    DIV0NULL(
        SUM(IFF(bb_type = 'ground_ball', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(bb_type)                           OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_gb_rate,

    -- ── Pitcher recent (30-day) PA outcome stats ──────────────────────────────

    DIV0NULL(
        SUM(woba_value) OVER (PARTITION BY pitcher ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        SUM(woba_denom) OVER (PARTITION BY pitcher ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS pitcher_30d_woba_against,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        COUNT(*) OVER (PARTITION BY pitcher ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS pitcher_30d_k_rate,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'walk', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING),
        COUNT(*) OVER (PARTITION BY pitcher ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING)
    ) AS pitcher_30d_bb_rate,

    -- ── Batter platoon splits (career, vs RHP and vs LHP) ────────────────────

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout' AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(p_throws = 'R', 1, 0))                              OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_rate_vs_rhp,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout' AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(p_throws = 'L', 1, 0))                              OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_rate_vs_lhp,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'walk' AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(p_throws = 'R', 1, 0))                         OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_bb_rate_vs_rhp,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'walk' AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(p_throws = 'L', 1, 0))                         OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_bb_rate_vs_lhp,

    DIV0NULL(
        SUM(IFF(p_throws = 'R', woba_value, NULL)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(p_throws = 'R', woba_denom, NULL)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_woba_vs_rhp,

    DIV0NULL(
        SUM(IFF(p_throws = 'L', woba_value, NULL)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(p_throws = 'L', woba_denom, NULL)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_woba_vs_lhp,

    -- Platoon exit velo and xwOBA: NULL on K/BB/HBP so AVG naturally computes over BIP only
    AVG(IFF(p_throws = 'R', launch_speed, NULL)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_exit_velo_vs_rhp,

    AVG(IFF(p_throws = 'L', launch_speed, NULL)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_exit_velo_vs_lhp,

    AVG(IFF(p_throws = 'R', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_xwoba_vs_rhp,

    AVG(IFF(p_throws = 'L', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS batter_career_avg_xwoba_vs_lhp,

    -- ── Pitcher platoon splits (career, vs RHB and vs LHB) ───────────────────

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout' AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(stand = 'R', 1, 0))                              OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_rate_vs_rhb,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout' AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(stand = 'L', 1, 0))                              OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_rate_vs_lhb,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'walk' AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(stand = 'R', 1, 0))                         OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_bb_rate_vs_rhb,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'walk' AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(stand = 'L', 1, 0))                         OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_bb_rate_vs_lhb,

    DIV0NULL(
        SUM(IFF(stand = 'R', woba_value, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(stand = 'R', woba_denom, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_woba_against_vs_rhb,

    DIV0NULL(
        SUM(IFF(stand = 'L', woba_value, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(stand = 'L', woba_denom, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_woba_against_vs_lhb,

    -- ── Pitcher contact quality allowed by pitch category ─────────────────────
    -- Captures "gets fastball crushed" vs "weak contact on breaking ball" profiles.
    -- AVG ignores NULL launch_speed (K/BB/HBP rows), so computes only over BIP on that pitch type.

    AVG(IFF(pitch_category = 'Fastball', launch_speed, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_exit_velo_fastball,

    AVG(IFF(pitch_category = 'Breaking', launch_speed, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_exit_velo_breaking,

    AVG(IFF(pitch_category = 'Offspeed', launch_speed, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_exit_velo_offspeed,

    AVG(IFF(pitch_category = 'Fastball', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_xwoba_fastball,

    AVG(IFF(pitch_category = 'Breaking', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_xwoba_breaking,

    AVG(IFF(pitch_category = 'Offspeed', estimated_woba_using_speedangle, NULL)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_xwoba_offspeed,

    -- K:BB ratios — directly encode discrimination power within not-in-play PAs.
    -- Computed inline since Snowflake can't reference same-level aliases.
    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(pa_outcome = 'walk',      1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_bb_ratio,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(pa_outcome = 'walk',      1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_bb_ratio,

    -- Platoon K:BB ratios
    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout' AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(pa_outcome = 'walk'      AND p_throws = 'R', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_bb_ratio_vs_rhp,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout' AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(pa_outcome = 'walk'      AND p_throws = 'L', 1, 0)) OVER (PARTITION BY batter ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS batter_career_k_bb_ratio_vs_lhp,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout' AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(pa_outcome = 'walk'      AND stand = 'R', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_bb_ratio_vs_rhb,

    DIV0NULL(
        SUM(IFF(pa_outcome = 'strikeout' AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(IFF(pa_outcome = 'walk'      AND stand = 'L', 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_k_bb_ratio_vs_lhb,

    -- ── Pitcher pitch-level aggregate rates (over PA-ending rows as proxy) ────

    DIV0NULL(
        SUM(is_swing) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        COUNT(*)      OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_swing_rate,

    DIV0NULL(
        SUM(IFF(is_swing = 1 AND is_contact = 0, 1, 0)) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        SUM(is_swing)                                    OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) AS pitcher_career_miss_rate,

    AVG(release_speed) OVER (PARTITION BY pitcher ORDER BY game_date, game_pk, at_bat_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS pitcher_career_avg_velo,

    -- targets
    pa_outcome,
    -- xwOBA target for Stage 2a regression (NULL on K/BB/HBP — non-null only on BIPs)
    estimated_woba_using_speedangle AS bip_xwoba,
    -- actual wOBA value per PA for cascade evaluation (0 for outs)
    COALESCE(woba_value, 0)         AS actual_woba_value

FROM pa
ORDER BY batter, game_date, at_bat_number;

-- Sanity checks
SELECT COUNT(*) AS total_pa FROM vw_model2_pa_features;
SELECT pa_outcome, COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM vw_model2_pa_features
GROUP BY pa_outcome
ORDER BY COUNT(*) DESC;
