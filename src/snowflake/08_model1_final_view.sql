USE ROLE ACCOUNTADMIN;
USE DATABASE STATCAST_DB;
USE SCHEMA MLB_RAW;
USE WAREHOUSE STATCAST_WH;

-- Combines flat features, rolling batter features (3 windows), zone-specific
-- and pitch-type-specific batter rates, pitcher aggregate rates, and the
-- 6-class target into a single view — one row per pitch, no JOINs.
-- All window functions read from the `flags` CTE which pre-computes
-- is_swing, is_contact, and pitch_category for each pitch.

CREATE OR REPLACE VIEW vw_model1_features AS

WITH flags AS (
    SELECT
        *,
        IFF(
            description IN (
                'swinging_strike', 'swinging_strike_blocked',
                'foul', 'foul_tip', 'foul_bunt', 'missed_bunt',
                'hit_into_play'
            ), 1, 0
        ) AS is_swing,
        IFF(
            description IN ('foul', 'foul_tip', 'foul_bunt', 'hit_into_play'),
            1, 0
        ) AS is_contact,
        CASE
            WHEN pitch_type IN ('FF', 'FT', 'SI', 'FC') THEN 'Fastball'
            WHEN pitch_type IN ('SL', 'CU', 'KC', 'CS', 'SV', 'ST') THEN 'Breaking'
            WHEN pitch_type IN ('CH', 'FS', 'FO', 'SC', 'KN', 'EP') THEN 'Offspeed'
            ELSE 'Other'
        END AS pitch_category,
        CASE
            WHEN description IN ('ball', 'blocked_ball', 'pitchout')    THEN 'Ball'
            WHEN description = 'called_strike'                          THEN 'Called_Strike'
            WHEN description IN (
                    'swinging_strike', 'swinging_strike_blocked',
                    'foul_tip', 'missed_bunt', 'bunt_foul_tip'
                 )                                                        THEN 'Swinging_Strike'
            WHEN description IN ('foul', 'foul_bunt')                    THEN 'Foul'
            WHEN description = 'hit_into_play'                           THEN 'In_Play'
            ELSE NULL
        END AS pitch_result
    FROM raw_statcast_pitches
    WHERE description NOT IN ('automatic_ball', 'automatic_strike')
)

SELECT
    -- identifiers (kept for joining/ordering, not features)
    game_pk,
    game_date,
    at_bat_number,
    pitch_number,
    batter,
    pitcher,

    -- flat features
    pitch_type,
    release_speed,
    release_spin_rate,
    pfx_x,
    pfx_z,
    plate_x,
    plate_z,
    zone,
    arm_angle,
    stand,
    p_throws,
    balls,
    strikes,
    outs_when_up,
    inning,
    bat_score,
    fld_score,
    bat_score_diff,
    home_team,
    IFF(on_1b IS NOT NULL, 1, 0)
        + IFF(on_2b IS NOT NULL, 1, 0)
        + IFF(on_3b IS NOT NULL, 1, 0)                       AS num_runners_on,
    IFF(on_2b IS NOT NULL OR on_3b IS NOT NULL, 1, 0)         AS runner_in_scoring_position,
    if_fielding_alignment,
    of_fielding_alignment,
    pitcher_days_since_prev_game,
    n_thruorder_pitcher,

    -- rolling features: recent (last 30 days)
    DIV0NULL(
        SUM(is_contact) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
        ),
        SUM(is_swing) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
        )
    ) AS contact_rate_recent_30d,
    DIV0NULL(
        SUM(is_swing) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
        )
    ) AS swing_rate_recent_30d,
    AVG(bat_speed) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
    ) AS avg_bat_speed_recent_30d,
    AVG(swing_length) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
    ) AS avg_swing_length_recent_30d,
    AVG(miss_distance) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
    ) AS avg_miss_distance_recent_30d,
    release_speed - AVG(bat_speed) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '1 day' PRECEDING
    ) AS bat_speed_velo_gap_recent_30d,

    -- rolling features: mid (31-100 days ago)
    DIV0NULL(
        SUM(is_contact) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '100 days' PRECEDING AND INTERVAL '31 days' PRECEDING
        ),
        SUM(is_swing) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '100 days' PRECEDING AND INTERVAL '31 days' PRECEDING
        )
    ) AS contact_rate_mid_31_100d,
    DIV0NULL(
        SUM(is_swing) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '100 days' PRECEDING AND INTERVAL '31 days' PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN INTERVAL '100 days' PRECEDING AND INTERVAL '31 days' PRECEDING
        )
    ) AS swing_rate_mid_31_100d,
    AVG(bat_speed) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '100 days' PRECEDING AND INTERVAL '31 days' PRECEDING
    ) AS avg_bat_speed_mid_31_100d,
    AVG(swing_length) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '100 days' PRECEDING AND INTERVAL '31 days' PRECEDING
    ) AS avg_swing_length_mid_31_100d,
    AVG(miss_distance) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '100 days' PRECEDING AND INTERVAL '31 days' PRECEDING
    ) AS avg_miss_distance_mid_31_100d,
    release_speed - AVG(bat_speed) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN INTERVAL '100 days' PRECEDING AND INTERVAL '31 days' PRECEDING
    ) AS bat_speed_velo_gap_mid_31_100d,

    -- rolling features: career (all prior pitches, pitch-ordered)
    DIV0NULL(
        SUM(is_contact) OVER (
            PARTITION BY batter
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        SUM(is_swing) OVER (
            PARTITION BY batter
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS contact_rate_career,
    DIV0NULL(
        SUM(is_swing) OVER (
            PARTITION BY batter
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY batter
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS swing_rate_career,
    AVG(bat_speed) OVER (
        PARTITION BY batter
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_bat_speed_career,
    AVG(swing_length) OVER (
        PARTITION BY batter
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_swing_length_career,
    AVG(miss_distance) OVER (
        PARTITION BY batter
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_miss_distance_career,
    release_speed - AVG(bat_speed) OVER (
        PARTITION BY batter
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS bat_speed_velo_gap_career,

    -- rolling features: distant (100+ days ago, season-to-date)
    DIV0NULL(
        SUM(is_contact) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '101 days' PRECEDING
        ),
        SUM(is_swing) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '101 days' PRECEDING
        )
    ) AS contact_rate_distant_100d_plus,
    DIV0NULL(
        SUM(is_swing) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '101 days' PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY batter ORDER BY game_date
            RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '101 days' PRECEDING
        )
    ) AS swing_rate_distant_100d_plus,
    AVG(bat_speed) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '101 days' PRECEDING
    ) AS avg_bat_speed_distant_100d_plus,
    AVG(swing_length) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '101 days' PRECEDING
    ) AS avg_swing_length_distant_100d_plus,
    AVG(miss_distance) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '101 days' PRECEDING
    ) AS avg_miss_distance_distant_100d_plus,
    release_speed - AVG(bat_speed) OVER (
        PARTITION BY batter ORDER BY game_date
        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '101 days' PRECEDING
    ) AS bat_speed_velo_gap_distant_100d_plus,

    -- pitcher pitch-mix tendency in this exact count, prior pitches only
    DIV0NULL(
        SUM(IFF(pitch_category = 'Fastball', 1, 0)) OVER (
            PARTITION BY pitcher, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY pitcher, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_fastball_rate_this_count,
    DIV0NULL(
        SUM(IFF(pitch_category = 'Breaking', 1, 0)) OVER (
            PARTITION BY pitcher, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY pitcher, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_breaking_rate_this_count,
    DIV0NULL(
        SUM(IFF(pitch_category = 'Offspeed', 1, 0)) OVER (
            PARTITION BY pitcher, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY pitcher, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_offspeed_rate_this_count,

    -- batter count-specific swing rate (pitch-level window — small per-count samples)
    DIV0NULL(
        SUM(is_swing) OVER (
            PARTITION BY batter, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY batter, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS batter_swing_rate_this_count,

    -- pitch sequencing: context from the previous pitch in this at-bat
    -- NULL on the first pitch of each at-bat (no prior pitch exists)
    LAG(pitch_category, 1) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
    ) AS prev_pitch_category,
    LAG(plate_x, 1) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
    ) AS prev_plate_x,
    LAG(plate_z, 1) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
    ) AS prev_plate_z,
    LAG(release_speed, 1) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
    ) AS prev_release_speed,
    LAG(pitch_result, 1) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
    ) AS prev_pitch_result,
    pitch_number - 1 AS pitches_in_ab,
    GREATEST(pitch_number - 1 - balls - strikes, 0) AS fouls_2strike_in_ab,
    COALESCE(SUM(IFF(pitch_category = 'Fastball', 1, 0)) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS fastballs_in_ab,
    COALESCE(SUM(IFF(pitch_category = 'Breaking', 1, 0)) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS breaking_in_ab,
    COALESCE(SUM(IFF(pitch_category = 'Offspeed', 1, 0)) OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS offspeed_in_ab,

    -- batter career miss distance by pitch type and zone context
    AVG(IFF(pitch_category = 'Fastball', miss_distance, NULL)) OVER (
        PARTITION BY batter
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_miss_distance_fastball_career,
    AVG(IFF(pitch_category = 'Breaking', miss_distance, NULL)) OVER (
        PARTITION BY batter
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_miss_distance_breaking_career,
    AVG(IFF(pitch_category = 'Offspeed', miss_distance, NULL)) OVER (
        PARTITION BY batter
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_miss_distance_offspeed_career,
    AVG(IFF(zone BETWEEN 1 AND 9, miss_distance, NULL)) OVER (
        PARTITION BY batter
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_miss_distance_in_zone_career,
    AVG(IFF(zone IN (11, 12, 13, 14), miss_distance, NULL)) OVER (
        PARTITION BY batter
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_miss_distance_chase_career,

    -- pitcher aggregate induced rates (all pitch types combined), prior pitches only
    DIV0NULL(
        SUM(is_swing) OVER (
            PARTITION BY pitcher
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY pitcher
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_induced_swing_rate,
    DIV0NULL(
        SUM(IFF(zone IN (11, 12, 13, 14) AND is_swing = 1, 1, 0)) OVER (
            PARTITION BY pitcher
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        SUM(IFF(zone IN (11, 12, 13, 14), 1, 0)) OVER (
            PARTITION BY pitcher
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_induced_chase_rate,
    DIV0NULL(
        SUM(IFF(is_swing = 1 AND is_contact = 1, 1, 0)) OVER (
            PARTITION BY pitcher
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        SUM(is_swing) OVER (
            PARTITION BY pitcher
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_induced_contact_rate,
    DIV0NULL(
        SUM(IFF(is_swing = 1 AND is_contact = 0, 1, 0)) OVER (
            PARTITION BY pitcher
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        SUM(is_swing) OVER (
            PARTITION BY pitcher
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_induced_miss_rate,

    -- pitcher x pitch type induced rates (this specific pitch category), prior pitches only
    DIV0NULL(
        SUM(is_swing) OVER (
            PARTITION BY pitcher, pitch_category
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY pitcher, pitch_category
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_swing_rate_this_pitch,
    DIV0NULL(
        SUM(IFF(zone IN (11, 12, 13, 14) AND is_swing = 1, 1, 0)) OVER (
            PARTITION BY pitcher, pitch_category
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        SUM(IFF(zone IN (11, 12, 13, 14), 1, 0)) OVER (
            PARTITION BY pitcher, pitch_category
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_chase_rate_this_pitch,
    DIV0NULL(
        SUM(IFF(is_swing = 1 AND is_contact = 1, 1, 0)) OVER (
            PARTITION BY pitcher, pitch_category
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        SUM(is_swing) OVER (
            PARTITION BY pitcher, pitch_category
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_contact_rate_this_pitch,
    DIV0NULL(
        SUM(IFF(is_swing = 1 AND is_contact = 0, 1, 0)) OVER (
            PARTITION BY pitcher, pitch_category
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        SUM(is_swing) OVER (
            PARTITION BY pitcher, pitch_category
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS pitcher_miss_rate_this_pitch,

    -- pitcher career average plate location by pitch type (pre-pitch proxy — no current-pitch location used)
    AVG(plate_x) OVER (
        PARTITION BY pitcher, pitch_type
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_plate_x_this_pitch_career,
    AVG(plate_z) OVER (
        PARTITION BY pitcher, pitch_type
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_plate_z_this_pitch_career,
    DIV0NULL(
        SUM(IFF(zone BETWEEN 1 AND 9, 1, 0)) OVER (
            PARTITION BY pitcher, pitch_type
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY pitcher, pitch_type
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS avg_in_zone_rate_this_pitch_career,

    -- pitcher career avg plate location by pitch type AND count (captures 3-0 vs 1-2 location tendencies)
    AVG(plate_x) OVER (
        PARTITION BY pitcher, pitch_type, balls, strikes
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_plate_x_this_pitch_this_count,
    AVG(plate_z) OVER (
        PARTITION BY pitcher, pitch_type, balls, strikes
        ORDER BY game_date, at_bat_number, pitch_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS avg_plate_z_this_pitch_this_count,
    DIV0NULL(
        SUM(IFF(zone BETWEEN 1 AND 9, 1, 0)) OVER (
            PARTITION BY pitcher, pitch_type, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        COUNT(*) OVER (
            PARTITION BY pitcher, pitch_type, balls, strikes
            ORDER BY game_date, at_bat_number, pitch_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    ) AS avg_in_zone_rate_this_pitch_this_count,

    -- kept around (not a model input) so the target can be regrouped
    -- ad hoc in Python without re-running this view
    description,

    -- target
    CASE
        WHEN description IN ('ball', 'blocked_ball', 'pitchout')   THEN 'Ball'
        WHEN description = 'called_strike'                         THEN 'Called_Strike'
        WHEN description IN (
                'swinging_strike', 'swinging_strike_blocked', 'foul_tip',
                'missed_bunt', 'bunt_foul_tip'
             )                                                       THEN 'Swinging_Strike'
        WHEN description IN ('foul', 'foul_bunt')                   THEN 'Foul'
        WHEN description = 'hit_into_play'                          THEN 'In_Play'
        WHEN description = 'hit_by_pitch'                           THEN 'Hit_By_Pitch'
        ELSE NULL
    END AS pitch_result_target

FROM flags
WHERE pitch_result_target IS NOT NULL
ORDER BY batter, game_date, at_bat_number, pitch_number;

-- Sanity checks after creating the view
SELECT COUNT(*) AS total_rows FROM vw_model1_features;
SELECT pitch_result_target, COUNT(*) FROM vw_model1_features GROUP BY pitch_result_target;
