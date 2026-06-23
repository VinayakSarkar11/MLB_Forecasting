USE ROLE ACCOUNTADMIN;
USE DATABASE STATCAST_DB;
USE SCHEMA MLB_RAW;
USE WAREHOUSE STATCAST_WH;

-- Model 1 — rolling batter tendencies (contact rate, swing rate, swing
-- mechanics) split into three NON-OVERLAPPING windows, a bat-speed vs.
-- pitch-velocity interaction term per window, and a pitcher pitch-mix
-- tendency by count.
--
-- Batter windows:
--   recent:  last 30 days
--   mid:     31-100 days ago
--   distant: everything before 100 days ago (season-to-date)
-- ORDER BY uses game_date only (not at_bat/pitch number), so every pitch
-- within the same calendar day shares the same trailing snapshot computed
-- as of the START of that day — leakage-safe without pitch-level
-- tiebreakers.
--
-- Pitcher pitch-mix window:
-- Partitioned by pitcher AND the exact count (balls, strikes), since the
-- question is "what does this pitcher tend to throw in THIS count" — uses
-- pitch-level ordering (at_bat/pitch number) since the per-count sample
-- size is much smaller than a full day of pitches.
--
-- NULLs are left as NULLs (no COALESCE fallback) — XGBoost/LightGBM both
-- handle missing values natively by learning a default split direction.
--
-- Note: Snowflake does not support the standard SQL named WINDOW clause,
-- so each OVER(...) below is written out in full rather than reused by name.

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
        END AS pitch_category
    FROM raw_statcast_pitches
),

engineered AS (
    SELECT
        game_pk,
        game_date,
        at_bat_number,
        pitch_number,
        batter,
        pitcher,
        balls,
        strikes,
        release_speed,

        -- recent: last 30 days
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

        -- mid: 31-100 days ago
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

        -- distant: 100+ days ago
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
        ) AS pitcher_offspeed_rate_this_count

    FROM flags
)

SELECT
    game_pk,
    game_date,
    at_bat_number,
    pitch_number,
    batter,
    pitcher,

    contact_rate_recent_30d,
    swing_rate_recent_30d,
    avg_bat_speed_recent_30d,
    avg_swing_length_recent_30d,
    avg_miss_distance_recent_30d,
    -- interaction: gap between this pitch's velocity and the batter's
    -- historical bat speed in this window — bigger gap implies the batter's
    -- bat may not catch up to pitches at this speed
    release_speed - avg_bat_speed_recent_30d AS bat_speed_velo_gap_recent_30d,

    contact_rate_mid_31_100d,
    swing_rate_mid_31_100d,
    avg_bat_speed_mid_31_100d,
    avg_swing_length_mid_31_100d,
    avg_miss_distance_mid_31_100d,
    release_speed - avg_bat_speed_mid_31_100d AS bat_speed_velo_gap_mid_31_100d,

    contact_rate_distant_100d_plus,
    swing_rate_distant_100d_plus,
    avg_bat_speed_distant_100d_plus,
    avg_swing_length_distant_100d_plus,
    avg_miss_distance_distant_100d_plus,
    release_speed - avg_bat_speed_distant_100d_plus AS bat_speed_velo_gap_distant_100d_plus,

    pitcher_fastball_rate_this_count,
    pitcher_breaking_rate_this_count,
    pitcher_offspeed_rate_this_count

FROM engineered
ORDER BY batter, game_date, at_bat_number, pitch_number;
