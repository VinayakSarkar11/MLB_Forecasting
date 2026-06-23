USE ROLE ACCOUNTADMIN;
USE DATABASE STATCAST_DB;
USE SCHEMA MLB_RAW;
USE WAREHOUSE STATCAST_WH;

-- Model 1 target — collapses `description` into a 6-class plate
-- discipline target.
--
-- Mapping decisions:
--   blocked_ball             -> Ball              (pitcher still missed the zone)
--   pitchout                 -> Ball
--   swinging_strike_blocked  -> Swinging_Strike    (batter swung and missed)
--   foul_tip                 -> Swinging_Strike    (caught foul tip is not an out
--                                                    on contact the way a normal
--                                                    foul is — closer to a swing-miss
--                                                    outcome than a live foul ball)
--   missed_bunt               -> Swinging_Strike
--   bunt_foul_tip             -> Swinging_Strike
--   foul_bunt                -> Foul
--   hit_by_pitch              -> Hit_By_Pitch       (separate class — not a plate
--                                                     discipline decision by either player)
--   automatic_ball/strike     -> NULL, excluded     (no pitch was actually thrown,
--                                                     e.g. pitch timer violation)

SELECT
    game_pk,
    game_date,
    at_bat_number,
    pitch_number,
    batter,
    pitcher,
    description,

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
        WHEN description IN ('automatic_ball', 'automatic_strike')  THEN NULL
        ELSE NULL
    END AS pitch_result_target

FROM raw_statcast_pitches
WHERE description NOT IN ('automatic_ball', 'automatic_strike')
ORDER BY batter, game_date, at_bat_number, pitch_number;
