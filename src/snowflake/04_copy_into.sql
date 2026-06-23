USE ROLE ACCOUNTADMIN;
USE DATABASE STATCAST_DB;
USE SCHEMA MLB_RAW;
USE WAREHOUSE STATCAST_WH;

COPY INTO raw_statcast_pitches
FROM @statcast_s3_stage
PATTERN = '.*\.csv'
FILE_FORMAT = (
    TYPE = 'CSV'
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
    NULL_IF = ('', 'NULL', 'null', 'None')
    EMPTY_FIELD_AS_NULL = TRUE
)
ON_ERROR = 'CONTINUE';

-- Verify row count after load
SELECT COUNT(*) AS total_pitches FROM raw_statcast_pitches;
