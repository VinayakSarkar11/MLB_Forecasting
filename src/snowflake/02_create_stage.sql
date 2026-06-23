-- Run this AFTER updating the AWS IAM trust policy.

USE ROLE ACCOUNTADMIN;
USE DATABASE STATCAST_DB;
USE SCHEMA MLB_RAW;
USE WAREHOUSE STATCAST_WH;

CREATE OR REPLACE STAGE statcast_s3_stage
    STORAGE_INTEGRATION = s3_statcast_integration
    URL = 's3://statcast-surge-raw-data-vs/raw/'
    FILE_FORMAT = (
        TYPE = 'CSV'
        FIELD_OPTIONALLY_ENCLOSED_BY = '"'
        SKIP_HEADER = 1
        NULL_IF = ('', 'NULL', 'null', 'None')
        EMPTY_FIELD_AS_NULL = TRUE
    );

-- Verify Snowflake can see the files in S3
LIST @statcast_s3_stage;
