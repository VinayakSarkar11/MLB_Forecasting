-- Run this in the Snowflake worksheet AFTER creating the IAM role in AWS.

USE ROLE ACCOUNTADMIN;
USE DATABASE STATCAST_DB;
USE SCHEMA MLB_RAW;
USE WAREHOUSE STATCAST_WH;

CREATE OR REPLACE STORAGE INTEGRATION s3_statcast_integration
    TYPE = EXTERNAL_STAGE
    STORAGE_PROVIDER = 'S3'
    ENABLED = TRUE
    STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::377028666088:role/snowflake-s3-role'
    STORAGE_ALLOWED_LOCATIONS = ('s3://statcast-surge-raw-data-vs/raw/');

-- Run this after creating the integration.
-- Copy STORAGE_AWS_IAM_USER_ARN and STORAGE_AWS_EXTERNAL_ID —
-- you'll need them to update the AWS IAM trust policy.
DESC INTEGRATION s3_statcast_integration;
