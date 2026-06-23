"""
End-to-end season ingestion: pybaseball → local CSV → S3 → Snowflake COPY INTO.

Pulls data month by month to keep memory manageable. Already-downloaded months
are skipped locally; Snowflake's COPY INTO skips already-loaded S3 files.

Usage:
    python -m src.ingestion.ingest_season --year 2023
    python -m src.ingestion.ingest_season --year 2023 --skip-extract
    python -m src.ingestion.ingest_season --year 2023 --skip-extract --skip-upload
"""

import argparse
import calendar
import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv
from pybaseball import statcast, cache

from src.ml.snowflake_client import get_connection

load_dotenv()
cache.enable()

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
S3_PREFIX = "raw"

SEASON_DATES = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
}


def extract_season(year: int) -> None:
    start_str, end_str = SEASON_DATES[year]
    print(f"Pulling Statcast data for {year} ({start_str} → {end_str})...")

    start_dt = pd.Timestamp(start_str)
    end_dt = pd.Timestamp(end_str)
    current = start_dt.replace(day=1)

    while current <= end_dt:
        month_last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = min(current.replace(day=month_last_day), end_dt)

        out_dir = RAW_DIR / f"year={year}" / f"month={current.month:02d}"
        out_path = out_dir / f"statcast_{year}_{current.month:02d}.csv"

        if out_path.exists():
            print(f"  {out_path.name} already exists locally — skipping.")
        else:
            print(f"  Fetching {current.strftime('%Y-%m')}...", end=" ", flush=True)
            df = statcast(
                start_dt=current.strftime("%Y-%m-%d"),
                end_dt=month_end.strftime("%Y-%m-%d"),
            )
            if df.empty:
                print("no data returned.")
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_path, index=False)
                print(f"{len(df):,} rows → {out_path.relative_to(RAW_DIR.parent.parent)}")

        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)

    print(f"Extraction complete for {year}.\n")


def upload_to_s3(year: int) -> None:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"],
    )

    year_dir = RAW_DIR / f"year={year}"
    csv_files = sorted(year_dir.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found under {year_dir}. Run extraction first.")
        return

    print(f"Uploading {len(csv_files)} file(s) for {year} to S3...")
    for local_path in csv_files:
        relative = local_path.relative_to(RAW_DIR)
        s3_key = f"{S3_PREFIX}/{relative}"
        print(f"  {relative} → s3://{S3_BUCKET}/{s3_key}")
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)

    print(f"S3 upload complete.\n")


def copy_into_snowflake() -> None:
    print("Running COPY INTO in Snowflake (already-loaded files are skipped automatically)...")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            COPY INTO raw_statcast_pitches
            FROM @statcast_s3_stage
            PATTERN = '.*\\.csv'
            FILE_FORMAT = (
                TYPE = 'CSV'
                FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                SKIP_HEADER = 1
                NULL_IF = ('', 'NULL', 'null', 'None')
                EMPTY_FIELD_AS_NULL = TRUE
            )
            ON_ERROR = 'CONTINUE'
        """)
        rows = cur.fetchall()
        loaded = sum(1 for r in rows if r[1] == "LOADED")
        skipped = sum(1 for r in rows if r[1] == "LOAD_SKIPPED")
        errors = [r for r in rows if r[1] not in ("LOADED", "LOAD_SKIPPED")]
        print(f"  Files loaded: {loaded} | already loaded (skipped): {skipped} | errors: {len(errors)}")
        if errors:
            for r in errors[:5]:
                print(f"    ERROR: {r[0]} — {r[2]}")

        cur.execute("SELECT COUNT(*) FROM raw_statcast_pitches")
        total = cur.fetchone()[0]
        print(f"  Total rows in raw_statcast_pitches: {total:,}\n")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest an MLB season into Snowflake.")
    parser.add_argument("--year", type=int, required=True, choices=list(SEASON_DATES.keys()),
                        help="Season year to ingest")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip pybaseball download (CSVs must already exist locally)")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip S3 upload (files must already be in S3)")
    args = parser.parse_args()

    if not args.skip_extract:
        extract_season(args.year)

    if not args.skip_upload:
        upload_to_s3(args.year)

    copy_into_snowflake()

    print("Done. Re-run 08_model1_final_view.sql in Snowflake to refresh the view with the new data.")


if __name__ == "__main__":
    main()
